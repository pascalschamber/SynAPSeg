import os
import shutil
import yaml
import numpy as np
import tifffile
import skimage.transform
import itertools


from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils import utils_czi as uCzi
from SynAPSeg.IO.writers import write_array
from SynAPSeg.IO.metadata_handler import MetadataParser


_STANDARD_FORMAT = "STCZYX"
_DISPLAY_FORMAT = "CZYX"
_RAW_IMG_NAME = 'raw' # name key of raw intensity image


# Main function
def write_output(
        ex_md, arr, predictions, pipeline, RUN_CONFIG, image_obj, skip_arr_names=None, 
        generate_summary=True, generate_mip=True, 
        overwrite_raw_img=True,
    ):
    """ Main function to write output images, metadata, and create thumbnail.
    """

    # Save previous results
    SEG_RESULTS_PATH = save_previous_segmentation_results(ex_md['output_dir'])

    # Generate image grid and create and thumbnail
    if generate_summary:
        mip_thumbnail = generate_image_grid(
            get_display_image_dict(arr, predictions, pipeline), pipeline, SEG_RESULTS_PATH, RUN_CONFIG, ex_md, image_obj
        )
    else:
        mip_thumbnail = None

    # write raw image array and predictions to disk - Save images and metadata
    OUTPUT_IMAGE_PYRAMID = RUN_CONFIG.get('OUTPUT_IMAGE_PYRAMID', False) # only applies for raw_img

    save_images_and_metadata(
        ex_md,
        arr,
        predictions,
        mip_thumbnail,
        ex_md["output_dir"],
        skip_files_str=skip_arr_names,
        generate_mip=generate_mip,
        OUTPUT_IMAGE_PYRAMID=OUTPUT_IMAGE_PYRAMID,
        overwrite_raw_img=overwrite_raw_img,
    )


def get_display_image_dict(arr, predictions, pipeline):
    """ merge raw image and prediction results into a dict and convert the predictions to display format """
    return ug.merge_dicts(
            {_RAW_IMG_NAME: to_display_format(arr, _STANDARD_FORMAT)}, 
            predictions_to_display_format(pipeline, predictions)
    )


def get_image_grid_cmaps(ex_md, image_obj, RUN_CONFIG, n_ch):
    
    # Colormap retrieval and processing
    cmapss = get_colormaps(ex_md, image_obj, RUN_CONFIG)
    cm = dict(zip(ex_md['data_metadata']['present_chs'], cmapss))
    fig_cmaps = update_colormap_with_null_channels(cm, ex_md, RUN_CONFIG)
        
    # ensure won't fail if wrong n cmaps supplied
    cyc = itertools.cycle(fig_cmaps)
    comp_img_cmaps = [cyc.__next__() for _ in range(n_ch)]
    return comp_img_cmaps


def downscale_if_large(image, max_area=1e6):
    """ 
    resize summary imgs to prevent wasted time in summary img generation for really large images 
    expects img format of YX
    Resize the image if its area (height * width) exceeds max_area.

    Parameters:
      image (ndarray): Input image (can be grayscale or color) as a NumPy array.
      max_area (float): Maximum allowed area (default 1e6 pixels).

    Returns:
      ndarray: The original image if its area is within max_area,
               or the resized image otherwise.
    """
    # Get the original dimensions (assumes image shape is (H, W) or (H, W, C))
    original_height, original_width = image.shape[:2]
    
    # Calculate the current area
    current_area = original_height * original_width
    
    # Only resize if the area exceeds the max_area threshold
    if current_area > max_area:
        # print(f'downscaling summary image size: {current_area} > {max_area}')
        
        # Compute the scaling factor such that the new area is max_area
        scale = np.sqrt(max_area / current_area)
        # Calculate new dimensions while maintaining aspect ratio
        new_height = max(1, int(round(original_height * scale)))
        new_width = max(1, int(round(original_width * scale)))
        output_shape = (new_height, new_width)
        
        # infer order from dtype - asume int32 is for labels
        resize_order = 0 if image.dtype in [np.int32, np.bool_] else 1
        anti_aliasing = False if resize_order == 0 else True
        # Resize using skimage.transform.resize with given kwargs
        resized_image = skimage.transform.resize(image,
                                order = resize_order,
                               output_shape=output_shape,
                               anti_aliasing=anti_aliasing,
                               preserve_range=True)
        # Optionally, convert back to the original data type
        return resized_image.astype(image.dtype)
    
    # Return the original image if no resizing is needed
    return image


def generate_image_grid(display_images, pipeline, SEG_RESULTS_PATH, RUN_CONFIG, ex_md, image_obj):
    """
    Build and save a result image grid.
    """
    thumbnail_img = np.zeros((64,64,3)) # random init
    C = max([a.shape[0] for a in display_images.values()]) + 1 # last col shows composite of image or segmentation outputs
    R = len(display_images) 
    grid_shape = (R, C)
    disp_img_list = []
    
    # get cmaps
    comp_img_cmaps = get_image_grid_cmaps(ex_md, image_obj, RUN_CONFIG, C)
    
    # init grid titles as array
    grid_titles = np.full(grid_shape, '', dtype='<U2048')
    
    disp_img_i = 0 # current_index of display image in list
    for r, (image_name, _arr) in enumerate(display_images.items()):
        for c in range(_arr.shape[0]):
            # build grid titles
            arr_ch = _arr[c]
            max_str = f" max=({arr_ch.max()})" if _arr.dtype == np.int32 else ''
            grid_titles[r, c] = f"{image_name} ch{c}{max_str}"
            
            # convert to z mip
            arr_mip = uip.mip(arr_ch, 0)
            # for large images, resize first since resolution will not matter for summary image
            arr_mip = downscale_if_large(arr_mip, max_area=2048**2)
            
            # prepare label predictions for display
            if (_arr.dtype == np.int32) and (image_name in pipeline.model_names):
                model_input_str = pipeline.get_model(image_name).get('model_input', 'raw') # get name of input to model
                # get idx of grid title and intensity img
                input_img_key = f"{model_input_str} ch{c}"
                int_img_idx = np.argwhere(grid_titles==input_img_key)[0]
                int_img_list_idx = int_img_idx[0]*C + int_img_idx[1]
                # display segmentation result as outlines overlayed on intensity img
                as_outlines = uip.mask_to_outlines(arr_mip)
                disp_img = up.overlay(disp_img_list[int_img_list_idx], as_outlines, 'red', (255,0,0))
                
            # normalize intensity image for display
            else:
                disp_img = (uip.normalize_01(arr_mip)*255).astype('uint8')
                disp_img = np.repeat(disp_img[..., np.newaxis], 3, -1)
                
            disp_img_list.append(disp_img)
            disp_img_i += 1
            
        # at end of col add composite image of channels
        grid_titles[r, -1] = f"{image_name} composite"
        ch_imgs = np.moveaxis(uip.mip(display_images[image_name], 1), 0, -1)
        
        if _arr.dtype == np.int32:
            # make a sum stack of segmentation channels, where each ch is assigned a uid
            as_bin = np.where(ch_imgs>0,1,0)
            as_labels = as_bin * np.arange(1, as_bin.shape[-1]+1)[np.newaxis, np.newaxis, :]
            comp_img = np.sum(as_labels, axis=-1).astype('int32')            
        else: 
            # apply arbitrary color mapping 
            comp_img = (up.create_composite_image_with_colormaps(ch_imgs, comp_img_cmaps)*255).astype('uint8')
        
        disp_img_list.append(comp_img)
        disp_img_i += 1

        if image_name == 'raw':
            # also grab composite image of the raw image to create thumbnail
            # resize so largest dim is <= 512  
            
            scale_factor = max(np.array(comp_img.shape[:-1])/512)
            target_shape = np.array(comp_img.shape[:-1])/scale_factor
            resized = uip.resize_by_format(comp_img, 'YXC', tuple(target_shape.astype('int')))
            thumbnail_img = (resized * 255).astype('uint8')
            
            # thumbnail_img = skimage.transform.resize(comp_img, output_shape=(512,512,3), anti_aliasing=True, preserve_range=True).astype('uint8')
            
    # plot the grid    
    up.plot_image_grid(
        disp_img_list, titles=grid_titles.flatten(), n_cols=C, n_rows=R,
        dpi=RUN_CONFIG.get('init_segmentation_results_dpi', 300),
        outpath = (SEG_RESULTS_PATH if not RUN_CONFIG['TESTING'] else None),
        noshow = (not RUN_CONFIG.get('SHOW_FIGURES', False)) if not RUN_CONFIG['TESTING'] else False,
    )
    
    return thumbnail_img


# Helper functions
def get_colormaps(ex_md, image_obj, RUN_CONFIG):
    """
    Retrieve colormaps for visualization based on metadata or use defaults.
    """
    # get default cmaps
    _default_cmaps = RUN_CONFIG['THUMBNAIL_CMAPS_DEFAULT']

    if len(ex_md['image_metadata']['ch_wavelengths']) == 0:
        return _default_cmaps
    else:
        try:
            return uCzi.get_cmaps_czi_intensity(image_obj, CMAPS_DEFAULT=_default_cmaps)
        except:
            return _default_cmaps

def update_colormap_with_null_channels(cm, ex_md, RUN_CONFIG):
    """
    Update colormaps to account for null channels.
    """
    if ex_md['data_metadata']['inserted_null_chs'] is not None:
        for chi in ex_md['data_metadata']['inserted_null_chs']:
            cm[chi] = RUN_CONFIG['THUMBNAIL_CMAPS_DEFAULT'][chi]
    return [el[1] for el in sorted(cm.items(), key=lambda x: x[0])]

def save_previous_segmentation_results(output_dir):
    """
    Save previous segmentation results by moving them to a 'previous' folder.
    """
    SEG_RESULTS_PATH = os.path.join(output_dir, 'init_segmentation_results.pdf')
    if os.path.exists(SEG_RESULTS_PATH):
        prev_res_dir = ug.verify_outputdir(os.path.join(output_dir, 'previous_segmentation_results'))
        shutil.move(SEG_RESULTS_PATH, os.path.join(prev_res_dir, f"init_segmentation_results_replaced_{ug.get_datetime()}.pdf"))
    return SEG_RESULTS_PATH


def save_images_and_metadata(ex_md, arr, predictions, mip_thumbnail=None, output_dir='', skip_files_str=None, generate_mip=True, OUTPUT_IMAGE_PYRAMID=False,
    overwrite_raw_img = True,
    ):
    """
    Save generated images and metadata to disk.
    """    
    # write thumbnail
    if mip_thumbnail is not None: 
        from PIL import Image
        im = Image.fromarray(mip_thumbnail)
        im.save(os.path.join(output_dir, f"thumbnail.png"))
    
    # write images      
    out_names = list(predictions.keys()) + ['raw_img']
    # ug.merge_dicts({'raw_img': arr}, predictions).items()
    
    for output_base_name in out_names:
        _ISRAW = (output_base_name == 'raw_img') # flag for processing the raw image
        outarr = arr if _ISRAW else predictions[output_base_name]

        if _ISRAW and not overwrite_raw_img:
            if os.path.exists(os.path.join(output_dir, 'raw_img.tiff')):
                continue

        if (skip_files_str is not None and skip_files_str in output_name) or outarr is None:
            print(f"found output to skip, not writing {output_base_name} to disk.")
            continue
        
        if not isinstance(outarr, list): # e.g. predictions are default returned as list, but raw img isnt
            outarr = [outarr] 
            
        n_items = len(outarr)
        is_pred = f"pred_" if output_base_name in predictions else ""
        
        for a_i, subarr in enumerate(outarr):
            # print('in save_images_and_metadata:', output_base_name, a_i, subarr.shape)
            i_list = "" if n_items==1 else f"_i{a_i}"
            output_name = f"{is_pred}{output_base_name}{i_list}"
            write_array(subarr, output_dir, output_name, _STANDARD_FORMAT, ex_md, OUTPUT_IMAGE_PYRAMID=(OUTPUT_IMAGE_PYRAMID and _ISRAW))
            
            # save a mip of raw_img, reducing any singleton dimensions as well - point is to make image easy to view in e.g. imagej
            if generate_mip and _ISRAW:
                mip_project_dims = "".join([d for d in ex_md['current_format'] if d not in 'CXY']) # project all dims except CXY when creating the mip
                raw_mip = uip.reduce_dimensions(arr, ex_md['current_format'], project_dims=mip_project_dims)
                mip_fmt = uip.subtract_dimstr(ex_md['current_format'], mip_project_dims)
                write_array(raw_mip, output_dir, f"mip_raw{i_list}", mip_fmt, ex_md)
    
    # handle archival and merging if example metadata already exists 
    ex_md = MetadataParser.maybe_merge_exmd(output_dir, ex_md, uuids=['image_path', 'scene_id', 'scene_name'])

    MetadataParser.write_metadata(output_dir, ex_md)
        
    print(f"results saved to: {output_dir}")


def to_display_format(arr, current_format):
    # first standardize the format
    arr = uip.transform_axes(arr, current_format, _STANDARD_FORMAT)
    # format to display dims
    take_dims = "".join([d for d in _STANDARD_FORMAT if d not in _DISPLAY_FORMAT])
    return uip.reduce_dimensions(arr, current_format=_STANDARD_FORMAT, take_dims=take_dims)


def predictions_to_display_format(pipeline, predictions):
    formatted = {}
    for model_name, model in pipeline.models:
        # display first output only
        formatted[model_name] = to_display_format(predictions[model_name][0], model.get_output_dims())
    return formatted


def show_pipeline_results(predictions, pipeline):
    """ coerce predicition results to a standardized format ('CZYX') so they can be further reduced and displayed as 2d images"""
    for name, result in predictions.items():
        dims = pipeline.get_model(name).get_output_dims()
        display_array = uip.transform_axes(result[0], dims, _STANDARD_FORMAT)   
        display_array = uip.reduce_dimensions(display_array, current_format=_STANDARD_FORMAT, take_dims='ST')       
        uip.pai(display_array)
        up.show_ch(display_array, axis=0)


def show_sd_results(chs, x1,x2,y1,y2, label_seg_input, sd_predictions, arr_mip):
    for i in chs:
        up.show(label_seg_input[0, x1:x2, y1:y2, i]) 
        up.show(sd_predictions[i][x1:x2, y1:y2])
        up.show(up.overlay_colored_outlines(sd_predictions[i][x1:x2, y1:y2], label_seg_input[0, x1:x2, y1:y2, i], 0.5))
        up.show(up.overlay_colored_outlines(sd_predictions[i][x1:x2, y1:y2], uip.norm_percentile(arr_mip[i, x1:x2, y1:y2], (1,99.8), None), 0.5))
