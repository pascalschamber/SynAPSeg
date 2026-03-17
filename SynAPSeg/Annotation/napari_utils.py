import re
import os
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) # add parent dir to sys.path
from utils import utils_general as ug
from utils import utils_image_processing as uip


import napari
import numpy as np

def dat(layer_name):
    """ convienence func that returns current viewer.layers[layer_name].data """
    return napari.current_viewer().layers[layer_name].data

def make_standard_vispy_cmaps():
    """ returns a dict which maps color names to a vispy cmap """
    import vispy.color

    return dict(
        red = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        green = vispy.color.Colormap([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        blue = vispy.color.Colormap([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        magenta = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 0.0, 1.0]]),
        gray = vispy.color.Colormap([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]),
    )


def get_napari_cmaps():
    
    std_cmaps = make_standard_vispy_cmaps()
    color_names = ['blue', 'green', 'red', 'magenta', 'gray']
    cmaps = {i: (color_names[i], std_cmaps[color_names[i]]) for i in range(len(color_names))} # same as below 
    # cmaps = dict(zip([0,1,2,3,4], [('blue', blue),('green', green),('red', red),('magenta', magenta),('gray',gray)]))
    return cmaps

from napari.utils import colormaps, color

viridis_hilo_cmap = colormaps.Colormap(
    colors=np.array(colormaps.ALL_COLORMAPS['viridis'].colors),
    name='viridis_HiLo',
    controls=colormaps.ALL_COLORMAPS['viridis'].controls,
    high_color = color.ColorValue(np.array([1, 0, 0, 1])),
    low_color = color.ColorValue(np.array([0, 0.55, 0.55, 1])),
)

def add_custom_colormaps_to_napari():
    to_add = [viridis_hilo_cmap]
    for c in to_add:
        napari.utils.colormaps.ALL_COLORMAPS[c.name] = c
        napari.utils.colormaps.AVAILABLE_COLORMAPS[c.name] = c

def get_single_color_cmap(color=[1,0,0,1]):
    """ get a cyclic cmap with 1 color for everything not == 0 """
    from napari.utils import CyclicLabelColormap

    return CyclicLabelColormap(
        colors=[[0,0,0,0], color],       # just red
        display_name='all_red',   # shown in the colormap dropdown
        name='red_cycle',         # internal name
        background_value=0        # map 0 → transparent
    )

def delete_layers(viewer, layers_to_remove):
    """ delete layers from viewer """
    for ln in layers_to_remove:
        if ln in viewer.layers:
            del(viewer.layers[ln])


def build_filemap(_FILE_MAP, get_contents_fxn, EXAMPLES_BASE_DIR, EXAMPLE_I):
    # pattern match filenames
    FILE_MAP = {k:[] for k in _FILE_MAP}
    get_image_list = []

    for imgtype, patterns in _FILE_MAP.items():
        for pattern in patterns:
            recur = True if get_contents_fxn.__name__ == 'get_contents_recursive' else False
            if get_contents_fxn.__name__ == 'get_contents': kwargs = dict(filter_str=pattern, pattern=True) 
            elif recur: kwargs = dict(pattern=pattern) 
            else: kwargs = {}

            fns = get_contents_fxn(EXAMPLES_BASE_DIR, **kwargs)
            filename = Path(fns[EXAMPLE_I]).name if not recur else fns[EXAMPLE_I].replace(EXAMPLES_BASE_DIR, '').strip('\\')
            get_image_list.append(filename)
            FILE_MAP[imgtype].append(filename)
    return FILE_MAP, get_image_list


def simple_viewer(
        image, 
        _FILE_MAP={
            'images':['.*img.tiff'], # pattern match filenames
            'labels':['.*mask.tiff'],
        }, 
        IMG_FORMAT='ZYX', 
        LABEL_INT_MAP={},
        get_contents_fxn = ug.get_contents,
    ):
    """ create a viewer to view images in a folder 
        
        Args
        ``````````
        EXAMPLE_I: index of image(s) in folder
        EXAMPLES_BASE_DIR: path to image folder
        _FILE_MAP = {
                'images':['.*img.tiff'], # pattern match filenames
                'labels':['.*mask.tiff'],
            }
        IMG_FORMAT: string mapping img axes to e.g. STCZYX dims
    """
    import numpy as np
    from utils import utils_image_processing as uip
    
    from Annotation.annotation_core import create_napari_viewer

    def add_img(im, img_name, IMG_FORMAT):
        if im.ndim==2:
            im = im[np.newaxis]
        image_dict[img_name] = im
        ndims = im.ndim
        exmd['data_metadata']['data_shapes'][img_name] = im.shape
        exmd['data_metadata']['data_formats'][img_name] = IMG_FORMAT[-ndims:]
    
    

    # init vars
    ###################################################################################
    _FILE_MAP = _FILE_MAP or {'images':['.*.tiff?']}
    get_image_list = [] # list of Path(filepath).name
    

    # load data, build metadata dict
    image_dict, exmd = {}, {
        'data_metadata':{'data_shapes':{}, 'data_formats':{}},
        'annotation_metadata': {'notes':''},
    }

    # handle different inputs for image
    ###################################################################################
    _ARRAY_INPUT, _INPUT_DICT = False, False # flags
    if isinstance(image, tuple):
        EXAMPLE_I, EXAMPLES_BASE_DIR = image
        FILE_MAP, get_image_list = build_filemap(_FILE_MAP, get_contents_fxn, EXAMPLES_BASE_DIR, EXAMPLE_I)
    elif isinstance(image, str):
        get_image_list = [image]
    elif isinstance(image, np.ndarray):
        _ARRAY_INPUT = True
        get_image_list = ['image.tiff']
        EXAMPLES_BASE_DIR = os.getcwd()
        FILE_MAP = {'images':['image.tiff']}
    elif isinstance(image, dict):
        # keys are image names, values are dict of img, fmt, type
        _INPUT_DICT = True
        md = image.pop('__meta__')
        EXAMPLES_BASE_DIR = md['EXAMPLES_BASE_DIR']
        FILE_MAP = md['FILE_MAP']
        get_image_list = md['get_image_list']

    else:
        raise ValueError(type(image))

    # load imgs
    ###################################################################################
    if _ARRAY_INPUT:
        add_img(image, 'image', IMG_FORMAT)
    if _INPUT_DICT:
        for img_name, img_dict in image.items():
            add_img(img_dict['array'], img_name, img_dict['format'])
    else:
        for fn in get_image_list:
            img_name = ug.get_prefix(fn)
            im = uip.imread(os.path.join(EXAMPLES_BASE_DIR, fn))
            im, _IMG_FORMAT = uip.collapse_singleton_dims(im, IMG_FORMAT[-im.ndim:])
            add_img(im, img_name, _IMG_FORMAT)
        
    image_dict['metadata'] = exmd

    for k,v in image_dict.items():
        print(k)
        if isinstance(v, np.ndarray):
            print(v.shape)


    # create napari viewer
    ###################################################################################
    viewer, widget_objects = create_napari_viewer(
        exmd, LABEL_INT_MAP, EXAMPLES_BASE_DIR, FILE_MAP, image_dict, get_image_list
    )
    return viewer, widget_objects, exmd
        

def _process_proj_contrain_objects_to_polygon_boundary(project):
    """ temp - not implemented """
    project.filter_example_contents(
        required_filenames = ['annotated_PYR_layers.tiff', 'HPC_boundary.csv', 'annotated_pred_stardist_ch0.tiff', 'annotated_pred_stardist_ch1.tiff', 'annotated_pred_stardist_ch2.tiff']
    )

    from shapely.geometry import Polygon as shapelyPolygon
    from IO.writers import write_array
    from IO.metadata_handler import MetadataParser

    boundary_shapes_name = 'HPC_boundary'
    obj_layer_name_base = 'annotated_pred_stardist_ch'
    outname = 'annotated_soma.tiff'
    outformat = 'CYX'
    get_chs = [0,1,2]

    for ex in project.examples:
        boundary_coords_df = pd.read_csv(os.path.join(ex.path_to_example, f"{boundary_shapes_name}.csv"))
        boundary_coords = boundary_coords_df[['axis-0', 'axis-1']].values
        boundary_polygon = shapelyPolygon(boundary_coords)
        polygons_per_label = {1:[boundary_polygon]}
        
        filtered_result = []
        for ch in get_chs:
            obj_layer_name = f'{obj_layer_name_base}{ch}'
            # lbl_img = viewer.layers[obj_layer_name].data
            lbl_img = uip.imread(os.path.join(ex.path_to_example, f"{obj_layer_name}.tiff"))
            lbl_img = uip.relabel(lbl_img)
            rpdf = uc.get_rp_table(lbl_img, np.zeros_like(lbl_img, dtype='uint8'))

            centroids_list = rpdf['centroid'].to_list()
            # centroids_list = np.array([np.array(c[::-1]) for c in centroids_list])

            assigned_ids_centroid, poly_subindices_centroid = uc.assign_labels(
                centroids_list, polygons_per_label
            )
            rpdf['roi_i'] = assigned_ids_centroid

            # filt labels
            keep_labels = rpdf[rpdf['roi_i']==1]['label'].to_list()
            lbl_img_filtered = uip.filter_label_img(lbl_img, keep_labels)
            filtered_result.append(lbl_img_filtered)
        
        filtered_result = np.stack(filtered_result, 0)
        # viewer.add_labels(lbl_img_filtered, name=f"filt_{obj_layer_name}")

        # write 
        write_array(filtered_result, ex.path_to_example, outname, ex_md=ex.exmd, fmt=outformat)
        MetadataParser.write_metadata(ex.path_to_example, ex.exmd)




