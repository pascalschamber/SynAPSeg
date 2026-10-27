from __future__ import annotations
from typing import Optional
import numpy as np
import os
import sys
import pandas as pd

from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.config.constants import STANDARD_FORMAT
from SynAPSeg.IO.metadata_handler import MetadataParser
from SynAPSeg.IO.project import Project, Example

import logging
logger = logging.getLogger("SynAPSeg")
logger.info("annotation_IO started.")

from copy import deepcopy
from typing import Optional, Dict, List, Tuple, Any

def _test():
    project = Project(r"D:\BygraveLab\Confocal data archive\Pascal\SEGMENTATION_DATASETS\VHL_VglutHomer")
    example = project.get_example("0000")
    include_only = ['raw_img.tiff', 'pred_stardist.tiff']
    exclude = None
    add_to_file_map = None
    fail_on_format_error = False
    get_label_int_map = False

    label_int_map, validated_filemap, image_dict, loaded_filenames = load_example_images(
        example,
        include_only=include_only,
        exclude=exclude,
        add_to_file_map=add_to_file_map,
        fail_on_format_error=fail_on_format_error,
        get_label_int_map=get_label_int_map,
        use_prefix_as_key=False,
    )

    exmd, path_to_example = image_dict.pop('metadata'), example.path_to_example

    inverted_file_map = {vv: k for k, v in validated_filemap.items() for vv in v}


    images_to_display, viewer_kwargs = preproc_images_for_display(
        exmd, validated_filemap, image_dict, loaded_filenames,
    )


def load_example_images(
    ex: Example,
    include_only: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    add_to_file_map: Optional[Dict] = None,
    fail_on_format_error: bool = False,
    get_label_int_map: bool = False,
    use_prefix_as_key: bool = True,
) -> tuple[dict, dict[str, list[str]], dict[str, Any], list[str]]:
    """
    Loads project example data for the annotator viewer.

    Args:
        use_prefix_as_key: Whether to use the prefix of the file name as the key.
            # TODO: this currently exists for compatibility with old code, but should be removed
            # need to update all code that calls this function to use the full file name as the key
    
    Returns:
        tuple: (label_int_map, validated_filemap, image_dict, loaded_filenames)
    """
    # 1. Setup Configuration
    # ---------------------------------------------------------
    type_patterns = {
        'ignore': ['thumbnail.png'],
        'shapeLayer_ROIs': [r'.*\.csv$', r'.*\.geojson$'], # polygons exported from napari `shapes` layer
        'images': [r'^raw_img', r'^mip_raw'],
        'labels': [r'^pred_'],
        'annotations': [r'^annotated_'],
        'ROIS': [r'^ROI'],
        'metadata': [r'^metadata\.ya?ml', r'.*\.txt$'],
    }

    # Merge user-defined patterns
    if add_to_file_map:
        for k, v in add_to_file_map.items():
            if k not in type_patterns:
                raise ValueError(f"Invalid filemap type: {k}. Must be one of {list(type_patterns.keys())}")
            type_patterns[k].extend(v if isinstance(v, list) else [v])

    # 2. Filter Filenames
    # ---------------------------------------------------------
    available_fns = set(ex.get_filenames(fail_on_empty=True))
    
    if include_only:
        available_fns &= set(include_only)
    if exclude:
        available_fns -= set(exclude)

    if not available_fns:
        raise ValueError("Filters removed all available filenames.")
    
    print(f"Files to process: {available_fns}")

    # 3. Categorize Files & Validate Metadata
    # ---------------------------------------------------------
    validated_filemap = {k: [] for k in type_patterns}
    
    # discover exported shapes layers
    present_shapeLayer_ROIs = ex.discover_napari_shapeLayerROIs()
    print(present_shapeLayer_ROIs)
    
    # We work on a copy of metadata to avoid mutating the original object state unexpectedly
    updated_md = deepcopy(ex.exmd)
    data_md = updated_md.setdefault('data_metadata', {})
    data_shapes = data_md.setdefault('data_shapes', {})
    data_formats = data_md.setdefault('data_formats', {})

    formats_cache = {} # For quick lookup later
    shapes_cache = {}

    for fn in available_fns:
        # A. Determine Type
        file_type = None
        for key, patterns in type_patterns.items():
            if ug.get_matches([fn], patterns, pattern=True):
                file_type = key
                break
        print(f"File {fn} is of type {file_type}")
        
        # Handle unknown files
        if not file_type:
            try:
                # Assuming handle_unknown_file returns a valid key string
                file_type = handle_unknown_file(os.path.join(ex.path_to_example, fn))
                # Add this specific file to patterns so it's caught next time/downstream if needed
                # (Optional, depending on handle_unknown_file behavior)
            except Exception:
                print(f"Skipping unknown file type: {fn}")
                continue
        
        # skip ignored types
        ####################
        if file_type in ['ignore', 'metadata']: 
            print(f"ignoring `{fn}` because of type <{file_type}>")
            continue

        # Get Shape & Format
        file_key = ug.get_prefix(fn)
        file_path = ex.get_path(fn)
        
        
        # napari shapes layer handling
        ##############################
        if file_type == 'shapeLayer_ROIs':
            if fn in present_shapeLayer_ROIs:
                validated_filemap[file_type].append(fn)
            else:
                print(f"Shape layer file {fn} not found in metadata")
            continue
                
        
        # tiff file handling
        ####################
        
        # Safely get shape from disk
        try:
            actual_shape = uip.get_tiff_shape(file_path)
        except Exception as e:
            print(f"Could not read shape for {fn}: {e}")
            actual_shape = None

        # Update Metadata if missing or mismatch (Logic simplified: Trust Disk > Metadata)
        if actual_shape:
            data_shapes[file_key] = list(actual_shape)
            shapes_cache[fn] = actual_shape
            
            # Estimate format if missing
            if file_key not in data_formats:
                fmt = uip.estimate_format(actual_shape)
                data_formats[file_key] = fmt
                print(f"\tInferred format for {fn}: {fmt}")
            
            formats_cache[fn] = data_formats[file_key]
            validated_filemap[file_type].append(fn)

        elif file_type == 'metadata':
            # Metadata files don't need shapes
            validated_filemap[file_type].append(fn)
            
        else:
            # Shape failed and not a metadata file
            msg = f"Shape validation failed for {fn}"
            if fail_on_format_error:
                raise ValueError(msg)
            print(f"Skipping {fn}: {msg}")

    # 4. Load Images
    # ---------------------------------------------------------
    files_to_load = ug.flatten_list(list(validated_filemap.values()))
    
    # Exclude metadata files from the image loader
    image_files_only = [f for f in files_to_load if not f.startswith('metadata.')]
    
    print(f"Loading: {image_files_only}")
    
    image_dict = MetadataParser.read_example(
        ex.path_to_example,
        load_files=image_files_only,
        exmd=updated_md,
        validate_shapes_and_format=True,
        use_prefix_as_key=use_prefix_as_key,
    )
    image_dict['metadata'] = updated_md

    # 5. Generate Label-to-Intensity Map (Optional)
    # ---------------------------------------------------------
    label_int_map = {}
    
    if get_label_int_map:
        try:
            # Find the primary raw image (priority: raw_img.ome.tiff > raw_img.tiff > mip_raw.tiff)
            raw_candidates = ['raw_img.ome.tiff', 'raw_img.tiff', 'mip_raw.tiff']
            raw_img_fn = next((c for c in raw_candidates if c in formats_cache), None)

            if raw_img_fn:
                raw_prefix = ug.get_prefix(raw_img_fn)
                
                # Get all label-like files
                label_files = validated_filemap.get('labels', []) + \
                              validated_filemap.get('annotations', []) + \
                              validated_filemap.get('ROIS', [])

                present_chs = data_md.get('present_chs', [])

                for l_fn in label_files:
                    l_prefix = ug.get_prefix(l_fn)
                    l_fmt = formats_cache.get(l_fn, [])
                    l_shape = shapes_cache.get(l_fn, [])
                    
                    # Check if label file has channels
                    if 'C' in l_fmt:
                        # Map every channel in the label image to the corresponding channel in raw image
                        for chi in present_chs:
                            ch_suffix = f"ch{chi}"
                            # Mapping: label_prefix_chX -> raw_prefix_chX
                            label_int_map[f"{l_prefix}_{ch_suffix}"] = f"{raw_prefix}_{ch_suffix}"
            else:
                print("Warning: No raw image found for Label Map generation.")

        except Exception as e:
            print(f"Failed to generate Label Map: {e}")
            if fail_on_format_error: raise e

    return label_int_map, validated_filemap, image_dict, files_to_load

def validate_mapping(arr, input_type, img_name):
    """ 
    check arr dtype matches mapping, standardize input type, and warn if conversion needs to occur 
        splits into either image or label types, print warning if input was different than check result
    """
    display_layer_class = get_assumed_mapping(arr, input_type)
    std_input_type = standardize_input_types(input_type)
    
    if display_layer_class != std_input_type: # convert to assumed mapping
        print(f"WARNING -- converting img {img_name} from {input_type} --> {display_layer_class} due to dtype constraints. see annotation_IO.validate_mapping for more info about dtype assumptions.")
    return display_layer_class

def standardize_input_types(input_type):
    """ This exists for backcompat. checks input type is supported, convert mips/3D to general `images` type as is now standard """
    if input_type in ['MIPS', '3D']:
        return 'images'
    elif input_type in ['annotations', 'ROIS', 'images', 'labels']:
        return input_type
    else:
        raise ValueError(input_type)

def get_assumed_mapping(arr, input_type: Optional['str'] = None):
    """
    Determine display type for the given array based on its dtype.

    Args:
        arr: np.ndarry
        input_type: optional str. if none uses array dtype to assume display type

    Assumptions:
        Uses same assumptions as Napari ..base.Layer:
            the layer is assumed to be “image”, unless data.dtype is one of 
            (np.int32, np.uint32, np.int64, np.uint64, np.bool_), in which case it is assumed to be “labels”.
        treats annotations and ROIs as a special case

    Returns:
        str: The assumed mapping type ("labels", "images").
    
    Raises:
        ValueError: If the array's dtype is unsupported.
    """
    label_dtypes = (np.int32, np.uint32, np.int64, np.uint64, np.bool_)
    
    # Check dtype and determine mapping
    if arr.dtype in label_dtypes:
        if input_type in ('annotations', 'ROIS'):
            return input_type
        return "labels"
    else:
        return "images"

def handle_unknown_file(filepath):
    """ get assumed mapping from a filepath, assuming it points to a tifffile. if exception is raised, defaults to image type"""
    try:
        a = np.zeros((), dtype=uip.get_tiff_dtype(filepath))
        return get_assumed_mapping(a, input_type=None)
    except:
        print(f"warning. failed to handle mapping unknown file to an image display type. Will be loaded as `images` type")
        return 'images'


def arr2chlist(arr, current_format: str):
    """explode channels axis of an array into a list of arrays (if present), otherwise just a 1 elem list
        returns list of arrays and updated format - C (if present).
    """
    c_i = current_format.find("C")
    if c_i == -1:
        return [arr], current_format
    return uip.unpack_array_axis(arr, c_i), current_format.replace("C", "")


def preproc_images_for_display(
    exmd, FILE_MAP, image_dict, get_image_list: Optional[list[str]] = None, warn=True
) -> tuple[dict, dict]:

    # handle inputs
    if get_image_list is None:
        get_image_list = ug.flatten_list(list(FILE_MAP.values()))

    if 'data_metadata' not in exmd:
        exmd['data_metadata'] = {'data_shapes': {}, 'data_formats': {}}

    # get data formats
    shapes, fmts = exmd['data_metadata'].get('data_shapes', {}), exmd['data_metadata'].get('data_formats', {})

    display_fmts = []
    # invert FILE_MAP dict so can look up how each image that was found should be handled
    # removed usage of filestem, so filename can now be used
    inverted_file_map = {vv: k for k, v in FILE_MAP.items() for vv in v}
    f_get_img_list = [n for n in get_image_list]

    # iter images - load into napari based on criteria defined in FILE_MAP
    images_to_display = []
    non_array_types_to_display = []
    for img_name, arr in image_dict.items(): # (filename, dataObject)

        if arr is None:
            if warn: print(f'ignoring {img_name} as arr is None')
            continue
        if img_name not in inverted_file_map: 
            if warn: print(f'ignoring {img_name} as it is not in FILE_MAP')
            continue
        if img_name in ['metadata']:
            continue
        if img_name not in f_get_img_list:
            if warn: print(f'ignoring {img_name} as it is not in get_image_list')
            continue

        _load_as = inverted_file_map[img_name]
        logger.info(f"processing image: {img_name} as type <{_load_as}>..")

        def get_ROI_export_metadata(exmd, img_name):

            roi_exports = exmd['data_metadata'].get('ROI_exports') or {}
            if roi_exports:
                roi_md = roi_exports.get(img_name) or {}
                return roi_md.get('parent_layer_name')

        from SynAPSeg.Plugins.ABBA import core_regionPoly as rp
        if isinstance(arr, rp.polyCollection):          
            shapes_data, features_df = geojsonPoly_to_napari_shapes_data(arr)
            images_to_display.append({
                'display_type': 'shapes',
                'poly_coords': shapes_data,
                'features': features_df,
                'shape_type': 'polygon',
                'ndim': 2, # polycollection only supports 2d polys 
                'parent_layer_name': get_ROI_export_metadata(exmd, img_name),
                'load_as': _load_as, 
                'img_name': ug.get_prefix(img_name), 
                'filename': img_name,
                'df': None,
            })

        elif isinstance(arr, pd.DataFrame):

            # load and process CSV into shapes layer
            from SynAPSeg.utils import utils_geometry as ugeom
            poly_coords, features_df = ugeom.parse_shapes_from_dataframe(arr) 

            images_to_display.append({
                'diplay_type': 'shapes',
                'poly_coords': poly_coords,
                'features': features_df,
                'shape_type': 'polygon',
                'ndim': poly_coords[0].shape[-1], # assumes all have same shape 
                'parent_layer_name': get_ROI_export_metadata(exmd, img_name),
                'load_as': _load_as, 
                'img_name': ug.get_prefix(img_name), 
                'filename': img_name,
                'df': arr,
            })
        elif isinstance(arr, np.ndarray):
            # added support for using filestem or filename to match format
            input_format = fmts.get(img_name)
            input_format2 = fmts.get(ug.get_prefix(img_name))
            any_found = [el for el in [input_format, input_format2] if el is not None]
            if len(any_found) == 0: 
                input_format = uip.estimate_format(arr.shape)
                print(f"WARNING -- could not find format for {img_name} in exmd, estimated format: {input_format} from shape {arr.shape}")
            else: 
                input_format = any_found[0]

            # standardize, then collapse singleton dims
            arr, current_format = uip.standardize_collapse(arr, input_format, STANDARD_FORMAT)

            logger.info(f"[{_load_as}], \'{img_name}\', shape: {arr.shape}, format: {current_format}, (input format: {input_format})")
            # check dtypes
            load_as = validate_mapping(arr, _load_as, img_name)

            # explode channels axis (if present), other wise just a 1 elem list
            arr_ch_list, current_format = arr2chlist(arr, current_format)

            # img arr normalizing
            NORMALIZE = False
            if load_as in ['images'] and NORMALIZE:

                logger.info(f"normalizing: {img_name} and converting to 8bit for proper colormap display")
                arr_ch_list = [
                    (uip.normalize_01(a) * 255).astype('uint8')
                    for a in arr_ch_list
                ]

            images_to_display.append({
                'display_type': load_as,
                'arr_ch_list': arr_ch_list, 
                'load_as': load_as, 
                'img_name': ug.get_prefix(img_name), 
                'current_format': current_format,
                'filename': img_name # with suffix, img_name here then is after get_prefix
            })
            display_fmts.append(current_format)
        else:
            if warn:
                print(f"ignoring {img_name} due to incompatible type: {type(arr)}")

    # build viewer args based on processed shapes
    viewer_kwargs = {'ndisplay':2, 'order':tuple(), 'axis_labels': []}

    disp_dims = set(list("".join(display_fmts))) # get unique display dimensions
    unused_dims = uip.subtract_dimstr(STANDARD_FORMAT, disp_dims) # get un-used dimensions
    disp_fmt = uip.subtract_dimstr(STANDARD_FORMAT, unused_dims) # set axes for viewer based on largest 
    viewer_kwargs['axis_labels'] = [f"<{d}> " for d in (disp_fmt)]
    viewer_kwargs['order'] = list(range(len(disp_fmt))) # e.g. 0,1,2 for ZYX

    # now we have to reformat all the arrays to match this format - performance improvement would be don't do arr2chlist until now
    # for imgd in images_to_display:
    #     imgd['arr_ch_list'] = [uip.transform_axes(a, imgd['current_format'], disp_fmt) for a in imgd['arr_ch_list']]
    #     imgd['current_format'] = disp_fmt

    return images_to_display, viewer_kwargs


def geojsonPoly_get_features_df(polycollection):
    features = []
    polyindex_to_napari_index = {} # dict mapping poly index in collection to list of indicies - required by napari where multipolys are split
    for polyindex, gjp in enumerate(polycollection):
        props = gjp.properties
        polyindex_to_napari_index[polyindex] = []
        if 'napari_shapes_layer_info' in props:
            napari_md = props['napari_shapes_layer_info']
            parse_keys = ['name', 'indicies', 'shape_types', 'custom', 'coordinate_system']
            extracted = {k: napari_md.get(k) for k in parse_keys}
            # need to separate multi polys into individual rows
            for i in range(len(extracted['indicies'])):
                row = {
                    'name': extracted['name'], 
                    'index': extracted['indicies'][i],
                    'type': extracted['shape_types'][i],
                    'custom': extracted['custom'][i],
                    # 'coordinate_system': extracted['coordinate_system']
                }
                features.append(row)
                
                polyindex_to_napari_index[polyindex].append(extracted['indicies'][i])
            
    features_df = pd.DataFrame(features)
    from SynAPSeg.utils import utils_geometry as ugeom
    features_df = ugeom.santize_napari_shapes_features_df(features_df)
    
    return features_df, polyindex_to_napari_index

def geojsonPoly_to_napari_shapes_data(polycollection) -> tuple[list[np.ndarray], pd.DataFrame]:
    
    features_df, polyindex_to_napari_index = geojsonPoly_get_features_df(polycollection)
    
    shapes_data = []
    for polyindex, gjp in enumerate(polycollection):
        napari_inds = polyindex_to_napari_index[polyindex]
        asShapely = gjp.to_shapely()
        print(type(asShapely))
        asNapari = convert_shapely_to_napari(asShapely)
        shapes_data.extend(asNapari)
        
    return shapes_data, features_df

def cart2mat(coords):
    """converts cartesian coordinates to matrix coordinates"""
    return np.column_stack((coords[:,1], coords[:,0]))
    
def convert_shapely_to_napari(shapely_obj):
    """
    Parses a Shapely Polygon or MultiPolygon (including holes) 
    and adds it to the napari viewer as a Shapes layer.
    """
    from shapely.geometry import Polygon, MultiPolygon
    
    # Standardize input to a list of Polygons
    if isinstance(shapely_obj, Polygon):
        polygons = [shapely_obj]
    elif isinstance(shapely_obj, MultiPolygon):
        polygons = list(shapely_obj.geoms)
    else:
        raise TypeError("Input must be a Shapely Polygon or MultiPolygon")
        
    shape_data = []
    
    for poly in polygons:
        # Extract exterior coordinates as a NumPy array (swap XY to YX if needed for image alignment)
        # Shapely stores coordinates as (X, Y). Napari usually expects (Y, X) or (Z, Y, X).
        ext_coords = np.array(poly.exterior.coords)
        
        if not poly.interiors: # No holes
            shape_data.append(cart2mat(ext_coords))
        else:
            # format as list of arrays: [exterior, hole1, hole2, ...] - unclear if napari supports holes
            poly_rings = [cart2mat(ext_coords)]
            for interior in poly.interiors:
                poly_rings.append(cart2mat(np.array(interior.coords)))
            shape_data.append(poly_rings)
    
    return shape_data



if __name__ == '__main__':
    from SynAPSeg.IO.project import Project
    proj = Project(r"J:\SEGMENTATION_DATASETS\2026_0914_AMI_cfosTimecourse")
    ex = proj.examples[0]
    
    LABEL_INT_MAP, FILE_MAP, image_dict, get_image_list = load_example_images(ex, use_prefix_as_key=False)
    exmd = image_dict.pop('metadata')
    images_to_display, viewer_kwargs = preproc_images_for_display(exmd, FILE_MAP, image_dict, ['annotated_shapes_constrained.geojson'])
