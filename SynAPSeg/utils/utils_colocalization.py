from tabulate import tabulate
import pandas as pd
import numpy as np
from timeit import default_timer as dt
from copy import deepcopy
import functools
from typing import List, Tuple, Optional, Dict, Any, Callable, Union
from skimage.measure import regionprops_table, _regionprops_utils
from skimage import measure
from skimage.morphology import skeletonize
from collections import deque
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy.ndimage import label as ndi_label
from shapely.geometry import Polygon, shape, Point
from shapely.ops import unary_union
from shapely.strtree import STRtree
from shapely.prepared import prep
# from skimage.draw import polygon2mask
import rasterio

from SynAPSeg.config.constants import SPATIAL_AXES



#TODO
###############
# this should be split into utils geometry for all polygon stuff and keep this focused on image arrays
# pre-filtering with STRtree may be faster for large images

# notes
########################################################
# measure.regionprops supported properties for 3d regions as of skimage.__version__='0.25.2'
#   area
#   area_bbox
#   area_convex
#   area_filled
#   axis_major_length
#   axis_minor_length
#   bbox
#   centroid
#   centroid_local
#   centroid_weighted
#   centroid_weighted_local
#   coords
#   equivalent_diameter_area
#   euler_number
#   extent
#   feret_diameter_max
#   image
#   image_convex
#   image_filled
#   image_intensity
#   inertia_tensor
#   inertia_tensor_eigvals
#   intensity_max
#   intensity_mean
#   intensity_min
#   intensity_std
#   label
#   moments
#   moments_central
#   moments_normalized
#   moments_weighted
#   moments_weighted_central
#   moments_weighted_normalized
#   slice
#   solidity

# Unsupported properties (for 3d objects):
#   eccentricity
#   moments_hu
#   moments_weighted_hu
#   orientation
#   perimeter
#   perimeter_crofton

# checkout colocalization methods in scikit-image described here: https://scikit-image.org/docs/dev/auto_examples/applications/plot_colocalization_metrics.html#sphx-glr-auto-examples-applications-plot-colocalization-metrics-py

_DEFAULT_REGION_PROPERTIES_TO_EXTRACT = [
        'label', 'area', 'bbox', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max',  'axis_major_length', 'axis_minor_length'
    ]

_3D_UNSAFE_PROPS = [
    'eccentricity', 'moments_hu', 'moments_weighted_hu', 'orientation', 'perimeter', 'perimeter_crofton', 'uc.circularity', 'circularity', 
    'axis_minor_length' # will raise domain error if obj ndim == 2 but in a 3d image
]

# get region props
##########################################################################################
def get_rp_table(
    label_arr, 
    intensity_arr, 
    ch_colocal_id: Optional[Dict] = None, 
    ch_axis: Optional[int] = -1,
    prt_str='', 
    rps_to_get: Optional[List] = None,
    get_object_coords=False, 
    additional_props: Optional[List[str]]=None,
    extra_properties: Optional[List[Callable]]=None,
    p=False,
    fmt: Optional[str]=None,
    check_unsafe_properties=True,
    ):
    ''' 
    use label image and intensity image to extract a dataframe containing region props for each channel specified 
        core implemention used is: skimage.measure.regionprops_table
        most time is spent in regionprops_table() (~80s, 4s is added through formatting bbox/centroid columns
        
        ARGS
        - label_arr (np.array[int]) --> label image containing objects
        - intensity_arr (np.array) --> intensity image
        - ch_colocal_id (dict) --> dictionary assigning colocal id to img channels by map the image channels to their respective colocal ids, 
            e.g. 0=dapi, 1=zif, 2=gfp, 3=zif+gfp
        - ch_axis (int): index of channel axis
        - prt_str (str) --> used for debugging
        - rps_to_get (list[str]) --> list of regionproperties to get, defaults to _DEFAULT_REGION_PROPERTIES_TO_EXTRACT == ['label', 'area', 'bbox', 'centroid', 'intensity_mean', 'intensity_min', 'intensity_max',  'axis_major_length', 'axis_minor_length']
        - get_object_coords (bool) --> get indicies of each point with a given label
        - additional_props (list, None) --> list of strings which regionprops will extract
        - extra_properties (Iterable of callables) --> custom fxns passed to skimage.measure.regionprops_table directly 
            note: the following special cases strings get mapped to local modules: 'uc.skewedness', 'uc.kurtosis', 'uc.circularity', 'uc.longest_skeleton_path'
        - fmt (Optional[str]) --> dimensions of array #TODO automatically handles rp extraction for ndimensional arrays
            inserts columns for dimensions outside ZYX (which as natively supported by skimage..regionprops_table)
            this superceeds ch_axis if also defined
        -check_unsafe_properties (bool) - if true will remove properties known to cause runtime errors with 3d data in some edge cases
        
    returns:
        pd.Dataframe
    '''
    from SynAPSeg.utils.utils_image_processing import nd_slice, safe_squeeze, has_singleton_dimenstions
    rp_st_time, len_init_prt_str = dt(), len(prt_str)  # for internal logging

    # TODO add fmt support to auto configure things
    if fmt is None:
        assert label_arr.shape == intensity_arr.shape
            
    # deal with input image being 2d only (only 1 channel)
    NDIM = int(label_arr.ndim)
    if NDIM == 2:
        label_arr, intensity_arr = np.expand_dims(label_arr, -1), np.expand_dims(intensity_arr, -1)
        ch_axis = 2

    if ch_axis is None:
        ch_axis = NDIM 
        label_arr, intensity_arr = label_arr[..., np.newaxis], intensity_arr[..., np.newaxis]
    elif ch_axis == -1:
        ch_axis = NDIM -1
        
    
    # determine if input is 3D - assumes first spaital axis is Z 
    spaital_shape = [s for i, s in enumerate(label_arr.shape) if i != ch_axis]
    IS_3D = label_arr.ndim==4 and (spaital_shape[0] > 1)

    # handle input region props    
    rps_to_get, _removed, extra_properties = _rp_table_handle_props(rps_to_get, get_object_coords, additional_props, extra_properties, IS_3D, check_unsafe_properties)
    
    if len(_removed) > 0:
        _msg = ' 3d input detected and' if IS_3D else ''
        prt_str+=(f'warning{_msg} unsafe properties. removed: {_removed}')

                
    ch_colocal_id = ch_colocal_id or {k:k for k in range(label_arr.shape[ch_axis])}
    # create the region props table and make modifications to simply the bbox/centroid columns and add the colocal id
    rpdfs = []
    for ch_i in ch_colocal_id.keys():
        indexer = nd_slice(label_arr, ch_axis, ch_i)
        
        lbls = label_arr[indexer]
        int_img = intensity_arr[indexer]

        if has_singleton_dimenstions(lbls): # when calculating props can get math domain error if shape of a dim is 1 so need to collapse singletons at this stage
            lbls = safe_squeeze(lbls, 2)
            int_img = safe_squeeze(int_img, 2)
        
        _df = pd.DataFrame(regionprops_table(
            lbls, 
            int_img, 
            properties=rps_to_get, 
            extra_properties=extra_properties
        ))
        _df = _df.assign(colocal_id=ch_colocal_id[ch_i])
        
        if 'bbox' in rps_to_get:
            _df = concat_cols(_df, 'bbox')
        if 'centroid' in rps_to_get:
            _df = concat_cols(_df, 'centroid')
        rpdfs.append(_df)
    
    rpdf = pd.concat(rpdfs, ignore_index=True)

    # add eccentricity - this is actually commonly refered to as 'aspect_ratio'
    if 'axis_major_length' in rpdf.columns and 'axis_minor_length' in rpdf.columns:
        rpdf = rpdf.assign(eccentricity = rpdf['axis_major_length']/rpdf['axis_minor_length'])
    
    prt_str+=(
        f"initial rpdf value counts {rpdf.value_counts(['colocal_id'])}\n"
        f"rps extracted in {dt() - rp_st_time}.\n"
    )

    
    if len_init_prt_str==0: # i.e. was called here and not from a func that didn't expect this output
        if p: 
            print(prt_str, flush=True)
        return rpdf
    
    return rpdf, prt_str

def concat_cols(df, col_name_base):
    """ combines cols that have the same prefix into a single column 
        e.g. mergeing values in 'bbox-0', 'bbox-1', etc. into 1 column 'bbox'
    """
    merge_cols = [c for c in df.columns if c.startswith(col_name_base)]
    df[col_name_base] = df[merge_cols].apply(lambda row: row.tolist(), axis=1)
    df = df.drop(columns=merge_cols)
    return df

def _rp_table_handle_props(rps_to_get, get_object_coords, additional_props, extra_properties, IS_3D, check_unsafe_properties):
    """ validation and curation of region prop attrs """
    # define region props to extract
    rps_to_get = rps_to_get or deepcopy(_DEFAULT_REGION_PROPERTIES_TO_EXTRACT)
    
    if get_object_coords: 
        rps_to_get.append('coords')
        
    if additional_props:
        assert isinstance(additional_props, list), f"got invalid type for additional_props: {additional_props}"
        rps_to_get = rps_to_get + additional_props

    _removed = [] # if 3d remove unsafe properties

    # handle string represesntation of local custom properties 
    _DEFAULT_EXTRA_PROPERTIES_MAP = {
        'uc.skewedness': skewedness, 'uc.kurtosis':kurtosis, 
        'uc.circularity':circularity, 'uc.longest_skeleton_path':longest_skeleton_path
    }
    
    if extra_properties is not None:
        _exprops = []
        for p in extra_properties:
            if callable(p):
                if IS_3D and check_unsafe_properties and p.__name__ in _3D_UNSAFE_PROPS:
                    _removed.append(p.__name__)
                else:
                    _exprops.append(p)
            elif isinstance(p, str):
                if IS_3D and check_unsafe_properties and p in _3D_UNSAFE_PROPS:
                    _removed.append(p)
                elif p in _DEFAULT_EXTRA_PROPERTIES_MAP.keys():
                    _exprops.append(_DEFAULT_EXTRA_PROPERTIES_MAP[p])
                else:
                    raise ValueError(f"property {p} is str and not in {_DEFAULT_EXTRA_PROPERTIES_MAP}. Else should be callable")
            else:
                raise ValueError(f"property {p} must be callable or a key(str) in {_DEFAULT_EXTRA_PROPERTIES_MAP}")
        extra_properties = _exprops    
                    
    rps_to_get = list(set(rps_to_get))

    # if 3d remove unsafe properties - note those passed via extra_props will already have been filtered out
    if IS_3D and check_unsafe_properties:
        for p in _3D_UNSAFE_PROPS:
            if p in rps_to_get:
                rps_to_get.remove(p)
                _removed.append(p)
    return rps_to_get, _removed, extra_properties


# colocalization
##########################################################################################
from typing import Iterable, Dict


def colocalize(
        colocalization_params: Iterable[Dict[str, Iterable[int] | int]],
        rpdf: pd.DataFrame,
        label_arr: np.ndarray,
        current_format: str,
        clc_axes_fmt: str = 'C',
        prt_str: str = '',
        intersection_threshold: float = 0.001,
        intersection_metric: Optional[Callable[[np.ndarray], float]] = None,
    ) -> Tuple[pd.DataFrame, str]:
    """
    Perform object-based colocalization across labeled segmentation channels. 
    Supports colocalization in arbitrary n-dimensional images across any number of channels.

    Parameters
    ----------
    colocalization_params : list of dict
        Each dict must define:
          - ``coChs``: channel indices to compare (last entry = base channel)
          - ``coIds``: colocal_id values corresponding to ``coChs`` (last entry = base)
          - ``assign_colocal_id``: new colocal_id for successful colocalizations
        e.g. dict(coIds=(0,1), coChs=(0,1), assign_colocal_id=3)

    rpdf : pandas.DataFrame
        Region properties table with required columns: ``label``, ``bbox``,
        and ``colocal_id``.

    label_arr : numpy.ndarray
        Labeled segmentation image with shape ``(..., C)`` where ``C`` is the
        channel axis. Values must be integer labels (0 = background).

    current_format : str
        Axis format string of ``label_arr`` (e.g. ``"YXC"``, ``"ZYXC"``).

    clc_axes_fmt : str, default="C"
        Axis in ``current_format`` corresponding to channels.

    prt_str : str, optional
        Status string appended with summary and timing information.

    intersection_threshold : float, default=0.001
        Minimum required intersection metric value for colocalization.

    intersection_metric : callable, optional
        Function accepting an array of stacked binary masks (last axis) and
        returning a scalar overlap metric. Defaults to multi-way IoU.
    
    Assumptions
    -----------
        - uses last provided channel index (in coChs) as base channel to compare all others to
        - We only consider an object colocalized if it has overlapping labels in all specified other channels.
        - bbox came from skimage.measure.regionprops, the order is (min_row, min_col, max_row, max_col)
    
    Implementation
    --------------
        For each specification in ``colocalization_params``, objects in ``rpdf``
        belonging to a base ``colocal_id`` are tested for spatial overlap with
        objects in other channels of ``label_arr``. The label with the largest
        overlap in each additional channel is selected, and a multi-mask
        intersection metric (default: IoU) is computed. If the metric exceeds
        ``intersection_threshold``, a new object with ``assign_colocal_id`` is
        appended to ``rpdf``.

    Returns
    -------
    rpdf : pandas.DataFrame
        Updated DataFrame with newly appended colocalized objects.

    prt_str : str
        Updated status string.
    """
    prt_str += f"Colocalization info:\n  base_coloc_counts: {get_colocal_id_counts(rpdf)}\n"

    label_arr = np.moveaxis(label_arr, current_format.index(clc_axes_fmt), -1) # move ch last

    for clc_props in colocalization_params:
        coChs, coIds, assign_colocal_id = clc_props['coChs'], clc_props['coIds'], clc_props['assign_colocal_id']
        rpdf, prt_str = get_colocalization(
            rpdf=rpdf, label_arr=label_arr,    
            coChs=coChs, coIds=coIds, assign_colocal_id=assign_colocal_id,  # type: ignore
            prt_str=prt_str, 
            intersection_threshold=intersection_threshold,
            intersection_metric=intersection_metric
        )
    return rpdf, prt_str

def calculate_iou(masks):
    """
    Calculate the Intersection over Union (IoU) for multiple masks.

    :param masks: A NumPy array (e.g. ZYXN or YXN) where N dim represents n_masks and is last dimesnion containing binary masks.
    :return: IoU value.
    """
    intersection = np.logical_and.reduce(masks, axis=-1).sum()
    union = np.logical_or.reduce(masks, axis=-1).sum()
    iou = intersection / union if union != 0 else 0
    return iou


def get_colocalization(
        rpdf: pd.DataFrame,
        label_arr: np.ndarray,
        coChs: Iterable[int],
        coIds: Iterable[int],
        assign_colocal_id: int,
        prt_str: str = '',
        intersection_threshold: float = 0.00,
        intersection_metric: Optional[Callable[[np.ndarray], float]] = None,
    ) -> Tuple[pd.DataFrame, str]:
    """
    Core colocalization implementation.  
    See :func:`colocalize` for the public API; this function should not be called directly.
    """
    from SynAPSeg.utils.utils_image_processing import _bbox2slice
    
    # basic input validation checks, more #TODO
    assert assign_colocal_id not in coIds, f"assign colocal id can not already exist"

    # setup
    ########################################
    st_time = dt()
        
    # parse args
    intersection_metric = calculate_iou if intersection_metric is None else intersection_metric
    
    n_clc_channels = len(coIds)
    base_coId, base_coCh = coIds[-1], coChs[-1] # uses last provided colocal_id as base for comparison
    other_coIds, other_coChs = coIds[:-1], coChs[:-1]
    len_other_coChs = len(other_coChs)
    iter_other_coChs = np.arange(len_other_coChs)

    # extract indicies in rpdf for base colocal id and setup other input values for indexing efficiency
    ch_df_mask = rpdf['colocal_id'] == base_coId
    ch_df_indicies = rpdf[ch_df_mask].index
    ch_bboxes = np.asarray(rpdf.loc[ch_df_mask, 'bbox'].to_list()) # slightly faster than np.stack(rpdf.loc[ch_df_mask, 'bbox'])
    ch_nuc_lbls = rpdf.loc[ch_df_mask, 'label'].values
    assert ch_bboxes.shape[0] == ch_df_indicies.shape[0] == ch_nuc_lbls.shape[0]
    
    # setup array to store results - cli, intersection_percent, *(largest_label for each other ch)
    result_default_cols = ['cli', 'intersection_percent']
    ch_intersecting_label_cols = [f'ch{coCh}_intersecting_label' for coCh in other_coChs]
    len_result_default_cols = len(result_default_cols) 
    results_arr = np.full((ch_bboxes.shape[0], len_result_default_cols + len(ch_intersecting_label_cols)), fill_value=-1.0)

    assert ch_bboxes.ndim==2, f"ch_bboxes.ndim = {ch_bboxes.ndim}"
    assert ch_bboxes.shape[1] == 0 or ch_bboxes.shape[1]//(label_arr.ndim-1)==2, f"shape of ch_bboxes is wrong got: {ch_bboxes.shape}"
    
    if min(ch_nuc_lbls) <= 0:
        raise ValueError(f"non-positive labels are not supported, got: {min(ch_nuc_lbls)}")
    
    # colocalization
    #######################################
    for row_i in np.arange(ch_bboxes.shape[0]):
        CONTINUE_FLAG = False
        cli, ch_bbox, base_nuc_lbl = ch_df_indicies[row_i], ch_bboxes[row_i], ch_nuc_lbls[row_i]
        ch_slc = _bbox2slice(ch_bbox)
        
        # extract bbox coords from nuclei image
        ch_nucleus_img_base = label_arr[(*ch_slc, base_coCh)] # extract bbox around this label
        base_non_zero = (ch_nucleus_img_base==base_nuc_lbl).nonzero() # get nonzero coords for current label in base ch (remove other labels in this channel if present)

        # generate a mask for the other colocalization channels from the base channel
        ch_nucleus_img_others = label_arr[(*ch_slc, other_coChs)]
        others_masked = np.zeros_like(ch_nucleus_img_others)
        for oCh_i in range(len(other_coChs)):
            o_masked = ch_nucleus_img_others[(*base_non_zero, oCh_i)]
            
            if o_masked.max() == 0: # skip if any ch has no lbl
                CONTINUE_FLAG = True
                break
            
            inter_labels, inter_counts = np.unique(o_masked.ravel()[o_masked.nonzero()[0]], return_counts=True) # flatten, take non-zero values, count unique
            largest_intersecting_label = inter_labels[np.argmax(inter_counts)] # take label w highest count
            others_masked[(*base_non_zero, oCh_i)] = np.where(o_masked==largest_intersecting_label, 1, 0)
            results_arr[row_i, len_result_default_cols+oCh_i] = largest_intersecting_label

        if CONTINUE_FLAG: 
            continue # if any of the other channels had no label overlapping with base, skip

        # get intersection percentage
        all_stacked = np.concatenate((others_masked, np.where(ch_nucleus_img_base[..., np.newaxis]==base_nuc_lbl, 1, 0)), axis=-1)
        intersection = intersection_metric(all_stacked)
        if intersection > intersection_threshold:
            # write to results
            results_arr[row_i, 0:len_result_default_cols] = cli, intersection
        

    # write results to input dataframe
    #######################################
    # convert successful results to dataframe and merge
    results = results_arr[(results_arr[:, 0]>-1), :]
    df_results = pd.DataFrame(results, columns=result_default_cols + ch_intersecting_label_cols).set_index('cli')
    merge_on_df = rpdf.loc[df_results.index.values, :]

    # remove the cols we are writing the results to if they already exist
    override_cols = [c for c in ch_intersecting_label_cols+['intersection_percent'] if c in merge_on_df.columns]
    coloc_df = (pd.merge(
        merge_on_df.drop(columns=override_cols), df_results, how='left', left_index=True, right_index=True)
        .assign(colocal_id = assign_colocal_id, ))

    # make cols if they don't exist in input rpdf
    for col in coloc_df.columns.to_list():
        if col not in rpdf:
            rpdf[col] = np.nan
    
    # add to input df
    rpdf_coloc = pd.concat([rpdf, coloc_df], ignore_index=True)
    prt_str += f"  colocal_id_counts: {get_colocal_id_counts(rpdf_coloc)}\n"
    prt_str += f"  colocalization completed in {dt()-st_time}.\n"
    return rpdf_coloc, prt_str




from typing import Iterable, Dict, Tuple, Optional, Callable, Union

# TODO test this implementation
def colocalize_by_distance(
        colocalization_params: Iterable[Dict[str, Iterable[int] | int]],
        rpdf: pd.DataFrame,
        distance_threshold: float = 10.0,
        pixel_size: float = 1.0,
        prt_str: str = '',
        **kwargs # Accept extra args to maintain compatibility with original calls
    ) -> Tuple[pd.DataFrame, str]:
    """
    Perform object-based colocalization via centroid distance thresholding.
    
    Parameters
    ----------
    distance_threshold : float
        Maximum distance between centroids to consider objects colocalized.
    pixel_size : float
        Scalar to convert pixel coordinates to physical units if threshold is in microns.
        Set to 1.0 if threshold is in pixels.
    """
    # prt_str += f"Colocalization (Distance) info:\n  base_coloc_counts: {get_colocal_id_counts(rpdf)}\n"

    for clc_props in colocalization_params:
        coChs, coIds, assign_colocal_id = clc_props['coChs'], clc_props['coIds'], clc_props['assign_colocal_id']
        rpdf, prt_str = get_colocalization_by_distance(
            rpdf=rpdf,
            coChs=coChs, 
            coIds=coIds, 
            assign_colocal_id=assign_colocal_id,  
            prt_str=prt_str, 
            distance_threshold=distance_threshold,
            pixel_size=pixel_size
        )
    return rpdf, prt_str

def get_colocalization_by_distance(
        rpdf: pd.DataFrame,
        coChs: Iterable[int],
        coIds: Iterable[int],
        assign_colocal_id: int,
        prt_str: str = '',
        distance_threshold: float = 10.0,
        pixel_size: float = 1.0,
        label_arr: Optional[np.ndarray] = None, # Kept for API compatibility, unused
    ) -> Tuple[pd.DataFrame, str]:
    """
    Core colocalization implementation using KD-Tree distance matching.
    """
    # 
    from scipy.spatial import cKDTree

    # 1. Validation and Setup
    assert assign_colocal_id not in coIds, f"assign colocal id cannot already exist"
    
    # Identify Base vs Other
    base_coId, base_coCh = coIds[-1], coChs[-1]
    other_coIds, other_coChs = coIds[:-1], coChs[:-1]
    
    # 2. Extract Base Coordinates
    # Filter for base objects
    base_mask = rpdf['colocal_id'] == base_coId
    if not base_mask.any():
        return rpdf, prt_str # No base objects to colocalize
        
    base_indices = rpdf[base_mask].index
    base_coords = _get_centroids(rpdf.loc[base_mask], pixel_size)
    
    # Initialize results tracking
    # We track: is_match (bool), plus label and distance for each other_channel
    n_base = len(base_indices)
    valid_match_mask = np.ones(n_base, dtype=bool)
    
    # Arrays to store results for the base objects
    # Shape: (n_base, n_other_channels)
    matched_labels = np.full((n_base, len(other_coChs)), fill_value=-1, dtype=int)
    matched_dists = np.full((n_base, len(other_coChs)), fill_value=np.inf, dtype=float)

    # 3. Iterate over "Other" channels and intersect
    for i, (oId, oCh) in enumerate(zip(other_coIds, other_coChs)):
        
        # Get potential partners
        other_mask = rpdf['colocal_id'] == oId
        if not other_mask.any():
            valid_match_mask[:] = False # If a required channel is empty, no full colocalization possible
            break
            
        other_df_subset = rpdf.loc[other_mask]
        other_coords = _get_centroids(other_df_subset, pixel_size)
        other_labels = other_df_subset['label'].values
        
        # Build Tree and Query
        # cKDTree is very fast for querying nearest neighbors
        tree = cKDTree(other_coords)
        
        # Query nearest neighbor for all base points
        # k=1 returns (distances, indices) for the nearest neighbor
        dists, idxs = tree.query(base_coords, k=1, distance_upper_bound=distance_threshold)
        
        # Identify hits (distance_upper_bound returns inf if no neighbor found)
        has_neighbor = dists != np.inf
        
        # Update the master validity mask (AND logic: must match ALL channels)
        valid_match_mask = valid_match_mask & has_neighbor
        
        if not valid_match_mask.any():
            break
            
        # Store metadata for valid hits (temporarily store all, filter later)
        # Note: idxs where no neighbor is found will be out of bounds (== len(data)), 
        # but we guard this with the valid_match_mask later.
        valid_idxs = idxs[has_neighbor]
        
        # We only record data where a neighbor was actually found to prevent index errors
        # Safe assignment:
        matched_dists[has_neighbor, i] = dists[has_neighbor]
        matched_labels[has_neighbor, i] = other_labels[valid_idxs]

    # 4. Construct Results DataFrame
    # Filter to only fully colocalized objects
    final_indices = base_indices[valid_match_mask]
    
    if len(final_indices) > 0:
        # Create result array
        # Columns: [cli, mean_distance, chX_label, chX_dist, chY_label, chY_dist...]
        
        # Dynamic column names
        res_cols = ['cli', 'mean_distance']
        for oCh in other_coChs:
            res_cols.extend([f'ch{oCh}_intersecting_label', f'ch{oCh}_distance'])
            
        # Extract valid data
        valid_labels = matched_labels[valid_match_mask]
        valid_dists = matched_dists[valid_match_mask]
        mean_dists = np.mean(valid_dists, axis=1)
        
        # Build the data list for DataFrame construction
        # Flatten label/dist pairs: zip(labels_col0, dists_col0, labels_col1, dists_col1...)
        interleaved_data = []
        for j in range(len(other_coChs)):
            interleaved_data.append(valid_labels[:, j])
            interleaved_data.append(valid_dists[:, j])
        
        # Stack: cli | mean_dist | interleaved_labels_dists
        results_matrix = np.column_stack((
            final_indices, 
            mean_dists, 
            *interleaved_data
        ))
        
        df_results = pd.DataFrame(results_matrix, columns=res_cols).set_index('cli')
        
        # 5. Merge and Append (Logic preserved from original)
        merge_on_df = rpdf.loc[final_indices].copy()
        
        # Remove columns if they already exist to avoid suffixes
        existing_cols = [c for c in res_cols if c in merge_on_df.columns and c != 'cli']
        merge_on_df = merge_on_df.drop(columns=existing_cols)
        
        coloc_df = pd.merge(
            merge_on_df, df_results, 
            how='left', left_index=True, right_index=True
        ).assign(colocal_id=assign_colocal_id)
        
        # Ensure new columns exist in original rpdf (fill NaN)
        for col in coloc_df.columns:
            if col not in rpdf.columns:
                rpdf[col] = np.nan
        
        rpdf_coloc = pd.concat([rpdf, coloc_df], ignore_index=True)
        
        # Update print string
        # prt_str += f"  colocal_id_counts: {get_colocal_id_counts(rpdf_coloc)}\n"
        
        return rpdf_coloc, prt_str

    return rpdf, prt_str

def _get_centroids(df: pd.DataFrame, pixel_size: float = 1.0) -> np.ndarray:
    """
    Helper to extract centroids from dataframe. 
    Prioritizes 'centroid' column, falls back to calculating center of 'bbox'.
    """
    # Option A: 'centroid' column exists (list/tuple/array)
    if 'centroid' in df.columns:
        coords = np.stack(df['centroid'].values)
    
    # Option B: Calculate from 'bbox'
    # Assumptions: bbox is (min_row, min_col, ..., max_row, max_col)
    else:
        bboxes = np.stack(df['bbox'].values)
        ndim = bboxes.shape[1] // 2
        mins = bboxes[:, :ndim]
        maxs = bboxes[:, ndim:]
        coords = (mins + maxs) / 2.0
        
    return coords * pixel_size



def filter_rpdf(
        threshold_dict, rpdf_coloc, all_colocal_ids,
        group_labels_col='img_name', return_labels=False, 
        clc_nucs_info=None, label_col='label'
    ):
    # TODO: this needs to be updated to work when data is aggregated from multiple images - where labels may be repeated from other images
    rpdf_coloc = rpdf_coloc.copy(deep=True)
    thresholded_rpdfs, valid_labels, invalid_labels, filtered_counts = {}, {}, {}, {'init':get_colocal_id_counts(rpdf_coloc, all_colocal_ids), 'final':None}
    for c in all_colocal_ids: filtered_counts[c] = {}

    for colocal_id in all_colocal_ids:
        cdf = rpdf_coloc.loc[rpdf_coloc['colocal_id'] == colocal_id]
        all_labels = set(cdf[label_col].to_list())

        # threshold regionprops
        if colocal_id in threshold_dict:
            cdf, rp_filtercounts = region_prop_table_filter(cdf, threshold_dict[colocal_id]) 
            for k,v in rp_filtercounts.items(): 
                print(colocal_id, k, v)
                print(rp_filtercounts)
                filtered_counts[colocal_id][k] = v
        thresholded_rpdfs[colocal_id] = cdf

    thresholded_rpdf = pd.concat(thresholded_rpdfs.values())
    filtered_counts['final'] = get_colocal_id_counts(thresholded_rpdf, all_colocal_ids)
    return thresholded_rpdf, filtered_counts

def region_prop_table_filter(cdf, t_dict) -> tuple[pd.DataFrame, dict[str, int]]:
    ''' filter region props dataframe using dictionary of min,max vals '''
    filter_counts_post_thresh = {"pre_filters": len(cdf)}
    t_dict = {k: _sanitize_range(v[0], v[1]) for k,v in t_dict.items()}

    for prop_name, (min_value, max_value) in t_dict.items():
        cdf = cdf.loc[(cdf[prop_name] > min_value) & (cdf[prop_name] < max_value)]
        filter_counts_post_thresh[f"post_{prop_name}"] = len(cdf)
        
    return cdf, filter_counts_post_thresh

def _sanitize_range(min_value, max_value) -> tuple[float, float]:
    """ convert none in min or max values to -inf or inf """
    min_limit = float("-inf") if min_value is None else min_value
    max_limit = float("inf") if max_value is None else max_value
    return min_limit, max_limit

def pretty_print_fcounts(filtered_counts):
    """ print counts after each filtering step as a str formatted table """
    count_tabulate = {k:v for k,v in filtered_counts.items() if k not in ['init','final']}
    df_idx = list(count_tabulate[list(count_tabulate.keys())[np.argmax(np.array([len(d) for d in count_tabulate.values()]))]].keys())
    count_f_df = pd.DataFrame.from_dict(count_tabulate).reindex(df_idx)
    print(tabulate(count_f_df.fillna(''), headers='keys'), flush=True)

def get_colocal_id_counts(rpdf, all_colocal_ids=None):
    """ all_colocal_ids is a sorted list of every colocal id found in ImgDB """
    vc = rpdf.value_counts('colocal_id').to_dict()
    all_colocal_ids = all_colocal_ids or sorted(list(vc.keys())) 
    return {i:0 if i not in vc else vc[i] for i in all_colocal_ids}


def _log_or_print(message, logger=None):
    """Helper function to log or print based on logger availability."""
    if logger:
        logger.info(message)
    else:
        print(message)

def separate_colocal_populations(
    all_rpdf, 
    imgdb, 
    merge_columns=['label', 'intensity_mean', 'area', 'eccentricity'], 
    image_id_column='img_i',
    logger=None,
):
    """
    Remove uncolocalized non-unique detections and create dataset from image means.
    Makes colocal_ids contain only non-colocalized spots by separating overlapping populations.
    
    Parameters:
    -----------
    all_rpdf : pd.DataFrame
        Combined dataframe containing region properties with colocal_id column
    imgdb : ImgDB
        Image database object containing colocalization configuration
    merge_columns : list
        Columns to transfer from intersecting regions (first must be 'label')
    image_id_column : str
        Column name to separate labels by their respective image origin
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataframe with separated colocalization populations
    """
    
    # Get initial counts for validation
    all_colocal_ids = list(imgdb.colocal_ids.keys())
    input_clc_counts = get_colocal_id_counts(all_rpdf, all_colocal_ids=all_colocal_ids)
    _log_or_print(f"Input colocal counts: {input_clc_counts}", logger)

    # Split data by image if needed
    if image_id_column is None:
        # Assumes IDs are unique within colocal_ids (single image)
        img_rpdfs = [all_rpdf]
    else:
        img_rpdfs = [
            all_rpdf[all_rpdf[image_id_column] == img_id] 
            for img_id in all_rpdf[image_id_column].unique()
        ]

    cleaned_labels_dfs = []
    
    for img_rpdf in img_rpdfs:
        _log_or_print(f"Processing img_rpdf with shape: {img_rpdf.shape}", logger)
        
        # Get present colocalization info for this image
        present_colocal_ids = sorted(img_rpdf['colocal_id'].unique())
        present_colocalizations = [
            coloc for coloc in imgdb.colocalizations 
            if coloc['assign_colocal_id'] in present_colocal_ids
        ]
        
        present_base_colocal_ids = {c["base_colocal_id"] for c in present_colocalizations}
        present_colocalized_ids = {c["assign_colocal_id"] for c in present_colocalizations}
        
        # Isolate colocal_ids that aren't involved in any colocalizations
        all_involved_ids = present_base_colocal_ids | present_colocalized_ids
        non_intersecting_ids = set(present_colocal_ids) - all_involved_ids
        non_itx_df = img_rpdf[img_rpdf['colocal_id'].isin(non_intersecting_ids)]

        # Track labels that will be removed from each colocal_id
        superseded_clc_lbls = {cid: set() for cid in present_colocal_ids}
        
        # Process each colocalization
        clc_dfs = []
        for coloc in present_colocalizations:
            processed_coloc_df = _process_single_colocalization(
                img_rpdf, coloc, merge_columns, superseded_clc_lbls, imgdb
            )
            clc_dfs.append(processed_coloc_df)
        
        # Combine all colocalized dataframes
        combined_clc_df = pd.concat(clc_dfs, ignore_index=True) if clc_dfs else pd.DataFrame()
        
        # Create clean base dataframes (removing superseded labels)
        clean_base_df = _create_clean_base_dataframes(
            img_rpdf, present_base_colocal_ids, superseded_clc_lbls
        )
        
        # Filter superseded labels from colocalized dataframes
        filtered_clc_df = _filter_superseded_colocalized_labels(
            combined_clc_df, present_colocalized_ids, superseded_clc_lbls
        )
        
        # Combine all cleaned dataframes for this image
        image_cleaned_df = pd.concat([
            non_itx_df,
            clean_base_df, 
            filtered_clc_df
        ], ignore_index=True)
        
        cleaned_labels_dfs.append(image_cleaned_df)
    
    # Combine all images
    cleandf = pd.concat(cleaned_labels_dfs, ignore_index=True)
    
    # Validation: Check for duplicate labels per colocal_id
    _validate_no_duplicate_labels(cleandf)
    
    # Print final counts
    final_clc_counts = get_colocal_id_counts(cleandf, all_colocal_ids=all_colocal_ids)
    _log_or_print(f"Input colocal counts: {input_clc_counts}", logger)
    _log_or_print(f"Final colocal counts: {final_clc_counts}", logger)

    return cleandf


def _process_single_colocalization(img_rpdf, coloc, merge_columns, superseded_clc_lbls, imgdb):
    """Process a single colocalization configuration."""
    coloc_id = coloc['assign_colocal_id']
    base_id = coloc['base_colocal_id']
    itx_ids = coloc['intersecting_colocal_ids']
    itx_col_labels = coloc['intersecting_label_columns']

    # potentially connect logger to display this info
    # print(f"Processing: base_id={base_id}, coloc_id={coloc_id}")
    # print(f"  intersecting_ids={itx_ids}, label_columns={itx_col_labels}")

    # Get base and colocalized dataframes
    base_df = img_rpdf[img_rpdf['colocal_id'] == base_id]
    clcdf = img_rpdf[img_rpdf['colocal_id'] == coloc_id].copy()

    # Merge intersecting region properties
    clcdf = _merge_intersecting_properties(
        clcdf, img_rpdf, itx_ids, itx_col_labels, merge_columns
    )
    
    # Update superseded labels tracking
    _update_superseded_labels(
        img_rpdf, clcdf, base_df, superseded_clc_lbls, base_id, coloc_id, imgdb
    )
    
    return clcdf


def _merge_intersecting_properties(clcdf, img_rpdf, itx_ids, itx_col_labels, merge_columns):
    """Merge intersecting region properties into colocalized dataframe."""
    for itx_id, itx_label_col in zip(itx_ids, itx_col_labels):
        # Get subset of properties to merge
        subset_columns = _handle_merge_columns(img_rpdf, merge_columns)
        subset = img_rpdf[img_rpdf['colocal_id'] == itx_id][subset_columns].copy()
        
        # Rename columns to avoid collisions
        suffix = f"clc{itx_id}"
        column_mapping = {col: f"{suffix}_{col}" for col in merge_columns}
        subset.rename(columns=column_mapping, inplace=True)
        
        # Merge based on intersecting label
        clcdf = clcdf.merge(
            subset, 
            left_on=itx_label_col, 
            right_on=column_mapping['label'], 
            how='left'
        )
    
    return clcdf

def _handle_merge_columns(img_rpdf, merge_columns):
    """ 
    clean and organize columns to merge 
        #TODO shouldn't need predefined merge columns and should be able to infer

    """
    _exclude_cols = ['label']
    cols = set(merge_columns)
    _merge_cols = [c for c in cols if (c in img_rpdf.columns and c not in _exclude_cols)]
    return ['label'] + _merge_cols



def _update_superseded_labels(img_rpdf, clcdf, base_df, superseded_clc_lbls, base_id, coloc_id, imgdb):
    """Update tracking of labels that should be removed from base and child colocalizations."""
    coloc_labels = set(clcdf['label'])
    base_labels = set(base_df['label'])
    
    # Validate data consistency
    base_only_labels = base_labels - coloc_labels
    overlapping_labels = coloc_labels & base_labels
    
    assert len(base_labels) - len(base_only_labels) == len(coloc_labels), \
        "Inconsistent label counts between base and colocalized data"
    assert sum(clcdf['label'].isin(overlapping_labels)) == len(overlapping_labels), \
        "Mismatch in overlapping labels"
    
    # Track labels to remove from base
    superseded_clc_lbls[base_id] = superseded_clc_lbls[base_id] | overlapping_labels

    # Track labels to remove from child colocalizations
    child_clc_ids = set(imgdb.get_inherited_colocalizations(coloc_id)) - {base_id}
    for child_id in child_clc_ids:
        child_labels = set(img_rpdf[img_rpdf['colocal_id'] == child_id]['label'])
        labels_to_remove = coloc_labels & child_labels
        superseded_clc_lbls[child_id] = superseded_clc_lbls[child_id] | labels_to_remove


def _create_clean_base_dataframes(img_rpdf, present_base_colocal_ids, superseded_clc_lbls):
    """Create cleaned base dataframes with superseded labels removed."""
    clean_base_dfs = []
    
    for base_id in present_base_colocal_ids:
        labels_to_remove = superseded_clc_lbls[base_id]
        base_mask = (img_rpdf['colocal_id'] == base_id) & (~img_rpdf['label'].isin(labels_to_remove))
        clean_base_dfs.append(img_rpdf[base_mask])
    
    return pd.concat(clean_base_dfs, ignore_index=True) if clean_base_dfs else pd.DataFrame()


def _filter_superseded_colocalized_labels(clc_df, present_colocalized_ids, superseded_clc_lbls):
    """Filter out superseded labels from colocalized dataframes."""
    if clc_df.empty:
        return clc_df
        
    for clc_id in present_colocalized_ids:
        labels_to_remove = superseded_clc_lbls[clc_id]
        if labels_to_remove:
            removal_mask = (clc_df['colocal_id'] == clc_id) & (clc_df['label'].isin(labels_to_remove))
            clc_df = clc_df[~removal_mask]
    
    return clc_df


def _validate_no_duplicate_labels(cleandf):
    """Validate that there are no duplicate labels per colocal_id."""
    label_counts = cleandf.groupby(['colocal_id', 'label']).size().reset_index(name='count')
    duplicates = label_counts[label_counts['count'] > 1]
    
    if not duplicates.empty:
        dups_per_colocal_id = duplicates.groupby('colocal_id').size().rename('num_duplicate_labels').reset_index()
        raise AssertionError(f"Found duplicate labels per colocal_id:\n{dups_per_colocal_id}")


def extract_labels_channel_intensities(
    GET_CLC_ID_CH_INTENSITIES, raw_img, mask, rpdf, 
    slice_cols_base = ['label'],
    slice_cols_str = 'intensity',
    ):
    """
    get intensity rps for each object in a 2d label mask from other channels in an intensity image (X, Y, C)
    params
    GET_CLC_ID_CH_INTENSITIES: A dictionary mapping colocalization IDs to lists of channels.
        Dict(int:list[int]) for objects in specified channel extract intensity of other channels within object's mask 
        e.g. {0:[1, 2]} = for synapses in ch0 get intensity in that region for chs 1 and 2
    raw_img: intensity image (X, Y, C).
    mask: The mask used to identify regions of interest. Mask must be 2d (X,Y) array.
    rpdf: The DataFrame to be updated with the processed data.
    """
    if GET_CLC_ID_CH_INTENSITIES is not None:
        for clc_id, _chs2get_list in GET_CLC_ID_CH_INTENSITIES.items():
            ch_img_raw = raw_img[..., _chs2get_list]
            stacked_labels = np.stack([mask]*len(_chs2get_list), -1)
            _clc_id_map = dict(zip(_chs2get_list, range(len(_chs2get_list))))
            ch_ints_rpdf = get_rp_table(stacked_labels, ch_img_raw, _clc_id_map)
            merged_df = pivot_columns_on_id(ch_ints_rpdf, slice_cols_base=slice_cols_base, slice_cols_str=slice_cols_str)
            merged_df = merged_df.assign(colocal_id=clc_id)
            rpdf = pd.merge(rpdf, merged_df, how='left', on=['colocal_id'] + slice_cols_base)
    return rpdf

def pivot_columns_on_id(
    ch_ints_rpdf: pd.DataFrame,
    slice_cols_base: List[str] = ['label'],
    slice_cols_str: str = 'intensity',
    id_col: str = 'colocal_id'
) -> pd.DataFrame:
    """
    Transform a pandas DataFrame containing intensity measurements associated with different colocalization IDs (colocal_id). The transformation pivots 
    the DataFrame so that each unique colocal_id has its own set of intensity columns, effectively spreading the data into a wider format.
    
    Parameters:
        ch_ints_rpdf (pd.DataFrame): DataFrame containing intensity data with a 'colocal_id' column.
        slice_cols_base (List[str], optional): Base columns to retain for slicing. Defaults to ['label'].
        slice_cols_str (str, optional): Prefix string to identify intensity columns. Defaults to 'intensity'.
        id_col (str, optional): grouping col to do the pivot on

    Returns:
        pd.DataFrame: A pivoted DataFrame with intensity columns suffixed by their corresponding colocal_id.
    """
    # slice just columns that contain 'slice_cols_str' (e.g. those that contain 'intensity')
    intensity_cols = [f"{el}" for el in ch_ints_rpdf.columns if el.startswith(slice_cols_str)]
    int_dfs = []
    for _clc_id in ch_ints_rpdf[id_col].unique():
        int_df = ch_ints_rpdf[ch_ints_rpdf[id_col]==_clc_id][slice_cols_base + intensity_cols]
        int_dfs.append(int_df.rename(columns=dict(zip(intensity_cols, [f"{el}_ch{_clc_id}" for el in intensity_cols]))))
    # merge the ch dfs on the label 
    merged_df = functools.reduce(lambda left, right: pd.merge(left, right, on=slice_cols_base, how='left'), int_dfs)
    return merged_df

def calculate_distances_from_point(df, point, centroid_col='centroid', distance_col='distance'):
    """
    Calculate the Euclidean distance from a given point to each centroid in the DataFrame.

    Parameters:
    - df: pandas DataFrame containing a 'centroid' column with coordinates (x, y) or (x, y, z).
    - point: Tuple representing the given point (e.g., (x0, y0) or (x0, y0, z0)).
    - centroid_col: Name of the column containing centroid coordinates. Default is 'centroid'.

    Returns:
    - df: pandas DataFrame with an additional 'distance' column.
    """

    def euclidean_distance(centroid, point):
        return np.sqrt(np.sum((np.array(centroid) - np.array(point))**2))

    df[distance_col] = df[centroid_col].apply(lambda centroid: euclidean_distance(centroid, point))

    return df


def convert_coordinates_image_to_geometric(coordinate_list):
    """
    Flip x and y coordinates in each numpy array in the given list.
        reason is that image coordinates and geometric/Cartesian coordinates use different coordinate systems
        Image Coordinate System: (0,0) is at the top-left corner, which is equivalent to (y, x)
        Cartesian/Geometric Coordinate System: bottom-left corner, Coordinates are in (x, y) format
        
    Parameters:
        coordinate_list: List of numpy arrays where each array has shape (N, 2)
                        representing x,y coordinates

    Returns:
        List of numpy arrays with flipped coordinates (y,x instead of x,y)
    """
    return [arr[:, ::-1] for arr in coordinate_list]


def plot_rp_values(
        rpdf, colocal_ids=None, 
        rps=['area', 'intensity_mean', 'eccentricity'], 
        clc_colors=['cyan', 'green', 'red']
    ):
    
    if colocal_ids is None:
        colocal_ids = sorted(list(rpdf['colocal_id'].unique()))
    else:
        colocal_ids = [el for el in colocal_ids if el in rpdf['colocal_id'].unique()]
    
    palette = dict(zip(colocal_ids, clc_colors))

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    for clc_id in colocal_ids:
        pltdf = rpdf[rpdf['colocal_id']==clc_id]
        ax.scatter(pltdf[rps[0]], pltdf[rps[1]], pltdf[rps[2]], ec=palette[clc_id], fc='none')

    x_min, x_max = 0, 200
    y_min, y_max = 5000, rpdf['intensity_mean'].max()
    z_min, z_max = rpdf['eccentricity'].min(), 1.5

    corners = np.array([
        [x_min, y_min, z_min],
        [x_min, y_min, z_max],
        [x_min, y_max, z_min],
        [x_min, y_max, z_max],
        [x_max, y_min, z_min],
        [x_max, y_min, z_max],
        [x_max, y_max, z_min],
        [x_max, y_max, z_max]
    ])

    # Lines to connect the corners and form the cube
    # Each pair of numbers corresponds to corners to connect
    lines = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # back face
        (4, 5), (5, 7), (7, 6), (6, 4),  # front face
        (0, 4), (1, 5), (3, 7), (2, 6)   # connecting lines
    ]

    # Plot lines
    for start, end in lines:
        ax.plot3D(*zip(*corners[[start, end]]), color='b', alpha=0.5)


    ax.set_xlabel(rps[0])
    ax.set_ylabel(rps[1])
    ax.set_zlabel(rps[2])
    ax.set_xlim(0,600)
    ax.set_zlim(1,3)
    ax.set_title(f"min/max X {x_min},{x_max}, Y: {y_min},{y_max}, Z: {z_min}, {z_max}")
    plt.show()

def plot_rps_kde(rpdf):
    
    

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    kde_cmaps = ['Blues_r', 'Greens_r']

    for clc_id in sorted(list(rpdf['colocal_id'].unique()))[:1]:
        pltdf = rpdf[rpdf['colocal_id']==clc_id]
        rps=['area', 'intensity_mean', 'eccentricity']
        x,y,z = pltdf[rps[0]], pltdf[rps[1]], pltdf[rps[2]]
        
        xyz = np.vstack([x,y,z])
        density = stats.gaussian_kde(xyz)(xyz) 

        idx = density.argsort()
        ax.scatter(x[idx], y[idx], z[idx], c=density[idx], cmap=kde_cmaps[clc_id])
    plt.show()

def plot_fancy_2d_hist(rpdf, attr1, attr2, plot_means=True, ax_scales=['linear', 'linear'], **joint_plot_kwargs):
    
    sns.jointplot(
        data=rpdf, x=attr1, y=attr2,
        marker="+", s=100, marginal_kws=dict(bins=25, fill=False),
        **joint_plot_kwargs,
    )
    
    # plot mean lines
    if plot_means:
        plt.axvline(rpdf[attr1].mean(), ls='--', c='k', alpha=0.4, zorder=-1)
        plt.axhline(rpdf[attr2].mean(), ls='--', c='k', alpha=0.4, zorder=-1)
    
    # set axis scale
    plt.xscale(ax_scales[0])
    plt.yscale(ax_scales[1])

    plt.show()


def plot_1d_hist(rpdf, attr, nbins=None):
    if nbins is None:
        if 'intensity' in attr:
            int_max = rpdf['intensity_min'].max()
            if int_max <= 1.0:
                nbins='auto'
            elif int_max < 256:
                nbins = int(int_max/10)
            else: # e.g. 16 bit intensities
                nbins = int(int_max/1000)
        else:
            nbins='auto'

    fig,ax=plt.subplots()
    hp = sns.histplot(data=rpdf, bins=nbins, x=attr, ax=ax)
    plt.show()

def plot_attrs_hist(
        rpdf, 
        plot_params=None, 
        id_vars=['label', 'colocal_id'], 
        palette=dict(zip(range(3),list('rgb'))), 
        hue='colocal_id', 
        col=None,
        save_path=None,
    ):
    """ 
    plot all histogram for each attr, separating attr by rows, and overlaying colocal_ids
    
    Args:
        plot_params: list of str (columns to plot)
            defaults to ['area', 'intensity_mean', 'axis_major_length', 'eccentricity']
    """

    if plot_params is None:
        plot_params = ['area', 'intensity_mean', 'axis_major_length', 'eccentricity']

    hist_df = pd.melt(rpdf, id_vars=id_vars, value_vars=plot_params, var_name='attribute', value_name='value')
    dp = sns.displot(data=hist_df, x='value', row='attribute', col=col, hue=hue, palette=palette, common_bins=False, facet_kws=dict(sharex=False, sharey=True))


from typing import Union, List, Tuple, Optional, Any


def sort_coordinates_by_distance(
    coordinates: Union[List[Tuple[float, ...]], np.ndarray],
    centroid: Optional[Tuple[float, ...]] = None,
    metric: Union[str, Any] = 'euclidean',
) -> Tuple[List[Tuple[float, ...]], Tuple[float, ...]]:
    """
    Sorts a list of 2D or 3D coordinates in order of increasing distance from a reference point.

    Parameters
    ----------
    coordinates : List[Tuple[float, ...]] or np.ndarray
        A list or NumPy array of shape (N, D), where D is 2 or 3.
    centroid : Optional[Tuple[float, ...]], optional
        A distinct (x, y) or (x, y, z) coordinate used as the reference point.
        If not provided, the centroid of the coordinates will be used.
    metric : str or callable, default 'euclidean'
        Distance metric to use ('euclidean', 'manhattan', 'chebyshev') or a custom function.

    Returns
    -------
    sorted_coordinates : List[Tuple[float, ...]]
        Input coordinates sorted by increasing distance to the centroid.
    used_centroid : Tuple[float, ...]
        The centroid used for distance calculations.

    Raises
    ------
    ValueError
        If input list is empty or has invalid shape.
    TypeError
        If input types are invalid.
    """
    # Convert input to numpy array
    if isinstance(coordinates, list):
        coords = np.array(coordinates)
    elif isinstance(coordinates, np.ndarray):
        coords = coordinates
    else:
        raise TypeError("coordinates must be a list of tuples or a NumPy array.")

    if coords.ndim != 2 or coords.shape[1] not in (2, 3):
        raise ValueError("coordinates must be of shape (N, 2) or (N, 3).")
    if coords.size == 0:
        raise ValueError("The list of coordinates is empty.")

    # Set centroid
    if centroid is None:
        used_centroid = tuple(np.mean(coords, axis=0))
    else:
        if not isinstance(centroid, tuple) or len(centroid) != coords.shape[1]:
            raise TypeError(f"centroid must be a tuple of length {coords.shape[1]}.")
        used_centroid = centroid

    centroid_arr = np.array(used_centroid)

    # Distance computation
    if callable(metric):
        distances = metric(coords, centroid_arr)
    elif metric == 'euclidean':
        distances = np.linalg.norm(coords - centroid_arr, axis=1)
    elif metric == 'manhattan':
        distances = np.sum(np.abs(coords - centroid_arr), axis=1)
    elif metric == 'chebyshev':
        distances = np.max(np.abs(coords - centroid_arr), axis=1)
    else:
        raise ValueError("Unsupported metric. Use 'euclidean', 'manhattan', 'chebyshev', or a callable.")

    # Sort and return
    sorted_indices = np.argsort(distances)
    sorted_coords = coords[sorted_indices]
    return [tuple(c) for c in sorted_coords], used_centroid


def filter_coordinates(
    coordinates: Union[List[Tuple[float, float]], np.ndarray],
    boundary: Tuple[float, float, float, float]
) -> Union[List[Tuple[float, float]], np.ndarray]:
    """
    Filters coordinates to include only those within the specified boundary.
    
    Parameters
    ----------
    boundary : Tuple[float, ...]
        Defines the rectangular boundary as (min_x, min_y, max_x, max_y).
    
    Returns
    -------
    filtered_coordinates : Same type as input
    """
    min_x, min_y, max_x, max_y = boundary
    mask = (coordinates[:, 0] >= min_x) & (coordinates[:, 0] <= max_x) & \
           (coordinates[:, 1] >= min_y) & (coordinates[:, 1] <= max_y)
    return coordinates[mask]





def assign_labels(
    object_centroids: Union[np.ndarray, List[Tuple[float, float]]],
    polygons_per_label: Dict[int, List[Polygon]]
) -> Tuple[List[int], List[int]]:
    """
    Given a list/array of object centroids (x, y) and a dictionary of label->[Polygons],
    return two lists of the same length:
        - assigned_labels: label ID if the point is inside that label's polygon(s), or 0 if no match.
        - polygon_label_subindex_found: index of the polygon within its label's list where the point was found, or -1 if no match.

    Parameters
    ----------
    object_centroids : np.ndarray or List[Tuple[float, float]]
        An array/list of shape (N, 2), each row is (x, y).
    polygons_per_label : Dict[int, List[shapely.geometry.Polygon]]
        A dictionary where the key is the label and the value is a list of polygons
        belonging to that label.

    Returns
    -------
    assigned_labels : List[int]
        A list of label IDs (or 0 if not found) for each centroid.
    polygon_label_subindex_found : List[int]
        A list of polygon sub-indices within their label's polygon list where the point was found, or -1 if not found.
        
    
    Example usage
    -------------
    # Sample object centroids
    object_centroids_ex = np.array([
        (1.5, 1.5),
        (3.0, 3.0),
        (5.0, 5.0),
        (99.0, 3.0)
    ])

    # Sample polygons per label
    polygons_per_label_ex = {
        1: [Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]), Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])],
        2: [Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])],
        3: [Polygon([(4, 4), (4, 6), (6, 6), (6, 4)])]
    }

    # Assign labels to centroids
    labels_ex, polygon_label_subindex_found = assign_labels(object_centroids_ex, polygons_per_label_ex)

    print(labels_ex)  # Output: [1, 2, 3]
    """
    from shapely.strtree import STRtree

    # Ensure object_centroids is a NumPy array for efficient processing
    if isinstance(object_centroids, list):
        object_centroids = np.array(object_centroids)
    elif not isinstance(object_centroids, np.ndarray):
        raise TypeError("object_centroids must be a NumPy array or a list of tuples.")

    if len(object_centroids) == 0:
        return [], []
    
    # Prepare a list of all polygons along with their corresponding labels and sub-indices
    all_polygons = []
    polygon_label_map = []
    polygon_label_subindex = []  # Indices of the polygons within their label key

    for label, polygons in polygons_per_label.items():
        all_polygons.extend(polygons)
        polygon_label_map.extend([label] * len(polygons))
        # Enumerate polygons to track their sub-indices within the label
        polygon_label_subindex.extend(list(range(len(polygons))))

    # Build a spatial index for all polygons
    # note: STRtree operates on the bounding boxes of the polygons, so this step only limits the possible polygons
    # that contain the point, then we check for actual containment with polygon.contains(point)
    spatial_index = STRtree(all_polygons)

    assigned_labels = []
    polygon_label_subindex_found = []  # List to store the sub-index of the matching polygon

    for centroid in object_centroids:
        point = Point(centroid)
        # Query the spatial index for possible containing polygons
        possible_polygons = spatial_index.query(point)  # Returns indices of polygons whose bounding boxes contain the point

        label_found = 0  # Default label if no polygon contains the point
        subindex_found = -1  # Default sub-index if no polygon contains the point

        for polygon_i in possible_polygons:
            polygon = all_polygons[polygon_i]

            if polygon.contains(point):
                # Retrieve the label corresponding to this polygon
                label_found = polygon_label_map[polygon_i]
                # Retrieve the sub-index of this polygon within its label
                subindex_found = polygon_label_subindex[polygon_i]
                break  # Stop after the first matching polygon is found

        assigned_labels.append(label_found)
        polygon_label_subindex_found.append(subindex_found)

    return assigned_labels, polygon_label_subindex_found

def assign_labels_vectorized(object_centroids, polygons_per_label):
    # TODO this may be faster than my implementation above, but I haven't tested it yet
    # on first glance doesn't seem to handle polygon sub-indices correctly
    
    if len(object_centroids) == 0:
        return [], []

    # 1. Flatten polygons and track their metadata
    all_polygons = []
    metadata = [] # stores (label, subindex)
    for label, polys in polygons_per_label.items():
        for i, poly in enumerate(polys):
            all_polygons.append(poly)
            metadata.append((label, i))

    # 2. Build the tree
    tree = STRtree(all_polygons)
    
    # 3. Convert input to Shapely Points
    # Shapely 2.0 can handle arrays of points efficiently
    points = [Point(c) for c in object_centroids]
    
    # 4. Query the tree for ALL points at once
    # This returns an array of [polygon_indices, point_indices]
    poly_idx, pt_idx = tree.query(points, predicate="contains")

    # Initialize results with defaults
    assigned_labels = [0] * len(object_centroids)
    subindices = [-1] * len(object_centroids)

    # 5. Map results back
    # Because 'contains' was passed to the query, 
    # we don't even need a second manual loop for polygon.contains()!
    for p_idx, t_idx in zip(poly_idx, pt_idx):
        label, sub = metadata[p_idx]
        assigned_labels[t_idx] = label
        subindices[t_idx] = sub
        
    return assigned_labels, subindices

def assign_labels_to_object_indices(
    object_indices_list: List[np.ndarray],
    polygons_per_label: Dict[int, List[Polygon]]
) -> Tuple[List[int], List[int]]:
    """
    Given a list/array of object indices (x, y) and a dictionary of label->[Polygons],
    return two lists of the same length:
        - assigned_labels: label ID if the object is inside that label's polygon(s), or 0 if no match.
        - polygon_label_subindex_found: index of the polygon within its label's list where the object was found, or -1 if no match.
        Note: tie-breakers are not implemented. obj is assigned to poly the first time point_in_poly(coord) returns true 
            In quant pipeline roi_handling (via uc.sort_coordinates_by_distance) the search order of coords starts 
            with the objects centroid and extends outward

    Parameters
    ----------
    object_indices : list of np.ndarray corresponding to an array of spatial coordinates for each object
            arrays must be of shape (N, 2), each row is (x, y).
    polygons_per_label : Dict[int, List[shapely.geometry.Polygon]]
        A dictionary where the key is the label and the value is a list of polygons
        belonging to that label.

    Returns
    -------
    assigned_labels : List[int]
        A list of label IDs (or 0 if not found) for each object index.
    polygon_label_subindex_found : List[int]
        A list of polygon sub-indices within their label's polygon list where the object was found, or -1 if not found.
        
    Example usage
    -------------
    # Sample object indices (x, y) in a 2D array
    object_indices_ex = [
        np.array([
        (1.5, 1.5),   # Should be in label 1, sub-index 0
        (3.0, 3.0),   # Could be in label 1, sub-index 1 or label 2, sub-index 0
        (5.0, 5.0),   # Should be in label 3, sub-index 0
        (99.0, 3.0)   # Should have no match
        ]),
        np.array([
        (3.0, 3.0),   
        (5.0, 5.0),   
        (99.0, 3.0)   
        ]),
        np.array([
        (5.0, 5.0),   
        (99.0, 3.0)   
        ])
        ]

    # Sample polygons per label
    polygons_per_label_ex = {
        1: [
            Polygon([(0, 0), (0, 2), (2, 2), (2, 0)]),  # Sub-index 0
            Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])   # Sub-index 1
        ],
        2: [
            Polygon([(2, 2), (2, 4), (4, 4), (4, 2)])   # Sub-index 0
        ],
        3: [
            Polygon([(4, 4), (4, 6), (6, 6), (6, 4)])   # Sub-index 0
        ]
    }

    # Assign labels to object indices
    labels_ex, polygon_subindices_ex = assign_labels_to_object_indices(object_indices_ex, polygons_per_label_ex)

    print("Assigned Labels:", labels_ex)
    print("Polygon Sub-Indices:", polygon_subindices_ex)
    """
    # Ensure object_indices is a NumPy array for efficient processing
    assert isinstance(object_indices_list, list), ("object_indices must be a NumPy array or a list of tuples.")
    if len(object_indices_list) == 0:
        return [], []
    
    # Prepare a list of all polygons along with their corresponding labels and sub-indices
    all_polygons = []
    polygon_label_map = []
    polygon_label_subindex = []  # Indices of the polygons within their label key

    for label, polygons in polygons_per_label.items():
        all_polygons.extend(polygons)
        polygon_label_map.extend([label] * len(polygons))
        # Enumerate polygons to track their sub-indices within the label
        polygon_label_subindex.extend(list(range(len(polygons))))

    
    if not all_polygons:
        # If there are no polygons, return 0 labels and -1 sub-indices for all object indices
        num_objects = len(object_indices_list)
        return [0] * num_objects, [-1] * num_objects

    # Prepare polygons for faster spatial operations
    prepared_polygons = [prep(polygon) for polygon in all_polygons]

    # Build a spatial index for all polygons
    spatial_index = STRtree(all_polygons)

    assigned_labels = []
    polygon_label_subindex_found = []  # List to store the sub-index of the matching polygon

    # iter each object
    for object_indicies in object_indices_list:
        label_found = 0  # Default label if no polygon contains the point
        subindex_found = -1  # Default sub-index if no polygon contains the point
        _running = True
        
        # iter each coord in an objects coordinates
        for obj_coord in object_indicies:
            point = Point(obj_coord)
            # Query the spatial index for possible containing polygons
            possible_polygons = spatial_index.query(point)

            for polygon_i in possible_polygons:
                # Retrieve the index of the polygon in all_polygons
                # polygon = all_polygons[polygon_i]
                if prepared_polygons[polygon_i].contains(point):
                    label_found = polygon_label_map[polygon_i]
                    subindex_found = polygon_label_subindex[polygon_i]
                    _running = False
                    break  # Stop after the first matching polygon is found
            if not _running:
                break

        assigned_labels.append(label_found)
        polygon_label_subindex_found.append(subindex_found)

    return assigned_labels, polygon_label_subindex_found


def compute_distances_to_rois(polygons_per_label, object_centroids):
    """
    Computes the distance from each centroid to the nearest ROI and determines if it's within a threshold.

    Parameters:
    ----------
    polygons_per_label : dict
        Dictionary where keys are ROI labels (indices) and values are lists of Shapely Polygon objects.
        Example:
            {
                1: [Polygon1, Polygon2, ...],
                2: [Polygon3, Polygon4, ...],
                ...
            }

    object_centroids : numpy.ndarray
        NumPy array of shape (n_objects, 2), where each row represents the (x, y) coordinates of a centroid.


    Returns:
    -------
    distances : numpy.ndarray
        Array of shape (n_objects,) containing the distance from each centroid to its nearest ROI.

    
    nearest_labels : numpy.ndarray
        Array of shape (n_objects,) containing the label of the nearest ROI for each centroid.
        If no ROI is within the threshold, the label will correspond to the nearest ROI regardless of the threshold.
    
    nearest_poly_subindex
    
    Example usage
    -------------
        # Sample polygons per label
        polygons_per_label_ex = {
            1: [Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), Polygon([(3, 3), (5, 3), (5, 5), (3, 5)])],
            2: [Polygon([(10, 10), (12, 10), (12, 12), (10, 12)])]
        }

        # Sample centroids
        object_centroids_ex = np.array([
            [1, 1],    # Inside ROI 1
            [4, 4],    # Inside ROI 1 (second polygon)
            [6, 6],    # Near ROI 1 second polygon
            [11, 11],  # Inside ROI 2
            [15, 15]   # Far from any ROI
        ])

        # Define distance threshold
        distance_threshold_ex = 2.5

        # Compute distances and threshold mask
        distances, within_threshold, nearest_labels = compute_distances_to_rois(
            polygons_per_label_ex,
            object_centroids_ex,
            distance_threshold_ex
        )

    """
    
    # Step 1: Flatten the polygons and keep track of their labels
    polygons = []
    labels = []
    polygon_label_subindex = []  # Indices of the polygons within their label key
    for label, polygons in polygons_per_label.items():
        polygons.extend(polygons)
        labels.extend([label] * len(polygons))
        # Enumerate polygons to track their sub-indices within the label
        polygon_label_subindex.extend(list(range(len(polygons))))
    
    if not polygons:
        raise ValueError("The polygons_per_label dictionary is empty. Please provide at least one polygon.")

    # Step 2: Build a spatial index for efficient querying
    tree = STRtree(polygons)
    
    # Step 3: Create a mapping from polygon id to its index for quick label retrieval
    poly_id_to_index = {id(poly): idx for idx, poly in enumerate(polygons)}
    
    # Step 4: Initialize output arrays
    n_objects = object_centroids.shape[0]
    distances = np.empty(n_objects)
    within_threshold = np.empty(n_objects, dtype=bool)
    nearest_labels = np.empty(n_objects, dtype=object)
    nearest_polygon_subindicies = np.empty(n_objects, dtype=object)

    # Step 5: Iterate over each centroid to find the nearest ROI
    for i, centroid in enumerate(object_centroids):
        point = Point(centroid)
        
        # Find the nearest polygon to the current point
        nearest_poly_index = tree.nearest(point)
        nearest_poly = polygons[nearest_poly_index]
                        
        # Calculate the distance to the nearest polygon
        min_dist = point.distance(nearest_poly)
        distances[i] = min_dist
        
        # Determine if the distance is within the specified threshold
        # within_threshold[i] = min_dist <= distance_threshold
        
        # Retrieve the label of the nearest polygon
        nearest_labels[i] = labels[nearest_poly_index]
        nearest_polygon_subindicies[i] = polygon_label_subindex[nearest_poly_index]
    
    return nearest_labels, nearest_polygon_subindicies, distances



def semantic_to_polygons_rasterio(sem_image):
    """
    Convert a 2D semantic segmentation image into polygon boundaries using 
    bbox cropping for performance and rasterio for edge-exact polygonization.
        using bbox cropping --> 20x speedup on large images with sparse labels
            7 sec vs 150 sec for (20352, 31800) array

    Parameters
    ----------
    sem_image : np.ndarray
        A 2D integer array of shape (H, W) where each pixel is a label.
        E.g., 0 = background, 1 = object1, 2 = object2, etc.
    
    Returns
    -------
    polygons_dict : dict
        Dictionary of the form: { label_value: [Polygon, Polygon, ...], ... }
        Each label key maps to one or more Shapely Polygons representing
        the regions of that label.
    """
    from rasterio.features import shapes as rf_shapes
    from scipy.ndimage import find_objects
    polygons_dict = {}
    
    slices = find_objects(sem_image) 

    for i, bbox_slice in enumerate(slices):
        if bbox_slice is None:
            continue
        
        label = i + 1  # find_objects is 1-indexed (index 0 is label 1)
        
        # 1. Crop the image to the bounding box
        # bbox_slice is (slice(y_min, y_max), slice(x_min, x_max))
        crop = sem_image[bbox_slice]
        y_offset, x_offset = bbox_slice[0].start, bbox_slice[1].start
        
        # 2. Create the local mask
        mask = (crop == label)
        
        # 3. Define the transform with the offset
        # Affine(a, b, c, d, e, f) where c is x-translation and f is y-translation
        # This automatically shifts the resulting polygons back to global coordinates
        transform = rasterio.Affine.translation(x_offset, y_offset)
        
        # 4. Extract shapes
        results = rf_shapes(
            mask.astype(np.uint8), 
            mask=mask, 
            transform=transform
        )
        
        region_polygons = []
        for geom, value in results:
            poly = shape(geom)
            if not poly.is_empty:
                region_polygons.append(poly)
        
        # 5. Merge fragments
        if region_polygons:
            merged = unary_union(region_polygons)
            if merged.geom_type == 'MultiPolygon':
                polygons_dict[label] = list(merged.geoms)
            else:
                polygons_dict[label] = [merged]
        else:
            polygons_dict[label] = []

    return polygons_dict




def compare_polygon_dicts(dict_orig, dict_new, tol=1e-7):
    """
    Compares two dictionaries of {label: [Polygons]} to ensure they are 
    spatially equivalent within a floating-point tolerance.
    """
    # 1. Check if they have the same set of labels
    if set(dict_orig.keys()) != set(dict_new.keys()):
        print(f"❌ Label mismatch! Orig labels: {dict_orig.keys()}, New labels: {dict_new.keys()}")
        return False

    for label, polys_orig in dict_orig.items():
        polys_new = dict_new.get(label, [])

        if len(polys_orig) != len(polys_new):
            print(f"❌ Label {label}: Different number of polygons ({len(polys_orig)} vs {len(polys_new)})")
            return False

        # Sort polygons by area or centroid to ensure we are comparing the same objects
        # (rf_shapes may return geometries in a different order)
        polys_orig = sorted(polys_orig, key=lambda p: (p.area, p.centroid.x))
        polys_new = sorted(polys_new, key=lambda p: (p.area, p.centroid.x))

        for p_orig, p_new in zip(polys_orig, polys_new):
            # Check Area
            if not math.isclose(p_orig.area, p_new.area, rel_tol=tol):
                print(f"❌ Label {label}: Area mismatch ({p_orig.area} vs {p_new.area})")
                return False
            
            # Check Centroid Coordinates (Rounding errors usually happen here)
            if not math.isclose(p_orig.centroid.x, p_new.centroid.x, rel_tol=tol) or \
               not math.isclose(p_orig.centroid.y, p_new.centroid.y, rel_tol=tol):
                print(f"❌ Label {label}: Centroid mismatch")
                return False

    print("✅ All geometries match perfectly (within tolerance).")
    return True


def plot_polygons_over_image(sem_image, polygons_dict):
    plt.figure(figsize=(12, 12))
    plt.imshow(sem_image, cmap="gray")
    # plt.gca().invert_yaxis()  # uncomment if you want cartesian-like axes
    
    colors = ["red", "green", "blue", "yellow", "magenta"]
    seen_labels = set()
    for i, (label, polys) in enumerate(polygons_dict.items()):
        for poly in polys:
            x, y = poly.exterior.xy
            color = colors[i % len(colors)]
            plt.plot(x, y, color=color, linewidth=2, label=f"Label {label}" if label not in seen_labels else None)
            seen_labels.add(label)
            
            
            # If there are holes:
            for hole in poly.interiors:
                hx, hy = hole.coords.xy
                plt.plot(hx, hy, color=color, linewidth=2, linestyle="--")

    plt.title("Polygons Overlaid on Semantic Label Image")
    # Show legend (optional—this can get crowded if many labels)
    plt.legend(loc="best")
    plt.show()


def test_semantic_to_polygons_rasterio():
    # Example usage
    sem_image = np.array([
        [1, 1, 0, 0, 0],
        [1, 0, 0, 2, 2],
        [1, 1, 1, 2, 2],
        [0, 1, 1, 2, 2],
        [0, 0, 0, 0, 0],
    ], dtype=np.uint8)

    polygons_per_label = semantic_to_polygons_rasterio(sem_image)
    plot_polygons_over_image(sem_image, polygons_per_label)


def format_geometric_to_image_coords(geom_coords, bounds):
    height, width = bounds
    image_coords = []
    
    if len(geom_coords) == 0:
        return image_coords
    
    if not isinstance(geom_coords, list):
        geom_coords = [geom_coords]
    
    assert isinstance(geom_coords, list) and isinstance(geom_coords[0], np.ndarray), 'input geom_coords is not list of array'
    
    
    for coords in geom_coords:
        # Convert (x, y) to (row, col)
        rows = coords[:, 1]  # Flip y-axis
        cols = coords[:, 0]
        
        # Ensure coordinates are within image bounds
        rows = np.clip(rows, 0, height - 1)
        cols = np.clip(cols, 0, width - 1)
        image_coords.append(np.vstack((rows, cols)).T)
    return image_coords

def filter_coordinate_set(exteriors, interiors):
    """
    This function removes any points in `exteriors` that also appear in `interiors`.
    
    Given two lists of 2D NumPy arrays:
    - `exteriors`: list of exterior coordinates (each array is shape (N, 2))
    - `interiors`: list of interior coordinates (each array is shape (M, 2))
    
    Returns a new list of filtered exterior arrays.
    """
    # 1. Gather all interior coordinates into a set of tuples
    interior_points_set = set()
    for interior_array in interiors:
        for point in interior_array:
            interior_points_set.add(tuple(point))  # convert [x, y] to (x, y) to store in a set

    # 2. For each exterior array, filter out points that are in the interior_points_set
    filtered_exteriors = []
    for exterior_array in exteriors:
        # Keep only the points not in the interior set
        filtered_array = np.array(
            [pt for pt in exterior_array if tuple(pt) not in interior_points_set]
        )
        filtered_exteriors.append(filtered_array)

    return filtered_exteriors




def convert_geometric_to_image_coords(geom, image_shape):
    """
    Converts a Shapely Polygon or MultiPolygon from geometric (x, y)
    coordinates to image (row, col) coordinates.

    Parameters
    ----------
    geom : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        The geometry in geometric coordinates.
    image_shape : Tuple[int, int]
        The shape of the image as (height, width).

    Returns
    -------
    ext_coords : List[np.ndarray]
        List of arrays of shape (N, 2) with (row, col) coordinates for exteriors.
    int_coords : List[np.ndarray]
        List of arrays of shape (M, 2) with (row, col) coordinates for all interiors.
    """
    from shapely.geometry import Polygon, MultiPolygon
    
    height, width = image_shape[:2]

    # Normalize to a list of polygons
    if isinstance(geom, Polygon):
        polygons = [geom]
    elif isinstance(geom, MultiPolygon):
        polygons = list(geom.geoms)
    else:
        raise TypeError(f"Expected Polygon or MultiPolygon, got {type(geom)}")

    ext_coords = []
    int_coords = []

    for poly in polygons:
        # Exterior
        ext = np.asarray(poly.exterior.coords)
        ext_img = format_geometric_to_image_coords(ext, (height, width))
        if len(ext_img)>1:
            raise ValueError(len(ext_img))
        ext_img = ext_img[0]
        ext_coords.append(ext_img)

        # Interiors (holes)
        for ring in poly.interiors:
            ring_arr = np.asarray(ring.coords)
            ring_img = format_geometric_to_image_coords(ring_arr, (height, width))
            if len(ring_img)>1:
                raise ValueError(len(ring_img))
            ring_img = ring_img[0]
            int_coords.append(ring_img)

    return ext_coords, int_coords

def combine_add_sub(add_list, subtract_list):
    """
    Given two lists of Boolean arrays:
    - add_list: list of Boolean arrays to be combined (added).
    - subtract_list: list of Boolean arrays to be subtracted.
    
    Returns a single Boolean array where:
    result = OR over all add_list  AND  NOT(OR over all subtract_list)
    """
    # 1. Combine (logical OR) all arrays in each list
    added_mask = np.logical_or.reduce(add_list)
    subtracted_mask = np.logical_or.reduce(subtract_list)
    
    result = added_mask & ~subtracted_mask # "subtract" means we want to turn False where subtracted_mask is True

    return result


import numba as nb

# match the enum values they use (for readability)
OUTSIDE = 0
INSIDE  = 1
VERTEX  = 2
EDGE    = 3


@nb.njit(cache=True)
def point_in_polygon_exact(xp, yp, x, y):
    """
    Numba clone of skimage's Cython point_in_polygon.

    Parameters
    ----------
    xp, yp : 1D float64 arrays
        Polygon vertex coordinates (x, y).
    x, y : float
        Point to test.

    Returns
    -------
    code : int
        OUTSIDE = 0, INSIDE = 1, VERTEX = 2, EDGE = 3
    """
    nr_verts = xp.shape[0]
    eps = 1e-12

    # Initialization
    x1 = xp[nr_verts - 1] - x
    y1 = yp[nr_verts - 1] - y

    l_cross = 0
    r_cross = 0

    for i in range(nr_verts):
        x0 = xp[i] - x
        y0 = yp[i] - y

        # vertex with eps tolerance
        if (-eps < x0 < eps) and (-eps < y0 < eps):
            return VERTEX

        # edge e = (i-1, i) straddles x-axis?
        if ((y0 > 0) != (y1 > 0)):
            # crosses ray to the right?
            num = (x0 * y1 - x1 * y0)
            den = (y1 - y0)
            if (num / den) > 0:
                r_cross += 1

        # reversed edge straddles x-axis?
        if ((y0 < 0) != (y1 < 0)):
            # crosses ray to the left?
            num = (x0 * y1 - x1 * y0)
            den = (y1 - y0)
            if (num / den) < 0:
                l_cross += 1

        x1 = x0
        y1 = y0

    # on edge if left and right crossings not of same parity
    if (r_cross & 1) != (l_cross & 1):
        return EDGE

    # inside if odd number of crossings
    if (r_cross & 1) == 1:
        return INSIDE

    # outside if even number of crossings
    return OUTSIDE

@nb.njit(cache=True, parallel=True)
def _polygon2mask_numba(rows, cols, mask):
    """
    Fill `mask` in-place for a single polygon given (rows, cols).
        note: without parallelism, execution time is slightly longer than skimage's for huge images

    Parameters
    ----------
    rows, cols : 1D float64 arrays
        Polygon vertices (row, col).
    mask : 2D bool array
        Output mask, modified in-place.
    """
    H, W = mask.shape

    # tight bounding box to limit work
    minr = int(np.floor(rows.min()))
    maxr = int(np.ceil(rows.max()))
    minc = int(np.floor(cols.min()))
    maxc = int(np.ceil(cols.max()))

    if minr < 0:
        minr = 0
    if minc < 0:
        minc = 0
    if maxr >= H:
        maxr = H - 1
    if maxc >= W:
        maxc = W - 1

    for r in nb.prange(minr, maxr + 1):
        for c in range(minc, maxc + 1):
            # point position test (use integer coords, as skimage does)
            code = point_in_polygon_exact(cols, rows, float(c), float(r))
            if code != OUTSIDE:  # INSIDE / VERTEX / EDGE all considered "inside"
                mask[r, c] = True


def polygon2mask_numba(image_shape, polygon):
    """
    Numba-accelerated polygon2mask intended to match
    skimage.draw.polygon2mask exactly.

    Parameters
    ----------
    image_shape : (H, W)
    polygon : (N, 2) array-like of (row, col) vertices

    Returns
    -------
    mask : (H, W) boolean ndarray
    """
    mask = np.zeros(image_shape, dtype=np.bool_)
    poly = np.asarray(polygon, dtype=np.float64)

    rows = poly[:, 0]
    cols = poly[:, 1]

    _polygon2mask_numba(rows, cols, mask)
    return mask


def _warmup_poly2mask_numba():
    a = np.array([[0,0],[1,1],[1,0]]).astype('float64')
    polygon2mask_numba((10,10), a)



def create_labeled_mask(polygons_per_label, mask_shape):
    """
    Creates a labeled mask where each pixel is assigned to rois defined by polygons.

    Parameters
    ----------
    polygons_per_label : Dict[int, List[Polygon]]
        Dictionary mapping ROI IDs to lists of Shapely Polygons.
    mask_shape : Tuple[int, int]
        The shape of the image as (height, width).
    
    Returns
    -------
    labeled_mask : numpy.ndarray
        An array of shape (height, width) with integer labels.
        0 represents 'unassigned'.
    """    
    _warmup_poly2mask_numba()

    labeled_mask = np.zeros(mask_shape, dtype=np.int32)

    for roi_id, polygons in polygons_per_label.items():

        # Only one full-size boolean mask allocated for the entire ROI
        roi_mask = np.zeros(mask_shape, dtype=bool)

        for polygon in polygons:
            ext_coords, int_coords = convert_geometric_to_image_coords(polygon, mask_shape)

            # ---- Add exteriors (OR in-place) ----
            for ec in ext_coords:
                tmp = polygon2mask_numba(mask_shape, ec)
                # tmp = polygon2mask(mask_shape, ec)
                np.logical_or(roi_mask, tmp, out=roi_mask)
                # tmp freed here

            # ---- Subtract interiors (clear True pixels in-place) ----
            for ic in int_coords:
                tmp = polygon2mask_numba(mask_shape, ic)
                # tmp = polygon2mask(mask_shape, ic)
                roi_mask[tmp] = False
                # tmp freed here

        # ---- Write into label image ONLY where unlabeled ----
        unlabeled = (labeled_mask == 0)
        labeled_mask[roi_mask & unlabeled] = roi_id

    return labeled_mask





def calculate_mean_intensity(image, labeled_mask, roi_ids, image_fmt):
    """
    Calculates the mean intensity for each ROI and each channel in an n-dimensional image.
    
    Parameters
    ----------
    image : numpy.ndarray
        The image array with dimensions specified by image_fmt (e.g., "TCZYX").
    labeled_mask : numpy.ndarray
        The labeled mask array, matching the spatial dimensions of the image (e.g., "ZYX").
    roi_ids : List[int]
        List of region-of-interest (ROI) IDs to compute mean intensities for.
    image_fmt : str
        String specifying the format of the image dimensions (e.g., "TCZYX").
                
    Returns
    -------
    mean_intensities : Dict[int, List[float]]
        Dictionary mapping ROI IDs to lists of mean intensities per channel.
    """
    # Ensure the spatial dimensions of the labeled_mask match the corresponding spatial dimensions in image
    spatial_axes = [i for i, c in enumerate(image_fmt) if c in "XYZT"]
    mask_shape = tuple(image.shape[i] for i in spatial_axes)
    assert labeled_mask.shape == mask_shape, f"Mismatch between labeled_mask and image spatial dimensions.\n\t{labeled_mask.shape} != {mask_shape}"
    
    # Determine the channel axis
    ch_axis = image_fmt.index('C')
    num_channels = image.shape[ch_axis]
    
    # Move the channel axis to the first position for easier iteration
    image_reordered = np.moveaxis(image, ch_axis, 0)
    
    # Create a dictionary to store mean intensities
    mean_intensities = {}
    
    # Compute mean intensities for each ROI ID
    for label in roi_ids:
        mask = labeled_mask == label
        if np.any(mask):
            means = [image_reordered[ch][mask].mean() for ch in range(num_channels)]
        else:
            means = [0.0 for _ in range(num_channels)]
        mean_intensities[label] = means
    
    # Handle 'unassigned' (label 0)
    mask_unassigned = labeled_mask == 0
    if np.any(mask_unassigned):
        means = [image_reordered[ch][mask_unassigned].mean() for ch in range(num_channels)]
    else:
        means = [0.0 for _ in range(num_channels)]
    mean_intensities[0] = means
    
    return mean_intensities

def get_size_spatial_dims(arr, fmt):
    return np.prod([arr.shape[fmt.index(dim)] for dim in SPATIAL_AXES if dim in fmt])

import shapely
def summarize_roi_properties(
        polygons_per_label:dict[int, shapely.Geometry], 
        image:np.ndarray, 
        labeled_mask:Optional[np.ndarray], 
        image_fmt:str, 
        PX_SIZES:Optional[dict[str,float|int]]=None

    ):
    """
    Polygon-based. Summarizes properties of each ROI, including area in pixels and mean intensity per channel.
    
    Parameters
    ----------
    polygons_per_label : Dict[int, List[Polygon]]
        Dictionary mapping ROI IDs to lists of Shapely Polygons.
    image : numpy.ndarray
        The image array of shape (height, width, channels).
    labeled_mask: np.ndarray, optional if you don't need pixel intensities in rois. 
        Otherwise must convert polygons to mask array or see summarize_roi_array_properties
        Mask created from converting polygons to mask e.g. create_labeled_mask(polygons_per_label, bounds)
    image_fmt: str
        string representation of the dimensions e.g. "STCZYX"
    PX_SIZES: dict[str, float]
        if none, defaults to {'X':1, 'Y':1, 'Z':1}
    
    Returns
    -------
    roi_df : pandas.DataFrame
        DataFrame with columns:
            - 'roi_i': ROI ID
            - 'area_px': Area in pixels
            - 'mean_intensity_ch_0', 'mean_intensity_ch_1', ..., 'mean_intensity_ch_N': Mean intensity per channel
    """
    if image is None: 
        raise NotImplementedError # should be handleable 
    
    C = image.shape[image_fmt.index('C')]
    channel_names = [i for i in range(C)]
    image_size = get_size_spatial_dims(image, image_fmt)


    # Calculate area in pixels for each ROI
    roi_areas = {k: sum([p.area for p in v]) for k,v in polygons_per_label.items()}
    bg_size = image_size if labeled_mask is None else labeled_mask.size
    roi_areas[0] = bg_size - sum(roi_areas.values())
    roi_df = pd.DataFrame.from_dict(roi_areas, orient='index', columns=['area_px'])
    roi_df[['area_um', 'area_mm']] = np.nan

    # convert pixel area/volume to um^2/um^3
    PX_SIZES = PX_SIZES or {'X':1, 'Y':1, 'Z':1}
    if len(polygons_per_label) > 0:
        um_per_px = np.prod(list(PX_SIZES.values()))
        roi_df['area_um'] = roi_df['area_px'] * um_per_px  # microns
        roi_df['area_mm'] = roi_df['area_um'] * (1e-6)  # convert microns -> mm 

    # Calculate mean intensity per ROI and channel
    if labeled_mask is not None:
        mean_intensities = calculate_mean_intensity(image, labeled_mask, polygons_per_label.keys(), image_fmt)
    else:
        mean_intensities = {0:[np.nan]*C, **{k:[np.nan]*C for k in polygons_per_label.keys()}}
    # merge intensities with roi props 
    mean_intensities_df = pd.DataFrame.from_dict(mean_intensities, orient='index', columns=[cn for cn in channel_names])
    roi_df = roi_df.join(mean_intensities_df, how="left")
    roi_df = roi_df.reset_index(names='i') # rename the index to the roi_id

    # now need to unpivot chanel intensities so colocal_ids represent single rows
    value_vars = [c for c in roi_df.columns if c in channel_names]
    id_vars = [c for c in roi_df.columns if c not in value_vars]
    roi_df = roi_df.melt(id_vars=id_vars, value_vars=value_vars, var_name='colocal_id', value_name='intensity_mean')

    # add prefix to colnames so roi_df cols are differentiable when merged in the summary df
    rename_cols = [c for c in roi_df if c != 'colocal_id']
    roi_df = roi_df.rename(columns=dict(zip(rename_cols, [f"roi_{c}" for c in rename_cols])))
    return roi_df


def summarize_roi_array_properties(
    image,
    roi,
    image_fmt,
    roi_fmt,
    coerce_roi_fmt=False,
    PX_SIZES=None,
    rps_to_get=None,
    get_object_coords=False,
    additional_props=None,
    extra_properties=None,
):
    """
    array-based summarize_roi_properties using regionprops instead of polygons so 3D rois can be properly handled
    args:
        coerce_roi_fmt: bool, False
            if True use uip.morph_to_target_shape to try and reshape/reformat (add extra dimensions) roi to match image
        PX_SIZES: dict[str, float]
            if none defaults to {'X':1, 'Y':1, 'Z':1}
    """
    from SynAPSeg.utils.utils_image_processing import morph_to_target_shape
    
    if len(image_fmt) > 4:
        raise ValueError(f"expects at most CZYX, but got {image_fmt}")
    if len(image_fmt) != image.ndim:
        raise ValueError(f"{image_fmt} != {image.shape}")

    if image_fmt != roi_fmt: # TODO move format
        if not coerce_roi_fmt:
            raise ValueError(
                f"formats must match but got image_fmt:{image_fmt}, roi_fmt: {roi_fmt}"
                "if coercion is desired set coerce_roi_fmt=True (e.g. YX -> ZYX)"
            )

        roi = morph_to_target_shape(roi, roi_fmt, image.shape, image_fmt)
        print(f"summarize_roi_array_properties - reshaped roi to match img: {roi_fmt} -> {image_fmt}")
        roi_fmt = image_fmt
    
    # run on bg alone and assign an unused label to background 
    roi_max_val = roi.max()
    roi_df_bg = get_rp_table(
                np.where(roi==0, roi_max_val+1, 0), 
                image, 
                ch_axis = image_fmt.index('C') if 'C' in image_fmt else None, 
                rps_to_get = rps_to_get,
                get_object_coords=get_object_coords,
                additional_props=additional_props,
                # extra_properties=extra_properties,  # these props can be really slow on the bg (esp. uc.longest_path)
            )
    roi_df_bg.loc[roi_df_bg['label']==roi_max_val+1, 'label'] = 0 # re-assign background label to label 0
    
    roi_df = get_rp_table(
                roi,
                # np.where(roi==0, roi_max_val+1, roi), # assign a label to background 
                image, 
                ch_axis = image_fmt.index('C') if 'C' in image_fmt else None, 
                rps_to_get = rps_to_get,
                get_object_coords=get_object_coords,
                additional_props=additional_props,
                extra_properties=extra_properties,
            )
    
    # merge bg back in 
    roi_df = pd.concat([roi_df_bg, roi_df], ignore_index=True)
    roi_df = roi_df.rename(columns={'area':'area_px', 'label':'i'})

    # convert pixel area/volume to um^2/um^3
    PX_SIZES = PX_SIZES or {'X':1, 'Y':1, 'Z':1}
    
    if len(roi_df) > 0:
        um_per_px = np.prod(list(PX_SIZES.values()))
        roi_df['area_um'] = roi_df['area_px'] * um_per_px  # microns
        roi_df['area_mm'] = roi_df['area_um'] * (1e-6)  # convert microns -> mm 

    # add prefix to colnames so roi_df cols are differentiable when merged in the summary df
    rename_cols = [c for c in roi_df if c != 'colocal_id']
    roi_df = roi_df.rename(columns=dict(zip(rename_cols, [f"roi_{c}" for c in rename_cols])))
    roi_df = roi_df.sort_values(['colocal_id', 'roi_i'])

    return roi_df


def skewedness(mask, intensity_image):
    """
    For normally distributed data, the skewness should be about zero. For unimodal continuous distributions, a skewness value greater than zero means that there is more weight in the right tail of the distribution.
    mask:  boolean array of the region (within its bounding‐box)
    intensity_image: numeric array of the same shape (cropped intensities)
    """
    # mask is True for pixels inside the region
    return stats.skew(intensity_image[mask], axis=None)

def kurtosis(mask, intensity_image):
    """
    Kurtosis is the fourth central moment divided by the square of the variance.
    mask:  boolean array of the region (within its bounding‐box)
    intensity_image: numeric array of the same shape (cropped intensities)
    """
    # mask is True for pixels inside the region
    return stats.kurtosis(intensity_image[mask], axis=None)

def circularity(mask, intensity_image):
    """
    mask:  boolean array of the region (within its bounding‐box)
    intensity_image: numeric array of the same shape (cropped intensities)
        note: perimeter is only defined for 2d objects 
    """
    if mask.ndim == 2:
        perimeter = _regionprops_utils.perimeter(mask, 4)
        return 4 * np.pi * np.sum(mask) / (perimeter ** 2) if perimeter > 0 else np.nan
    else:
        return np.nan







def longest_skeleton_path(regionmask, intensity_image=None):
    """
    Calculate the sum of longest path lengths in a skeletonized region.
    
    For regions with disjoint components, skeletonizes each component
    and sums the longest path from each component.
    
    Compatible with skimage.measure.regionprops and regionprops_table.
    Works for both 2D and 3D (ZYX format) binary masks.
    
    Parameters
    ----------
    regionmask : ndarray
        Boolean mask of the region (2D or 3D)
    intensity_image : ndarray, optional
        Not used, but required for regionprops compatibility
        
    Returns
    -------
    float
        Sum of longest path lengths across all disjoint components
    """
    
    # Handle empty masks
    if not np.any(regionmask):
        return 0.0
    
    # Determine if 2D or 3D
    is_3d = regionmask.ndim == 3
    
    # Skeletonize the region
    if is_3d:
        skeleton = skeletonize(regionmask.astype(np.uint8))
    else:
        skeleton = skeletonize(regionmask.astype(bool))
    
    # If skeleton is empty, return 0
    if not np.any(skeleton):
        return 0.0
    
    # Label connected components in the skeleton
    # This handles disjoint parts
    if is_3d:
        labeled_skeleton, num_components = ndi_label(skeleton, structure=np.ones((3, 3, 3)))
    else:
        labeled_skeleton, num_components = ndi_label(skeleton, structure=np.ones((3, 3)))
    
    # Process each connected component separately
    total_path_length = 0.0
    
    for component_label in range(1, num_components + 1):
        component_mask = labeled_skeleton == component_label
        # max_path_length, longest_path_coords, total_length = _calculate_longest_path_for_component(component_mask)
        max_path_length = _calculate_longest_path_for_component(component_mask)
        total_path_length += max_path_length
    
    return total_path_length


def _calculate_longest_path_for_component(skeleton_mask, max_iter=10) -> float:
    """
    Helper function to calculate longest path for a single connected component.
    
    Parameters
    ----------
    skeleton_mask : ndarray
        Boolean mask of a single connected skeleton component
    is_3d : bool
        Whether the mask is 3D
    max_iter : int
        Maximum number interesting nodes to search for the longest path search
        
    Returns
    -------
    tuple
        - length of longest path using BFS
        - coordinates of longest path taken
        - total length of skeleton including branching segments
    """

    is_3d = skeleton_mask.ndim == 3

    # Get coordinates of skeleton pixels
    coords = np.argwhere(skeleton_mask)
    
    if len(coords) == 0:
        return 0.0
    
    # Single pixel component
    if len(coords) == 1:
        return 0.0
    
    # Build adjacency graph
    # For each skeleton pixel, find its neighbors
    adjacency = {}
    coord_to_idx = {tuple(c): i for i, c in enumerate(coords)}
    
    if is_3d:
        # 26-connectivity for 3D
        offsets = [(dz, dy, dx) 
                   for dz in [-1, 0, 1] 
                   for dy in [-1, 0, 1] 
                   for dx in [-1, 0, 1] 
                   if not (dz == 0 and dy == 0 and dx == 0)]
    else:
        # 8-connectivity for 2D
        offsets = [(dy, dx) 
                   for dy in [-1, 0, 1] 
                   for dx in [-1, 0, 1] 
                   if not (dy == 0 and dx == 0)]
    
    for idx, coord in enumerate(coords):
        neighbors = []
        for offset in offsets:
            neighbor = tuple(coord + np.array(offset))
            if neighbor in coord_to_idx:
                neighbor_idx = coord_to_idx[neighbor]
                # Calculate Euclidean distance
                dist = np.linalg.norm(np.array(offset))
                neighbors.append((neighbor_idx, dist))
        adjacency[idx] = neighbors
    
    # Find longest path using BFS from each endpoint/branch point
    def bfs_longest_path_trace(start_idx):
        # We now store parent pointers to reconstruct the path
        # visited maps node_idx -> distance
        visited = {start_idx: 0.0} 
        parent = {start_idx: None} 
        queue = deque([start_idx])
        
        max_dist = 0.0
        furthest_node = start_idx
        
        while queue:
            curr = queue.popleft()
            curr_dist = visited[curr]
            
            # Track the furthest node found
            if curr_dist > max_dist:
                max_dist = curr_dist
                furthest_node = curr
            
            for neighbor, edge_dist in adjacency.get(curr, []):
                if neighbor not in visited:
                    visited[neighbor] = curr_dist + edge_dist
                    parent[neighbor] = curr
                    queue.append(neighbor)
        
        # Reconstruct path by backtracking from furthest_node to start
        path_indices = []
        curr = furthest_node
        while curr is not None:
            path_indices.append(curr)
            curr = parent[curr]
            
        return max_dist, path_indices
    
    # Find endpoints (degree 1) and branch points (degree > 2)
    # These are good starting points for longest path search
    degrees = {idx: len(neighbors) for idx, neighbors in adjacency.items()}
    interesting_nodes = [idx for idx, deg in degrees.items() if deg == 1 or deg > 2]
    
    # If no interesting nodes, use all nodes (shouldn't happen often)
    if not interesting_nodes:
        interesting_nodes = list(adjacency.keys())
    
    # Find longest path by trying BFS from multiple starting points
    global_max_dist = 0.0
    best_path_indices = []

    # Limit starts for performance
    for start in interesting_nodes[:min(max_iter, len(interesting_nodes))]: # limit to e.g. 10 for performance
        dist, path_idxs = bfs_longest_path_trace(start)
        if dist > global_max_dist:
            global_max_dist = dist
            best_path_indices = path_idxs

    # Convert indices back to real coordinates
    longest_path_coords = np.array([coords[i] for i in best_path_indices])
    
    # calculate total length of skeleton (i.e. including branching segments)
    total_length = 0.0
    for node_idx, neighbors in adjacency.items():
        for neighbor_idx, dist in neighbors:
            total_length += dist

    # Divide by 2 because every edge is counted twice (once for each direction)
    total_length /= 2.0
    
    # return (global_max_dist, longest_path_coords, total_length)
    return global_max_dist



# ============================================================================
# Tests and Examples
# ============================================================================
def display_longest_skeleton_path():
    """ 
    test example showing longest skeleton path
    example shows length of branching segments is not included
    """
    import utils_image_processing as uip
    import utils_plotting as up
    dends3d = uip.read_img(r"J:\SEGMENTATION_DATASETS\2025_0928_hpc_psd95andrbPV_zstacks\examples\0002\dends_filt.tiff")
    up.show(dends3d)

    regionmask = np.where(dends3d==1, 18, 0)
    skeleton_mask = skeletonize(regionmask)
    up.show(skeleton_mask)
    is_3d = True

    global_max_dist, longest_path_coords, total_length = _calculate_longest_path_for_component(skeleton_mask, max_iter=np.inf)

    print(f"global_max_dist: {global_max_dist}")
    print(f"total_length: {total_length}")


    path_coords = longest_path_coords
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot all skeleton points (faintly)
    all_coords = np.argwhere(skeleton_mask)
    ax.scatter(all_coords[:, 2], all_coords[:, 1], all_coords[:, 0], c='gray', alpha=0.1, s=1)

    # Plot the longest path (boldly)
    if len(path_coords) > 0:
        # Assuming Z, Y, X order in numpy
        ax.plot(path_coords[:, 2], path_coords[:, 1], path_coords[:, 0], c='red', linewidth=3)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

    up.show(skeleton_mask, def_title='skeleton_mask')
    regionmask[tuple(path_coords.T)] = 20
    up.show(regionmask, def_title='longest_path')

def test_longest_skeleton_path():
    """
    Test the longest_skeleton_path function with known shapes.
    """
    print("Testing longest_skeleton_path function...\n")
    
    # Test 1: 2D straight line (horizontal)
    print("Test 1: 2D horizontal line")
    line_2d = np.zeros((10, 20), dtype=bool)
    line_2d[5, 2:18] = True  # 16 pixels long
    result = longest_skeleton_path(line_2d)
    expected = 15.0  # Distance is n-1 for n pixels in a line
    print(f"  Expected: ~{expected:.1f}, Got: {result:.2f}")
    print(f"  {'✓ PASS' if abs(result - expected) < 1.0 else '✗ FAIL'}\n")
    
    # Test 2: 2D diagonal line
    print("Test 2: 2D diagonal line")
    diag_2d = np.zeros((20, 20), dtype=bool)
    for i in range(15):
        diag_2d[i+2, i+2] = True  # 15 pixels diagonal
    result = longest_skeleton_path(diag_2d)
    expected = 14 * np.sqrt(2)  # sqrt(2) per diagonal step
    print(f"  Expected: ~{expected:.1f}, Got: {result:.2f}")
    print(f"  {'✓ PASS' if abs(result - expected) < 1.5 else '✗ FAIL'}\n")
    
    # Test 3: 2D rectangle (should skeletonize to its medial axis)
    print("Test 3: 2D rectangle")
    rect = np.zeros((20, 50), dtype=bool)
    rect[5:15, 5:45] = True
    result = longest_skeleton_path(rect)
    # Rectangle should skeletonize to ~40 pixels horizontally
    expected_min, expected_max = 35.0, 42.0
    print(f"  Expected: {expected_min:.1f}-{expected_max:.1f}, Got: {result:.2f}")
    print(f"  {'✓ PASS' if expected_min <= result <= expected_max else '✗ FAIL'}\n")
    
    # Test 4: 3D straight line (z-axis)
    print("Test 4: 2x 3D vertical line (z-axis)")
    line_3d = np.zeros((15, 10, 10), dtype=bool)
    line_3d[2:14, 5:8, 5:8] = True  # 12 pixels along z
    line_3d[2:14, 2, 2] = True  # 12 pixels along z
    result = longest_skeleton_path(line_3d)
    expected = 22.0  # 2x11 steps
    print(f"  Expected: ~{expected:.1f}, Got: {result:.2f}")
    print(f"  {'✓ PASS' if abs(result - expected) < 1.0 else '✗ FAIL'}\n")
    
    # Test 5: Disjoint components (NEW TEST)
    print("Test 5: Disjoint components - two separate lines")
    disjoint = np.zeros((30, 50), dtype=bool)
    # Line 1: horizontal, 20 pixels
    disjoint[10, 5:25] = True
    # Line 2: horizontal, 15 pixels (separate)
    disjoint[20, 30:45] = True
    
    result = longest_skeleton_path(disjoint)
    expected = 19.0 + 14.0  # Sum of both lines (20-1 + 15-1)
    print(f"  Expected: ~{expected:.1f}, Got: {result:.2f}")
    print(f"  {'✓ PASS' if abs(result - expected) < 2.0 else '✗ FAIL'}\n")
    
    # Test 6: Integration with regionprops
    print("Test 6: Integration with regionprops_table")
    # Create a labeled image with two objects
    labeled = np.zeros((30, 40), dtype=int)
    # Object 1: horizontal line
    labeled[10, 5:25] = 1  # 20 pixels
    # Object 2: vertical line
    labeled[5:20, 30] = 2  # 15 pixels
    
    # Use regionprops_table with our custom function
    props = measure.regionprops_table(
        labeled,
        properties=('label',),
        extra_properties=(longest_skeleton_path,)
    )
    
    print("  Label | Longest Path")
    print("  ------|-------------")
    for i in range(len(props['label'])):
        print(f"    {props['label'][i]:3d} | {props['longest_skeleton_path'][i]:8.2f}")
    
    # Verify results
    obj1_expected = 19.0  # 20 pixels = 19 steps
    obj2_expected = 14.0  # 15 pixels = 14 steps
    
    pass1 = abs(props['longest_skeleton_path'][0] - obj1_expected) < 1.0
    pass2 = abs(props['longest_skeleton_path'][1] - obj2_expected) < 1.0
    
    print(f"  {'✓ PASS' if pass1 and pass2 else '✗ FAIL'}\n")
    
    print("All tests completed!")
    
    # Test 7: Disjoint components within a single labeled region
    print("\nTest 7: Single label with disjoint parts")
    labeled_disjoint = np.zeros((30, 50), dtype=int)
    # Both lines belong to the same label (1)
    labeled_disjoint[10, 5:25] = 1  # 20 pixels
    labeled_disjoint[20, 30:45] = 1  # 15 pixels
    
    props = measure.regionprops_table(
        labeled_disjoint,
        properties=('label',),
        extra_properties=(longest_skeleton_path,)
    )
    
    result = props['longest_skeleton_path'][0]
    expected = 19.0 + 14.0  # Sum of both disjoint parts
    print(f"  Label 1 path length: {result:.2f}")
    print(f"  Expected: ~{expected:.1f}, Got: {result:.2f}")
    print(f"  {'✓ PASS' if abs(result - expected) < 2.0 else '✗ FAIL'}")