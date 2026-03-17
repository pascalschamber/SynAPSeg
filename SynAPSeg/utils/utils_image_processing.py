
from tifffile import imread, imwrite, TiffFile
import numpy as np
import matplotlib.pyplot as plt
import scipy
from skimage import morphology

from typing import Optional, Tuple, Any, Iterable, Mapping, Union, Dict
import numba as nb
from numba import cuda
from skimage.measure import regionprops
from skimage.segmentation import find_boundaries

from SynAPSeg.utils import utils_colocalization as uc
from SynAPSeg.utils import utils_plotting as up

# import cv2   #!!! DO NOT USE CV2 FOR non-8bit IMAGES, does not read channels correctly

def get_tiff_metadata(filepath: str, return_all: bool = False) -> dict:
    """
    Read the metadata from a TIFF image.
        Note: by default only returns the first metadata, even if the tiff has multiple shaped_metadata

    returns:
        dict: metadata of the tiff file
            e.g. {'axes': 'STCZYX', 'shape': [2, 5, 3, 10, 256, 256], ...}
    """
    from tifffile import TiffFile

    with TiffFile(filepath) as tiff:
        if tiff.shaped_metadata is not None:
            
            if len(tiff.shaped_metadata) > 1:
                if return_all:
                    return({i: d for i,d in enumerate(tiff.shaped_metadata)})
                
                print('tiff has multiple shaped_metadata, returning first')
            
            return(tiff.shaped_metadata[0])
    return {}


def get_tiff_shape(file_path: str) -> Tuple:
    """
    Get the shape a TIFF image without loading whole array into memory.
        also supports ome.tiff
    
    if tifffile was written with other metadata this func doesn't get it e.g.:
        imwrite(output_file, data, metadata={'axes': axes})
    Args:
        file_path (str): Path to the TIFF image file.
    Raises:
        ValueError if tiff is not shaped or cannot be read
    Returns:
        tuple: The shape of the image.
    """
    is_ometiff = file_path.lower().endswith(('ome.tiff', 'ome.tif'))
    if is_ometiff:
        from SynAPSeg.IO.readers import PyramidOMEReader
        ome = PyramidOMEReader(file_path)
        return tuple(ome.shape)
    
    with TiffFile(file_path) as tiff:
        if tiff.is_shaped:
            # shaped_metadata (tuple) --> e.g. ({'axes': 'STCZYX', 'shape': [2, 5, 3, 10, 256, 256]},)
            return tuple(tiff.shaped_metadata[0]['shape'])
        
    raise ValueError('tiff is not shaped')
        
    
def get_tiff_shapes(file_paths: list[str]) -> list[list | Any]:
    """
    handles multiple files simulatenously, using get_tiff_shape to get shapes of TIFF images using the tifffile library.
        the imagesize lib only gets height and width.
    
    Args:
        file_paths (list): A list of file paths to the TIFF images.
    
    Returns:
        list: A list of tuples where each tuple contains the shape (n dimensional) of the corresponding image.
    """
    shapes = []
    for file_path in file_paths:
        try:
            shape = get_tiff_shape(file_path)
            shapes.append(shape)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            shapes.append(None)
    return shapes


def get_tiff_dtype(path):
    """ Get the dtype of a TIFF image without loading whole array into memory. """
    with TiffFile(path) as tif:
        return tif.series[0].dtype  # dtype of the first image series

def to_binary(arr: np.ndarray, thresh=None):
    """convert image to binary integer array (e.g. float32 arr in range 0,1 )
        
    Parameters:
    - arr: np.ndarray, can be boolean array e.g. to_binary(arr!=0) returns array where all values that are not equal to 0 are 1
    - thresh: float or int, values greater than this value are set to 1, otherwise set to 0. defaults to 0
    """
    thresh = 0 if thresh is None else thresh
    
    if arr.dtype == bool: # handle passing an expression that generates a boolean array (e.g. arr < 50)
        result = arr
    else:
        result = arr > thresh
        
    return np.where(result, 1, 0)


def to_grayscale(image, weights=None, axis=-1, warn=True):
    """ convert an image with an arbitrary number of channels (default = last axis) to 2d grayscale image"""
    nCh = image.shape[axis]
    
    if nCh == 1:
        if warn: print('image is already only a single channel, returning input.')
        return np.take(image, 0, axis=axis)
    
    if weights is None: 
        weights = np.array([1/nCh] * nCh)
    assert len(weights) == nCh
    
    return np.tensordot(image, weights, axes=([axis], [0]))

def rgb_to_grayscale(rgb_image):
    """
    Convert a 3-channel RGB image to a grayscale image.
    
    Parameters:
    - rgb_image: A numpy array of shape (height, width, 3) representing an RGB image.
    
    Returns:
    - A numpy array of shape (height, width) representing the grayscale image.
    """
    assert rgb_image.ndim==3
    if rgb_image.shape[-1]==1:
        print('image is already only a single channel, returning input.')
        return rgb_image[...,0]
    
    # Define the weights for each channel
    weights = np.array([0.2989, 0.5870, 0.1140])
    # Apply the weights to the channels
    grayscale_image = np.dot(rgb_image[...,:3], weights)
    return grayscale_image
    

def min_max_normalization(arr: np.ndarray, minv=None, maxv=None, eps=1e-20, clip: bool|int|float = False):
    """ min-max normalize an array to range [0, 1] with optional clipping"""
    minv = np.min(arr) if minv is None else minv
    maxv = np.max(arr) if maxv is None else maxv

    normed = (arr - minv)/(maxv - minv + eps) # to prevent zero-division warning

    if clip is not False:
        clip = 1 if isinstance(clip, bool) else clip
        normed = np.clip(normed, 0, clip)
    
    return normed

normalize_01 = min_max_normalization # alias


def normalize_dtype(array):
    """
    Normalizes a NumPy array into the range [0, 1] by dividing the array by its dtype's maximum value.
    """
    # Get the maximum value possible for the array's dtype
    max_value = get_max_value_dtype(array)
    # Normalize the array
    normalized = array.astype(np.float32) / max_value
    return normalized

def get_max_value_dtype(array):
    """
    Returns the maximum possible value based on the data type of the input NumPy array.

    Parameters:
    - array: A NumPy array whose data type's maximum value is to be found.

    Returns:
    - The maximum value that can be represented by the data type of the input array.
    """
    # Check if the array is of integer type
    if np.issubdtype(array.dtype, np.integer):
        return np.iinfo(array.dtype).max
    # Check if the array is of floating-point type
    elif np.issubdtype(array.dtype, np.floating):
        return np.finfo(array.dtype).max
    else:
        raise TypeError("The function supports only integer and floating-point data types.")
    

def _norm_percentile(a, pmin=1, pmax=99.9, axis=None, clip=False):
    """ helper function for percentile normalization using approach based on csbdeep's implementation """
    return min_max_normalization(
        a, 
        minv=np.percentile(a, pmin, axis=axis, keepdims=True), 
        maxv=np.percentile(a, pmax, axis=axis, keepdims=True), 
        clip=clip
    )

def norm_percentile(image, norm, ch_axis:Optional[int]=None, clip=True):
    """ 
    Perform percentile normalization along the specified channel axis 
        using approach based on csbdeep's implementation.

    Parameters
    ----------
    image : ndarray
        Input image array.
    norm : dict, tuple, or list
        Normalization parameters:
        - Dict: Map channels to nmin/nmax.
        - Tuple or list: Global (nmin, nmax) for all channels.
    ch_axis : int, optional
        Axis corresponding to channels. Default is -1.
    clip : bool, optional
        Whether to clip values to [nmin, nmax]. Default is True.
    
    Returns
    -------
    out_arrn : ndarray
        Normalized image array.
    """

    # If no channel axis or 2D image, normalize directly
    if ch_axis is None or image.ndim == 2:
        assert len(norm) == 2, 'norm argument must be an iterable of length 2'
        return _norm_percentile(image, pmin=norm[0], pmax=norm[1], axis=None, clip=clip)
    
    # Handle normalization parameters
    n_channels = image.shape[ch_axis]
    if isinstance(norm, (tuple, list)):
        nd = {i: {'nmin': norm[0], 'nmax': norm[1]} for i in range(n_channels)}
    elif isinstance(norm, dict):
        nd = norm
    else:
        raise ValueError(f"Invalid norm argument: {norm}")
    
    # Prepare output array
    out_arrn = np.zeros_like(image, dtype='float32')
    
    # Normalize along the specified channel axis
    for i in range(n_channels):
        # Create a slice object to handle arbitrary ch_axis
        slices = [slice(None)] * image.ndim
        slices[ch_axis] = i
        slices = tuple(slices)

        # Normalize the selected channel
        out_arrn[slices] = _norm_percentile(
            image[slices], pmin=nd[i]['nmin'], pmax=nd[i]['nmax'], axis=None, clip=clip)
    
    return out_arrn



def norm_chs(image, norm_method=None, ch_index=-1, out_dtype=float, **norm_kwargs):
    """
    Wraps different normalization methods for applying them along an axis

        Args:
            norm_method: optional, callable. If none, defaults to min-max normalization.
    """
    # Convert image to float for processing
    out = np.zeros_like(image).astype(out_dtype)
    
    # Adjust for negative indices
    ch_index = image.ndim + ch_index if ch_index < 0 else ch_index
    
    # Iterate over each channel
    for ch in range(image.shape[ch_index]):
        # Select all dimensions as slices
        slc = [slice(None)] * image.ndim
        # Update the slice for the current channel
        slc[ch_index] = ch

        all_zeros = np.all(image[tuple(slc)] == 0)
        if all_zeros: # if all zeros to avoid introducing nans just insert zeroarray with out normalizing
            out[tuple(slc)] = image[tuple(slc)]
        elif norm_method is not None:
            # Apply the normalization method to the current channel
            out[tuple(slc)] = norm_method(image[tuple(slc)], **norm_kwargs)
        else:
            # If no normalization method is normalize_01
            out[tuple(slc)] = normalize_01(image[tuple(slc)])
    
    return out


def to_8bit(arr):
    """ convience function to quickly convert an array to 8bit dtype. Note: normalizes range to (0, 1) then multiplies by 255 """
    return (normalize_01(arr) * 255).astype('uint8')


def convert_16bit_image(image, NORM=True, CLIP=None):
    """use util in scipy or skimage instead - e.g. see skimage.util.img_as_uint """
    # if NORM, normalize to img min/max, else use max possible value for 16 bit image
    # if CLIP, set min, max directly, useful for standardizing image display
    ndims = image.ndim
    if ndims==2:
        image = np.expand_dims(image,-1)
    assert image.ndim == 3
    assert image.shape[-1] < image.shape[0] and image.shape[-1] < image.shape[1] # assert chs last
    array = []
    for i in range(image.shape[-1]):
        if CLIP is None:
            ch_min, ch_max = (image[...,i].min(), image[...,i].max()) if NORM else (0, 2**16)
        else: # set max px value
            ch_min, ch_max = CLIP[0], CLIP[1]

        ch = ((image[...,i]-ch_min)/(ch_max - ch_min))*255
        ch = np.clip(ch, 0, 255)

        array.append(ch)
    if ndims ==2:
        return array[0].astype('uint8')
    else:
        return np.stack(array, axis=-1).astype('uint8')
    
    
def filter_label_img_old(segmented_image, labels_to_keep):
    """ remove objects from an image if label is not in labels_to_keep list"""
    mask = np.zeros(segmented_image.shape, dtype=bool)
    # Iterate over each label you wish to keep and update the mask
    for label in labels_to_keep:
        mask = mask | (segmented_image == label)
    # Apply the mask to the segmented image, Set pixels not in `labels_to_keep` to 0 (or any other background value you prefer)
    filtered_image = np.where(mask, segmented_image, 0)
    return filtered_image


def filter_label_img(segmented_image, labels_to_keep):
    """ 
    remove objects from an image if label is not in labels_to_keep list
        using numba - ~250x faster for large images
        however np.isin(labels, keep_lbls) might be faster for large and dense label images
    """
    # if no labels to keep, return empty image
    if len(labels_to_keep) == 0:
        return np.zeros_like(segmented_image).astype(np.int32)
    
    filtered_image = np.copy(segmented_image).astype(np.int32)
    labels_to_keep = np.array(labels_to_keep, dtype=np.int32)
    filter_label_img_nb(filtered_image, labels_to_keep) # modifies in place, hence why np.copy
    return filtered_image


@nb.njit
def create_value_dict(values):
    """
    Create a Numba-typed dictionary for O(1) membership checks.
    using a dict since average O(1) membership lookup is fast if you have a large values list.
    assume int32 as the data type. 
    """
    d = nb.typed.Dict.empty(
        key_type=nb.types.int32,
        value_type=nb.types.boolean
    )
    for v in values:
        d[v] = True
    return d

@nb.njit
def filter_label_img_nb(arr, values):
    """
    This function is wrapped by "filter_label_img", use that version since it inits varriables properly
    Given an nD array 'arr' and a 1D array 'values',
    set every element of 'arr' that is NOT in 'values'
    to zero, returning the filtered array.
        NOTE: arr is updated in-place
    
    time is faster for numba if keep_values > 20, if keep_values is 1000 long it is ~ 45x faster
    
    """
    # Create the dictionary of valid values (O(len(values))).
    valid_dict = create_value_dict(values)

    arr_1d = arr.ravel()  # Flattened view of arr
    size = arr_1d.size

    for i in range(size):
        x = arr_1d[i]
        # If x is not in the dictionary, set to 0
        if x not in valid_dict:
            arr_1d[i] = 0
    
    return arr

def init_filter_label_img_nb():
    arr = np.random.randint(0,1000,(128,128,128), dtype=np.int32)
    keep_values = np.array(list(range(50)), dtype=np.int32)
    filter_label_img_nb(arr, keep_values)


def filter_area_objects(segmented_image, size_range, preserve_labels=True, relabel_input=True, connectivity=None):
    """
    Filter objects from a segmented image by size (inclusive).
        if preserve_labels=True and relabel_input=True can remove non-connected 
        small artifacts while preserving og labels

    Parameters:
    - segmented_image: np.ndarray, segmented image where objects have non-zero values.
    - size_range: tuple, (min_area, max_area), keep objects >= min_area, and <= max_area
    - preserve_labels: bool, whether to retain the labels from the input image (works even if relabel_input is True)
    - relabel_input: bool, ensures small artifacts that share a label with a large one are removed
    - connectivity: int, connectivity for relabeling, defaults to ndim

    Returns:
    - np.ndarray, filtered image with objects within the given size range
    """
    from skimage.measure import regionprops
    
    size_range = _sanitize_size_range(size_range)
    if size_range is None:
        return segmented_image

    # Label connected components - essential to remove small artifacts
    if relabel_input:
        labels = relabel(segmented_image, connectivity=connectivity)
    else:
        labels = segmented_image
    
    # get labels in size range
    props = regionprops(labels)

    keep_lbls = [
        p.label for p in props 
        if size_range[0] <= p.area <= size_range[1]
    ]

    if len(keep_lbls) == 0:
        return np.zeros_like(labels)
    
    filtered = filter_label_img(labels, keep_lbls)
    
    # retain labels used in input image
    if preserve_labels: 
        filtered = filtered.astype(bool) * segmented_image

    return filtered


def _sanitize_size_range(size_range):
    """
    Sanitize size range for object detection.
    """
    if size_range is None:
        return None

    size_range = list(size_range)
    if len(size_range) != 2: 
        raise ValueError(f"size range must be of length 2, got:{size_range}")
        
    if size_range[0] is None or size_range[0] == -1:
        size_range[0] = 0
    if size_range[1] is None or size_range[1] == -1 or size_range[1] == 'inf':
        size_range[1] = np.inf
    if size_range[0] == 0 and size_range[1] == np.inf:
        size_range = None
    
    return size_range

def test_filter_area_objects():
    labeled_image = sample_labeled_image()
    filtered_image = filter_area_objects(labeled_image, 1, np.inf)

def sample_labeled_image():
    """
    Creates a 100x100 image with 3 distinct objects:
    - Object A: 4 pixels (Small)
    - Object B: 10 pixels (Medium)
    - Object C: 20 pixels (Large)
    """
    img = np.zeros((100, 100), dtype=int)
    
    # Object A (Area = 4)
    img[0:2, 0:2] = 1 
    
    # Object B (Area = 10)
    img[3:5, 0:5] = 2
    
    # Object C (Area = 20)
    img[6:8, 0:10] = 3 

    img[50:70, 50:70] = 5
    img[72, 72] = 5
    img[45:50, 45:50] = 4
    img[50:55, 50:55] = 4
    
    return img



def remove_small_objs(arr, min_size, connectivity=1):
    """ 
    removes objects < min_size, handles binary or labeled images. 
    
    Likely faster than filter_area_objects if only removing small objects"""
    from skimage.morphology import remove_small_objects

    nNonZero_labels = len(unique_nonzero(arr))
    
    if nNonZero_labels == 0:
        return arr
    
    r = relabel(arr) if nNonZero_labels == 1 else arr # must label binary imgs first
    r = remove_small_objects(r, min_size, connectivity=connectivity) # !!! will not to work correctly if duplicated object ids !!!
    r = np.where(r>0, 1, 0) if nNonZero_labels == 1 else r
    return r

    

def extract_largest_objects(segmented_image, n=1, label=True, connectivity=None):
    """
    Extract the largest `n` objects from a segmented image.

    Parameters:
    - segmented_image: 2D array, segmented image where objects have non-zero values.
    - n: int, the number of largest objects to extract.
    - label: bool, whether to relabel the input image
    - connectivity: Optional[int], defaults to ndim

    Returns:
    - A 2D array with only the largest `n` objects.
    """
    from skimage import measure

    if label:
        labels = measure.label(segmented_image, connectivity=connectivity or segmented_image.ndim)
    else:
        labels = segmented_image

    props = measure.regionprops(labels)
    
    # If there are fewer objects than n, just return all objects
    n = min(n, len(props))
    
    # Find the largest `n` objects based on area
    largest_props = sorted(props, key=lambda x: x.area, reverse=True)[:n]
    
    # Initialize a mask for the largest `n` objects
    mask = np.zeros_like(segmented_image, dtype=int)
    
    # Iterate through the largest `n` objects and update the mask
    new_label = 1
    for prop in largest_props:
        mask[labels == prop.label] = new_label
        new_label +=1
    
    return mask

def _extract_largest_object(segmented_image):
    """ get the largest single object """
    labeled = relabel(segmented_image)
    if labeled.max() == 0:
        return segmented_image  # nothing found
    largest = 1 + np.argmax(np.bincount(labeled.flat)[1:])
    return labeled == largest





def mask_to_outlines(binary_mask):
    """convert binary mask to outlines using sobel filter"""
    # bin mask to outlines
    sobel_h, sobel_v = scipy.ndimage.sobel(binary_mask, axis=0), scipy.ndimage.sobel(binary_mask, axis=1)
    magnitude = np.where(np.hypot(sobel_h, sobel_v) != 0, 1, 0).astype(int)
    outline = morphology.skeletonize(magnitude)
    return outline

def correct_annotated_labels(sd_mask, ch_img, size_min=0):
    """correct annotations where same label was used in non-touching objects, where ndi.label fails to separate"""
    import scipy.ndimage as ndi
        
    corrected_mask = np.zeros_like((sd_mask))
    unique_labels = np.unique(sd_mask)
    current_val = 1
    ccc = 0
    for ul in unique_labels:
        if ul == 0: 
            continue
        labels, count = ndi.label(np.where(sd_mask==ul, 1, 0))
        ccc += count
        rpdf, _ = uc.get_rp_table(labels, ch_img, ch_colocal_id={0:0}, prt_str='__')
        for uni_l in np.unique(labels):
            if uni_l == 0: continue
            # size exclusion
            rpdf_row = rpdf[rpdf['label']==uni_l]
            if rpdf_row['area'].values[0] >= size_min:
                corrected_mask = np.where(labels==uni_l, current_val, corrected_mask)
                current_val +=1
    return corrected_mask


def relabel(label_image: np.ndarray, connectivity: Optional[int] = 1, return_split_ids=False) -> np.ndarray:
    """
    Relabel every spatially disconnected region in ``label_image`` with a unique
        sequential ID, preserving background as 0.
        uses bounding box approach to make this faster than skimage implementation

    Parameters
    ----------
    label_image : ndarray of int or bool
        2D or 3D array whose non zero values give the original object IDs.
    connectivity : {1, 2, 3}, optional
        Pixel connectivity passed to ``scipy.ndimage.generate_binary_structure``  
        (1=nearest neighbors: 4 conn. in 2D / 6 conn. in 3D).

    Returns
    -------
    out : ndarray of int32
        Same shape as ``label_image``.  Each connected component has a distinct,
        consecutive, 1‑based label.  Background remains 0.
    """
    label_image = np.asarray(label_image)
    if label_image.ndim < 2:
        raise ValueError("Input must be at least 2D")
    
    from scipy import ndimage as ndi

    # --- allocate output once -----------------------------------------------
    out = np.zeros(label_image.shape, dtype=np.int32)

    # One bounding box per original label (None for labels that are absent)
    bboxes = ndi.find_objects(label_image)

    struct = ndi.generate_binary_structure(label_image.ndim, connectivity or label_image.ndim)
    next_id = 1  # first unused global label
    split_ids = {} # dict mapping og ids to new ids, if they were split

    for lbl, slc in enumerate(bboxes, start=1):
        if slc is None:          # this label is not present
            continue

        view = label_image[slc]
        mask = (view == lbl)     # pixels belonging to this original label
        if not mask.any():
            continue             # defensive; should never fire

        # Connected‑component labeling inside the small view
        local, n_cc = ndi.label(mask, structure=struct)

        if return_split_ids:
            if n_cc>1:
                split_ids[lbl] = [next_id-1+i for i in range(1, n_cc+1)]

        # Offset local IDs so they remain unique in the global image
        out_slice = out[slc]
        out_slice[local > 0] = local[local > 0] + next_id - 1
        next_id += n_cc
    
    if return_split_ids:
        return out, split_ids
    return out



def shrink_object_perimeters(
    label_image,
    intensity_image,
    threshold,
    connectivity=1,
    iter_max=1,
    find_boundaries_mode='inner',
    remove_islands = True,
    ):
    """
    Shrink each labeled object by peeling away boundary pixels whose intensity
    is below a user-specified threshold.
    
    Parameters
    ----------
    label_image : 2D int array
        Input label map (0 = background, >0 = object IDs).
    intensity_image : 2D numeric array
        A co-registered intensity image.
    threshold : float or callable
        If float, a constant intensity threshold.
        If callable, will be called as thr = threshold(intensities_of_object),
        and must return a float.
    connectivity : int, optional
        Connectivity for boundary-finding (1 = 4-connectivity, 2 = 8-connectivity).
    iter_max: int, optional
        maximum number of iterations to before boundry thresholding. e.g. if 1, max lbl can be reduced is 1 pixel
    find_boundaries_mode: str, optional
        passed to find_boundaries
    remove_islands: bool
        if True keep only the largest single object each iteration
    
    Returns
    -------
    new_labels : 2D int array
        A new label map of the same shape, with the same object IDs but
        “shrunken” where outer pixels were below threshold.
    """
    new_labels = np.zeros_like(label_image)
    props = regionprops(label_image, intensity_image)

    for prop in props:
        lbl = prop.label
        _bbox = prop.bbox # bbox is formated (minr, minc, maxr, maxc) or (minz, minr, minc, maxz, maxr, maxc)

        bbox_slices, shrunk_mask = _shrink_single_label(
            label_image,
            intensity_image,
            lbl, 
            _bbox,
            threshold,
            connectivity=connectivity,
            iter_max=iter_max,
            find_boundaries_mode=find_boundaries_mode,
            remove_islands = remove_islands,
        )
        new_labels[bbox_slices] += shrunk_mask

    return new_labels


def _shrink_single_label(
    label_image,
    intensity_image,
    lbl,
    bbox,
    threshold,
    connectivity=1,
    iter_max=1,
    find_boundaries_mode='inner',
    remove_islands = True,
    restore_background_labels = False,
    ):
    """
    Shrink one labeled region by peeling off low‐intensity boundary pixels.
        works on cropped region for efficiency
    
    Args:
        restore_background_labels
            if applying this outside of main func (shrink_object_perimeters), may want to keep other labels that were in bbox

    Returns
        the bbox slices and the new (cropped) mask to paste back.
    """
    # extract object sub‐region
    bbox_slices = _bbox2slice(bbox)

    sub_labels = label_image[bbox_slices]
    obj_mask  = (sub_labels == lbl)

    # compute threshold for this object
    sub_int   = intensity_image[bbox_slices]
    thr = threshold(sub_int[obj_mask]) if callable(threshold) else threshold
    threshed_out = (sub_int < thr)

    # iteratively peel away boundary pixels below threshold
    mask = obj_mask.copy()
    for _ in range(iter_max):
        boundary  = find_boundaries(np.pad(mask, 1), # without padding this fxn fails when object is fully flush with border
                                    mode=find_boundaries_mode,
                                    connectivity=connectivity)
        
        boundary = boundary[_bbox2slice([1]*boundary.ndim + [-1]*boundary.ndim)] # un-pad
        to_remove = boundary & threshed_out
        if not to_remove.any():
            break
        mask[to_remove] = False

        if remove_islands:
            mask = _extract_largest_object(mask) # remove little islands that were above threshold

    # re‐apply the label value
    mask = mask * lbl

    if restore_background_labels: # if any non this lbl were included add back in 
        mask = np.where(obj_mask, mask, sub_labels)

    return bbox_slices, mask

def _bbox2slice(bbox):
    """ convert a 2d or 3d bbox to slice. assumes bbox is formated (minr, minc, maxr, maxc) or (minz, minr, minc, maxz, maxr, maxc)"""
    ndim = len(bbox)//2
    as_slice = tuple([slice(bbox[i%ndim],bbox[i%ndim+ndim]) for i in range(ndim)])
    return as_slice



def read_img(img_path, fmt=None, collapse=False):
    """ 
    read image from path 

    args:
        fmt: string format of dims present in raw image
        collapse: if fmt is provided, standardize dim order and collapse singleton dims
    
    """
    img = _imread(img_path)
    if fmt and collapse:
        img, fmt = standardize_collapse(img, fmt, 'STCZYX')
        return img, fmt
    return img

def _imread(img_path):
    """ helper function for loading tiffs or other types with PIL  """
    # TODO move to image parser and handle different types 
    if img_path.endswith('.tif') or img_path.endswith('.tiff'):
        return imread(img_path)
    
    if img_path.endswith('.npy'):
        return np.load(img_path)
    
    from PIL import Image
    with Image.open(img_path) as im:
        return np.array(im)
    

def load_pdf(pdf_path, page=0, dpi=300):
    """ loads a pdf as an array"""
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, dpi=dpi)  # adjust dpi for resolution
    image = pages[page]  # get 1 page

    # Convert to NumPy array
    array = np.array(image)
    return array

def load_svg(svg_path):
    """ loads a svg as an array"""
    import cairosvg
    from PIL import Image
    import io
    # Convert SVG to PNG in memory
    png_bytes = cairosvg.svg2png(url=svg_path)
    image = Image.open(io.BytesIO(png_bytes))

    # Convert to NumPy array
    array = np.array(image)
    return array


def load_image_data(
    path: str,
    key: Optional[str] = None,
    lazy_loading: bool = False
) -> np.ndarray:
    """Helper function to load image data from file.
        # taken from https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/util.py#L972
    Args:
        path: The filepath to the image data.
        key: The internal filepath for complex data formats like hdf5.
        lazy_loading: Whether to lazyly load data. Only supported for n5 and zarr data.

    Returns:
        The image data.
    """
    import imageio
    if key is None:
        image_data = imageio.imread(path)
    else:
        with open_file(path, mode="r") as f:
            image_data = f[key]
            if not lazy_loading:
                image_data = image_data[:]
    return image_data


def crop_img(bbox, img, pad, chs):
    """crop image based on bbox, with padding, and select channels"""
    if isinstance(pad, int): pad = [pad, pad]
    elif isinstance(pad, list) and len(pad) == 2:
        assert all([isinstance(el, int) for el in pad]), pad
    xmax, ymax, _ = img.shape
    x,y,X,Y = np.array(bbox) + np.array([-pad[0], -pad[1], pad[0], pad[1]])
    x,y,X,Y = max(x, 0), max(y,0), min(X, xmax), min(Y, ymax)
    crop = img[x:X,y:Y, chs]
    return crop


def max_intensity_projection(array: np.ndarray, axis: int = 0):
    """
    Creates a maximum intensity projection of a multidimensional array over a specified axis.
    """
    return np.max(array, axis=axis)

# Create an alias for the max_intensity_projection function
mip = max_intensity_projection 



    
def pai(array, asstr=False):
    """prints the info for a list of arrays. info includes: min, mean, max, shape, dtype"""
    if isinstance(array, list):
        return [print_array_info(a, asstr) for a in array]
    return print_array_info(array, asstr)

def print_array_info(array, asstr=False):
    """ helper function for pai 'print.array.info' for single array """
    res = f"shape = {array.shape}\nmin/mean/max = {array.min(), array.mean(), array.max()}\ndtype = {array.dtype}\n"
    if not asstr:
        print(res)
    return res

def crop_or_pad(array, target_shape=(256, 256)):
    """Crop or pad an 2d array to a target shape. crop is from top-left corner."""
    result = np.zeros(target_shape, dtype=array.dtype)
    mins = np.minimum(np.array(array.shape), target_shape)
    result[:mins[0], :mins[1]] = array[:mins[0], :mins[1]]
    return result



def segment_neurites(base_img, PLOT=False, n_objs=1, sigma=5, tophat_radius=15, norm_dict ={0: {'nmin': 1, 'nmax': 99.8}}, median_connectivity=2, median_iter=1, labels_binary_dilate = 7):
    """
    implementation of morphological filters to segement neurite branches
    
    Args:
        base_img: input image
        n_objs: take largest n objs
        sigma: gaussian filter sigma
        tophat_radius: tophat radius
        norm_dict: normalization dict
        median_connectivity: connectivity for median filter
        median_iter: number of iterations for median filter
        labels_binary_dilate: binary dilation for labels
    """
    import skimage
    from scipy import ndimage as ndi
    import SynAPSeg.utils.utils_plotting as up 
    

    median_struct = ndi.iterate_structure(ndi.generate_binary_structure(base_img.ndim, median_connectivity), median_iter)

    pred_input_preproc = base_img.copy()
    pred_input_preproc = skimage.filters.median(pred_input_preproc, footprint=median_struct)
    pred_input_preproc = skimage.filters.gaussian(pred_input_preproc, sigma=(sigma, sigma), truncate=3.5)
    # pred_input_preproc = bg_subtract_tophat(pred_input_preproc, tophat_radius=tophat_radius)

    pred_input_preproc_norm, _ = normalize(pred_input_preproc, norm_dict)
    pred_input_preproc_norm = np.clip(pred_input_preproc_norm, 0, 1)
    
    thresh = skimage.filters.threshold_triangle(pred_input_preproc_norm)
    neurites_img = pred_input_preproc_norm>thresh
    neurites_img = skimage.morphology.binary_dilation(neurites_img, footprint=np.ones((labels_binary_dilate,labels_binary_dilate)))
    neurites_img = skimage.morphology.label(neurites_img).astype('int32')
    
    props = skimage.measure.regionprops(neurites_img)
    largest_obj_area = max(props, key=lambda x: x.area).area
    largest_label = max(props, key=lambda x: x.area).label

    # Step 4: Create a mask for the largest n objects
    mask = extract_largest_objects(neurites_img, n=n_objs)
    
    if PLOT:
        fig,axs=plt.subplots(1,4, figsize=(20,10))
        up.show(base_img, def_title='og_img', ax=axs[0])
        up.show(pred_input_preproc_norm, def_title='pred_input_preproc_norm', ax=axs[1])
        up.show(neurites_img, def_title=f'neurites_labeled_img (largest area:{largest_obj_area})', ax=axs[2])
        up.show(base_img * to_binary(mask), def_title='neurites_int_img', ax=axs[3])
    return mask


def plot_all_threshold_methods(vol):
    """
    Plot the results of different thresholding methods on a 3D volume.
    """
    import inspect
    from skimage.exposure import histogram
    from skimage import filters

    # Global algorithms.
    thresh = lambda x: x
    methods = {
            'Isodata': thresh(filters.threshold_isodata),
            'Li': thresh(filters.threshold_li),
            'Mean': thresh(filters.threshold_mean),
            # 'Minimum': thresh(filters.threshold_minimum),
            'Otsu': thresh(filters.threshold_otsu),
            'Triangle': thresh(filters.threshold_triangle),
            'Yen': thresh(filters.threshold_yen),
        }
    
    hist = histogram(vol.reshape(-1), 256, source_range='image')
    for name, func in methods.items():
        # Use precomputed histogram for supporting functions
        sig = inspect.signature(func)
        _kwargs = dict(hist=hist) if 'hist' in sig.parameters else {}
        plot_thresh(vol, title=name, thresh_func=func, **_kwargs)
    


def plot_thresh(volume, title='', thresh_func=None, **func_kwargs):
    """
    Plot the results of a thresholding method on a 3D volume

    Args:
        volume (np.ndarray): 3D numpy array representing raw intensity values.
        title (str, optional): Title for the plot. Defaults to ''.
        thresh_func (callable, optional): Thresholding function. Defaults to filters.threshold_otsu if None.
        **func_kwargs: Additional keyword arguments to pass to the thresholding function.
    """
    from skimage import exposure
    import matplotlib.pyplot as plt
    from skimage import filters
    
    if thresh_func is None:
        thresh_func = filters.threshold_otsu
    
    val = thresh_func(volume, **func_kwargs)
    hist, bins_center = exposure.histogram(volume)

    plt.figure(figsize=(9, 4))
    ax = plt.subplot(131)
    up.show(volume, ax=ax)
    plt.axis('off')
    ax = plt.subplot(132)
    up.show(np.where(volume < val, 0, volume), ax=ax)
    plt.axis('off')
    plt.subplot(133)
    plt.plot(bins_center, hist, lw=2)
    plt.axvline(val, color='k', ls='--')
    plt.yscale('log')
    plt.xscale('log')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()


def refine_segmentation(volume, threshold, sigma=1.0, min_size=500, compactness=0.01, min_distance=5, footprint=None):
    """
    Refines 3D segmentation by applying intensity thresholding, filtering small objects,
    and reducing over-segmentation using compact watershed.
    
    Args:
        volume (np.ndarray): 3D numpy array representing raw intensity values.
        threshold (float or None): Intensity threshold for segmentation. If None, Otsu's method is used.
        sigma (float): Standard deviation for Gaussian smoothing (optional, default is 1.0).
        min_size (int): Minimum size of objects to retain (default is 500).
        compactness (float): Compactness parameter for watershed to control over-segmentation (default is 0.01).
        min_distance (int): Minimum distance between peaks for watershed markers (default is 5).
    
    Returns:
        refined_labels (np.ndarray): 3D array with refined segmentation labels.
        num_objects (int): Number of refined segmented objects.
    """
    from scipy.ndimage import label, gaussian_filter, binary_dilation
    from skimage.morphology import remove_small_objects
    from skimage.segmentation import watershed
    from skimage.feature import peak_local_max
    from scipy.ndimage import distance_transform_edt

    # Optional: Smooth the volume
    smoothed_volume = gaussian_filter(volume, sigma=sigma)
    
    
    # Apply intensity threshold
    binary_mask = smoothed_volume > threshold
    
    # Remove small objects and fill gaps
    binary_mask = remove_small_objects(binary_mask, min_size=min_size)
    binary_mask = binary_dilation(binary_mask, structure=footprint)
    
    # Compute the distance map
    distance_map = distance_transform_edt(binary_mask)
    
    # Detect local maxima for watershed markers
    local_maxi_coords = peak_local_max(
        distance_map, min_distance=min_distance, threshold_abs=0.1 * distance_map.max(), labels=binary_mask
    )
    markers = np.zeros_like(binary_mask, dtype=np.int32)
    markers[tuple(local_maxi_coords.T)] = np.arange(1, len(local_maxi_coords) + 1)
    
    # Apply compact watershed
    refined_labels = watershed(-distance_map, markers, mask=binary_mask, compactness=compactness)
    
    # Return the refined labels and object count
    num_objects = len(np.unique(refined_labels)) - 1  # Exclude background
    return refined_labels, num_objects        
        


def adjust_mean_intensity(source_image, target_image):
    """
    Adjust the source image so that its mean intensity matches the target mean intensity.

    Parameters:
    - source_image: 2D numpy array, the source image whose intensity is to be adjusted.
    - target_image: 2D numpy array, the source image whose intensity is the desired mean intensity.

    Returns:
    - A 2D numpy array of the adjusted image.
    """
    # Calculate the current mean intensity of the source image
    # Calculate the difference between the current and target mean intensities
    intensity_difference = np.mean(target_image) - np.mean(source_image)

    # Adjust the source image's intensity
    adjusted_image = source_image + intensity_difference

    # Clip values to ensure they stay within the valid range (e.g., 0 to 255 for uint8 images)
    adjusted_image = np.clip(adjusted_image, target_image.min(), target_image.max())

    return adjusted_image




def unpack_array_axis(arr, axis=0):
    """
    Unpack a numpy array along a specified axis. Returns a list of arrays.
    """
    inds = list(range(arr.shape[axis]))
    return [np.take(arr, i, axis=axis) for i in inds]



try:
    import numba as nb
    @nb.njit
    def numba_label(input):
        structuring_element = np.array([[0, 1, 0],
                                        [1, 1, 1],
                                        [0, 1, 0]])
        labeled_array = np.zeros(input.shape, dtype=np.int32)
        label_count = 0
        rows, cols = input.shape
        
        def find_neighbors(r, c):
            neighbors = []
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if (dr == 0 and dc == 0) or not (0 <= r + dr < rows and 0 <= c + dc < cols):
                        continue
                    if structuring_element[dr + 1, dc + 1]:
                        neighbors.append((r + dr, c + dc))
            return neighbors

        def flood_fill(r, c, label):
            stack = [(r, c)]
            while stack:
                cr, cc = stack.pop()
                if labeled_array[cr, cc] != 0:
                    continue
                labeled_array[cr, cc] = label
                for nr, nc in find_neighbors(cr, cc):
                    if input[nr, nc] != 0 and labeled_array[nr, nc] == 0:
                        stack.append((nr, nc))

        for r in range(rows):
            for c in range(cols):
                if input[r, c] != 0 and labeled_array[r, c] == 0:
                    label_count += 1
                    flood_fill(r, c, label_count)

        return labeled_array, label_count

    @nb.njit
    def correct_annotated_labels_numba(sd_mask, size_min=0): #, size_min, labels_arr, areas_arr):
        """correct annotations where same label was used in non-touching objects, where ndi.label fails to separate"""
        corrected_mask = np.zeros_like(sd_mask, dtype=np.int32)
        unique_labels = np.unique(sd_mask)
        current_val = 1
        
        for ul in unique_labels:
            if ul == 0: continue
            labels, count = numba_label(sd_mask == ul)
            
            for uni_l in np.unique(labels):
                if uni_l == 0: continue
                mask = labels==uni_l
                if mask.sum() > size_min:
                    corrected_mask += mask.astype('int') * current_val
                    current_val = current_val +1
        return corrected_mask
    
    def init_correct_annoted_labels_numba():
        input_array = np.array([[0, 0, 2, 2, 0],
                            [1, 2, 0, 0, 1],
                            [0, 0, 1, 0, 0],
                            [1, 0, 1, 1, 1],
                            [0, 2, 0, 0, 0]], dtype=np.int32)
        return correct_annotated_labels_numba(input_array)
except ImportError:
    pass

def find_extent(volume):
    """
    Finds the extent of a 2D or 3D image/volume where pixel values are > 0.
    
    For 3D: returns (z_min, z_max, y_min, y_max, x_min, x_max)
    For 2D: returns (y_min, y_max, x_min, x_max)
    """
    # Get the coordinates where the pixel values are greater than zero
    non_zero_coords = np.argwhere(volume > 0)
    if non_zero_coords.size == 0:
        raise ValueError("No non-zero pixels/voxels found.")
    
    coords_min = non_zero_coords.min(axis=0)
    coords_max = non_zero_coords.max(axis=0)
    
    if volume.ndim == 3:
        z_min, y_min, x_min = coords_min
        z_max, y_max, x_max = coords_max
        return (z_min, z_max, y_min, y_max, x_min, x_max)
    elif volume.ndim == 2:
        y_min, x_min = coords_min
        y_max, x_max = coords_max
        return (y_min, y_max, x_min, x_max)
    else:
        raise ValueError("Only 2D or 3D arrays supported.")


def find_extent_and_crop(volumes):
    """
    Finds the extent of a 2D or 3D image/volume where pixel values are > 0,
    and returns the cropped image(s) based on this extent.
    If multiple volumes, pass as list, uses 1st array to determine extent
    
    If you pass in a single array, you get back a single cropped array;
    if you pass in a list of arrays, you get back a list of cropped arrays.
    All volumes must have the same dimensionality.

    Parameters:
    volume (numpy.ndarray): A 3D numpy array representing the volume.

    Returns:
    cropped_volume (numpy.ndarray): The cropped 3D volume.
    extent (tuple): The (z_min, z_max, y_min, y_max, x_min, x_max) of the extent.
    """
    # Wrap single volume in a list for uniform handling
    is_list = isinstance(volumes, list)
    vols = volumes if is_list else [volumes]
    
    # Determine extent from the first volume
    extent = find_extent(vols[0])
    
    # Crop according to dimensionality
    if vols[0].ndim == 3:
        z0, z1, y0, y1, x0, x1 = extent
        cropped = [v[z0:z1+1, y0:y1+1, x0:x1+1] for v in vols]
    else:  # 2D
        y0, y1, x0, x1 = extent
        cropped = [v[y0:y1+1, x0:x1+1] for v in vols]
    
    return cropped if is_list else cropped[0]
        

def create_patch_iterator(images, format="czyx", patch_size=(256, 256)):
    """
    Create an iterator that yields random square patches from a list of images.

    Parameters:
        images (list(numpy.ndarray) or numpy.ndarray): Input image arrays.
        format (str): Format of the image (e.g., "czyx", "zyx", "yx").
        patch_size (tuple): Desired patch size as (height, width).
    
    Returns:
        Iterator: An iterator that yields random patches from the image.
        
    # TODO merge with coord_patch_iterator and 
    """
    # Parse the format to identify spatial dimensions
    format = format.lower()
    if "y" not in format or "x" not in format:
        raise ValueError("Unsupported format. Must include 'y' and 'x'.")

    y_index = format.index("y")
    x_index = format.index("x")
    
    # handle case where images is a single np array rather than expected list
    if not isinstance(images, list):
        images = [images]
    
    # all images must have same shape
    uShapes = set([im.shape for im in images])
    assert len(uShapes)==1, f"all shapes must match, got {uShapes}."
    
    # Get spatial dimensions using the first image provided
    image = images[0]
    spatial_shape = (image.shape[y_index], image.shape[x_index])
    
    if any(dim < size for dim, size in zip(spatial_shape, patch_size)):
        raise ValueError("Patch size is larger than the image dimensions.")

    def patch_generator():
        while True:  # Infinite loop to allow arbitrary patch generation
            # Randomly select top-left corner for cropping
            start_y = np.random.randint(0, spatial_shape[0] - patch_size[0] + 1)
            start_x = np.random.randint(0, spatial_shape[1] - patch_size[1] + 1)
            
            # Create slicing indices for cropping
            slices = [slice(None)] * image.ndim  # initialize as full slices
            slices[y_index] = slice(start_y, start_y + patch_size[0])
            slices[x_index] = slice(start_x, start_x + patch_size[1])
            
            yield [img[tuple(slices)] for img in images]
    
    return patch_generator()


def create_patches(images, format="czyx", patch_size=(256, 256), n_patches=1):
    """
    Create random N-dimensional patches from a list of images.

    Parameters:
        images (list(numpy.ndarray) or numpy.ndarray): Input image arrays.
        format (str): Format of the image (e.g., "czyx", "zyx", "yx").
        patch_size (tuple): Desired patch size for spatial dimensions. 
                           Use None for a dimension to use the full size.
        n_patches (int): Number of patches to generate.
    
    Returns:
        list: List of patch arrays, where each element is a list of patches 
              (one per input image).
    """
    # Parse the format to identify spatial dimensions
    format = format.lower()
    if "y" not in format or "x" not in format:
        raise ValueError("Unsupported format. Must include 'y' and 'x'.")

    # Find indices of all spatial dimensions present in format
    spatial_dims = []
    spatial_indices = []
    dim_names = ['z', 'y', 'x']
    
    for dim_name in dim_names:
        if dim_name in format:
            spatial_dims.append(dim_name)
            spatial_indices.append(format.index(dim_name))
    
    # Ensure patch_size matches number of spatial dimensions
    if len(patch_size) != len(spatial_dims):
        raise ValueError(
            f"patch_size length ({len(patch_size)}) must match number of "
            f"spatial dimensions ({len(spatial_dims)}: {spatial_dims})"
        )
    
    # Handle case where images is a single np array rather than expected list
    if not isinstance(images, list):
        images = [images]
    
    # All images must have same shape - since we use shape of first one to determine boundaries
    uShapes = set([im.shape for im in images])
    assert len(uShapes) == 1, f"All shapes must match, got {uShapes}."
    
    # Get spatial dimensions using the first image provided
    image = images[0]
    spatial_shape = tuple(image.shape[idx] for idx in spatial_indices)
    
    # Process patch_size: replace None with full dimension, pad if needed
    processed_patch_size = []
    padding_needed = []
    
    for i, (dim_size, patch_dim) in enumerate(zip(spatial_shape, patch_size)):
        if patch_dim is None:
            # Use full dimension size
            processed_patch_size.append(dim_size)
            padding_needed.append((0, 0))
        elif dim_size < patch_dim:
            # Pad to reach patch_dim
            processed_patch_size.append(patch_dim)
            total_pad = patch_dim - dim_size
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            padding_needed.append((pad_before, pad_after))
        else:
            # No padding needed
            processed_patch_size.append(patch_dim)
            padding_needed.append((0, 0))
    
    # Apply padding if needed
    if any(pad != (0, 0) for pad in padding_needed):
        # Create padding spec for all dimensions
        pad_width = [(0, 0)] * image.ndim
        for i, idx in enumerate(spatial_indices):
            pad_width[idx] = padding_needed[i]
        
        # Pad all images
        images = [np.pad(img, pad_width, mode='reflect') for img in images]
        
        # Update spatial shape after padding
        spatial_shape = tuple(images[0].shape[idx] for idx in spatial_indices)

    patches = []
    for _ in range(n_patches):
        # Randomly select top-left corner for cropping in all spatial dimensions
        starts = [
            np.random.randint(0, spatial_shape[i] - processed_patch_size[i] + 1)
            if spatial_shape[i] > processed_patch_size[i] else 0
            for i in range(len(spatial_dims))
        ]
        
        # Create slicing indices for cropping
        slices = [slice(None)] * images[0].ndim  # initialize as full slices
        for i, idx in enumerate(spatial_indices):
            slices[idx] = slice(starts[i], starts[i] + processed_patch_size[i])
        
        patches.append([img[tuple(slices)] for img in images])
    
    return patches



def coord_patch_iterator(images, format="czyx", patch_size=(256, 256)):
    """
    Create an iterator that yields YX slices for patches of an n-d image.
        format must include 'y' and 'x' in any order

    Parameters:
        images (list(numpy.ndarray) or numpy.ndarray): Input image arrays.
        format (str): Format of the image (e.g., "czyx", "zyx", "yx").
        patch_size (tuple): Desired patch size as (height, width) (e.g. y,x).
    
    Returns:
        Iterator: An iterator that yields slices for patches covering the image.
    
    Notes:
        # TODO make patch_size n-dimensional i.e. merge with coord_patch_iterator's random patch implementation

    """
    # input validation
    #######################################################
    assert len(patch_size) == 2, "Patch size must be a tuple of length 2 (y,x | height, width)."
    
    # Parse the format to identify spatial dimensions
    format = format.lower()
    if "y" not in format or "x" not in format:
        raise ValueError("Unsupported format. Must include 'y' and 'x'.")

    y_index = format.index("y")
    x_index = format.index("x")
    
    # Handle case where images is a single np array rather than expected list
    if not isinstance(images, list):
        images = [images]
    
    # All images must have same shape
    u_shapes = set(im.shape for im in images)
    if len(u_shapes) != 1:
        raise ValueError(f"All shapes must match, got {u_shapes}.")
    

    # Get spatial dimensions using the first image provided
    #######################################################
    image = images[0]
    spatial_shape = (image.shape[y_index], image.shape[x_index])
    
    if any(dim < size for dim, size in zip(spatial_shape, patch_size)):
        raise ValueError("Patch size is larger than the image dimensions.")

    def patch_coord_generator():
        # Determine the number of patches along each dimension
        num_patches_y = (spatial_shape[0] + patch_size[0] - 1) // patch_size[0]
        num_patches_x = (spatial_shape[1] + patch_size[1] - 1) // patch_size[1]
        
        for patch_y in range(num_patches_y):
            for patch_x in range(num_patches_x):
                # Calculate the start and end coordinates for this patch
                start_y = patch_y * patch_size[0]
                start_x = patch_x * patch_size[1]
                end_y = min(start_y + patch_size[0], spatial_shape[0])
                end_x = min(start_x + patch_size[1], spatial_shape[1])
                
                # Create slicing indices for cropping
                slices = [slice(None)] * image.ndim  # Initialize as full slices
                slices[y_index] = slice(start_y, end_y)
                slices[x_index] = slice(start_x, end_x)
                
                yield tuple(slices)
    
    return patch_coord_generator()



def euclidean_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points with XY coordinates.

    Parameters:
    - point1: Tuple (x1, y1)
    - point2: Tuple (x2, y2)

    Returns:
    - The Euclidean distance between the two points
    """
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def manhattan_distance(point1, point2):
    """
    Calculate the Manhattan (street) distance between two points with XY coordinates.
    simulates a "street" or "grid-like" distance where movement is restricted to vertical and horizontal directions only
    
    Parameters:
    - point1: Tuple (x1, y1) representing the first point
    - point2: Tuple (x2, y2) representing the second point

    Returns:
    - The Manhattan distance between the two points
    """
    x1, y1 = point1
    x2, y2 = point2
    return abs(x2 - x1) + abs(y2 - y1)

def align_shapes(target, roi):
    """
    copys roi array in array of shape target, e.g. if target is smaller crops rois, if larger fills top left corner with roi 
    arrays must have same number of dimensions
    #TODO not sure if this function is really useful

    old description
    Adjust the shape of the ROI array to match the target array.
    
    Parameters:
        target (np.ndarray): The target array whose shape needs to be matched.
        roi (np.ndarray): The ROI array that needs resizing.
    
    Returns:
        np.ndarray: The resized ROI array with the same shape as the target array.
    """
    target_shape = target.shape
    roi_shape = roi.shape

    if roi_shape == target_shape:
        print("ROI is already the same shape as target.")
        return roi

    # Initialize a new array with the target shape, filling it with zeros
    aligned_roi = np.zeros(target_shape, dtype=roi.dtype)

    # Determine the overlap region to copy
    overlap_slices = tuple(slice(0, min(dim_t, dim_r)) for dim_t, dim_r in zip(target_shape, roi_shape))

    # Copy the ROI data into the aligned array
    aligned_roi[overlap_slices] = roi[overlap_slices]

    return aligned_roi


def split_image_into_substacks(image, stack_size, discard_incomplete=False, axis=0):
    """
    Splits an array into substacks along a specified axis based on the stack_size.

    Parameters:
        image (numpy.ndarray): Input image.
        stack_size (int): Number of slices in each substack along the specified axis.
        discard_incomplete (bool): If True, discard incomplete stacks at the end.
        axis (int): Axis along which to bin the image into substacks (default is 0).

    Returns:
        list of numpy.ndarray: List of 3D substacks.
    """
    # Get the length along the specified axis
    z_dim = image.shape[axis]

    # Create indices for slicing along the axis
    indices = [range(i, min(i + stack_size, z_dim)) for i in range(0, z_dim, stack_size)]

    # Use np.take to slice along the axis
    substacks = [np.take(image, ind, axis=axis) for ind in indices]

    # Discard incomplete stacks if specified
    if discard_incomplete:
        substacks = [stack for stack in substacks if stack.shape[axis] == stack_size]

    return substacks










def map_arr_values(arr: np.ndarray, mapping: dict):
    """
    Map ranges or specific values in a NumPy array to new values based on a mapping dictionary.
    
    Parameters:
    arr (np.ndarray): The input array to be mapped.
    mapping (dict): A dictionary where the keys are either tuples (defining ranges) 
                    or single values, and the values are the mapped values.
    
    Returns:
    np.ndarray: A new array with the mapped values.
    
    
    Example usage:
        arr = np.array([[0, -2, -1], [1, -1, 3], [-3, 2, 4]])
        mapping = {
            (-np.inf, -1): 2,  # Map all values less than -1 to 2
            -1: 1,             # Map -1 to 1
            (2, np.inf): 5     # Map values greater than 2 to 5
        }
        mapped_arr = map_arr_values(arr, mapping)

    """
    def _sanitize(key):
        """ allow string inf to be converted to np.inf and -inf"""
        if isinstance(key, str):
            key = key.lower()
            if key == 'inf':
                return np.inf
            elif key == '-inf':
                return -np.inf
            else:
                float(key)
        return key
    
    mapped_arr = np.copy(arr)  # Create a copy of the original array to avoid modifying in place
    
    for key, value in mapping.items():

        if isinstance(key, (tuple, list)):  # Key is a range (min, max)
            # Map values that fall within the range (inclusive of min and max)
            mapped_arr[(arr >= _sanitize(key[0])) & (arr <= _sanitize(key[1]))] = value
        else:  # Key is a single value
            # Map the exact value
            mapped_arr[arr == _sanitize(key)] = value
    
    return mapped_arr


def nd_slice(arr: np.ndarray, axes, indices) -> Tuple[slice]:
    """Returns n-dimensional slice indexer along array axes at given indices.
    if you want multiple indices along a single axis use np.take(a, inds, axis)
    
    Args:
        arr: The n-dimensional array.
        axes: An integer, a list of integers, or a tuple of integers specifying the axes.
        indices: An integer, a list of integers, or a tuple of integers specifying the indices for the axes.
        
    Returns:
        A tuple of slices that can be used for indexing.
    """
    # Ensure axes and indices are iterable (list or tuple)
    if isinstance(axes, (int, tuple)):
        axes = list(axes) if isinstance(axes, tuple) else [axes]
    if isinstance(indices, (int, tuple)):
        indices = list(indices) if isinstance(indices, tuple) else [indices]
    
    if len(axes) != len(indices):
        raise ValueError("axes and indices must have the same length")

    # Step 1: Create a list of slice(None) for each axis
    indexer = [slice(None)] * arr.ndim

    # Step 2: For each (axis, index) pair, replace the slice
    for ax, idx in zip(axes, indices):
        indexer[ax] = idx

    return tuple(indexer)

def index_along_axes(arr, axes, indices, return_indexer=False) -> np.ndarray | Tuple[Tuple[slice]]:
    """
    Given an array `arr` and two equal-length tuples:
      - `axes` (which axes to index along),
      - `indices` (which indices to take on those axes),
    returns the sliced array or if return_indexer is True, a tuple(*slice).

    For example, if `arr.shape == (5, 6, 7)`, 
    and we want to index axis=0 at index=2, and axis=2 at index=5,
    we can do:
    
        result = index_along_axes(arr, (0, 2), (2, 5))
    
    which is effectively `arr[2, :, 5]`.
    """
    indexer = nd_slice(arr, axes, indices)

    # Step 3: Convert to a tuple and index
    if return_indexer:
        return tuple(indexer)
    else:
        return arr[tuple(indexer)]
    


def insert_inplace(arr, values, index, axis):
    """handle arbitrary n-dimensional arrays, you can construct the slice dynamically based on the axis and index"""
    # select the entire dimension
    slices = nd_slice(arr, axis, index)
    # Perform the insertion by assigning values
    arr[tuple(slices)] = values
    

def transform_axes(array: np.ndarray, current_format: str, target_format: str) -> np.ndarray:
    """
    Transforms the dimensions of a multidimensional array to match a desired target format,
    adding singleton dimensions where necessary.
    
    Parameters:
    - array (np.ndarray): The input array to be transformed.
    - current_format (str): The current axis format of the array (e.g., 'xyztc').
    - target_format (str): The desired axis format of the array (e.g., 'ctxyz').
    
    Returns:
    - np.ndarray: The transformed array.
    
    Example:
        Assume we have an array in the shape of (X, Y, Z, T) = (64, 64, 30, 10) representing 'xyzt' format
        We want to transform it to 'ctxyz' format, which would have the shape (1, 10, 64, 64, 30)

        input_array = np.random.rand(64, 64, 30, 10) 
        current_format = 'xyzt'
        target_format = 'ctxyz'

        transformed_array = transform_axes(input_array, current_format, target_format)
        print("Original shape:", input_array.shape)
        print("Transformed shape:", transformed_array.shape)
        
    """
    current_format = current_format.upper()
    target_format = target_format.upper()
    # check if array matches current format
    if current_format == target_format:
        return array
    
    assert array.ndim == len(current_format), f"current_format doesn't match array dimensions, {current_format} != {array.shape}"
    # Determine the shape with missing axes added as singleton dimensions
    current_shape = list(array.shape)
    for axis in target_format:
        if axis not in current_format:
            current_format += axis
            current_shape.append(1)
    array = array.reshape(current_shape)

    # Find the index positions of each axis in the target format based on current format
    target_order = [current_format.index(axis) for axis in target_format]

    # Use numpy's transpose function to reorder the axes
    transformed_array = np.transpose(array, axes=target_order)
    
    return transformed_array

def reduce_dimensions(arr, current_format, take_dims='', project_dims='', return_current_format=False) -> np.ndarray | tuple[np.ndarray, str]:
    """
    Reduces dimensions of an array by either slicing first position or projecting along specified axes.
    
    Parameters:
    arr : np.ndarray
        The input array to be reduced.
    current_format : str
        A string where each character represents an axis label in `arr` (e.g., "XYZC" for a 4D array).
    take_dims : str
        A string containing the axis labels along which to take the first element 
        e.g., reduce_dimensions(arr, current_format, take_dims='Z') on ZYX array of shape (3, 32, 32) would return arr of shape (32,32)
    project_dims : str, optional
        A string containing the axis labels along which to apply maximum intensity projection 
        e.g.,  reduce_dimensions(arr, current_format, project_dims="Z") on ZYX array of shape (3, 32, 32) would return z projection of shape (32,32)
    
    Returns:
    np.ndarray
        The reduced array.
    
    Raises:
    AssertionError
        If there are duplicate axes specified in both `take_dims` and `project_dims`.
    
    Notes:
    - The function first reduces `arr` by slicing along `take_dims` axes, then applies 
      maximum intensity projection along `project_dims` axes. This ensures the axes 
      for projection remain valid after slicing.
    - Axes should be specified in `current_format` order to avoid unexpected results.
    """
    
    # Ensure no duplicate axes between take and project
    reduce_dims = take_dims + project_dims
    assert len(set(reduce_dims)) == len(reduce_dims), f"axes cannot be duplicated, but got dims {reduce_dims}"
    if len(set(current_format).intersection(set(reduce_dims))) == 0: # these axes do not exist, must already be taken
        return arr
        
    # Iterate over `take_dims` first to slice those axes
    for axis_label in take_dims:
        axis_index = current_format.index(axis_label)
        arr = np.take(arr, indices=0, axis=axis_index)
        # Update current_format to reflect removed axis
        current_format = current_format[:axis_index] + current_format[axis_index + 1:]
    
    # Now iterate over `project_dims` to apply max projection
    for axis_label in project_dims:
        axis_index = current_format.index(axis_label)
        arr = np.max(arr, axis=axis_index)
        # Update current_format to reflect removed axis
        current_format = current_format[:axis_index] + current_format[axis_index + 1:]
    
    if return_current_format:
        return arr, current_format
    return arr

def morph_to_target_format(arr: np.ndarray, current_fmt: str, target_fmt: str) -> np.ndarray:
    """ helper that wraps morph_to_target_shape, infering target shape """
    target_shape = tuple([arr.shape[i] for i in [current_fmt.index(d) for d in target_fmt]])
    return morph_to_target_shape(arr, current_fmt, target_shape, target_fmt)
    

def morph_to_target_shape(arr: np.ndarray, current_fmt: str, target_shape: tuple, target_fmt: str) -> np.ndarray:
    """
    Adjust an array's shape and axis order to match a target format and shape.
    
    Expands (via broadcasting) or reduces (via projection) as needed.

    Args:
        arr: Input ndarray (e.g. (64, 64)).
        current_fmt: Axis format of arr (e.g. 'YX').
        target_shape: Desired shape (e.g. (4, 16, 64, 64)).
        target_fmt: Desired axis format (e.g. 'CZYX').

    Returns:
        Transformed ndarray matching target shape and format.
    """
    if arr.shape == target_shape:
        return arr
    
    current_ax_shapes = {ax: arr.shape[current_fmt.index(ax)] for ax in current_fmt}
    target_ax_shapes = {ax: target_shape[target_fmt.index(ax)] for ax in target_fmt}

    # check dimension shapes are compatible
    shared_axes = set(target_fmt).intersection(current_fmt)
    _compat = [current_ax_shapes[ax] == target_ax_shapes[ax] for ax in shared_axes]
    assert all(_compat), f"arr and intensity_img shape incompatable\n\tcurrent_ax_shapes:{current_ax_shapes}\n\ttarget_ax_shapes:{target_ax_shapes}"
    
    # Expand or reduce
    # if target is larger, insert current into target - otherwise reduce target to current
    if len(target_ax_shapes) > len(current_ax_shapes):

        reshaped = transform_axes(arr, current_fmt, target_fmt) # reshape to target format by inserting singleton dimensions
        if reshaped.shape == target_shape:
            return reshaped
    
        # broadcast to the full target shape - # e.g. insert an object of format "YX" into an object of format "CYZX"
        transformed = np.zeros(target_shape, dtype=arr.dtype)
        transformed[...] = reshaped
    else:
        # reduce target to current
        axes_diff = "".join(set(current_fmt).difference(target_fmt)) # axes to project over
        transformed = reduce_dimensions(arr, current_fmt, project_dims=axes_diff)
        transformed = transform_axes(transformed, subtract_dimstr(current_fmt, axes_diff), target_fmt)

    return transformed



def apply_to_chunks(arr: np.ndarray, current_format: str, iter_dims: str, f, out_dtype=None, *f_args, **f_kwargs):
    """ 
    Apply a function to specific axes (chunks) of an array.

    This function iterates over specified dimensions of an input array (`iter_dims`) 
    and applies a given function `f` to each chunk (slice) of the array. The resulting 
    processed chunks are combined into an output array of the same shape as the input.
    
    Assumes function doesn't reduce (consume) any axis. 
        e.g. applying mip function isn't supported since output shape changes, though it wouldn't fail

    Parameters
    ----------
    arr : np.ndarray
        The input array to process. Its dimensions must match the order specified in `current_format`.
    current_format : str
        A string describing the order of axes in the input array (e.g., "CZYX").
    iter_dims : str
        A subset of `current_format` specifying the axes over which to iterate. The function `f` will be applied to chunks 
        corresponding to these dimensions.
    f : callable
        The function to apply to each chunk. It must accept an array as its first argument and 
        return a processed array of the same shape.
    out_dtype : data-type, optional
        The data type for the output array. If not specified, the output will have the same 
        dtype as `arr`.
    **f_kwargs : dict
        Additional keyword arguments to pass to the function `f`.

    Returns
    -------
    np.ndarray
        An array of the same shape as `arr`, with `f` applied to each chunk.

    Example
    -------
    Apply a normalization function to each channel and slice of an image:
    
    def normalize_01(x):
         return (x - np.min(x)) / (np.max(x) - np.min(x))
    input_data = np.random.rand(3, 5, 256, 256)  # Shape: CZYX
    res = apply_to_chunks(input_data, 'CZYX', 'CZ', normalize_01, out_dtype=np.float32)

    """
    # ensure current format contains same num dims as input
    assert arr.ndim == len(current_format)
    
    # init array to store results
    result_dtype = arr.dtype if out_dtype is None else out_dtype
    result = np.zeros_like(arr, dtype=result_dtype)
    
    # determine how to slice chunks
    iter_axes = tuple([iter_dims.index(d) for d in iter_dims])
    iter_shape = tuple(arr.shape[current_format.index(d)] for d in iter_dims)
    iter_indicies = np.ndindex(iter_shape)
    
    # apply over chunks
    for indices in iter_indicies:
        slice_index = index_along_axes(arr, iter_axes, indices, return_indexer=True)
        chunk = arr[slice_index]
        result[slice_index] = f(chunk, *f_args, **f_kwargs)
        
    return result


def collapse_singleton_dims(arr: np.ndarray, current_format: str) -> tuple[np.ndarray, str]:
    """
    Collapses singleton dimensions in an array and updates the format string.

    Parameters:
    -----------
    arr : numpy.ndarray
        The input array with potentially singleton dimensions (dimensions of size 1).
    current_format : str
        A string representing the current format of the dimensions in the array.
        Each character corresponds to a dimension in the array.

    Returns:
    --------
        - collapsed_array: numpy.ndarray with singleton dimensions removed.
        - updated_format: str representing the updated format string.

    Example:
    --------
    arr = np.zeros((1, 1, 1, 3180, 1, 3180))
    current_format = "STCZYX"
    collapse_singleton_dims(arr, current_format)
    (array with shape (3180, 3180), "ZX")
    
    """
    assert isinstance(arr, np.ndarray)
    if arr.ndim != len(current_format):
        raise ValueError(f"Array dimensions ({arr.shape}) do not match the length of the format string ({current_format}).")
    
    # Get indices of non-singleton dimensions
    non_singleton_indices = [i for i, size in enumerate(arr.shape) if size > 1]
    updated_format = "".join(current_format[i] for i in non_singleton_indices)
    
    # Collapse the array by keeping only non-singleton dimensions
    collapsed_array = arr.squeeze()
    
    return collapsed_array, updated_format
                

def standardize_collapse(arr, input_format: str, standard_format: str):
    """standardize, then collapse singleton dims""" 
    arr = transform_axes(arr, input_format, standard_format)
    arr, current_format = collapse_singleton_dims(arr, standard_format)
    return arr, current_format

def has_singleton_dimenstions(arr):
    """ check if array has singleton dimensions """
    return 1 in arr.shape                 

def safe_squeeze(arr, min_n_dim):
    """ squeeze an array if dims don't drop below min_n_dim 

        raises:
            ValueError if not safe to squeeze
    """
    sq = arr.squeeze()
    if sq.ndim >= min_n_dim:
        return sq
    raise ValueError(f"cannot safely squeeze array of shape: {arr.shape} and keep it above min n_dim of {min_n_dim}")


def subtract_dimstr(s1: str, s2: str):
    """
    Removes characters from s1 that appear in s2 while preserving the order of remaining characters.

    Parameters:
        s1 (str): The input string to filter.
        s2 (str): The string containing characters to remove from s1.

    Returns:
        str: The filtered string.
        
    Example:
        subtract_dimstr('STCZYX', 'ST') --> 'CZYX'
    """
    return ''.join([char for char in s1 if char not in s2])                
                


def estimate_format(shape, default_format: str = 'STCZYX', channel_max: int = 4) -> str:
    """
    Estimate a dimension format string for an arbitrary image shape.

    Heuristics:
    - 1D: use last dim from default_format (typically 'X').
    - 2D: always 'YX'.
    - For n >= 3:
        * Identify the two largest dimensions and label them Y, X in their
          original order.
        * All other dimensions are assigned from the remaining letters of
          default_format[-n:], with special handling for 'C' (channel):
              - Prefer to assign 'C' to a small (<= channel_max) dimension.
              - Prefer dims > 1 over dims == 1 (1's are often S/T).
              - Prefer the **rightmost** such dim (channels often trail).
              - If no such candidate exists, 'C' is dropped rather than
                being assigned to a large spatial-like axis.
        * If no 'C' exists in the default tail (e.g., 3D shapes), but the
          last overall axis is small (<= channel_max) and non-spatial,
          we promote that axis to 'C' (e.g. (H, W, 3) -> 'YXC').

    Examples:
        (3, 23, 1024, 1024)   -> 'CZYX'
        (23, 3, 1024, 1024)   -> 'ZCYX'
        (3, 1024, 1024)       -> 'ZYX'   # could be CYX but we do not assume C
        (1024, 1024, 4)       -> 'YXC'   # here we assume C since last dim size is less than default channel max size
        (1024, 1024, 5)       -> 'YXZ'   
        
        (1, 1, 3, 23, 1024, 1024)  -> 'STCZYX'
        (1, 1, 23, 3, 1024, 1024)  -> 'STZCYX'
    """
    shape = tuple(shape)
    n = len(shape)

    if n == 0 or n == 1:
        raise ValueError("Cannot estimate format for an empty or 1D shape.")
    if n == 2:
        # Simple 2D image
        return 'YX'
    if n > len(default_format):
        raise NotImplementedError(
            f"Shapes larger than {len(default_format)} dims "
            f"are not currently supported (got {n})."
        )

    # --- 1) Identify spatial dims (Y, X) as the two largest ---
    idx_sorted = sorted(range(n), key=lambda i: shape[i], reverse=True)
    y_idx, x_idx = sorted(idx_sorted[:2])  # preserve order in shape

    axis_for_index = [''] * n
    axis_for_index[y_idx] = 'Y'
    axis_for_index[x_idx] = 'X'

    remaining_indices = [i for i in range(n) if i not in (y_idx, x_idx)]

    # --- 2) Build axis pool from default tail, minus Y and X ---
    tail = list(default_format[-n:])  # e.g. n=6 -> ['S','T','C','Z','Y','X']
    pool = [ax for ax in tail if ax not in ('Y', 'X')]  # non-spatial axes

    # --- 3) Optionally inject 'C' if strongly indicated but absent ---
    # This is mainly for 3D HW-C cases (e.g. (H, W, 3)).
    if 'C' not in pool and remaining_indices:
        last_idx = remaining_indices[-1]
        # Strong "channel" cue: last overall axis is small and non-spatial
        if last_idx == n - 1 and shape[last_idx] <= channel_max:
            if pool:
                pool[0] = 'C'
            else:
                pool.append('C')

    # --- 4) Special handling for channels ('C') ---
    if 'C' in pool and remaining_indices:
        # Dim candidates that look like channels
        candidate_small = [i for i in remaining_indices if shape[i] <= channel_max]
        c_idx = None

        if candidate_small:
            # Prefer dims > 1 (1's are often S/T), among those pick the rightmost.
            non_one = [i for i in candidate_small if shape[i] > 1]
            if non_one:
                c_idx = max(non_one)  # rightmost >1 small dim
            else:
                # Only size-1 dims; if we must, use the rightmost of them.
                c_idx = max(candidate_small)
        else:
            # No "small" dims; try dims significantly smaller than spatial dims
            min_spatial = min(shape[y_idx], shape[x_idx])
            candidate_thin = [i for i in remaining_indices if shape[i] <= min_spatial / 4]
            if candidate_thin:
                c_idx = max(candidate_thin)  # rightmost thin dim

        if c_idx is not None:
            axis_for_index[c_idx] = 'C'
            remaining_indices.remove(c_idx)
            pool.remove('C')
        else:
            # No reasonable channel candidate; drop 'C' entirely
            pool.remove('C')

    # --- 5) Assign remaining axes from pool to remaining dims ---
    # We keep:
    #   - pool order = order in default_format tail (e.g. S,T,Z,...),
    #   - index order = ascending, so leading dims tend to get S/T.
    for idx, ax in zip(sorted(remaining_indices), pool):
        axis_for_index[idx] = ax

    fmt = ''.join(axis_for_index)
    if '' in axis_for_index:
        raise RuntimeError(f"Incomplete format assignment for shape {shape}: {axis_for_index}")

    return fmt


def test_estimate_format():
    test_cases = [
        (3, 23, 1024, 1024),        # -> CZYX
        (23, 3, 1024, 1024),        # -> ZCYX
        (3, 1024, 1024),            # -> ZYX
        (1024, 1024, 3),            # -> YXC
        (1024, 1024, 23),           # -> YXZ
        (5, 5, 23, 3, 1024, 1024),  # -> STZCYX
        (1, 1, 23, 3, 1024, 1024),  # -> STZCYX
    ]

    for test in test_cases:
        res = estimate_format(test)
        print(f"{test} --> {res}")


# convenience f     unc tions
def unique_nonzero(array):
    """ returns unique values that are not 0 """
    return np.unique(array[array != 0]) # this is substantially faster than previous implementation when array is sparse

def nunique(array):
    """ returns number of unique values that are not 0 """
    return len(unique_nonzero(array))

def allzero(arr: np.ndarray):
    """ checks if all elements of an array are zero, this method is faster than 'not np.any(arr)'"""
    return np.all(arr == 0)


def propagate_labels(volume, overlap_threshold=0.33):
    """
    Propagates labels in a 3D volume along the z-axis, i.e. align object ids across slices
    
    Args:
        volume (np.ndarray): 3D volume with shape (Z, Y, X) containing labeled objects slice by slice.
        overlap_threshold (float): if object overlap with parent > thresh then it is considered part of same object, otherwise it is considered unique
        
    Returns:
        np.ndarray: 3D volume with propagated labels for consistent object tracking along the z-axis.
    """
    
    # Get the dimensions of the volume
    Z, Y, X = volume.shape
    
    # Output array for propagated labels
    propagated_volume = np.zeros_like(volume)
    next_label = np.unique(volume[0]).max()  # To keep track of the next available label ID
    propagated_volume[0] = volume[0] # Initialize with the first slice
                
    # Iterate through slices
    for z in list(range(1, Z))[:]:
    
        current_slice = volume[z]
        previous_slice = propagated_volume[z - 1]
        
        current_labels, unique_labels = current_slice, unique_nonzero(current_slice)
        current_slices = scipy.ndimage.find_objects(current_labels)

        # if bool(0):
        #     up.show(current_slice)
        #     up.show(previous_slice)
        #     print(unique_labels)

        # Iterate through the current labels
        for li, label_id in list(enumerate(unique_labels)):
            label_region = current_slices[li]
            label_area = current_labels[label_region] == label_id
            
            # get non-zero coords in this region
            origin = np.array([current_slices[li][0].start, current_slices[li][1].start])
            non_zero_coords = np.argwhere(label_area != 0) + origin
            
            # Check for any overlap with the previous slice
            overlap_region = previous_slice[non_zero_coords[:,0], non_zero_coords[:,1]] # arr of prev px values in this region
            overlap_labels = unique_nonzero(overlap_region) # Exclude background
            
            if overlap_labels.size == 0: # no overlap, so assign new obj id
                propagated_label = next_label
                next_label += 1
            else:
                # get coordiates of this label, sort points by closest to this labels centroid                       
                sorted_coordinates, used_centroid = uc.sort_coordinates_by_distance(non_zero_coords, None, 'euclidean')
                non_zero_coords = np.array(sorted_coordinates)
                
                # iterate over coordiates of this label, starting with point closest to this labels centroid
                overlap_region = previous_slice[non_zero_coords[:,0], non_zero_coords[:,1]]
                
                # get label of point in prev layer closest to the centroid
                non_zero_indices = np.flatnonzero(overlap_region)
                propagated_label = overlap_region[non_zero_indices[0]]
                
                # check minimum overlap requirement met, otherwise assume it is a new object
                overlap_p = np.sum(overlap_region==propagated_label)/overlap_region.size
                
                if not (overlap_p > overlap_threshold): 
                    propagated_label = next_label
                    next_label += 1
                                            
                
                # if bool(0):
                #     sarr = np.zeros_like(current_slice)
                #     sarr[non_zero_coords[:,0], non_zero_coords[:,1]] = to_binary(previous_slice[non_zero_coords[:,0], non_zero_coords[:,1]])
                #     sarr[non_zero_coords[:,0], non_zero_coords[:,1]] += 2
                #     up.show(sarr, def_title=f"{np.unique(sarr)}")
            
            
            # Assign the propagated label to the output volume
            propagated_volume[z][non_zero_coords[:,0], non_zero_coords[:,1]] = propagated_label
    return propagated_volume



def slice_by_multiple(array, axis, multiple):
    """
    Slice an array along a specified axis by a multiple of a given factor.
    
    Parameters:
        array (np.ndarray): The input array to be sliced.
        axis (int): The axis along which to slice.
        multiple (int): The factor by which to slice along the axis.
        
    Returns:
        np.ndarray: The sliced array.
    
    Example:
        img = np.random.rand(8, 11, 8, 8)
        sliced_img = slice_by_multiple(img, axis=1, multiple=4)
        print(img.shape)       # Original shape: (8, 11, 8, 8)
        print(sliced_img.shape)  # Sliced shape: (8, 8, 8, 8)
    """
    axis_length = array.shape[axis]
    max_length = axis_length // multiple * multiple
    slices = tuple(slice(None) if i != axis else slice(0, max_length) for i in range(array.ndim))
    return array[slices]



def pad_image_to_multiple(image, multiple=256):
    """Calculate and apply necessary padding to reach multiple of patch_size in YX dims (assuming last 2 dims)"""
    assert image.ndim in [2, 3], f"image must be 2D or 3D, got {image.ndim}"

    pad_height = (multiple - image.shape[-2] % multiple) % multiple
    pad_width = (multiple - image.shape[-1] % multiple) % multiple
    # Apply symmetric padding to both dimensions
    if image.ndim==2:
        padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    elif image.ndim==3:
        padded_image = np.pad(image, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    else: 
        raise ValueError(image.shape, image.ndim)
    return padded_image

def flatten_patches(patches):
    """flatens all but last 2 dims e.g. converts (N x M x X x Y) array --> (n x X x Y )
    example: patches = (5, 5, 256, 256), returns (25, 256, 256)"""
    return patches.reshape(-1, *patches.shape[-2:])


def sample_along_dimension(array, sample_percentage, axis=0):
    """
    Samples a percentage of elements along a specified dimension in an array.

    Parameters:
        array (np.ndarray): The input array of any shape.
        sample_percentage (float): The percentage of elements to sample (0-1).
        dimension (int): The dimension along which to sample.

    Returns:
        np.ndarray: An array containing the sampled elements along the specified dimension.
    """
    if not (0 <= sample_percentage <= 1):
        raise ValueError("sample_percentage must be between 0 and 100.")
    if axis < 0 or axis >= array.ndim:
        raise ValueError(f"dimension must be between 0 and {array.ndim - 1} (inclusive).")

    # Calculate the number of samples
    num_elements = array.shape[axis]
    num_samples = int(sample_percentage * num_elements)

    # Sample indices without replacement
    sample_indices = np.random.choice(num_elements, num_samples, replace=False)

    # Take samples along the specified dimension
    sampled_array = np.take(array, sample_indices, axis=axis)

    return sampled_array
       

def resize_by_format(img: np.ndarray, fmt: str, target_shape: tuple, order=None) -> np.ndarray:
    """
    Resize an image based on its format string by resizing YX dims to target_shape.
        e.g. img.shape = (3,512,512), fmt='CYX', target_shape=(128,128) returns img of shape (3, 128, 128)

    Parameters:
    - img: np.ndarray, input image
    - fmt: str, format string like 'CYX', 'YX', etc.
    - target_shape: tuple, new shape for Y and X (height, width) e.g. (512, 512)
    - order: int, passed to skimage.transfrom.resize 

    Returns:
    - resized_img: np.ndarray, image resized along YX dimensions
    """
    from skimage.transform import resize

    fmt = fmt.upper()
    assert 'Y' in fmt and 'X' in fmt, f"Format must contain both Y and X, but got fmt: {fmt}"
    

    # Get the current shape and axis indices
    shape = list(img.shape)
    y_idx, x_idx = fmt.index('Y'), fmt.index('X')
    assert len(shape) == len(fmt), f"lengths mismatch: {shape} != {fmt}"

    # Move Y and X to the last two dims
    axes_order = [i for i in range(len(shape)) if i not in (y_idx, x_idx)] + [y_idx, x_idx]
    img_transposed = np.transpose(img, axes_order)

    # Flatten any leading dimensions (if present)
    leading_shape = img_transposed.shape[:-2]
    img_reshaped = img_transposed.reshape(-1, img_transposed.shape[-2], img_transposed.shape[-1])

    # Resize each slice individually
    order = order if order else 0 if img.dtype == np.int32 else 1
    antialiasing = False if order == 0 else True
    resized_slices = [
        resize(slice_, target_shape, order=order, preserve_range=True, anti_aliasing=antialiasing)
        for slice_ in img_reshaped
    ]
    resized_stack = np.stack(resized_slices)

    # Restore leading dimensions
    resized_stack = resized_stack.reshape(*leading_shape, *target_shape)

    # Invert the axes transformation
    inverse_order = np.argsort(axes_order)
    resized_final = np.transpose(resized_stack, inverse_order)

    return resized_final.astype(img.dtype)


def get_highest_density_bbox(
    label_img: np.ndarray,
    crop_shape: Tuple[int, int] = (256, 256)
) -> Tuple[slice, slice]:
    """
    Find the bounding box (as slices) of the region with the highest object
    density in a 2D label image.

    Parameters
    ----------
    label_img : np.ndarray
        2D array with integer labels; 0 is background, unique object labels > 0.
    crop_shape : tuple[int, int]
        Shape (height, width) of the desired crop region.

    Returns
    -------
    Tuple[slice, slice]
        A tuple of (row_slice, col_slice) specifying the crop region.

    Raises
    ------
    ValueError
        If crop_shape is larger than the image in any dimension.
    """
    H, W = label_img.shape
    ch, cw = crop_shape

    if ch > H or cw > W:
        raise ValueError("Crop shape is larger than the input image.")

    # Binary mask of object pixels
    occ = (label_img != 0).astype(np.int64)

    # Compute integral image
    ii = occ.cumsum(axis=0).cumsum(axis=1)
    

    def window_sum(r0: int, c0: int) -> int:
        r1, c1 = r0 + ch - 1, c0 + cw - 1
        s = ii[r1, c1]
        if r0 > 0: s -= ii[r0 - 1, c1]
        if c0 > 0: s -= ii[r1, c0 - 1]
        if r0 > 0 and c0 > 0: s += ii[r0 - 1, c0 - 1]
        return s

    best_sum, best_r0, best_c0 = -1, 0, 0
    for r0 in range(H - ch + 1):
        for c0 in range(W - cw + 1):
            s = window_sum(r0, c0)
            if s > best_sum:
                best_sum = s
                best_r0, best_c0 = r0, c0

    # Recenter the crop on the densest region's center
    center_r = best_r0 + ch // 2
    center_c = best_c0 + cw // 2

    r0 = center_r - ch // 2
    c0 = center_c - cw // 2
    r1 = r0 + ch
    c1 = c0 + cw

    # Shift to stay within bounds
    if r0 < 0: r1, r0 = r1 - r0, 0
    if c0 < 0: c1, c0 = c1 - c0, 0
    if r1 > H: r0, r1 = r0 - (r1 - H), H
    if c1 > W: c0, c1 = c0 - (c1 - W), W

    return slice(r0, r1), slice(c0, c1)




def create_synthetic_test_objects_data(
    nObjs = 100,
    size = 256
    ):
    """ create intensity and label image with circular objects """
    intensity_image = np.zeros((size, size))
    label_image = np.zeros((size, size), dtype=int)
    GT_label_image = np.zeros((size, size), dtype=int)

    for o in range(nObjs):
        lbl_val = o+1
        center = np.random.randint(10, 246, 2)
        radius_true = np.random.randint(4, 8)
        radius_annotated = radius_true + np.random.randint(1, 2)

        # Create intensity image with bright circle
        y, x = np.ogrid[:size, :size]
        dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)


        intensity_image[dist_from_center <= radius_true] = 128
        intensity_image[dist_from_center <= radius_true/2] = 235  # Brighter center

        label_image[dist_from_center <= radius_annotated] = lbl_val # non gt
        GT_label_image[dist_from_center <= radius_true] = lbl_val

    # Add some noise
    intensity_image += np.random.randint(0, 20, intensity_image.shape)

    return intensity_image, label_image





def nine_odd(n):
    """Ensure PSF size is an odd integer (RL deconvolution works best that way)."""
    return int(n) if int(n) % 2 else int(n) + 1

def gaussian_psf(size= nine_odd(15), sigma=2.0):
    """
    Create a normalized 2D Gaussian point-spread function.
    """
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return psf / psf.sum()


def enhance_features(
    img,
    denoise_sigma=1.0,
    psf=None,
    psf_sigma=2.0,
    psf_size=15,
    rl_iterations=20,
    peak_size=3,
    valley_size=15,
    clip_output=True,
    return_intermediates=False,
):
    """
    Process a 2D image with:
      1) Denoising (Gaussian blur)
      2) 2D deconvolution (Richardson–Lucy)
      3) Subtract valleys from peaks to enhance features

    Parameters
    ----------
    img : ndarray (H, W)
        Input 2D image.
    denoise_sigma : float
        Std dev for Gaussian denoising. Set to 0/None to skip.
    psf : ndarray or None
        Point spread function for deconvolution. If None, a Gaussian PSF is generated.
    psf_sigma : float
        Sigma for generated Gaussian PSF (ignored if `psf` is provided).
    psf_size : int
        Size of the PSF kernel (odd integer, ignored if `psf` is provided).
    rl_iterations : int
        Number of Richardson–Lucy iterations.
    peak_size : int
        Neighborhood size for maximum filter (peaks).
    valley_size : int
        Neighborhood size for minimum filter (valleys/background).
    clip_output : bool
        If True, negative values after subtraction are clipped to 0.
    return_intermediates : bool
        If True, return dict with intermediate results.

    Returns
    -------
    out : ndarray or dict
        Enhanced image, or dict with intermediates if `return_intermediates=True`.
    """
    from scipy import ndimage as ndi
    from skimage.restoration import richardson_lucy
    
    img = np.asarray(img, dtype=np.float32)

    # 1. Denoise
    denoised = ndi.gaussian_filter(img, sigma=denoise_sigma) if denoise_sigma else img

    # 2. Deconvolution
    if psf is None:
        psf = gaussian_psf(size=nine_odd(psf_size), sigma=psf_sigma)
    deconv = richardson_lucy(denoised, psf, num_iter=rl_iterations, clip=False)

    # 3. Peaks - Valleys subtraction
    peaks = ndi.maximum_filter(deconv, size=peak_size)
    valleys = ndi.minimum_filter(deconv, size=valley_size)
    enhanced = peaks - valleys
    if clip_output:
        enhanced = np.clip(enhanced, 0, None)

    if return_intermediates:
        return {
            "denoised": denoised,
            "deconvolved": deconv,
            "peaks": peaks,
            "valleys": valleys,
            "enhanced": enhanced,
        }
    return enhanced



def dilate_instance_labels_plane(label_vols:list[np.ndarray], dilation_radius):
    """
    Dilate 3D instance labels in XY only (Z unaffected).
    Each instance is dilated independently to avoid label collisions.

    Parameters
    ----------
    label_vols : list[np.ndarray]
        List of 3D label volumes, each with shape (Z, Y, X).
        Each element must be integer labels where 0=background and
        positive ints correspond to unique objects.
    dilation_radius : int
        Number of pixels to dilate in XY. (0, 1, 2, ...)

    Returns
    -------
    list[np.ndarray]
        New list with XY-dilated instance labels.
    """

    if dilation_radius <= 0:
        return [arr.copy() for arr in label_vols]

    # Structuring element for XY-only dilation:
    # shape = (1, (2r+1), (2r+1))
    struct = np.zeros((1, 2 * dilation_radius + 1, 2 * dilation_radius + 1), dtype=bool)
    struct[0] = True  # only dilate within XY plane

    out_vols = []

    for vol in label_vols:
        vol = vol.copy()
        Z, Y, X = vol.shape

        # unique instance ids, ignore background 0
        instance_ids = np.unique(vol)
        instance_ids = instance_ids[instance_ids != 0]

        new_vol = np.zeros_like(vol)

        for inst_id in instance_ids:

            # Get bounding box (fast way)
            zyx = np.where(vol == inst_id)
            zmin, zmax = zyx[0].min(), zyx[0].max()
            ymin, ymax = zyx[1].min(), zyx[1].max()
            xmin, xmax = zyx[2].min(), zyx[2].max()

            # Expand bounding box by dilation radius (XY only)
            ymin_e = max(0, ymin - dilation_radius)
            ymax_e = min(Y - 1, ymax + dilation_radius)
            xmin_e = max(0, xmin - dilation_radius)
            xmax_e = min(X - 1, xmax + dilation_radius)

            # Extract chunk
            chunk = vol[zmin : zmax + 1, ymin_e : ymax_e + 1, xmin_e : xmax_e + 1]
            mask = chunk == inst_id

            # Dilate in XY for each Z slice independently (struct has depth=1)
            dilated_mask = binary_dilation(mask, structure=struct)

            # Insert back — but avoid overwriting already-filled voxels
            # (Ensures touching objects keep distinct IDs)
            target = new_vol[zmin : zmax + 1, ymin_e : ymax_e + 1, xmin_e : xmax_e + 1]
            target[dilated_mask & (target == 0)] = inst_id

        out_vols.append(new_vol)

    return out_vols


def dilate_thin_instances_xy(
    volume: np.ndarray,
    radius: int,
    thin_radius_thresh: float = 1.5,
) -> np.ndarray:
    """
    Dilate thin instance labels in a 3D ZYX label volume, only in XY.

    Parameters
    ----------
    volume : np.ndarray
        3D array with shape (Z, Y, X). 0 = background, >0 = instance IDs.
    radius : int
        Dilation radius (in pixels) in X and Y. If 0, returns a copy of volume.
    thin_radius_thresh : float, optional
        Maximum distance-transform radius (in pixels) for an instance to be
        considered "thin". Only instances whose max XY distance <= this
        threshold will be dilated.

    Returns
    -------
    np.ndarray
        New label volume with thin instances slightly dilated in XY.
    """
    if radius <= 0:
        return volume.copy()

    IS_2D = False 
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
        IS_2D = True # flag to return as 2d 

    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D or 2D array. ndim={volume.ndim}")

    base = volume  # do not modify
    new_vol = base.copy()

    z_max, y_max, x_max = base.shape
    labels = np.unique(base)
    labels = labels[labels != 0]  # exclude background

    # 3D structuring element: 1 x (2r+1) x (2r+1), active only in the central Z plane
    structure = np.zeros((1, 2 * radius + 1, 2 * radius + 1), dtype=bool)
    structure[0, :, :] = True

    for lab in labels:
        mask = base == lab
        if not mask.any():
            continue

        # ---- 1) Check if this object is "thin" in XY via distance transform ----
        # Compute max XY distance across slices
        max_dt = 0.0
        for z in range(mask.shape[0]):
            slice_mask = mask[z]
            if slice_mask.any():
                dist = ndimage.distance_transform_edt(slice_mask)
                max_dt = max(max_dt, dist.max())
        if max_dt > thin_radius_thresh:
            # Skip "thick" instances
            continue

        # ---- 2) Get tight ZYX bounding box for this label ----
        zs, ys, xs = np.where(mask)
        z0, z1 = zs.min(), zs.max()
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()

        # Pad XY by radius (Z unchanged; no Z dilation)
        y0p = max(0, y0 - radius)
        y1p = min(y_max - 1, y1 + radius)
        x0p = max(0, x0 - radius)
        x1p = min(x_max - 1, x1 + radius)

        # Extract ROI from base (for mask and collision checks) and new_vol (for writing)
        roi_mask = mask[z0 : z1 + 1, y0p : y1p + 1, x0p : x1p + 1]
        roi_base = base[z0 : z1 + 1, y0p : y1p + 1, x0p : x1p + 1]
        roi_new = new_vol[z0 : z1 + 1, y0p : y1p + 1, x0p : x1p + 1]

        # ---- 3) Dilate in XY only inside this ROI ----
        dilated = ndimage.binary_dilation(roi_mask, structure=structure)

        # Only grow into background (0) in the *original* base volume
        grow_region = dilated & (roi_base == 0)

        # Write back into new_vol
        roi_new[grow_region] = lab
    
    if IS_2D:
        new_vol = new_vol[0]

    return new_vol


def batch_dilate_thin_instances_xy(
    volumes: list[np.ndarray],
    radius: int,
    thin_radius_thresh: float = 1.5,
) -> list[np.ndarray]:
    """
    Apply dilate_thin_instances_xy to a list of ZYX label volumes.
    """
    return [dilate_thin_instances_xy(v, radius, thin_radius_thresh) for v in volumes]


def test_dilate_thin_instances_xy():
    a = np.array(
        (
            [
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0, 0.0, 0.0],
                    [0.0, 0, 2.0, 0, 0.0],
                    [0.0, 0.0, 0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 2.0, 2.0, 0.0],
                    [0.0, 2.0, 2.0, 2.0, 0.0],
                    [0.0, 2.0, 2.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 2.0, 2.0, 2.0, 0.0],
                    [0.0, 0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
                [
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                ],
            ]
        )
    )

    batch_dilate_thin_instances_xy([a], 1)



# ---------------------------------------
# Host helpers
# ---------------------------------------
def _make_ball_heights_and_mask(radius: int, dtype=np.float32):
    """
    Build the non-flat spherical height map and a uint8 mask indicating
    which offsets are inside the ball footprint.
    """
    R = float(radius)
    d = 2 * radius + 1
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    r2 = xx*xx + yy*yy
    inside = r2 <= (radius * radius)

    h = np.zeros((d, d), dtype=dtype)
    # h >= 0; h(0,0)=0; increases toward the edge
    h[inside] = R - np.sqrt(R * R - r2[inside].astype(np.float64))
    mask = inside.astype(np.uint8)  # 1=inside, 0=outside
    return h.astype(dtype, copy=False), mask

# ---------------------------------------
# CUDA kernels (non-flat morphology)
# ---------------------------------------
@cuda.jit
def _erode_nonflat_ball(img, hmap, mask, out, radius):
    """
    Non-flat grayscale erosion:
    (f ⊖ h)(x) = min_s [ f(x+s) - h(s) ], only for s inside the ball mask.
    """
    y, x = cuda.grid(2)
    H, W = img.shape
    if y >= H or x >= W:
        return

    best = np.float32(1e20)  # large +ve sentinel that fits in float32

    for oy in range(-radius, radius + 1):
        yy = y + oy
        if yy < 0 or yy >= H:
            continue
        hy = oy + radius
        for ox in range(-radius, radius + 1):
            hx = ox + radius
            # Skip offsets outside the spherical footprint
            if mask[hy, hx] == 0:
                continue
            xx = x + ox
            if xx < 0 or xx >= W:
                continue
            hh = hmap[hy, hx]
            val = img[yy, xx] - hh
            if val < best:
                best = val

    out[y, x] = best


@cuda.jit
def _dilate_nonflat_ball(img, hmap, mask, out, radius):
    """
    Non-flat grayscale dilation:
    (g ⊕ h)(x) = max_s [ g(x+s) + h(s) ], only for s inside the ball mask.
    """
    y, x = cuda.grid(2)
    H, W = img.shape
    if y >= H or x >= W:
        return

    best = np.float32(-1e20)  # large -ve sentinel

    for oy in range(-radius, radius + 1):
        yy = y + oy
        if yy < 0 or yy >= H:
            continue
        hy = oy + radius
        for ox in range(-radius, radius + 1):
            hx = ox + radius
            if mask[hy, hx] == 0:
                continue
            xx = x + ox
            if xx < 0 or xx >= W:
                continue
            hh = hmap[hy, hx]
            val = img[yy, xx] + hh
            if val > best:
                best = val

    out[y, x] = best


def subtract_rolling_ball_gpu(img: np.ndarray, radius: int = 10, block=(16, 16)):
    """
    GPU-accelerated rolling-ball background subtraction using Numba CUDA.
    """
    if img.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported.")

    if not cuda.is_available():
        raise RuntimeError("No CUDA-capable GPU detected (cuda.is_available() is False).")

    H, W = img.shape
    orig_dtype = img.dtype
    img32 = img.astype(np.float32, copy=False)

    # Build height map + mask and move to device
    hmap, mask = _make_ball_heights_and_mask(radius, dtype=np.float32)

    d_img  = cuda.to_device(img32)
    d_hmap = cuda.to_device(hmap)
    d_mask = cuda.to_device(mask)
    d_tmp  = cuda.device_array((H, W), dtype=np.float32)
    d_bg   = cuda.device_array((H, W), dtype=np.float32)

    grid = ((H + block[0] - 1) // block[0],
            (W + block[1] - 1) // block[1])

    _erode_nonflat_ball[grid, block](d_img, d_hmap, d_mask, d_tmp, radius)
    _dilate_nonflat_ball[grid, block](d_tmp, d_hmap, d_mask, d_bg, radius)

    bg = d_bg.copy_to_host()
    out = img32 - bg
    np.maximum(out, 0.0, out)  # clip at 0 in-place

    if np.issubdtype(orig_dtype, np.integer):
        info = np.iinfo(orig_dtype)
        np.minimum(out, float(info.max), out)
        out = out.astype(orig_dtype, copy=False)
    else:
        out = out.astype(orig_dtype, copy=False)

    return out

def subtract_rolling_ball_cpu(img, radius=10, num_threads=None):
    """ cpu impl. rolling ball background subtraction"""
    from skimage.restoration import rolling_ball
    bg  = rolling_ball(img, radius=radius, num_threads=num_threads)
    out = img - bg
    out[out < 0] = 0
    return out.astype(img.dtype, copy=False)

def subtract_rolling_ball(img:np.ndarray, radius=10, block=(16, 16), num_threads=None):
    """ 
    rolling-ball background subtraction 
        uses gpu version if available, else cpu version 
    
    """
    if img.ndim != 2:
        raise ValueError("Only 2D grayscale images are supported.")

    if not cuda.is_available():
        return subtract_rolling_ball_cpu(img, radius=radius, num_threads=num_threads)
    return subtract_rolling_ball_gpu(img, radius=radius, block=block)



def bounds2slice(bounds, fmt='minmax'):
    """
    Convert a list of ints representing axis min/max bounds to a tuple of slices for easy indexing into arrays.
        supports N-dimensional data

    Args:
        bounds : sequence of int
            bounding coordinates.
            For fmt='minmax': [min0, max0, min1, max1, ...]
            For fmt='minmin': [min0, min1, ..., max0, max1, ...]
        fmt : str
            Format of the bounding box: 'minmax' or 'minmin'

    Returns:
        tuple of slice
            Tuple of slices that can be used for array indexing.
    """
    bounds = np.array(bounds)
    assert len(bounds) % 2 == 0, "bounds must have even number of elements"
    n_dims = len(bounds) // 2

    if fmt == 'minmax':
        return tuple(slice(bounds[2 * i], bounds[2 * i + 1]) for i in range(n_dims))
    elif fmt == 'minmin':
        return tuple(slice(bounds[i], bounds[n_dims + i]) for i in range(n_dims))
    elif fmt == 'bbox':
        assert bounds.ndim==2
        n_dims = bounds.shape[1]
        _bbox = []
        for i in range(n_dims):
            _bbox.extend([bounds[:, i].min(), bounds[:, i].max()])
        return bounds2slice(_bbox, fmt='minmax')

    else:
        raise ValueError(f"Unsupported format: {fmt}")




def fill_outline_labels(labels: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """
    Fill outlined labeled regions so that each object's interior is assigned the
    same label value as its outline.

    Parameters
    ----------
    labels : np.ndarray
        Labeled image with integer labels (0 = background). Each object is currently
        just an outline (one-pixel-ish thick). Works for 2D, 3D, ... ND.
    connectivity : int, optional
        Neighborhood connectivity used for hole filling:
        - 1 = face-connected (4-connectivity in 2D, 6 in 3D)
        - 2 = include edges/corners (8 in 2D, 26 in 3D), etc.

    Returns
    -------
    np.ndarray
        A copy of `labels` where each outlined object is filled solid with its label.

    Notes
    -----
    - If two outlines enclose overlapping interiors, the last one processed will win.
      Typically outlines don’t overlap; if yours can, resolve that upstream.
    - For very large images with many labels, see the bbox-optimized version below.
    """
    from scipy.ndimage import binary_fill_holes, generate_binary_structure, find_objects

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("`labels` must be an integer array (0 = background).")

    out = np.zeros_like(labels)
    lbl_ids = unique_nonzero(labels)

    # Connectivity struct for fill_holes
    struct = generate_binary_structure(out.ndim, connectivity)

    # find obj bboxes to operate on small crops, then slice back into img
    bboxes = find_objects(labels)

    for lbl in lbl_ids:
        bbox = bboxes[lbl-1]
        crop = labels[bbox]
        mask = (crop == lbl)
        if not mask.any():
            continue
        # Fill the interior (holes) of this outline
        filled = binary_fill_holes(mask, structure=struct)
        # Assign label to entire filled region (outline + interior)
        crop[filled] = lbl
        out[bbox] = crop

    return out


import numpy as np
from scipy import ndimage

def check_unique_object_labels(labeled_image, connectivity=1):
    """
    Check if all objects in a labeled image have unique labels.
    
    This function uses vectorized operations on the entire image at once,
    avoiding expensive per-label loops and mask creation.
    
    The approach:
    1. Relabel the entire image to get guaranteed unique connected components
    2. For each original label, check if it maps to multiple new labels
    3. If any original label maps to >1 connected component, it's shared
    
    Parameters:
    -----------
    labeled_image : numpy.ndarray
        A 2D or 3D array where each connected component should have a unique
        integer label. Background should be labeled as 0.
    connectivity : int, optional
        Connectivity for determining connected components:
        - For 2D: 1 (4-connectivity) or 2 (8-connectivity, default)
        - For 3D: 1 (6-connectivity), 2 (18-connectivity), or 3 (26-connectivity)
    
    Returns:
    --------
    dict : A dictionary containing:
        - 'is_unique': bool - True if all objects have unique labels
        - 'num_original_labels': int - Number of unique non-zero labels in input
        - 'num_actual_objects': int - Actual number of connected components
        - 'shared_labels': list - Labels that are shared by multiple objects
    
    Examples:
    ---------
    >>> # Good labeling - each object has unique label
    >>> img = np.array([[1, 1, 0, 2, 2],
    ...                 [1, 0, 0, 0, 2],
    ...                 [0, 0, 3, 3, 3]])
    >>> result = check_unique_object_labels(img)
    >>> result['is_unique']
    True
    
    >>> # Bad labeling - label 1 used for two separate objects
    >>> img = np.array([[1, 1, 0, 2, 2],
    ...                 [0, 0, 0, 0, 2],
    ...                 [1, 1, 0, 0, 0]])
    >>> result = check_unique_object_labels(img)
    >>> result['is_unique']
    False
    >>> result['shared_labels']
    [1]
    """
    labeled_image = np.asarray(labeled_image)
        
    # Get the number of original labels
    original_labels = np.unique(labeled_image)
    original_labels = original_labels[original_labels != 0]
    num_original_labels = len(original_labels)
    
    # Relabel the entire image based on connectivity (one operation!)
    binary_mask = labeled_image > 0
    relabeled = relabel(binary_mask, connectivity=connectivity)
    num_actual_objects = len(unique_nonzero(relabeled))
    
    # Flatten both arrays for easier processing
    original_flat = labeled_image.ravel()
    relabeled_flat = relabeled.ravel()
    
    # Only consider non-background pixels
    non_zero_mask = original_flat != 0
    original_flat = original_flat[non_zero_mask]
    relabeled_flat = relabeled_flat[non_zero_mask]
    
    # Create a mapping: for each original label, collect all relabeled values
    # Using a vectorized approach with unique and sorting
    
    # Stack the arrays and get unique pairs
    pairs = np.column_stack([original_flat, relabeled_flat])
    unique_pairs = np.unique(pairs, axis=0)
    
    # Count how many unique relabeled values each original label has
    original_in_pairs = unique_pairs[:, 0]
    unique_originals, counts = np.unique(original_in_pairs, return_counts=True)
    
    # Labels with counts > 1 are shared across multiple objects
    shared_labels = unique_originals[counts > 1].tolist()
    
    is_unique = len(shared_labels) == 0
    
    return {
        'is_unique': is_unique,
        'num_original_labels': int(num_original_labels),
        'num_actual_objects': int(num_actual_objects),
        'shared_labels': shared_labels
    }



def unpad(arr: np.ndarray, pad: int) -> np.ndarray:
    """Remove uniform padding from all axes of an N-dimensional array.

    This function inverts np.pad(x, pad) when the same integer pad width
    was applied to all axes on both sides.

    Args:
        arr (np.ndarray): Padded input array.
        pad (int): Number of pixels removed from each side of every axis.

    Returns:
        np.ndarray: Unpadded array.

    Raises:
        ValueError: If pad is too large for the array shape.
    """
    if any(s <= 2 * pad for s in arr.shape):
        raise ValueError("Pad size too large for one or more array dimensions.")

    slices = tuple(slice(pad, -pad) for _ in arr.shape)
    return arr[slices]


def zero_border_nd(
    arr: np.ndarray,
    pad: Union[int, Mapping[str, int]],
    *,
    fmt: str = "zyx",
    axes: Iterable[str] = None,
    sides: Union[str, Mapping[str, str]] = "both",
    copy: bool = True,
) -> np.ndarray:
    """
    Zero-out border regions of an n-D array along selected axes.

    Parameters:
    ----------
    arr : np.ndarray
        Input array of any dtype and dimensionality. The array's dimension
        order is described by `fmt` (e.g., "tczyx").
    pad : int or dict[str, int]
        Padding size(s) to zero from each selected axis edge.
        - If int: the same pad is used for all selected axes.
        - If dict: per-axis padding using axis letters from `fmt`, e.g. {'x': 5, 'y': 10}.
    fmt : str, default "zyx"
        Format string describing axis order of `arr`. Must have length == arr.ndim,
        with unique letters (e.g., "tczyx", "bzyx", "zyx").
    axes : iterable[str], optional
        Which axes (by letter from `fmt`) to apply zeroing on.
        If None, defaults to all axes in `fmt`.
        Examples: {'x','y'} to only affect the XY edges.
    sides : {"both","min","max"} or dict[str, {"both","min","max"}], default "both"
        Which edge(s) to zero per axis. If a single string, it's applied to all
        selected axes. If a dict, you can specify per-axis sides, e.g. {'y':'min','x':'both'}.
    copy : bool, default True
        If True, operate on a copy and return it; if False, modify `arr` in place.

    Returns:
    -------
    np.ndarray
        Array with specified edge regions set to zero.

    Notes:
    -----
    - If pad <= 0 for an axis, that axis is ignored.
    - If pad >= axis length and sides == "both", that entire axis is zeroed.
    - Unspecified axes (e.g., 't','c') remain unchanged.

    Examples:
        1) tczyx array: zero a 5-pixel border only on X and Y (both edges)
        clean = zero_border_nd(img, pad=5, fmt="tczyx", axes=("y","x"))

        2) Only remove at the "min" edge of Y (top), but both edges of X
        clean = zero_border_nd(
            img, pad={'y': 8, 'x': 3}, fmt="tczyx", axes=("y","x"),
            sides={'y': 'min', 'x': 'both'}
        )

        3) 3D ZYX: remove a 4-voxel border on all three axes, but only the far side ("max")
        clean = zero_border_nd(vol, pad=4, fmt="zyx", sides="max")

        4) Leave batch/time/channel dims untouched; trim only spatial dims in "bctzyx"
        clean = zero_border_nd(arr, pad={'y': 6, 'x': 6, 'z': 2}, fmt="bctzyx", axes=("z","y","x"))
    """
    if arr.ndim != len(fmt):
        raise ValueError(f"`fmt` length ({len(fmt)}) must match arr.ndim ({arr.ndim}).")
    if len(set(fmt)) != len(fmt):
        raise ValueError("`fmt` must contain unique axis letters.")
    axis_to_idx: Dict[str, int] = {ax: i for i, ax in enumerate(fmt)}

    # Normalize `axes`
    if axes is None:
        axes = tuple(fmt)
    else:
        axes = tuple(axes)
        unknown = set(axes) - set(fmt)
        if unknown:
            raise ValueError(f"Unknown axes in `axes`: {unknown}. Valid axes: {set(fmt)}")

    # Normalize `pad`
    if isinstance(pad, int):
        pad_map: Dict[str, int] = {ax: pad for ax in axes}
    else:
        pad_map = dict(pad)
        # Ensure pads exist for all selected axes; default to 0 if missing
        for ax in axes:
            pad_map.setdefault(ax, 0)

    # Normalize `sides`
    valid_sides = {"both", "min", "max"}
    if isinstance(sides, str):
        if sides not in valid_sides:
            raise ValueError(f"`sides` must be one of {valid_sides} or a dict.")
        side_map: Dict[str, str] = {ax: sides for ax in axes}
    else:
        side_map = dict(sides)
        for ax in axes:
            side_map.setdefault(ax, "both")
        bad = {ax: s for ax, s in side_map.items() if s not in valid_sides}
        if bad:
            raise ValueError(f"Invalid side specifiers: {bad}. Allowed: {valid_sides}")

    out = arr.copy() if copy else arr
    shape = out.shape

    for ax in axes:
        p = int(pad_map.get(ax, 0))
        if p <= 0:
            continue
        aidx = axis_to_idx[ax]
        n = shape[aidx]
        if n == 0:
            continue
        p = min(p, n)  # guard if pad exceeds axis length
        which = side_map[ax]

        # Zero the "min" edge
        if which in ("min", "both"):
            slc = [slice(None)] * out.ndim
            slc[aidx] = slice(0, p)
            out[tuple(slc)] = 0

        # Zero the "max" edge
        if which in ("max", "both"):
            slc = [slice(None)] * out.ndim
            slc[aidx] = slice(n - p, n)
            out[tuple(slc)] = 0

    return out