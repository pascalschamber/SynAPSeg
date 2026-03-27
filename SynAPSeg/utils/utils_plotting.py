import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Tuple
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
import os
import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import imageio
from typing import Optional, List, Dict, Any
from PIL import Image
import napari
from scipy.ndimage import binary_dilation
import skimage
from scipy.stats import pearsonr
import io
from IPython.display import Image as IPyImage, display

from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.config.constants import STANDARD_FORMAT, DISPLAY_FORMAT

cv2 = ug.try_import('cv2')


def get_colors(cmap='hsv', n=None, l=None):
    """ 
    get n colors from a colormap, 
        or if l is a list return dict mapping the elemnents to colors 
    """
    if l is None:
        assert n is not None
    if n is None:
        assert l is not None
        n = len(l)
    
    cmap = plt.get_cmap(cmap)
    colors = [cmap(i) for i in np.linspace(0, 1, n)]
    if l is not None:
        return dict(zip(l, colors))
    return colors

def save_images_as_gif(X, X2=None, filename='output.gif', duration=0.1, auto_format_output=True):
    # Assuming X is your 4D numpy array of images with shape (num_images, height, width, channels)
    # Example: X.shape -> (100, 64, 64, 3)
    if X.ndim == 3: # e.g. time series with 1 channel
        X = np.repeat(X[:, :, :, np.newaxis], 3, axis=3)
    if X2 is not None:
        if X2.ndim == 3:
            X2 = np.repeat(X2[:, :, :, np.newaxis], 3, axis=3)
        X = np.concatenate((X, X2), axis=2)
    with imageio.get_writer(filename, mode='I', duration=duration) as writer:
        for i in range(X.shape[0]):
            writer.append_data(X if not auto_format_output else (X[i]/X[i].max()*255).astype('uint8'))

def save_fig(output_path, dpi=300):
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)

def create_composite_image_with_colormaps(image, colormaps):
    """
    Creates a composite image from a n-channel image using specified colormaps for each channel.
    
    Parameters:
    image (numpy.ndarray): Input image of shape (512, 512, C).
    colormaps (list): A list of colormaps or color lists for each channel.
    
    Returns:
    numpy.ndarray: Composite image of shape (512, 512, 3).
    """
    assert image.ndim == 3
    
    # Initialize the composite image
    composite_image = np.zeros((image.shape[0], image.shape[1], 3))
    
    for i in range(image.shape[-1]):
        # Generate the colormap from the provided colors
        color_map = colormaps[i]
        if isinstance(color_map, list):
            assert len(color_map) == 2
            cmap = LinearSegmentedColormap.from_list(f"custom_cmap_{i}", color_map)
        elif isinstance(color_map, str):
            cmap = LinearSegmentedColormap.from_list(f"custom_cmap_{i}", ['black', color_map])
        elif isinstance(color_map, LinearSegmentedColormap):
            cmap = color_map
        else: 
            raise ValueError(cmap)
        
        # Ensure pixel values are in range [0, 1]
        if image[:,:,i].max() > 1:
            image_normalized = uip.normalize_01(image[:,:,i])
        else:
            image_normalized = image[:,:,i]
        
        # Apply the colormap
        colored_channel = cmap(image_normalized)
        
        # Extract RGB components (ignore alpha channel if present)
        for j in range(3):  # RGB channels
            composite_image[:, :, j] += colored_channel[:, :, j]
    
    # Clip values to keep them within the [0, 1] range
    composite_image = np.clip(composite_image, 0, 1)
    
    return composite_image

def get_label_colormap(alpha=None):
    ''' colormap for a label image from pyclesperanto's implementation but with ability to get px labels'''
    from numpy.random import MT19937
    from numpy.random import RandomState, SeedSequence
    import matplotlib
    
    rs = RandomState(MT19937(SeedSequence(3)))
    lut = rs.rand(65537, 3)
    lut[0, :] = 0
    # these are the first four colours from matplotlib's default
    lut[1] = [0.12156862745098039, 0.4666666666666667, 0.7058823529411765]
    lut[2] = [1.0, 0.4980392156862745, 0.054901960784313725]
    lut[3] = [0.17254901960784313, 0.6274509803921569, 0.17254901960784313]
    lut[4] = [0.8392156862745098, 0.15294117647058825, 0.1568627450980392]
    
    # add alpha channel
    if alpha is not None:
        lut = np.hstack([lut, np.ones([len(lut),1])*alpha])
        lut[0] = [np.nan, np.nan, np.nan, 0]

    cmap = matplotlib.colors.ListedColormap(lut)
    return cmap


def add_weighted_np(src1, alpha, src2, beta, gamma):
    """
    Blend two images using a weighted sum.

    Parameters:
    - src1: First source image.
    - alpha: Weight for the first image.
    - src2: Second source image.
    - beta: Weight for the second image.
    - gamma: Scalar added to each sum.

    Returns:
    - Blended image as a NumPy array.
    """
    # Ensure the images are floats to prevent data type overflow/underflow
    src1 = src1.astype(np.float32)
    src2 = src2.astype(np.float32)
    
    # Perform the weighted sum
    blended = src1 * alpha + src2 * beta + gamma
    
    # Clip the values to the valid range [0, 255] and convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    return blended

def overlay(
    image: np.ndarray,
    mask: np.ndarray,
    image_cmap: str = 'red',
    mask_color: Tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.5, 
    resize: Tuple[int, int] = (1024, 1024)
    ) -> np.ndarray:
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay
    Params:
        image: Training image. should be normalized before hand
        mask: Segmentation mask.
        color: Color for segmentation mask rendering.
        alpha: Segmentation mask's transparency.
        resize: If provided, both image and its mask are resized before blending them together.
    
    Returns:
        image_combined: The combined image. (x,y,3)
        
    """
    if image.ndim != 3:
        image = np.stack([image]*3, 0)
    elif image.ndim == 3:# convert ch last to ch first
        if image.shape[-1] == 1: # e.g. a 3dim with only 1 ch
            image = np.stack([image[..., 0]]*3, -1)
        if image.shape[-1] == 3: 
            image = np.moveaxis(image, -1, 0)
        else: raise ValueError(image.shape)
    mask_color = np.asarray(mask_color).reshape(3, 1, 1)
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=mask_color)
    image_overlay = masked.filled()
    try:
        import cv2
        func = cv2.addWeighted
    except ImportError:
        func = add_weighted_np
    image_combined = func(image, 1 - alpha, image_overlay, alpha, 0)
    image_combined = np.moveaxis(image_combined, 0, -1)
    return image_combined

def convert_to_float_arr(image, norm=False):
    """
    Convert an image to floating-point representation in the range [0, 1]
    without altering the relative pixel values.

    Parameters:
        image (numpy.ndarray): Input image of any dtype.

    Returns:
        numpy.ndarray: Floating-point image in the range [0, 1].
    """
    # Get the dtype of the input image
    dtype = image.dtype
        
    # Define the maximum value based on the dtype
    if np.issubdtype(dtype, np.integer):
        if norm:
            max_value = image.max()
        else:
            max_value = np.iinfo(dtype).max  # Maximum value for integer types
    elif np.issubdtype(dtype, np.floating):
        if norm:
            max_value = image.max()
        else:
            max_value = 1.0  # Floating-point images are assumed to already be in range [0, 1]
    else:
        raise ValueError(f"Unsupported image dtype: {dtype}")
    
    # Convert to float and normalize by the maximum value
    image_float = image.astype(np.float32) / max_value
    
    return image_float


def overlay_colored_outlines(lbl_img, int_img, lbl_alpha=1.0, convert_to_float=False, norm=False, warn=True):
    """
    in most cases would actually want to use up.mask_to_outline_contours
    int_img should be in range(0,1)
    
    """
    
    bin_mask = uip.mask_to_outlines(lbl_img)
    colored_outlines = np.where(bin_mask>0, skimage.morphology.dilation(lbl_img), 0)
    ulabels = np.unique(colored_outlines)
    n_colors = len(ulabels)
    colors = [get_label_colormap()(i / (n_colors - 1)) for i in range(n_colors)] if n_colors>1 else [get_label_colormap()(0.5)]
    
    # image must be type floating point in range (0,1), if its 
    if convert_to_float:
        int_img = convert_to_float_arr(int_img, norm=norm)
    
    if warn and int_img.max() > 1.0:
        print('warning - in overlay_colored_outlines - int_img max > 1.0, should set convert_to_float=True to ensure proper image display')

    # convert colorspace to dtype of intensity image
    if np.issubdtype(int_img.dtype, np.integer):
        img_max_val = int_img.max()
        colors = [np.array(c) * img_max_val for c in colors]
    elif np.issubdtype(int_img.dtype, np.floating):
        img_max_val = 1.0
        
        
    # add alpha channel
    if int_img.ndim == 2:
        overlay = np.stack([int_img]*3 + [np.full_like(int_img, img_max_val)], -1)
    elif int_img.ndim == 3:
        overlay = np.concatenate([int_img, np.expand_dims(np.full_like(int_img[...,0], img_max_val), -1)], -1)
    else: 
        raise ValueError(int_img.shape)
    assert overlay.shape[-1] == 4

    for uli, ul in enumerate(ulabels):
        if ul==0: continue
        lbl_coords = np.where(colored_outlines==ul)
        lbl_color = colors[uli]
        overlay[lbl_coords] = lbl_color
    return overlay


def mask_to_outline_contours(
    intensity_img, label_image, 
    iterations=0, alpha=1, linewidth=1, 
    ax=None, label_cmap=False, c_func=None, image_cmap='gray', 
    mpl_poly_kwargs=None,
    imshow_kwargs=None
    ):
    """
    create a matplotlib axis with outlines of label_image objects overlayed on the intensity image

    args:
        intensity_img: intensity_img
        label_image: label_img or list of label_imgs
        iterations: number of iterations to perform on the label image
        alpha: alpha value for the outlines
        linewidth: linewidth for the outlines
        ax: matplotlib axis to plot on
        label_cmap: if False c_func is used, if True uses label_colormap with many colors 
        c_func: used if label_cmap False, if none defaults to lambda x: (1,0,0,alpha) i.e. red, else function that takes as input a float and returns a rgba tuple
        mpl_poly_kwargs: kwargs passed to matplotlib.patches.Polygon
        imshow_kwargs: kwargs passed to ax.imshow
    """
    ########################################################
    
    _ax_in = not (ax is None) # True if ax is provided
    if not _ax_in:
        fig,ax = plt.subplots(figsize=(10,10))
    
    ax.imshow(intensity_img, cmap=image_cmap, **imshow_kwargs or {})

    label_image = label_image if isinstance(label_image, list) else [label_image]
    flatten_list = lambda ell: [el for ell in ell for el in ell]
    contours_list = flatten_list([get_label_image_contours(li, iterations=iterations) for li in label_image])
    
    ax = plot_contours(contours_list, ax, alpha=alpha, linewidth=linewidth, label_cmap=label_cmap, c_func=c_func, mpl_poly_kwargs=mpl_poly_kwargs)
    
    if not _ax_in: 
        plt.show()

    return ax


def get_label_image_contours(label_image, iterations=0) -> list[list[np.ndarray]]:
    """ 
    helper function to convert a 2d label image to a contours list 
    Returns:
        list[list[np.ndarray]]: list of list of contours for each label in the label image
        if multiple objects (non-connected components) have the same label there will be multiple contours in the list
    """
    
    assert label_image.ndim == 2, f"only 2d label image input supported"

    uni = np.unique(label_image)
    lbls = uni[np.nonzero(uni)]

    contours_list = []
    for lbl in lbls:
        
        lbl_img =  label_image==lbl

        # Perform dilation to expand the object by i pixels
        if iterations > 0:
            dilated_image = binary_dilation(lbl_img, structure=np.ones((3,3)), iterations=iterations)
        else: 
            dilated_image = (lbl_img != 0)*1

        # Find contours using OpenCV
        contours, _ = cv2.findContours(dilated_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c[:, 0, :] for c in contours] # Remove redundant axis
        contours_list.append(contours)
    return contours_list

def plot_contours(contours_list, ax, alpha=1, linewidth=1, label_cmap=False, c_func=None, mpl_poly_kwargs=None):
    """ helper function to plot contours list as matplotlib patches adding them to ax """
    n_colors = len(contours_list)
    if label_cmap is True:
        colors = [get_label_colormap(alpha)(i+1) for i in range(n_colors)] if n_colors>1 else [get_label_colormap(alpha)(0.5)]
    else:
        c_func = (lambda x: (1,0,0,alpha)) if c_func is None else c_func # all red by default
        colors = [c_func(i / (n_colors - 1)) for i in range(n_colors)] if n_colors>1 else [c_func(1)]
        
    for ci, contours in enumerate(contours_list):
        kwargs = ug.merge_dicts(dict(edgecolor=colors[ci], fill=None, linewidth=linewidth), mpl_poly_kwargs or {})
        if kwargs.get('fill'):
            kwargs['facecolor'] = colors[ci]
        
        for contour in contours:
            polygon = plt.Polygon(contour, **kwargs)
            ax.add_patch(polygon)
    return ax


import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Iterator, Dict, Any 

def subplots(
    n_imgs: int, 
    n_cols: Optional[int] = None, 
    n_rows: Optional[int] = None, 
    size_per_dim: float = 4.0,
    **subplot_kwargs
) -> Tuple[plt.Figure, plt.Axes, Iterator[Tuple[int, plt.Axes]]]:
    """
    Creates a flexible Matplotlib subplot grid and returns an iterator 
    for easy plotting.

    Args
    ----
        n_imgs: int
            Number of images to plot
        n_cols: Optional[int]
            Number of columns in the grid
        n_rows: Optional[int]
            Number of rows in the grid
        size_per_dim: float
            Size of the figure in inches per dimension
        **subplot_kwargs: Optional[Dict[str, Any]]
            Additional keyword arguments to pass to plt.subplots
    """
    
    # Calculate layout logic
    if n_cols is None:
        if n_rows is None:
            n_cols = math.ceil(np.sqrt(n_imgs))
        else:
            n_cols = math.ceil(n_imgs / n_rows)
            
    if n_rows is None:
        n_rows = math.ceil(n_imgs / n_cols)

    assert size_per_dim is not None, "size_per_dim must be provided"
    if isinstance(size_per_dim, (int, float)):
        scale = np.array([size_per_dim, size_per_dim])
    else:
        scale = np.asarray(size_per_dim)

    # Create figure and axes
    fig, axs = plt.subplots(
        n_rows, 
        n_cols, 
        figsize= scale * np.array([n_cols, n_rows]),
        **subplot_kwargs
    )

    # Standardize axs into a flat numpy array for easier iteration
    if n_imgs == 1:
        axs_flat = np.array([axs])
    else:
        axs_flat = np.array(axs).flatten()

    # Generator function to yield index and axis
    def axis_iterator() -> Iterator[Tuple[int, plt.Axes]]:
        for i in range(n_imgs):
            yield i, axs_flat[i]
            
        # Optional: Hide unused axes if n_imgs < n_rows * n_cols
        for i in range(n_imgs, len(axs_flat)):
            axs_flat[i].axis('off')

    return fig, axs, axis_iterator()

def to_display_format(arr, current_format=None):
    from SynAPSeg.utils.utils_image_processing import morph_to_target_format
    # validate format 
    current_format = validate_format(arr, current_format=current_format)
    # standardize the format
    disp_arr = morph_to_target_format(arr, current_format, 'YXC').squeeze()
    if disp_arr.ndim < 2: raise ValueError(disp_arr.shape)
    return disp_arr
    
    
    # arr = uip.transform_axes(arr, current_format, STANDARD_FORMAT)
    # # format to display dims
    # take_dims = "".join([d for d in STANDARD_FORMAT if d not in DISPLAY_FORMAT])
    # return uip.reduce_dimensions(arr, current_format=STANDARD_FORMAT, take_dims=take_dims, project_dims="Z")

def validate_format(arr, current_format=None):
    if current_format and len(current_format) == len(arr.shape):
        return current_format
    from SynAPSeg.utils.utils_image_processing import estimate_format
    return estimate_format(arr.shape, default_format=STANDARD_FORMAT)



def plot_image_grid(
        images:List[np.ndarray], 
        masks: Optional[List[np.ndarray]]=None, 
        n_cols: Optional[int]=None, 
        n_rows: Optional[int]=None, 
        size_per_dim=8, 
        labels:bool=False, 
        cmap:Optional[Any]=None, 
        titles: Optional[List[str]]=None, 
        outpath: Optional[str]=None, 
        noshow:bool = False, 
        dpi=300, 
        suptitle: Optional[str]=None, 
        tight_layout=True,
        convert_to_float=False, 
        norm=False, 
        mask_overlay_fxn = None,
        mask_display_function_kwargs={'iterations':0, 'linewidth':1, 'label_cmap':False, 'alpha':1, 'c_func':lambda x: (1,0,0,1)},
        auto_int32_label_cmap = True, # if true and image is dype int32 use label cmap by default
        im_show_kwargs = None, # kwargs passed to imshow, only passed in not using an ovelay fxn
    ) -> tuple[plt.Figure, np.ndarray]:
    ''' 
    plot a x,y grid of images 
        if n_cols and n_rows are not provided, will default to square layout (e.g. 4 images -> 2x2 grid)
    
    args:
        images: list of numpy arrays
        masks: optional list of numpy arrays, if provided will overlay on images
        mask_alpha: deprecated, use mask_display_function_kwargs
        mask_color: deprecated, use mask_display_function_kwargs
        n_cols: number of columns in the grid
        n_rows: number of rows in the grid
        size_per_dim: size of each subplot in inches
        labels: if True uses label cmap by default
        cmap: colormap for images
        titles: optional list of titles for each image
        outpath: optional path to save the figure
        noshow: if True does not show the figure
        dpi: dpi for saving the figure
        suptitle: optional title for the figure
        tight_layout: if True applies tight layout to the figure
        convert_to_float: if True converts images to float
        norm: if True normalizes images
        mask_overlay_fxn: function to use for mask overlay, defaults to mask_to_outline_contours
        mask_display_function_kwargs: kwargs passed to mask_overlay_fxn
        auto_int32_label_cmap: if True and image is dtype int32 use label cmap by default
        im_show_kwargs: kwargs passed to imshow, only passed in not using an ovelay fxn
    '''

    # handle input args
    cmap = get_label_colormap() if labels else cmap
    mask_overlay_fxn = mask_to_outline_contours if mask_overlay_fxn is None else mask_overlay_fxn

    # determine rows, cols of image plotting grid based on input 
    n_imgs = len(images)

    if n_cols is None:
        if n_rows is None:
            n_cols = math.ceil(np.sqrt(n_imgs)) # default to square layout
        else:
            n_cols = math.ceil(n_imgs/n_rows)
    n_rows = n_rows or math.ceil(n_imgs/n_cols)
    
    # use shape to build the plot object
    fig,axs = plt.subplots(n_rows, n_cols, figsize=np.array([n_cols, n_rows])*size_per_dim)
    
    # add images to the figure
    for img_i in range(n_imgs):

        # get axis to position the image
        if n_cols==1 and n_rows==1:
            ax = axs
        elif n_cols==1 or n_rows==1:
            ax = axs[img_i]
        else:
            img_x = img_i // n_cols
            img_y = img_i % n_cols
            ax = axs[img_x, img_y]
            
        # handle display of image
        ############################################################################################
        #     if masks were provided use overlay fxn
        #     if img is int32 render as mask
        #     else treated as intensity image
        if (not masks) or (masks[img_i] is None) or (mask_overlay_fxn is False):
            use_lbl_cmap = auto_int32_label_cmap and (images[img_i].dtype == np.int32)
            _cmap = get_label_colormap() if use_lbl_cmap else cmap
            ax.imshow(images[img_i], cmap=_cmap, **im_show_kwargs or {})
        else:            
            mask_overlay_fxn(images[img_i], masks[img_i], ax=ax, **mask_display_function_kwargs)
        
        # fetch title if provided
        try:
            t = titles[img_i]
        except:
            t = None
            
        ax.set_title(t, fontsize='x-large')
        ax.axis('off')
        
    if tight_layout:    fig.tight_layout()
    if suptitle:        fig.suptitle(suptitle)
    if outpath:         fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    if not noshow:      plt.show()
    
    return fig, axs


def show_channels(img, clip_max=None, rgb=['red', 'green', 'blue']):
    print(img.shape)
    assert img.ndim == 3
    # normalize, or apply label color map
    if img.dtype == np.int32: # labels
        cmaps = [get_label_colormap()]
    elif img.dtype == np.uint16: # images
        if clip_max is None: clip_max = img.max() # equates to min/max normalization
        img = uip.convert_16bit_image(img, NORM=True, CLIP=(0,clip_max))

    if img.dtype == np.uint8:
        cmaps = [generate_custom_colormap(c, img) for c in rgb]

    for i in range(img.shape[-1]):
        plt.imshow(img[...,i], cmap=cmaps[i])
        plt.show()

def show_ch(img, cmaps=None, axis=-1):
    """new version simplfied, assumes channels are last"""
    n_ch = img.shape[axis]
    cmaps = dict(zip([c for c in range(n_ch)], ['red', 'green', 'blue', 'magenta'])) if cmaps is None else cmaps
    
    for i in range(n_ch):
        im = np.take(img, i, axis)
        show(im, 
             cmap = None if img.dtype==np.int32 else LinearSegmentedColormap.from_list(f"custom_cmap_{i}", ['black', cmaps[i]]), 
             def_title=f"ch-{i}")


def show_stack_new(img, ch, cmap='magma', figsize=(32,16), nmin=0.1, nmax=99.9):
    # show a single channel of 4dim, CZYX, mip with normalized intensities
    plt.figure(figsize=figsize)
    plt.imshow(np.max(img[ch],axis=0), 
            cmap=cmap,
            vmin=np.percentile(img[ch],nmin),
            vmax=np.percentile(img[ch],nmax)
            )
    plt.show()


def generate_custom_colormap(color_name, image):
    # create a custom color map from black to specific color, where px at max value appear white
    alpha=1
    color_dict = {'red':0, 'green': 1, 'blue': 2}

    if color_name not in color_dict.keys():
        raise ValueError("color_name must be 'red', 'green', or 'blue'")

    try: # handle integer 
        dtype_max, dtype_min = np.iinfo(image.dtype).max, np.iinfo(image.dtype).min
    except ValueError: # and float dtypes
        dtype_max, dtype_min = np.finfo(image.dtype).max, np.finfo(image.dtype).min

    n = int(dtype_max-dtype_min)  # Number of color steps

    # Generate list of RGBA tuples ranging from black to the specified color
    ch_color = color_dict[color_name]
    color_list = [[0, 0, 0, 1]] + [[i/n if j==ch_color else 0 for j in range(3)]+[alpha] for i in range(1, n-1)] + [[255,255,255,1]]

    # # Create a color map from this list
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('black_to_' + color_name, color_list)
    
    return cmap


def apply_colormap(image, cmap, NORM=False):
    if NORM: # Normalize image to range 0-1
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
    
    # Apply colormap (this converts image to RGBA)
    image_colored = cmap(image)
    
    return image_colored


def plot_image_hist(img):
    assert img.ndim == 3
    for i in range(img.shape[-1]):
        ch = img[..., i]        
        plt.hist(ch.ravel(), bins=255)
        plt.yscale('log')
        plt.show()


def show(
    img,
    ax=None,
    cmap=None,
    alpha=None,
    interpolation=None,
    def_title=None,
    figsize=(14, 14),
    hide_axis=False,
    **im_show_kwargs,
    ):
    """ display a 2D or 3D image using matplotlib's imshow """

    if isinstance(img, list):
        _show_handle_list_img_input(
            img,ax=ax,cmap=cmap,alpha=alpha,interpolation=interpolation,def_title=def_title,
            figsize=figsize,hide_axis=hide_axis,**im_show_kwargs
        )
        return

    img = np.array(img)
    alpha = 1.0 if alpha is None else alpha

    # handle different image dimensions
    ####################################
    if img.ndim == 2: # handle 2d
        pass
    elif (img.ndim==3 and (img.shape[-1]==3 or img.shape[-1]==4)): # handle 2d 3ch
        if img.shape[-1]==4:
            alpha = None
    else: # if ZYX or ZYXC
        img = uip.mip(np.array(img), axis=0)

    # handle custom colormaps
    if img.dtype.type == np.uint16:
        img = uip.convert_16bit_image(img)
    elif img.dtype == np.int32:
        cmap = get_label_colormap()
        interpolation = 'nearest'

    ax_flag = False if ax is None else True
    if not ax_flag: 
        fig,ax = plt.subplots(figsize=figsize)

    ax.imshow(img, cmap=cmap, interpolation=interpolation, alpha=alpha, **im_show_kwargs or {})

    if def_title is not None: ax.set_title(str(def_title))
    if hide_axis: ax.axis('off')

    if not ax_flag: 
        plt.show()
    else: 
        return ax


def _show_handle_list_img_input(img, ax=None, cmap=None, alpha=None, interpolation=None, def_title=None, figsize=(14,14), hide_axis=False):
    for im in img:
        show(im,ax=ax,cmap=cmap,alpha=alpha,interpolation=interpolation,def_title=def_title,figsize=figsize,hide_axis=hide_axis)


def show_gif(images, figsize=(12,12), duration=200, loop=0, cmap='gray', hide_axis=True):
    """
    Display a looping GIF inline in a Jupyter notebook from a list of images.

    Parameters
    ----------
    images : list of np.ndarray or PIL.Image.Image
        List of 2D or 3D images to display as frames in the GIF.
    duration : int, optional
        Duration (in milliseconds) of each frame. Default is 200.
    loop : int, optional
        Number of times the GIF should loop (0 = infinite). Default is 0.
    cmap : str, optional
        Colormap used for NumPy arrays (ignored for PIL images). Default is 'gray'.

    Returns
    -------
    None
        Displays the GIF inline in the notebook.
    """
    frames = []
    
    assert all(im.ndim==2 for im in images)

    for i, img in enumerate(images):
        fig, ax = plt.subplots(figsize=figsize)
        t = f"Frame {i}"
        show(img, ax=ax, cmap=cmap, def_title=t, hide_axis=hide_axis)
        
        
        # Save figure to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf))

    # Create GIF in memory
    gif_bytes = io.BytesIO()
    frames[0].save(
        gif_bytes,
        format='GIF',
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop
    )
    gif_bytes.seek(0)

    # Display inline
    display(IPyImage(data=gif_bytes.read(), format='png'))


def show_in_napari(imgs, lbl_img_contours=1):
    """show arrays in napari """
    viewer = napari.Viewer()
    imgs = [imgs] if isinstance(imgs, np.ndarray) else imgs
    assert isinstance(imgs, list)
    
    for img in imgs:
        # if filepath, load it
        if isinstance(img, str):
            if not os.path.exists(img): 
                raise FileNotFoundError(f"{img}")
            
            from SynAPSeg.IO.image_parser import ImageParser
            filepath = img
            parser = ImageParser.create_parser(filepath)
            _, img = parser.load_image()
            fmt = uip.estimate_format(img.shape)
            img, fmt = uip.standardize_collapse(img, fmt, 'STCZYX')
            print(f"loaded filepath: {filepath}.\n  <{img.dtype.name}> shape: {img.shape} -> interpreted format: {fmt}")
            
        if img.dtype == np.int32:
            lbl_layer = viewer.add_labels(img)
            lbl_layer.contour = lbl_img_contours
        else:
            viewer.add_image(img)
    napari.run()




def show_timepoints(all_mips_arr, n_channels, show_tps=[0, -1]):
    # show all chanels at different timepoints, expects shape: (T, C, Y, X)
    fig,axs=plt.subplots(nrows=n_channels, ncols=len(show_tps), figsize=np.array([18,8])*np.array([len(show_tps),n_channels]))
    for show_tp in show_tps:
        for chi in range(n_channels):
            ax = axs[chi, show_tp]
            ax.imshow(all_mips_arr[show_tp, chi, ...]/all_mips_arr[show_tp, chi, ...].max())
            ax.axis('off')
    plt.tight_layout()
    plt.show()


def show_overlayed_timepoints(all_mips_arr, n_channels, show_tps=[0, -1], save_path=None):
    # show overlayed timpoints for each channel, expects shape: (T, C, Y, X)
    fig,axs=plt.subplots(nrows=n_channels, figsize=np.array([18,8])*np.array([1,n_channels]))
    for chi in range(n_channels):
        ax = axs[chi]
        overlay_tps = np.zeros_like(all_mips_arr[0,0, ...])
        for show_tp in show_tps:
            overlay_tps += all_mips_arr[show_tp, chi, ...]
        overlay_tps /= overlay_tps.max()
        ax.imshow(overlay_tps)
        ax.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def hist(img, scale='symlog'):
    plt.hist(img.ravel())
    plt.yscale(scale)
    plt.show()


def show_filter_rpdf_results(rpdf, rpdf_filtered, label_img, int_img, clc_id_map, ax=None, color_outlines_kept=(255, 0, 0), color_outlines_removed=(128, 128, 128)):
    """
        Plot detection outlines kept/removed by filtering region props overlayed on intensity image

        Example usage:
            import skimage
            import utils_colocalization as uc
            IMG_CH = 0
            rpdf = uc.get_rp_table(label_img_preproc, all_mips[0,IMG_CH,:,:], ch_colocal_id={0:0}, prt_str='')
            rpdf = rpdf.assign(eccentricity = rpdf['axis_major_length']/rpdf['axis_minor_length'])

            int_img = skimage.img_as_float(all_mips[0,IMG_CH,:,:])
            threshold_dict = {0:{'intensity_max':[20000,None], 'eccentricity':[1.8,None]}}
            rpdf_filtered, fc = uc.filter_nuclei(threshold_dict, rpdf, [0])
            uc.pretty_print_fcounts(fc)

            fig,ax = plt.subplots(1,3, figsize=(20,20))
            up.show_filter_rpdf_results(rpdf, rpdf_filtered, label_img_preproc, int_img, {0:0}, ax=ax[0])
            plt.show()
    """
    # handle multiple axes
    MULTI_AX = True if isinstance(ax, np.ndarray) else False 
    
    # handle 2d input
    if int_img.ndim==2:
        int_img = np.expand_dims(int_img, -1)
        label_img = np.expand_dims(label_img, -1)

    for ax_i, (clc_id, int_img_ch) in enumerate(clc_id_map.items()):
        labels_kept = rpdf_filtered[rpdf_filtered['colocal_id']==clc_id]['label'].to_list()
        labels_removed = list(set(rpdf[rpdf['colocal_id']==clc_id]['label'].values) - set(labels_kept))
        pred_img = uip.filter_label_img(label_img[..., int_img_ch], labels_kept)
        pred_img_removed = uip.filter_label_img(label_img[..., int_img_ch], labels_removed)

        mask_outlines = uip.mask_to_outlines(pred_img)
        ch_pred_overlay = overlay(int_img[..., int_img_ch], mask_outlines, 'red', mask_color=color_outlines_kept)
        mask_outlines_removed = uip.mask_to_outlines(pred_img_removed)
        ch_pred_overlay = overlay(ch_pred_overlay, mask_outlines_removed, 'red', mask_color=color_outlines_removed)
        
        show(ch_pred_overlay, ax=ax[ax_i] if MULTI_AX else ax)


def legend_huestyle(
        ax, 
        HUE:str, PALETTE_HUE:dict, 
        STYLE:str, PALETTE_STYLE:dict, STYLE_LABELS:Optional[dict]=None
    ):
    """ 
        custom legend where ncol=2, nrow = len(PALETTE_HUE | PALETTE_STYLE)
            First Row: Hue Variables (Patches) &  row 2 is Style

        args:
            STYLE_LABELS, optional: keys must be in PALETTE_STYLE and values are reformatted labels 
    """
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    STYLE_LABELS = STYLE_LABELS or {}
    
    # First Row: Hue Variables (Patches)
    hue_elements = [
        Patch(facecolor=PALETTE_HUE[k], edgecolor='black', label=k) for k in PALETTE_HUE
    ]

    # Second Row: Marker Styles (Lines)
    style_elements = [
        (Line2D([0], [0], marker=PALETTE_STYLE[k], color='w', 
            markerfacecolor='black', markersize=10, 
            label=STYLE_LABELS.get(k) or k)) 
        for k in PALETTE_STYLE
    ]

    legend_elements = hue_elements + style_elements

    # Create the legend, order in legend_elements ensures row 1 is Hue and row 2 is Style
    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.02, 1), ncol=2,
        columnspacing=1.5, handletextpad=0.5,
        frameon=True, 
        title=f"{HUE.capitalize()} | {STYLE.capitalize()}"
    )


def seaborn_legend(g, new_labels):
    if isinstance(new_labels, list):
        for t, l in zip(g._legend.texts, new_labels): # update texts in the existing legend
            t.set_text(l)
    elif isinstance(new_labels, dict):
        for t in g._legend.texts:
            txt = str(t._text)
            if txt in new_labels:
                t.set_text(new_labels[txt])

def seaborn_annotate_correlations(data, corr_cols, subset_var, subset_attrs, **kws):
    """ Function to annotate each facet with the Pearson correlation coefficient and format according to subset var
    could be useful function generally, as this usually takes a lot of code to implement otherwise
    works by applying a function to each subset of the data ('palette' when plotting but here called subset_var)
    `````````````````````````````````````````````````````````````````````````````````````````````````````````````````
    example use case
    palette = dict(zip([0,1, 3, 4],['green', 'red', 'purple', 'brown']))
    g = sns.lmplot(
        data=dfp, x='manual', y='synapsedist_v2', hue='colocal_id', col='variable',
        sharex=False, sharey=False,
        col_wrap=4, palette=palette, ci=None,
        # height=4, scatter_kws={"s": 50, "alpha": 1}
    )
    # Apply the annotation function to each facet
    g.map_dataframe(seaborn_annotate_correlations, corr_cols=['manual', 'synapsedist_v2'], subset_var = 'colocal_id',
        subset_attrs = {
            0:dict(xy=(0.05, 0.95), color='green'), 
            1:dict(xy=(0.05, 0.9), color='red'),
            3:dict(xy=(0.05, 0.85), color='purple'),
            4:dict(xy=(0.05, 0.8), color='brown')
        }
    )
    """
    current_var = data[subset_var].unique()
    assert len(current_var) == 1
    current_attrs = subset_attrs[current_var[0]]
    r, _ = pearsonr(data[corr_cols[0]], data[corr_cols[1]])
    ax = plt.gca()
    ax.annotate(f'Pearson r: {r:.2f}', xycoords=ax.transAxes,
                fontsize=10, fontweight='bold', ha='left', va='center', **current_attrs)


def find_files(root_dir, search_string):
    """
    Search through the directory structure starting at `root_dir`
    to find all files that contain the specified `search_string`.

    Parameters:
    - root_dir (str): The root directory to start the search.
    - search_string (str): The substring to search for within file names.

    Returns:
    - list: A list of paths to the files that contain the search string.
    """
    matching_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if search_string in filename:
                matching_files.append(os.path.join(dirpath, filename))
    return matching_files


def svg_to_png(svg_path, png_path, size=(800, 800)):
    """
    Convert an SVG file to a PNG file using cairosvg and resize it.

    Parameters:
    - svg_path (str): Path to the input SVG file.
    - png_path (str): Path to the output PNG file.
    - size (tuple): The (width, height) to resize the images to.
    """
    cairosvg = ug.try_import('cairosvg')
    
    # Convert SVG to PNG
    cairosvg.svg2png(url=svg_path, write_to=png_path)
    
    # Open the newly created PNG and resize it
    img = Image.open(png_path)
    img = img.resize(size)
    img.save(png_path)

def create_gif(image_paths, gif_path, duration=2000):
    """
    Create a GIF from a list of image paths.

    Parameters:
    - image_paths (list): List of image file paths to include in the GIF.
    - gif_path (str): Path to save the output GIF file.
    - duration (int): Duration of each frame in the GIF in microseconds.
    """
    images = []
    for image_path in image_paths:
        images.append(imageio.imread(image_path))
    imageio.mimsave(gif_path, images, duration=duration)


def delete_files(file_paths):
    """
    Delete list of files from the filesystem.

    Parameters:
    - file_paths (list): List of file paths to delete.
    """
    failures = [] # where failed to remove
    for file_path in file_paths:
        try:
            os.remove(file_path)
        except Exception as e:
            failures.append([file_path, e])
    print(f"Deleted: {len(file_paths)-len(failures)}/{len(file_paths)} file_paths successfully.")
    return failures if len(failures) > 0 else None


def svgs_to_gif(
        root_directory = r"D:\BygraveLab\ConfocalImages\HTL_Homer-Traf3_Longitudinal-imaging\predictions\examples",
        output_gif_path = r"D:\BygraveLab\ConfocalImages\HTL_Homer-Traf3_Longitudinal-imaging\predictions\sdPred_Traf3-mCh_v3.1.gif",
        filter_str  = 'sdPred_Traf3-mCh_v3.1.svg',
        duration = 2000,
        size=(600, 600),
    ):
    """
    convert a number of .svg images into a gif
    root_directory - Specify the directory to search and the output GIF path
    """

    # Find SVG files
    svg_files = sorted(find_files(root_directory, filter_str))
    png_files = []

    # Convert SVG files to PNG
    for svg_file in svg_files:
        png_file = svg_file.replace('.svg', '.png')
        svg_to_png(svg_file, png_file, size=size)  # Specify your desired size here
        png_files.append(png_file)

    # Create GIF from PNG files
    create_gif(png_files, output_gif_path, duration=duration)
    print(f"GIF created successfully: {output_gif_path}")
    delete_files(png_files)


def plot_areas(binary_image):
    """plot the areas of objects in a binary image"""
    # Label the objects in the binary image
    labeled_image = label(binary_image)

    # Calculate properties of labeled regions
    regions = regionprops(labeled_image)

    # Extract areas of each object
    areas = [region.area for region in regions]
    print("Areas of objects:", areas)

    # Plot the labeled objects
    plt.figure(figsize=(12, 8))

    # Plot the labeled image
    plt.subplot(1, 2, 1)
    plt.imshow(labeled_image, cmap=get_label_colormap(), interpolation='nearest')
    plt.title("Labeled Objects")
    plt.axis("off")

    # Plot the histogram of area sizes
    plt.subplot(1, 2, 2)
    plt.hist(areas, bins=255, color='blue', edgecolor='black', alpha=0.7)
    plt.title("Histogram of Object Areas")
    plt.xlabel("Area (pixels)")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.xscale('log')

    plt.tight_layout()
    plt.show()


def jittered_scatterplot(data, x, y, jitter_strength=0.1, ax=None, order=None, **kwargs):
    """
    Wraps seaborn.scatterplot but adds horizontal jitter to categorical x-axis values.

    Parameters:
    - data: pd.DataFrame
    - x: str — column name for x-axis (categorical)
    - y: str — column name for y-axis
    - jitter_strength: float — range of jitter to apply around each x category
    - ax: matplotlib Axes — optional, axis to plot on
    - **kwargs: passed to sns.scatterplot (e.g., hue, style, etc.)
    """
    import seaborn as sns
    if ax is None:
        ax = plt.gca()

    # Map each unique x-value to a numeric position
    x_unique = data[x].unique() if order is None else order
    x_map = {val: i for i, val in enumerate(x_unique)}
    data = data.copy()
    data['_x_numeric'] = data[x].map(x_map)

    # Apply jitter
    jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(data))
    data['_x_jittered'] = data['_x_numeric'] + jitter

    # Plot
    sns.scatterplot(
        data=data,
        x='_x_jittered',
        y=y,
        ax=ax,
        **kwargs
    )

    # Set proper x-ticks
    ax.set_xticks(list(x_map.values()))
    ax.set_xticklabels(list(x_map.keys()))
    ax.set_xlabel(x)

    # Clean up
    data.drop(columns=['_x_numeric', '_x_jittered'], inplace=True, errors='ignore')
    
    return ax


def get_ars(df, col=None, **filters):
    """
    Filter a DataFrame by arbitrary column-value pairs and optionally return a specific column.

    Args:
        df (pd.DataFrame): The input DataFrame to filter.
        col (str, optional): Name of the column to return. If None, the filtered DataFrame is returned.
        **filters: Arbitrary keyword arguments specifying column-value pairs to filter on. 
            Values can be of any type supported by `pandas.DataFrame.query`.

    Returns:
        pd.DataFrame or pd.Series: 
            - If `col` is None, returns the filtered DataFrame.
            - If `col` is specified, returns the corresponding column (as a Series) from the filtered DataFrame.

    Example:
        get_ars(df, age='P60', reg_id=123)
        get_ars(df, col='nuclei_count', group='control', brain_region='CA1')
    """
    if filters:
        query_str = " & ".join(
            [f"{k} == '{v}'" if isinstance(v, str) else f"{k} == {v}" for k, v in filters.items()]
        )
        res = df.query(query_str)
    else:
        res = df

    return res if col is None else res[col]


def apply_comp_ars(comp_fxn, df, col, grp_vars, comp_var, comp_vals, min_obs=2, **comp_fxn_kwargs):
    """
    comp_var is column containing comp_vals which differentiate groups for comparing values
    last grp val and comp_vals are used to select df regions for plotting and comparing
    grp_vars = ['name', 'st_level', 'parent_reg_id', 'acronym', 'reg_id']
    comp_vals = ['young', 'old'] if comp_vals is None else comp_vals
    """
    import pandas as pd
    assert len(comp_vals) == 2
    assert isinstance(comp_var, str)

    results = []
    for dfn, _df in df.groupby(grp_vars):
        filters = [{comp_var:comp_val} for comp_val in comp_vals]
        A, B = [get_ars(_df, col=col, **_filts).values for _filts in filters]
        if len(A)<min_obs or len(B)<min_obs:
            continue
        _d = dict(zip(grp_vars, dfn))
        _d[col] = comp_fxn(A, B, **comp_fxn_kwargs)
        _d['A'] = np.mean(A)
        _d['B'] = np.mean(B)
        _d['A/B'] = _d['A']/_d['B']
        _d['nA'] = len(A)
        _d['nB'] = len(B)
        results.append(_d)
    return pd.DataFrame(results)


def plot_effect_sizes(
        df,
        y_vars = 'acronym',
        figsize=(15,6),
        x_vars = ['count', 'area', 'intensity_mean'],
        comp_var = 'treatment',
        comp_vals = ['young', 'old'],
        hue_fxn = lambda x: 'old' if x <0 else 'young',
        hue_palette = dict(zip(['young', 'old'], ['gray', 'k'])),
        EFTYPE = 'hedges',
        var2lblmap = {'intensity_mean': 'Intensity'},
        outpath=None,
        legend_title = None,
        min_obs = 2,
    ):
    import pingouin as pg
    import seaborn as sns
    import matplotlib.patches as mpatches

    orientation = 'vertical' if isinstance(x_vars, list) else 'horizontal'
    iter_vars = x_vars if orientation == 'vertical' else y_vars  
    fixed_var = y_vars if orientation == 'vertical' else x_vars  
    
    figsize = figsize if figsize is not None else (5 * len(iter_vars), 6)
    ncols, nrows = [len(iter_vars), 1] if isinstance(x_vars, list) else [len(iter_vars), 1][::-1]
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize, sharey=True)
    axes = [axes] if len(iter_vars) == 1 else axes # If there is just one axis (when len(x_vars)==1), convert axes to a list.
    
    # Iterate over the x variables and the corresponding axes.
    results = {v:None for v in iter_vars}
    for ax, var in zip(axes, iter_vars):
        _xvar = var if isinstance(x_vars, list) else fixed_var
        _yvar = var if isinstance(y_vars, list) else fixed_var

        # calculate effect size
        data = apply_comp_ars(
            pg.compute_effsize, df, var, [fixed_var], comp_var, comp_vals,  
            min_obs=min_obs, eftype=EFTYPE
        )
        # data = apply_comp_ars(pg.compute_effsize, df, var, min_obs=2, eftype=EFTYPE)
        data['hue'] = data[var].apply(hue_fxn)
        
        # plot
        sns.barplot(data=data, y=_yvar, x=_xvar, ax=ax, legend=False, hue='hue', palette=hue_palette)
        
        # format
        def get_var_name(v): 
            lbl_name = var2lblmap.get(v, v) #v if v not in var2lblmap else var2lblmap[v] 'Intensity' if v == 'intensity_mean' else v
            return lbl_name.capitalize()
            
        ax.set_title(get_var_name(var) + f" ({EFTYPE}\' g)")
        ax.set_xlabel(get_var_name(_xvar))
        ax.set_ylabel(get_var_name(_yvar))
        # add a vertical line at 0 if orientation is horizontal
        if orientation == 'vertical':
            ax.axvline(x=0, color='gray', linestyle='--')
        else:
            ax.axhline(y=0, color='gray', linestyle='--')
        results[var] = data
    
    patches = [mpatches.Patch(color=color, label=label) for label, color in hue_palette.items()]
    ax.legend(handles=patches, title=legend_title or 'Legend', bbox_to_anchor=(1.05, 1.0))
    
    plt.tight_layout()
    sns.despine(fig)
    
    if outpath:
        save_fig(outpath, dpi=300)
    plt.show()

    return results


def set_global_textsize_defaults(global_textsize=8, override_defaults: dict = {}) -> None:
    """ 
    set textsize of all common matplotlib figure text elements to a global_textsize through plt.rcParams.update.
        can pass override values through override_defaults
    """
    # setup global figure settings
    defaults = {
        'font.family': 'sans-serif',
        'font.serif': ['Arial'],
        'axes.titlesize': global_textsize,
        'axes.labelsize': global_textsize,
        'xtick.labelsize': global_textsize,
        'ytick.labelsize': global_textsize,
        'legend.fontsize': global_textsize,
        'figure.titlesize': global_textsize,
    }
    defaults.update(override_defaults)
    plt.rcParams.update(defaults)


def palette_cat(plot_df, hue_var):
    """ create a categorical palette for each unique value in the hue_var column using glasbey.create_palette """
    import glasbey
    return dict(zip(plot_df[hue_var].unique(), glasbey.create_palette(palette_size=plot_df[hue_var].nunique())))


def make_legend(
    palette,
    title=None,
    ax=None,
    kind="patch",
    size=10,
    edgecolor="none",
    linewidth=2,
    ncol=1,
    **legend_kwargs
):
    """
    Create a legend from a palette mapping {label: color}.
    
    Parameters
    ----------
    title : str
        Title of the legend.
    palette : dict
        Mapping of label -> color.
    ax : matplotlib axis, optional
        Axis to attach the legend to. Defaults to current axis.
    kind : str, optional
        One of {"patch", "circle", "line"}. Controls legend marker type.
    size : float, optional
        Size of circle or patch handles.
    edgecolor : str, optional
        Edge color for patch or circle.
    linewidth : float, optional
        Line width for `kind='line'`.
    ncol : int
        Number of legend columns.
    legend_kwargs : dict
        Additional kwargs passed to `ax.legend()`.
    """
    # move this to utils plotting 
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Circle
    from matplotlib.lines import Line2D

    ax = ax or plt.gca()

    handles = []

    for label, color in palette.items():

        if kind == "patch":
            handle = Patch(facecolor=color, edgecolor=edgecolor)

        elif kind == "circle":
            handle = Line2D(
                [0], [0],
                marker="o",
                markersize=size,
                markerfacecolor=color,
                markeredgecolor=edgecolor,
                linestyle="None",
            )

        elif kind == "line":
            handle = Line2D(
                [0], [0],
                color=color,
                linewidth=linewidth
            )

        else:
            raise ValueError(f"Invalid kind '{kind}'. Use 'patch', 'circle', or 'line'.")

        handles.append(handle)

    legend = ax.legend(
        handles,
        list(palette.keys()),
        title=title,
        ncol=ncol,
        **legend_kwargs,
    )

    return legend

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def add_scalebar(
    ax, 
    pixel_to_unit, 
    title=None, 
    length_units=10, 
    unit_of_length="μm",
    location=(0.95, 0.05), 
    color='white', 
    bar_height=0.01, 
    fontsize=10,
    title_y_adjust=0,
    scalebar_kwargs=None, 
    text_kwargs=None,
    ):
    """
    Adds a scale bar to a matplotlib axis.

    Parameters:
    - ax: The matplotlib axis to draw on.
    - pixel_to_unit: Scale factor (units per pixel).
    - title: defaults to f"{length_units} {unit_of_length}". String label to place under the bar (e.g., '10 μm').
    - length_units: The physical length the bar represents (in the same units as pixel_to_unit).
    - unit_of_length: The unit of the length the bar represents (e.g., 'μm').
    - location: Tuple (x, y) in relative axes coordinates (0 to 1) for the bar's right-end position.
    - color: Color of the bar and text.
    - bar_height:  thickness of the scale bar expressed as a fraction of the image height, defaults to 1%.
    - fontsize: Size of the title text.
    - title_y_adjust: add this value to the y coordinate of the label
    - scalebar_kwargs: Additional kwargs passed to patches.Rectangle
    - text_kwargs: Additional kwargs passed to ax.text
    """
    if title is None:
        title = f"{length_units} {unit_of_length}"
    # Calculate bar length in pixel coordinates
    bar_width_pixels = length_units / pixel_to_unit
    
    # Get axis limits to convert relative location to data coordinates
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    
    # Calculate position in data coordinates. For images, Y axis is inverted (0 at top)
    pos_x = xmin + location[0] * (xmax - xmin)
    pos_y = ymin + location[1] * (ymax - ymin)
    
    # Create the rectangle (scale bar) anchor it such that pos_x is the right edge
    _scalebar_kwargs = dict(
        xy = (pos_x - bar_width_pixels, pos_y), 
        width = bar_width_pixels, 
        height = ((ymax - ymin) * bar_height),
        linewidth=0,  # using linewidth adds to border making it bigger than it should be
        edgecolor=color, 
        facecolor=color,
        transform=ax.transData
    )
    _scalebar_kwargs.update(scalebar_kwargs or {})
    
    scale_bar = patches.Rectangle(**_scalebar_kwargs)
    ax.add_patch(scale_bar)
    

    # Add the title underneath
    _text_kwargs = dict(
        x = pos_x - (bar_width_pixels / 2), 
        y = pos_y + title_y_adjust, 
        s = title, 
        color=color, 
        fontsize=fontsize,
        ha='center', 
        va='center',
        transform=ax.transData,
    )
    _text_kwargs.update(text_kwargs or {})

    ax.text(**_text_kwargs)
