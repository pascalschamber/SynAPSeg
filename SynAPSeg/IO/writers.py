import os
import tifffile
import struct


def write_array(arr, outdir, output_name, fmt=None, ex_md=None, tiff_metadata=None, OUTPUT_IMAGE_PYRAMID=False):
    """ 
    write an array to disk as tiff file using zlib compression (lossless) and cache its metadata in ex_md 

    Saves the provided NumPy array as a compressed TIFF image using zlib compression.
    If an image pyramid is requested via metadata and the output name is "raw_img", the array
    will be saved in OME-TIFF format with channel metadata and pixel scaling for compatibility
    with tools like QuPath and ABBA. Metadata is cached in the `ex_md` dictionary.

    Args:
        arr (np.ndarray): Array to be written to disk.
        outdir (str): Output directory where the file will be saved.
        output_name (str): Base name of the output file (extension will be added automatically).
        fmt (str, optional): Axis format string (e.g., 'ZYXC'); required for OME-TIFF output.
        ex_md (dict, optional): Metadata dictionary to store shape and format information.
            If None, a default dictionary will be created.
        tiff_metadata (dict, optional): Additional metadata to embed into the TIFF file.
        OUTPUT_IMAGE_PYRAMID (bool): write as ome.tiff, currently only in abba compat. format (YXC)

    Raises:
        AssertionError: If `fmt` is not provided when saving as OME-TIFF.
        struct.error: If compression fails due to TIFF size limits; a fallback to BigTIFF is used.

    Returns:
        None
    
    """
    # init empty if not provided
    if not ex_md:
        ex_md = {'data_metadata': {
            'data_shapes': {}, 'data_formats': {}}, 
            'image_metadata':{'scaling':{'X':1.0, 'Y':1.0, 'Z':1.0}}, 
            'channel_info':{}
        }
            
    output_filetype = 'ome.tiff' if OUTPUT_IMAGE_PYRAMID else 'tiff'
    if output_name.endswith(output_filetype): # prevent double extension
        output_name = output_name[:-len(output_filetype)-1]
        
    outpath = os.path.join(outdir, f"{output_name}.{output_filetype}")
    
    # save raw img as OME.TIFF for abba/qupath compatibility 
    if OUTPUT_IMAGE_PYRAMID:
        
        from SynAPSeg.utils.utils_image_processing import collapse_singleton_dims, transform_axes
        
        # get px scaling
        px_size = 1.0 # default
        scaling_info = ex_md['image_metadata'].get('scaling')
        if isinstance(scaling_info, str):
            from ast import literal_eval
            scaling_info = literal_eval(scaling_info)
        # TODO need to handle unit - czi store in meters, but qupath wants ome metadata as microns
        # for now assumes all units are in meters !!
        if isinstance(scaling_info, dict) and 'X' in scaling_info.keys():
            px_size = scaling_info['X']
        

        # reduce singleton dims and reformat so ch last
        assert fmt is not None, "fmt is required for ome.tiff output"
        arr, fmt = collapse_singleton_dims(arr, fmt)
        out_fmt = fmt.replace('C', '') + 'C'
        arr = transform_axes(arr, fmt, out_fmt) # move ch last
        fmt = out_fmt

        # get channel names
        channel_names = list(ex_md['data_metadata'].get('channel_info', {}).values())
        if not channel_names:
            channel_names = [f'channel_{i}' for i in range(arr.shape[fmt.index('C')])]

        arr2ome(outpath, arr, fmt, channel_names, px_size)
        
    # code adapted from napari.utils.io.imsave_tiff
    elif arr.dtype == bool:
        tifffile.imwrite(outpath, arr, metadata={"axes":fmt} if fmt else {"shape":arr.shape})
    else:
        _kwargs = dict(
            compression='zlib', # same compression settings used by napari
            compressionargs={'level': 1},
        )
        _kwargs['metadata'] = {"axes":fmt, "shape":arr.shape} if fmt else {"shape":arr.shape}  # store fmt in tiff metadata
        if tiff_metadata: 
            _kwargs['metadata'].update(tiff_metadata)
            
        try:
            tifffile.imwrite(outpath, arr, **_kwargs)
        except struct.error: # regular tiffs don't support compressed data >4GB
            tifffile.imwrite(outpath, arr, bigtiff=True, **_kwargs)
    
    # add metadata
    ex_md['data_metadata']['data_shapes'][output_name] = list(arr.shape)
    ex_md['data_metadata']['data_formats'][output_name] = fmt



def arr2ome(outpath, arr, fmt, channel_names, px_size, ch_last=True):
    """ write an array to disk as a multi-scale OME.TIFF 
        args:
            outpath (str)
            arr (np.ndarry)
            fmt (str)
            channel_names (list(str))
            px_size (float): pixel size in microns
            ch_last() : channels last
    """
    # TODO switch to implementation provided by aiscimgio
    # from aicsimageio.writers import OmeTiffWriter
    
    from SynAPSeg.IO.ome_pyramid_writer import generate_image_pyramids, pyramid_upgrade
    from SynAPSeg.utils.utils_image_processing import collapse_singleton_dims, transform_axes
    
    # check chs length and move to channels last - as ch last fmt is required for ABBA
    if ch_last:
        out_fmt = fmt.replace('C', '') + 'C'
        arr = transform_axes(arr, fmt, out_fmt) # move ch last
        C = arr.shape[out_fmt.index('C')]
        if not channel_names:
            channel_names = [f'channel_{i}' for i in range(C)]
        if len(channel_names) != C:
            raise ValueError(f"Number of channel names ({channel_names}: n={len(channel_names)}) does not match number of channels ({C})")
        
        # convert to OME TIFF
        generate_image_pyramids.main([arr[...,i] for i in range(C)], outpath, px_size=px_size)
        pyramid_upgrade.main(outpath, channel_names)
    else:
        raise NotImplementedError



if __name__ == '__main__':

    

    # USE THIS LIB INSTEAD
    # unsure how to capture dimensions with this format either
    # they both read identically as CYX format
    from aicsimageio.writers import OmeTiffWriter
    from aicsimageio.types import PhysicalPixelSizes
    import numpy as np

    # Your data (C, Y, X) or (C, Z, Y, X)
    data = np.random.randint(0, 255, (3, 512*16, 512*16), dtype=np.uint8)

    # It handles the pyramid levels automatically
    OmeTiffWriter.save(
        data, 
        r"C:\Users\pasca\Downloads\testwide.ome.tiff", 
        dim_order="CYX",
        scale_num_levels=3,  # It will validate if this is possible
        scale_factor=2.0,
        physical_pixel_sizes = PhysicalPixelSizes(1.0, 1.0, 1.0),
        channel_names=['ch1', 'ch2', 'ch3']

    )

    
    

