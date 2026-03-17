from aicsimageio import AICSImage
import numpy as np
import os
import xml
import pandas as pd
import ast
import re
import scipy.ndimage as ndi
from tqdm import tqdm
import copy

    
from . import utils_general as ug


def read_czi(image_path):
    ''' load czi image and extract scene ids (number of images) '''
    from _aicspylibczi import PylibCZI_CDimCoordinatesOverspecifiedException

    try:
        czi = AICSImage(image_path) # get AICSImage object
    
    except PylibCZI_CDimCoordinatesOverspecifiedException:
        raise PylibCZI_CDimCoordinatesOverspecifiedException('exception for czi multifile scenes - use read_czi_multifilescenes_mosiac instead')
        czi, arr, dims = read_czi_multifilescenes_mosiac(image_path, get_dims = 'STCZYX')

    scene_ids = czi.scenes
    print('dims', czi.dims, 'shape', czi.shape, '\nscene_ids:', scene_ids)
    return czi, scene_ids

def czi_scene_to_array(
        czi_img, 
        scene_i=0, 
        czi_fmt_str="STCZYX", 
        czi_fmt_timepoint=None, # deprecated
        czi_fmt_slice=None, 
        ch_last=True, rotation=None, bgr2rgb=False, moveax=None):
    ''' extract a image data from a czi file, and convert to numpy array '''
    czi_img.set_scene(scene_i)
    slic = get_slice_from_string(czi_fmt_slice) if czi_fmt_slice is not None else None
    arr = czi_img.get_image_data(czi_fmt_str)[slic]
    if rotation is not None:
        arr = ndi.rotate(arr, rotation, axes=(2,1), reshape=False, order=0, prefilter=False)
    if bgr2rgb:
        arr = np.stack([arr[i] for i in [2,1,0]], 0)
    arr = np.moveaxis(arr, 0, -1) if ch_last else arr # reshape so channels last
    if moveax is not None:
        arr = np.moveaxis(arr, moveax[0], moveax[1])
    
    print(arr.shape)
    return arr

def get_slice_from_string(slice_str):
    """ Slice the numpy array based on a slice string (e.g. ":, 0, :, :"), returning tuple that can index into numpy array """
    # Split the slice string and remove spaces
    slices = [s.strip() for s in slice_str.split(',')]
    def get_slice_parts(s):
        # Convert the slice strings to actual slice objects or integers
        return slice(*list(map(lambda x: int(x) if x else None, s.split(':')))) if ':' in s else int(s)
    return tuple([get_slice_parts(s) for s in slices])

def print_keys_from(d, keys, prefix=''):
    for k in keys:
        print(f"{prefix}{k}: {d.get(k)}")

def czi_metadata2xmlFile(md, outpath='czimetadata.xml'):
    """ write metadata to xml file. md can be accessed through czi.metadata attribute"""
    assert isinstance(md, xml.etree.ElementTree.Element)
    with open(outpath, 'w') as f:
        f.write(xml.etree.ElementTree.tostring(md, encoding='unicode'))


    


def get_czi_ch_wavelengths(czi, common_wavelengths=[405, 488, 561, 639]):
    """ 
    extract laser wavelengths for all channels by parsing the czi's metadata.
        matches each of the common_wavelengths to the closest excitation wavelength or if 
        not found to the closest detection wavelength, so they do not need to be exact.
        If more channels are provided than the number of wavelengths, they are appended to the end.
    """
    ch_wavelength = {}
    wl_set = set(common_wavelengths.copy())
    n_exhausted = 0 # for keeping track of channels if more provided than common wavelengths
    channels = czi.metadata.find('Metadata/Information/Image/Dimensions/Channels') 
    for chan in channels:
        # need to use excitation wavelenth and get closest match to common, as LightSourcesSettings/LightSourceSettings/Wavelength, can be inaccurate apparently if there are multiple per channel
        if chan.find('ExcitationWavelength') is not None:
            wl = chan.find('ExcitationWavelength').text
        else:
            wl = float(chan.find('DetectionWavelength/Ranges').text.split('-')[0])
        exWavelength = int(wl)
        if len(wl_set) == 0:
            print('common wavelenths exhausted, appending to end')
            closest_match = max(common_wavelengths) + 1 + n_exhausted
            n_exhausted +=1
        else:
            closest_match = _find_closest_and_remove([exWavelength], wl_set)[0]
        ch_wavelength[chan.attrib['Name']] = closest_match
    return ch_wavelength

def _find_closest_and_remove(input_list, input_set):
    """ For each element in the input_list, finds the closest number in the set by 
    calculating the absolute difference.
    """
    input_list, input_set = copy.deepcopy(input_list), copy.deepcopy(input_set)
    output, closest_matches, unmatched = [], [], []
    for number in input_list:
        if len(input_set) == 0:
            unmatched.append(number)
            continue
        closest = min(input_set, key=lambda x: abs(x - number))
        output.append(closest)
        closest_matches.append(number)
        input_set.remove(closest) # Remove the found closest number from the set
    return output, closest_matches, unmatched


def find_missing_czi_indicies(expected_wl, czi):
    """ check if less channels in image than expected
        e.g. when running pipeline on heterogeneous images where one channel may be missing

    Args:
        expected_wl (dict): e.g. {'EGFP': 488, 'AF568': 561}
        czi (): czi object

    Raises:
        ValueError: if can't match expected wavelengths to the ones it found

    Returns:
        None or list of indicies which are unmatched
    """
    NULL_CHANNELS = None
    chwl = get_czi_ch_wavelengths(czi) # map channel names to wavelengths
    
    if len(expected_wl) != len(chwl):
        s_chwl = sorted(chwl.items(), key=lambda x: x[1])
        ch_names = czi.channel_names
        sorted_indices = [ch_names.index(elem[0]) for elem in s_chwl]

        found_wl = {v:k for k,v in chwl.items()}  
        # missing wavelengths 
        found, closest_matches, unmatched_wl = _find_closest_and_remove(list(found_wl.keys()), set(expected_wl.keys()))
        # find index missing should be in the array
        expects_not_found = [el for el in expected_wl.keys() if el not in found]
        s_expected = sorted(expected_wl.keys())
        unmatched_inds = [s_expected.index(v) for v in expects_not_found]
        if len(unmatched_inds) == 0: 
            raise ValueError('failed to parse missing indicies\n\texpected wls: {expected_wl}\n\tfound wls: {found_wl}')
        NULL_CHANNELS = unmatched_inds
    return NULL_CHANNELS



def _extract_czi_channel_info(czi, CMAPS_DEFAULT=None, laser_wavelengths=None):
    CMAPS_DEFAULT = ['blue', 'green' , 'red', 'magenta'] if CMAPS_DEFAULT is None else CMAPS_DEFAULT
    laser_wavelengths = [405, 488, 561, 639] if laser_wavelengths is None else laser_wavelengths
    WAVELENGTH2CMAP = dict(zip(laser_wavelengths, CMAPS_DEFAULT))
    
    ch_names = czi.channel_names
    chwl = get_czi_ch_wavelengths(czi) # map ch names to wavelength
    chNames2cmap = {k:WAVELENGTH2CMAP[chwl[k]] for k in ch_names} # map ch_names to a colormap
    
    return chwl, chNames2cmap

def get_cmaps_czi_intensity(czi, CMAPS_DEFAULT = ['blue', 'green' , 'red', 'magenta']):
    """ get colormaps based on laser intensity """
    chwl, chNames2cmap = _extract_czi_channel_info(czi, CMAPS_DEFAULT=CMAPS_DEFAULT)
    
    CMAPS = []
    for k,v in sorted(chwl.items(), key=lambda x: x[1]):
        CMAPS.append(chNames2cmap[k])
    return CMAPS


def czi_arrange_channels(czi, arr, channel_axis=-1):
    """Reformat channels to be in order of wavelength (e.g., 405, 488, 594, 647).

    Parameters:
    czi : CZI file object
        CZI file containing metadata and channel information.
    arr : ndarray
        Multi-channel image data from the CZI file.
    channel_axis : int, optional
        Axis index of the channels in the mip array (default is -1, last axis).

    Returns:
    sorted_arr : ndarray
        Image data with channels rearranged according to their wavelengths.
    """
    ch_names = czi.channel_names
    chwl = get_czi_ch_wavelengths(czi)  # map channel names to wavelengths
    s_chwl = sorted(chwl.items(), key=lambda x: x[1])
    
    # Get indices of channels in the order they should be sorted
    sorted_indices = [ch_names.index(elem[0]) for elem in s_chwl]
    
    # Use np.take to reorder the channels
    sorted_arr = np.take(arr, sorted_indices, axis=channel_axis)
    
    return sorted_arr






def get_czi_zooms(czi):
    # return czi.metadata.find('Metadata/HardwareSetting/ParameterCollection/Zoom').text # this is different for some reason, but seems wrong based on info in zen
    channels = czi.metadata.find('Metadata/Information/Image/Dimensions/Channels') #.getchildren()
    zooms = []
    for chan in channels:
        zooms.append([float(chan.find('LaserScanInfo/ZoomX').text), float(chan.find('LaserScanInfo/ZoomY').text)])
    return np.array(zooms)

def get_czi_mag(czi):
    return float(czi.metadata.find(
        'Metadata/Information/Instrument/Objectives/Objective/NominalMagnification'
    ).text)

def get_czi_scaling(czi):
    output = {}
    try:
        distances = czi.metadata.find('Metadata/Scaling/Items').findall('Distance')
        for el in distances:
            axis = el.attrib['Id']
            output[axis] = float(el.find('Value').text)
    except Exception as e:
        print(f'czi_metadata_get_scaling failed. error: {e}')
    return output


def get_channel_ex_detect_wavelength(chan_metadata):
    """ parse ExcitationWavelength and DetectionWavelength ranges for a channel given its metadata """
    return (
        f"exλ: {chan_metadata.find('ExcitationWavelength').text}" + \
        f", detectionλ: {[round(float(el),0) for el in chan_metadata.find('DetectionWavelength/Ranges').text.split('-')]}"
    )


def get_laser_settings_by_channel(czi, laserLinePattern = 'MTBLKM980LaserLine(\d+)'):
    md_str = 'Metadata/Experiment/ExperimentBlocks/AcquisitionBlock/SubDimensionSetups/MultiTrackSetup/Track'
    md_dict = {}
    for track_i, track in enumerate(czi.metadata.findall(md_str)):
        assert track.attrib['IsActivated'] == 'true'
        # print([el for el in track.iter()])

        tracklasersettings = track.findall('DetectionModeSetup/Zeiss.Micro.LSM.Acquisition.Lsm880ChannelTrackDetectionMode/TrackLaserSettings/ParameterCollection')
        for pc in tracklasersettings:
            if pc.find('IsEnabled').text == 'true':
                trackId = pc.attrib['Id']
                intensity = pc.find('Intensity').text
                laserLine = re.match(laserLinePattern, trackId).groups(1)[0]
                print(trackId, intensity, laserLine)
                if trackId not in md_dict:
                    md_dict[trackId] = {'track_laser_settings':[]}
                else:
                    raise ValueError('did not expect tracks would have the same laser line ids.')
                md_dict[trackId] = dict(zip(['intensity', 'laserLine', 'channels', 'track_i'],[intensity, laserLine, [], track_i]))
    
    # get gain from channels and add laser intensity to output dict
    channels = czi.metadata.find('Metadata/Information/Image/Dimensions/Channels')
    channels_dict = {}
    for ch_i, ch in enumerate(channels):
        gain = ch.find('DetectorSettings/Voltage').text
        lightSourceId = ch.find('LightSourcesSettings/LightSourceSettings/LightSource').attrib['Id']
        lightSourceWavelength = ch.find('LightSourcesSettings/LightSourceSettings/Wavelength').text
        print(ch.attrib, gain, lightSourceId, lightSourceWavelength)
        if lightSourceId not in md_dict:
            raise ValueError(lightSourceId, 'not in', md_dict)
        md_dict[lightSourceId]['channels'].append(ug.merge_dicts(ch.attrib, dict(zip(['gain'], [gain]))))
        ch_name = ch.attrib['Name']
        if ch_name in channels_dict:
            raise ValueError ('ch should not already exist')
        channels_dict[ch_name] = {'gain':gain, 'intensity':md_dict[lightSourceId]['intensity']}
    return channels_dict


def print_czi_metadata(czi_filepath):
    # deprecated, might still work with other czi files
    c = AICSImage(czi_filepath)
    md = c.ome_metadata.dict()
    images_metadata = md['images']
    print(f"{czi_filepath}\n\tnum images:{len(images_metadata)}")
    for imd in images_metadata[:1]:
        print_keys_from(imd['pixels'], ['dimension_order', 'size_c', 'size_t'])
        print_keys_from(imd['pixels'], ['physical_size_x', 'physical_size_y', 'physical_size_z'])
        for ch in imd['pixels']['channels']:
            print(ch['id'])
            print_keys_from(ch, ['fluor', 'name', 'excitation_wavelength', 'emission_wavelength'], prefix='\t')


def extract_metadata(czi):
    metadata = {}
    for n, f in {
        'ch_wavelengths': get_czi_ch_wavelengths,
        'zooms': get_czi_zooms,
        'mag': get_czi_mag,
        'scaling' : get_czi_scaling,
        'shape': lambda x: x.shape,
    }.items():
        try:
            res = f(czi)
            metadata[n] = str(res) #if not isinstance(res, dict) else res # convert to str unless its a dictionary
        except:
            metadata[n] = None
    return metadata
                

def compile_czi_metadata(paths) -> pd.DataFrame:       
    """ 
    compile metadata from czi files into a dataframe
    
    Parameters
    ----------
    paths : list of str
        list of paths to czi files
    
    Returns
    -------
    mddf : pd.DataFrame
        dataframe containing metadata for each czi file
    
    """    
    mds = []
    for image_path in tqdm(paths, desc="Compiling CZI files", unit="file"):
        fn = os.path.basename(image_path)
        try:
            # load image data
            czi = AICSImage(image_path) # get AICSImage object
            scene_ids = czi.scenes
            md = extract_metadata(czi)
            
            # explode scaling dict
            scaling = ast.literal_eval(md.pop('scaling'))
            md[f"scaling_unit"] = 'μm'
            for k,v in scaling.items():
                md[f"scaling_{k}"] = v * 1e6 # convert to μm
            
            # explode shape 
            shape = ast.literal_eval(md.pop('shape'))
            shape_order = czi.dims.order
            for i, s in enumerate(shape):
                md[f"{shape_order[i]}"] = s
            
            # merge in other metadata
            md = ug.merge_dicts(
                {'fn':fn, 'path':image_path,  'scene_ids':scene_ids}, 
                md, 
                {'error':np.nan}
            )
            
        except Exception as e:
            print(f"failed to read {image_path}.\n\terror: {e}")
            md = {'fn':fn, 'path':image_path, 'error':str(e)}
        mds.append(md)
    
    mddf = pd.DataFrame(mds)
    return mddf
        


def read_czi_multifilescenes_mosiac(czi_path, get_dims = 'STCZYX'):
    """ 
    resolves aicsimageio issue reading czi files where contiguous scenes are saved across multiple files during acquisition 
    this implementation is not robust 

    returns stitched array
        reader, arr, get_dims
    """
    from aicsimageio.readers.czi_reader import CziReader
    from aicspylibczi import CziFile

    uri = czi_path
    reader = CziReader(uri)
    scenes = reader.scenes
    reader.set_scene(0)

    
    tiles, dims = reader._get_image_data(reader._fs, uri, 0)
    
    with reader._fs.open(reader._path) as open_resource:
        czi = CziFile(open_resource.f)
        data_dims_shape = czi.get_dims_shape()[0]
        dd = {k:v-1 for (k,v) in dims if k not in get_dims}
        tile_info_to_bboxes = czi.get_all_mosaic_tile_bounding_boxes(
            **dd
        )
        current_scene_i = data_dims_shape['S'][0] # assumes issue is b/c scene indicies are spread out over multiple files
        mosaic_scene_bbox = czi.get_mosaic_scene_bounding_box(current_scene_i)

    arr = reader._stitch_tiles(
        tiles,
        data_dims = reader.mapped_dims,
        data_dims_shape = data_dims_shape,
        tile_bboxes = tile_info_to_bboxes,
        final_bbox = mosaic_scene_bbox,
    )

    return reader, arr, get_dims



if __name__ == '__main__':
    
    # test compile_czi_metadata
    #########################################################################################
    img_dir = r"\\rstore.it.tufts.edu\tusm_bygravelab01$\Confocal data archive\Pascal\tests"
    pattern = re.compile(r".*psd95.*\.czi", re.IGNORECASE)
    
    # get paths
    paths = ug.get_contents_recursive(img_dir, pattern=pattern)
    print(f"found {len(paths)} files")
    # compile metadata
    czi_metadata = compile_czi_metadata(paths[:5])

    # filter metadata
    czi_metadata = czi_metadata.sort_values(['scaling_Z'])
    subdf = czi_metadata.copy(deep=True)
    subdf = subdf[subdf['scaling_Z'] < 0.5]
    subdf = subdf[subdf['Z'] > 5]
    for i, row in subdf.iterrows():
        print(row['path'])
        print(f"fn: {os.path.basename(row['path'])}")
        print(row['scaling_Z'])
        print(f"shape: {row['Z']} x {row['Y']} x {row['X']}")
        print('\n')