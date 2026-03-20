from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import copy
import re
import ast
import pprint
import os
import sys
from typing import Any, Dict, Optional, Tuple, List

from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_czi as uCzi
from SynAPSeg.IO.metadata_handler import MetadataParser, process_channel_info


# we focus on extracting this metadata from images. Availability is dependent on image type.
IMAGE_OBJECT_METADATA_TEMPLATE = {
    'ch_wavelengths': {},
    'zooms': None,
    'mag': None,
    'shape': None,
    'init_image_format_target': 'STCZYX' # will try to parse input image so all these dimensions are filled
}

def fill_empty_metadata(found_metadata=None):
    """initalize or fill missing keys in metadata"""
    found_metadata = {} if found_metadata is None else found_metadata
    for k,v in IMAGE_OBJECT_METADATA_TEMPLATE.items():
        if k not in found_metadata.keys():
            found_metadata[k] = v
    return found_metadata

def update_format(current_format, removed_axes=''):
    """track removed dimensions"""
    for el in removed_axes:
        current_format = current_format.replace(el, '')
    return current_format




def create_parser( 
                    file_path: str, 
                    params: Optional[dict]=None, 
                    load_kwargs: Optional[dict]=None
):
    """
    Factory method to create the appropriate image parser
        based on regex patterns that match the end of the file path.
    Args:
        file_path:
        params: TODO describe, consider changing to reflect its really config used in .run()
    TODO:
        make params optionally unless using in seg pipeline
    """
    
    parsers = {
        r"\.czi$": CZIImageParser,
        r"\.(tiff|tif)$": TIFFImageParser,   # TODO this is going to intercept OME.TIFFs but may want to actually read them with the aicsimgio lib
        r"\..*$": _determine_general_parser(),          
        # r"\.ims$": IMSImageParser,  # TODO, not currently implemented
    }

    for pattern, parser_cls in parsers.items():
        if re.search(pattern, file_path, flags=re.IGNORECASE):
            return parser_cls(file_path, params=params, load_kwargs=load_kwargs)
    
    parser_cls = _determine_general_parser()
    
    try:
        return parser_cls(file_path, params=params, load_kwargs=load_kwargs)
    except Exception as e:
        raise  ValueError (f"no suitable parser could be created based on image path: {file_path}") from e
    

class ImageParser(ABC):
    """
    Abstract base class for image parsers.
    
    Example usage:
        file_path = 'image.vsi'
        image_parser = ImageParser.create_parser(file_path)    
        img_obj, image_data = parser.load_image()
        
    """

    def __init__(self, file_path, params: Optional[dict]=None, load_kwargs: Optional[dict]=None):
        self.parser_type = self.__class__.__name__
        self.file_path = file_path
        self.params = params or {}
        self.load_kwargs = load_kwargs or {}

        self.error_msg = None
        self.metadata = None
        self.fatal_error = False
    
    @classmethod
    def create_parser(cls,                              # keeping here for back-compat
                      file_path: str, 
                      params: Optional[dict]=None, 
                      load_kwargs: Optional[dict]=None
    ):
        """
        Factory method to create the appropriate image parser
            based on regex patterns that match the end of the file path.
        Args:
            file_path:
            params: TODO describe, consider changing to reflect its really config used in .run()
        TODO:
            make params optional unless using in seg pipeline
        """
        return create_parser(file_path, params=params, load_kwargs=load_kwargs) 
    
    def __str__(self):
        return pprint.pformat(vars(self))
        
    @abstractmethod
    def load_image(self, load_kwargs: Optional[dict]=None):
        """Load the image from the source defined in parameters."""
        pass

    @abstractmethod
    def extract_metadata(self, image_obj):
        """Extract metadata specific to the image type."""
        pass

    @abstractmethod
    def format_prediction_input(self, img_obj, arr, ex_md, RUN_CONFIG):
        """Preprocess arr to standardized format """
        pass
    
    def maybe_input_arr_format_func(self, arr, ex_md):
        """
        allows passing a function in ex_md to reformat an array after it is read, before it is passed to prediction models
        function passed must accept 2 arguments (an array and the example metadata dict) and return only an array
        """
        if 'format_prediction_input_func' in ex_md:
            return ex_md['format_prediction_input_func'](arr, ex_md)
        return arr
    
    def _fill_empty_channel_info(self, arr_shape):
        """ check if params data_metadata channel info is empty, if so fill with generic names for shape of C axis """
        
        if self.params['data_metadata']['channel_info'] is None:
            nch = 1
            if 'C' in self.params['data_metadata']['input_image_format']:
                nch = arr_shape[self.params['data_metadata']['input_image_format'].index('C')]
            
            _chinfo = {i: f"channel_{i}" for i in range(nch)}
            self.params['data_metadata']['channel_info'] = _chinfo


    def handle_channel_info(self, arr_shape):
        """
        updates self.params['data_metadata']['channel_info'] if empty, parses channel info from filenames
        """
        # handle empty channel info 
        self.params['data_metadata']['input_image_format'] = self.params['data_metadata']['input_image_format'].upper()
        self._fill_empty_channel_info(arr_shape)
        # check if need to pattern match ch names for channel info wls/names
        self.params['data_metadata']['channel_info'] = process_channel_info(self.params['data_metadata']['channel_info'], Path(self.params['image_path']).name)


    def run(self):
        """
        Runs full data loading steps: loads the image, extracts metadata, axes formating
        returns:
            img_obj: the image object
            arr (np.ndarray): formatted image array with STCZYX axes
            self.params: the example metadata dict
        """
        img_obj, arr, self.error_msg = self.try_load_image()
        if img_obj is None:
            img_obj = arr # just a pointer

        self.params['image_metadata'] = fill_empty_metadata(self.extract_metadata(img_obj))   
        self.handle_channel_info(arr.shape)
        
        if self.error_msg:
            return False, self.error_msg
        else:
            arr, checkShape, self.error_msg = self.validate_shape(arr) 

        return img_obj, arr, self.params
    
    def try_load_image(self):
        """ attempt reading image object and fetching image array"""
        error_msg = None
        try:
            img_obj, arr = self.load_image(self.load_kwargs)
        except Exception as e:
            _emsg_base = f"error in load_image\nskipping {self.file_path}\nerror msg: "
            img_obj, arr, error_msg = None, None, f"{_emsg_base}{e}"
        return img_obj, arr, error_msg
    
    def validate_shape(self, arr):
        """check arr shape matches expected/defined input"""
        
        # whole section needs a rework, except for tranforming to target
        min_z = self.params.get('MIN_Z', None)
        valid_T = self.params.get('VALID_SHAPE_T', None)
        valid_S = self.params.get('VALID_SHAPE_S', None)
        skip_format_mismatches = self.params.get('SKIP_FORMAT_MISMATCHES', False)
        _emsg_base = f"error in validate_shape\nskipping {self.file_path}\nerror msg: "
                
        try:
            # TODO: should try to estimate format if not provided!
            print('input_image_format:', self.params['data_metadata']['input_image_format'])
            print('init_image_format_target:', self.params['image_metadata']['init_image_format_target'])
            arr = uip.transform_axes(
                arr, 
                self.params['data_metadata']['input_image_format'], 
                self.params['image_metadata']['init_image_format_target']
            )
            self.params['current_format'] = self.params['image_metadata']['init_image_format_target']
            print('transformed format (current_format):', self.params['current_format'])
            S, T, C, Z, Y, X = arr.shape
            
            # valid_shape = (T == valid_T and S == valid_S and Z >= min_z)
            valid_shape = all([
                (T >= valid_T) if valid_T is not None else True,
                (S >= valid_S) if valid_S is not None else True,
                (Z >= min_z) if min_z is not None else True
            ])
            if not valid_shape and skip_format_mismatches:
                return arr, False, f"{_emsg_base}Invalid constraints: given format:{self.params['current_format']} and T={valid_T}, S={valid_S}, Z>={min_z} << but got shape: {arr.shape}. skipping..."
            
            return arr, True, None
        except ValueError as e:
            if skip_format_mismatches:
                return None, False, f"{_emsg_base}Shape mismatch (S, T, C, Z, Y, X != arr.shape), skipping.."
            raise ValueError(e)

    
    def try_format_prediction_input(self, img_obj, arr, ex_md, RUN_CONFIG):
        """ apply common elements to arr formating"""
        # reduce specified dimensions
        arr = uip.reduce_dimensions(arr, ex_md['current_format'], take_dims=ex_md['take_dims']) # used to be arr[0,0]
        ex_md['current_format'] = update_format(ex_md['current_format'], removed_axes=ex_md['take_dims'])

        # reorder channels 
        arr = self.format_prediction_input(img_obj, arr, ex_md, RUN_CONFIG)
        
        # potentially fill missing channels if know mapping from C inds to expectations    
        
        if RUN_CONFIG['MD_TEMPLATE']['data_metadata']['channel_info']:
            d = process_channel_info(RUN_CONFIG['MD_TEMPLATE']['data_metadata']['channel_info'], Path(ex_md['image_path']).name)
        else: # feed back in present channels if not known
            # ?? this confuses me why above uses run_config and below uses ex_md
            # TODO: also implement a default way that maps channel inds to general names (e.g. ch0 -> Synaptopodin)
            # note: this may need to be created outside of this function to handle cases where n ch differs.
            d = process_channel_info(ex_md['data_metadata']['channel_info'], Path(ex_md['image_path']).name)
        exp_inds = {i:k for i,k in enumerate(sorted(d.keys()))}
        
        ch_axis = ex_md['current_format'].index('C')
        ex_md['data_metadata']['present_chs'] = list(range(arr.shape[ch_axis]))
        ex_md['data_metadata']['inserted_null_chs'] = None
        
        
        if RUN_CONFIG.get('COERCE_SHAPE', True) and (len(exp_inds) != len(ex_md['data_metadata']['present_chs'])) and len(ex_md['data_metadata']['channel_info'])>0: # must be czi
            print(ex_md['data_metadata'])
            # check if more channels in image than specified (remove channels)
            channel_info_expected = RUN_CONFIG.DATA_METADATA['channel_info']                  # !!!!!
            if len(channel_info_expected) < len(ex_md['data_metadata']['present_chs']):
                get_these_wls = list(RUN_CONFIG.DATA_METADATA['channel_info'].keys())

                present_ch_info = ast.literal_eval(ex_md['image_metadata']['ch_wavelengths'])
                present_ordered_ch_info = sorted(present_ch_info.items(), key=lambda x: x[1])
                present_ordered_chs = [el[1] for el in present_ordered_ch_info]
                # need to correct for differences in provided wls and ones found in czi (e.g. gave 568 instead of 561)
                get_these_wls_real, closest_matches, unmatched = uCzi._find_closest_and_remove(get_these_wls, set(present_ordered_chs))

                ch_names = [el[0] for el in present_ordered_ch_info if el[1] in get_these_wls_real]
                get_these_ch_inds = [present_ordered_chs.index(wl) for wl in get_these_wls_real]
                
                # update array and metadata
                ch_axis = ex_md['current_format'].index('C')
                arr = np.take(arr, get_these_ch_inds, ch_axis) # only take the channels that matched to the specified wavelengths
                ex_md['data_metadata']['present_chs'] = list(range(arr.shape[ch_axis]))
                ex_md['data_metadata']['inserted_null_chs'] = None
                
            else:
                # check if less channels in image (add empty arrays)
                expected_wl = ast.literal_eval(ex_md['image_metadata']['ch_wavelengths']) # since it comes as str, need to convert to dict
                maybe_missing_inds = uCzi.find_missing_czi_indicies(expected_wl, img_obj)
                ch_axis = ex_md['current_format'].index('C')

                if maybe_missing_inds is not None:
                    ex_md['data_metadata']['inserted_null_chs'] = maybe_missing_inds
                    ex_md['data_metadata']['present_chs'] = [i for i in exp_inds if i not in maybe_missing_inds]

                    # TODO this needs to be tested 
                    arr2 = arr.copy()
                    for i in maybe_missing_inds:
                        mpty = np.zeros_like(np.take(arr, indices=slice(0, 1), axis=ch_axis)) # new 3/18
                        arr2 = np.insert(arr2, i, mpty, axis=ch_axis)
                        # arr2 = np.insert(arr2, i, np.zeros_like(arr[0:1, ...]), axis=ch_axis) # replaced 3/18
                    arr = arr2
        
        mip_project_dims = "".join([d for d in ex_md['current_format'] if d not in 'CXY']) # project all dims except CXY when creating the mip
        arr_mip = uip.reduce_dimensions(arr, ex_md['current_format'], project_dims=mip_project_dims)
        uip.print_array_info(arr)
        return arr, arr_mip

        
        



class IMSImageParser(ImageParser):
    """
    Parser for handling .ims images.
    """

    def load_image(self, load_kwargs):
        from imaris_ims_file_reader.ims import ims
        imsFile = ims(self.file_path)
        arr = imsFile[:]
        return imsFile, arr
    
    def extract_metadata(self, imsFile):
        return {}

    def format_prediction_input(self, img_obj, arr, ex_md, RUN_CONFIG):
        return self.maybe_input_arr_format_func(arr, ex_md)
        


class TIFFImageParser(ImageParser):
    """
    Parser for handling .tiff images.
    """

    def load_image(self, load_kwargs: Optional[dict]=None):
        import tifffile
        im = tifffile.imread(self.file_path)
        return None, im
    def extract_metadata(self, im):
        return {}
    def format_prediction_input(self, img_obj, arr, ex_md, RUN_CONFIG):
        return self.maybe_input_arr_format_func(arr, ex_md)

class GeneralImageParser(ImageParser):
    """
    Parser for handling images not explicitly handled (e.g. czi, tiff, ims files).
    """

    def load_image(self, load_kwargs: Optional[dict]=None):
        import imageio
        im = imageio.v2.imread(self.file_path)
        return None, im
    def extract_metadata(self, im):
        return {}
    def format_prediction_input(self, img_obj, arr, ex_md, RUN_CONFIG):
        return self.maybe_input_arr_format_func(arr, ex_md)




def _has_bioformats() -> bool:
    """determine if bioformats_jar is installed"""
    try:
        import bioformats_jar
        return True
    except ImportError:
        return False
        
def _determine_general_parser() -> type[ImageParser]:
    """determine which general parser to use based on bioformats_jar availability"""
    if _has_bioformats():
        return AICSImageParser
    return GeneralImageParser

def get_aics_reader():
    """ determine reader for aics loader """
    if _has_bioformats():
        from aicsimageio.readers import BioformatsReader
        return BioformatsReader
    return None
    



class CZIImageParser(ImageParser):
    """
    Parser for handling .czi images.
    """
    # TODO: convert implementation to use the parser based on the official czi lib I implemented in IO.readers
    
    def read_file(self, load_kwargs: Optional[dict]=None):
        """ 
        # TODO: UPDATE object types relevant to czis
        Returns an AICSImage object, potentially using the BioformatsReader. 
            Note may fail if reading image where scenes are saved across multiple files during acquisition 
            Therefore load_image is prefered
        """ 
        from aicsimageio import AICSImage
        
        self.load_kwargs.update(load_kwargs or {})
        
        czi, scene_ids = uCzi.read_czi(self.file_path)
        
        return czi
    

    def load_image(self, load_kwargs: Optional[dict]=None):
        """ load czi object and extract image array"""
        from _aicspylibczi import PylibCZI_CDimCoordinatesOverspecifiedException
        
        try:
            czi, scene_ids = uCzi.read_czi(self.file_path)
            scene_id = self.load_kwargs.get('scene_id', 0)
            arr = uCzi.czi_scene_to_array(czi, scene_id, None, None, None, ch_last=False, rotation=None, bgr2rgb=False, moveax=None)
        
        except PylibCZI_CDimCoordinatesOverspecifiedException:
            print('caught exception for czi multifile scenes')
            czi, arr, dims = uCzi.read_czi_multifilescenes_mosiac(self.file_path, get_dims = 'STCZYX')

        
        return czi, arr
        
        
    def extract_metadata(self, czi):
        metadata = {}
        for n, f in {
            'ch_wavelengths': uCzi.get_czi_ch_wavelengths,
            'zooms': uCzi.get_czi_zooms,
            'mag': uCzi.get_czi_mag,
            'scaling' : uCzi.get_czi_scaling,
            'shape': lambda x: x.shape,
        }.items():
            try:
                res = f(czi)
                metadata[n] = str(res) #if not isinstance(res, dict) else res # convert to str unless its a dictionary
            except:
                metadata[n] = None
        return metadata

    def format_prediction_input(self, img_obj, arr, ex_md, RUN_CONFIG):
        arr = uCzi.czi_arrange_channels(img_obj, arr, channel_axis=ex_md['current_format'].index('C'))
        arr = self.maybe_input_arr_format_func(arr, ex_md)
        return arr


class AICSImageParser(ImageParser):
    """
    Optional Parser for handling images using bioformats (if bioformats_jar is installed) 
    """    
    def read_file(self, load_kwargs: Optional[dict]=None):
        """ Returns an AICSImage object, potentially using the BioformatsReader """ 
        from aicsimageio import AICSImage
        
        self.load_kwargs.update(load_kwargs or {})
        
        img_obj = AICSImage(self.file_path, reader=get_aics_reader())
        
        return img_obj
    
    def set_scene(self, img_obj):
        
        scene_id = self.load_kwargs.get('scene_id', 0)
        if hasattr(img_obj, 'set_scene'):
            try:
                img_obj.set_scene(scene_id)
            except:
                print(f'failed to set scene (scene_id: {scene_id})')
        

    def load_image(self, load_kwargs: Optional[dict]=None):
                
        # might be able to just wrap the existing czi parser since obj seem to be identical, though metadata will likely be different
        
        # Create an AICSImage object using the BioformatsReader
        img_obj = self.read_file(load_kwargs=load_kwargs)
        
        self.set_scene(img_obj)
                
        # Access the image data as numpy array (always?)
        image_data = img_obj.data # TCZYX

        return img_obj, image_data
    

    def extract_metadata(self, img_obj):
        
        # md = img_obj.metadata
        
        metadata = {}
        for n, f in {
            'ch_wavelengths': self.get_channel_wavelengths,
            'scaling' : self.get_scaling,
            'zooms': self.get_zooms,
            'mag': self.get_mag,
            'shape': self.get_shape,
            'format': self.get_format,
            # 'channels': self.get_channels,
        }.items():
            try:
                res = f(img_obj)
                if isinstance(res, tuple): # tuple's need yaml handling, so avoid
                    res = list(res)
                metadata[n] = res #if not isinstance(res, dict) else res # convert to str unless its a dictionary
            except:
                metadata[n] = None
        return metadata
        
    def get_scaling(self, img_obj):
        scaling = {}
        
        scene_index = img_obj.current_scene_index
        
        _from = img_obj.metadata.images[scene_index].pixels
        for k, attr in {dim: f'physical_size_{dim}' for dim in 'xyz'}.items():
            k = k.upper()
            try:
                unit_key = f"{attr}_unit"
                unit = getattr(_from, unit_key).value  # -> 'µm'
                scaling[k] = getattr(_from, attr)
            except:
                scaling[k] = None
        return scaling

    def get_shape(self, img_obj):
        return img_obj.dims.shape

    def get_format(self, img_obj):
        return img_obj.dims.order

    def get_channels(self, img_obj):
        scene_index = img_obj.current_scene_index
        return img_obj.metadata.images[scene_index].pixels.channels
    
    def get_mag(self, img_obj):
        return img_obj.metadata.instruments[0].objectives[0].nominal_magnification
    
    def get_zooms(self, img_obj):
        return [detec.zoom for detec in img_obj.metadata.instruments[0].detectors]
    
    # def get_formated_channels_md(self, img_obj):
    #     # seems to be nested types in (pxmd.channels[0]).__dict__
    #     # need to consider how to handle 
    
    def get_channel_wavelengths(self, img_obj, common_wavelengths: Optional[list[int]]=None) -> dict:
        """ 
        get mapping of channel names to common excitation wavelengths 
        
        args:
            common_wavelengths: if not provided defaults to [405, 488, 561, 639]    
        
        """
        common_wavelengths = common_wavelengths or [405, 488, 561, 639]
        wl_set = set(common_wavelengths)
        n_exhausted = 0 # for keeping track of channels if more provided than common wavelengths
        
        chs = self.get_channels(img_obj)
        ch_info = {}                        # -> {'EGFP-T1': 488.0, 'EBFP2-T2': 384.0, 'mCher-T2': 587.0}
        for c_i, ch in enumerate(chs):
            wl = ch.excitation_wavelength   # -> 384.0
            
            # find closest common wavelength 
            if len(wl_set) == 0:
                print('common wavelenths exhausted, appending to end')
                closest_match = max(common_wavelengths) + 1 + n_exhausted
                n_exhausted +=1
            else:
                closest_match, matched_wl, unmatched = ug.find_closest_and_remove([wl], wl_set)  # -> [405], [384.0], []
                if len(closest_match)!= 1:
                    raise ValueError(closest_match) # this shouldn't occur
                closest_match = closest_match[0]
            
            name = ch.name                  # -> 'EBFP2-T2'
            ch_info[name] = closest_match
                
        return ch_info
    
    def get_channels_inds_in_wl_order(self, img_obj) -> list[int]:
        """ returns indicies of channels in wavelength order """ 
        ch_names = img_obj.channel_names                                                        # -> ['EGFP-T1', 'EBFP2-T2', 'mCher-T2']
        ch_ordered = sorted(self.get_channel_wavelengths(img_obj).items(), key=lambda x: x[1])  # -> [('EBFP2-T2', 384.0), ('EGFP-T1', 488.0), ('mCher-T2', 587.0)]
        ch_inds = [ch_names.index(tup[0]) for tup in ch_ordered]                                # -> [1, 0, 2]
        return ch_inds
    
    def arrange_channels(self, img_obj, arr, channel_axis):
        """Reformat channels to be in order of wavelength (e.g., 405, 488, 594, 647).

        Parameters:
        img_obj : file object
            containing metadata and channel information.
        arr : ndarray
            Multi-channel image data array
        channel_axis : int, optional
            Axis index of the channels in the array (default is -1, last axis).

        Returns:
        sorted_arr : ndarray
            Image data with channels rearranged according to their wavelengths.
        """
        
        # Get indices of channels in the order they should be sorted
        reordered_channel_inds = self.get_channels_inds_in_wl_order(img_obj) # -> [1, 0, 2]
        
        # can now use np.take to reorder the channels in wavelength order 
        sorted_arr = np.take(arr, reordered_channel_inds, axis=channel_axis)
        
        return sorted_arr

            
    def format_prediction_input(self, img_obj, arr, ex_md, RUN_CONFIG):
        arr = self.arrange_channels(img_obj, arr, channel_axis=ex_md['current_format'].index('C'))
        arr = self.maybe_input_arr_format_func(arr, ex_md)
        return arr
    




# TODO: not invoked, does this need to be used anywhere?
class ImageConfigInterpreter:
    def __init__(self, config: Dict[str, Any], run_config: Optional[Dict[str, Any]] = None):
        self.config = config
        self.run_config = run_config or {}

    def resolve(self, image_path: str, arr_shape: tuple) -> Dict[str, Any]:
        resolved = self.config.copy()
        data_md = resolved.setdefault("data_metadata", {})

        fmt = data_md.get("input_image_format", "STCZYX").upper()
        data_md["input_image_format"] = fmt

        if data_md.get("channel_info") is None:
            nch = 1
            if "C" in fmt:
                c_index = fmt.index("C")
                nch = arr_shape[c_index]
            data_md["channel_info"] = {i: f"channel_{i}" for i in range(nch)}

        data_md["channel_info"] = process_channel_info(
            data_md["channel_info"], Path(image_path).name
        )

        return resolved

    def init_ex_md(self, image_i, image_path, scene_id=None, scene_name=None) -> dict:
        image_i = Path(image_path).stem if image_i is None else image_i
        assert isinstance(image_i, (int, str))
        _example_i = str(image_i).zfill(4) if isinstance(image_i, int) else image_i

        ex_md = copy.deepcopy(self.run_config['MD_TEMPLATE'])
        ex_md['COLOCALIZE_PARAMS'] = self.run_config.get('COLOCALIZE_PARAMS', None)
        ex_md['image_path'] = image_path
        ex_md['scene_id'] = scene_id
        ex_md['scene_name'] = scene_name
        ex_md['example_i'] = _example_i
        ex_md['output_dir'] = os.path.join(self.run_config['OUTPUT_DIR_EXAMPLES'], _example_i)

        for a in ['take_dims', 'project_dims']:
            ex_md[a] = self.run_config[a]


        # this should be moved to the metadata handler
        if not self.run_config['IS_NEW_RUN']:
            _ex_md = MetadataParser.try_get_metadata(ex_md['output_dir'], {'metadata': ['metadata.yml']}, silent=True)
            if len(_ex_md) > 0:
                ex_md['annotation_metadata'] = _ex_md['annotation_metadata']

        return ex_md



# def test():
if __name__ == '__main__' and bool(0):
    from SynAPSeg.utils import utils_plotting as up
    from SynAPSeg.utils import utils_general as ug
            #     'ch_wavelengths': uCzi.get_czi_ch_wavelengths,
            # 'zooms': uCzi.get_czi_zooms,
            # 'mag': uCzi.get_czi_mag,

    fp = r"D:\BygraveLab\Confocal data archive\Pascal\VHL_VglutHomer\2024_0328_VHL1_mark--VHL1-1.1--.czi"
    
    parser = create_parser(fp)
    czi_obj = parser.read_file()
    

    czi_parser = CZIImageParser(fp)
    czi_obj = czi_parser.read_file()
    czi_parser.extract_metadata(czi_obj)

    parser = AICSImageParser(fp)
    img_obj, image_data = parser.load_image()
    print(image_data.shape)
    ex_md = parser.extract_metadata(img_obj)
    print(ex_md)
    parser.get_channels(img_obj)
    
    imgmd = img_obj.metadata.images[img_obj.current_scene_index]
    pxmd = imgmd.pixels
    
    from ome_types._autogenerated.ome_2016_06.pixels import Pixels

    
    
    common_wavelengths = [405, 488, 561, 639]
    present_wls = list(parser.get_channel_wavelengths(img_obj).values())
    missing_wls = set(common_wavelengths).difference(present_wls)
    missing_inds = [common_wavelengths.index(wl) for wl in missing_wls]


    up.show_ch(image_data[0], axis=0)

    arr = parser.arrange_channels(img_obj, image_data, channel_axis=ex_md['format'].index('C'))
    up.show_ch(arr[0], axis=0)
    
    
    