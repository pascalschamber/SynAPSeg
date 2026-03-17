"""
This module handles loading examples into the analysis pipeline and manages configuration parameters

Overview
----------
• Each ExampleDispatcher is responsible for loading its own data (using an external DataLoader), running through the pipeline, and marking itself complete. 
• The DispatcherCollection wraps a list of these dispatchers so the overall analysis can iterate over each example. 
• Logging and error handling are built in, and processing time can be tracked.
"""

import gc
from ast import Call
import os
import sys
from pathlib import Path
from copy import deepcopy
from numpy import imag
import tifffile
import traceback
import yaml
import numpy as np
import traceback
from typing import Any, Tuple, Dict, Optional, Callable

from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.config.constants import STANDARD_FORMAT, DISPLAY_FORMAT, PX_UNITS_CONVERSION_TO_UM, CHANNEL_MAX
from SynAPSeg.IO.metadata_handler import MetadataParser, GroupExtractor, GroupParser
from SynAPSeg.IO.BaseConfig import BaseConfig
from SynAPSeg.common.Logging import setup_default_logger, rich_str
from SynAPSeg.IO.BaseDispatcher import DispatcherBase
from SynAPSeg.IO.project import Project, Example


# TODO ensure consistency by keeping config attributes inside config, not assigning them to the disp object

class ExampleDispatcher(DispatcherBase):
    """ Represents a single example in the analysis project.
    Attributes:
        image_path (str): Path to the example directory.
        config (dict): Configuration parameters (from ConfigManager).
        index (int): Unique index for this example.
        scene_id (optional): For multi-scene images (e.g., czi files).
        metadata (dict): Loaded metadata for the example.
        img_obj: The image object loaded by a parser.
        arr: The image data as a NumPy array.
        state (dict): Dictionary for storing intermediate results.
        status (str): Current processing status.
        log (list): Log messages for this example.
        start_time (float): Timestamp when processing started.
        end_time (float): Timestamp when processing finished.
    """
    def __init__(self, config, disp_i=None, logger=None, parse_config=True):
        self.config = config
        self.disp_i = disp_i
        self.ex_i = self.config.ex_i
        self.attach_logger(logger)

        self.exmd = None
        self.state = {}
        self.status = "pending"
        self.description = "" # append to with run specific information/metadata
        self.start_time = None
        self.end_time = None
        self.EXIT_FLAG = False # track if need to skip if error during loading 

        # load exmd
        self.path_to_example = self.config.params["path_to_example"] = os.path.join(self.config.EXAMPLES_DIR, self.ex_i) 
        self.exmd = MetadataParser.try_get_metadata(self.config.path_to_example)

        if parse_config:
            self.parse_config(self.config)

    def parse_config(self, config):
        """ parse config and relevant example metadata attributes """

        # input validation
        ###############################################################################
        assert isinstance(self.exmd, dict), f"exmd must be a dictionary, but got: {type(self.exmd)}"


        # specifically declare the config variables that are used during data loading stages
        ###############################################################################
        # assert keys must not be None
        assert_keys = [
            "OBJECTS_IMAGE_NAME",
            "INTENSITY_IMAGE_NAME",
            "FILE_MAP"
        ]
        self.validate_reqs(self.config, assert_keys, "required keys error in dispatcher.config")
        self.validate_reqs(self.exmd['data_metadata'], ['data_formats'], "required keys error in dispatcher.exmd['data_metadata']")      
        
        # handle params that are okay to be none
        self.config.params["annotated_file_patterns"] = (self.config.get('annotated_file_patterns') or []) + ['pred_.*_ch.*']        
        self.config.params["COLOCALIZATIONS"] = config.get("COLOCALIZATIONS", None)
        self.config.params["EXTRACT_GROUPS"] = self.config.get("EXTRACT_GROUPS", None)
        self.config.params['GET_CHS'] = self.config.get("GET_CHS", [])
        self.config.params['scene_name'] = self.config.get("scene_name", None)
        self.config.params['DATASHAPES'] = self.config.get("DATASHAPES") or {}

        self.exmd['COLOCALIZE_PARAMS'] = {'colocalizations':self.config.COLOCALIZATIONS}
        self.exmd["scene_id"] = self.exmd.get('scene_id', 0)
        self.exmd["scene_name"] = self.exmd.get('scene_name', None)

        # legacy exmd support
        # path name compatibility
        if self.exmd.get('image_path', None) is None:
            self.exmd['image_path'] = deepcopy(self.exmd['czi_path'])

        
        
        # validation functions
        #####################################################################
        # ensure format of filemap 
        self.validate_FILEMAP(self.config['FILE_MAP'], "dispatcher.config `FILE_MAP` validation error")

        # if data formats provided, override those in exmd
        self.handle_datashapes_formats()

        # extract pixel/voxel sizes -> self.config.params["PX_SIZES"]
        self.parse_pixel_spacing()

        # extract categorical variables defined in EXTRACT_GROUP_MAPS from the og image path
        self.config.EXTRACT_GROUPS, self.config.extracted_fn_groups = self.handle_extract_groups(self.config.EXTRACT_GROUPS)

        # extract filenames of annotated images and add to FILE_MAP
        self.get_annotated_predictions()

        # add descriptive info about this example
        exmd_ch_info = self.exmd.get('data_metadata', {}).get('channel_info', None)
        self.description += (
            f"{Path(self.config.path_to_example).stem} -- (scene_name: {self.config.scene_name}\nexmd_ch_info: {exmd_ch_info}\n"
            f"{os.path.join(*list(Path(self.exmd['image_path']).parts)[:-2])}\n" 
            f"{os.path.join(*list(Path(self.exmd['image_path']).parts)[-2:])}\n"
            f"extracted_fn_groups: {self.config.extracted_fn_groups}\n"
        )

        self.logger.debug(f"dispatcher {self.ex_i} parsed config.")

    def parse_pixel_spacing(self):
        """ 
            Populate config.params['PX_SIZES'] from either:
            - config.PX_SIZE_XY (user override), OR
            - exmd['image_metadata']['scaling'] (CZI metadata)
                note: Assumes dimension is given in meters, as that is default of zeiss microscopes czi files 

            -------------------------
            CONFIG INPUT PARAMETERS:
            • config.PX_SIZE_XY     (float or None)
            • config.PX_SIZES_UNIT  (str or None)
            • exmd['image_metadata']['scaling']  (stringified dict)
            -------------------------

            CONFIG OUTPUT PARAMETERS SET:
            • config.params['PX_SIZES'] = {'X': float, 'Y': float, 'Z': float}
            -------------------------
        """

        import ast

        # -------------------------------------------------------------
        # INPUT PARAMS READ FROM CONFIG / EXMD
        # -------------------------------------------------------------
        px_sizes = {'X':1, 'Y':1, 'Z':1} # default if scaling extraction from exmd fails 
        px_xy = self.config.get("PX_SIZE_XY")
        px_z = self.config.get("PX_SIZE_Z")
        px_unit = self.config.get("PX_SIZES_UNIT") or 'm'

        # get scaling from metadata, if it exists. this is usually a stringified dict like "{'X': 0.108, 'Y': 0.108, 'Z': 0.3}"
        scaling_raw = ((self.exmd or {}).get('image_metadata') or {}).get('scaling')

        # parse pixel size unit and convert to um - default uses czi format where unit is meters
        if px_unit not in PX_UNITS_CONVERSION_TO_UM.keys():
            raise KeyError(f"Got {px_unit} but PX_SIZES_UNIT must be a key in: {PX_UNITS_CONVERSION_TO_UM}")

        # CASE 1: User explicitly provided PX_SIZE_XY
        # -------------------------------------------------------------
        if px_xy is not None:
            px_sizes = {'X': px_xy, 'Y': px_xy, 'Z': px_z or 1.0}
            self.logger.info(f"setting PX_SIZES using ({px_sizes}) provided by config (PX_SIZE_XY, PX_SIZE_Z)")

        # CASE 2: Use metadata-derived pixel sizes
        # -------------------------------------------------------------
        else:
            try:                
                scale_factor = PX_UNITS_CONVERSION_TO_UM[px_unit]
                px_sizes= {dim: s * scale_factor for dim, s in ast.literal_eval(scaling_raw).items()}
                self.logger.info(f"setting PX_SIZES using self.exmd['image_metadata']['scaling'] ({scaling_raw}) and scale_factor ({scale_factor})")
            except Exception as e:
                self.logger.warning(f"failed to extract PX_SIZES from from exmd. using default.\nException:\n{e}")

        self.config.params["PX_SIZES"] = px_sizes
        self.logger.info(f"PX_SIZES={px_sizes}")

    
        


    def handle_datashapes_formats(self):
        # TODO rename or split into funcs that describe init setup of missing file map kwargs as well
        # allow user to spec array data formats, legacy for early examples need to manually add data formats
        
        if not self.config['DATASHAPES']:
            return

        for k,v in self.config['DATASHAPES'].items():
            self.exmd['data_metadata']['data_formats'][k] = v

    def handle_extract_groups(self, extract_groups_config: dict):
        """ handles parsing of config's extract group attributes and initializes and runs the GroupExtractor 
            uses GroupExtractor object to extract categorical variables defined in EXTRACT_GROUP_MAPS from the og image path

            Args:
                extract_groups_config (dict): 
        """
        if extract_groups_config is None:
            return None, None

        # input validation
        if isinstance(extract_groups_config, GroupExtractor): # handle situation where group extract config was already parsed into extractor object
            group_extractor = extract_groups_config

        else:  # parse extractor object from config parameters  
            assert isinstance(extract_groups_config, list)
            assert all(isinstance(d, dict) for d in extract_groups_config)
            assert all({'var', 'attrib', 'mapping'}.issubset(d.keys()) for d in extract_groups_config) # check for required_keys

            group_extractor = GroupExtractor(
                [GroupParser(d['var'], d['attrib'], d['mapping']) for d in extract_groups_config]
            )

        extracted_fn_groups = group_extractor.extract_groups(self.exmd)

        return group_extractor, extracted_fn_groups

    # main methods - called during iteration
    ###################################################################################################################
    def load(self, data_loaders: Optional[dict[str, Callable]] = None) -> dict[str, Any]:
        """
        Loads the image and metadata for this example.

        Args:
            data_loaders (None or dict(callable)): values are functions operate on dict returned by MetadataParser.read_example 
                used to inject modifications to the data after loading, keys must point to a `img_key` in this dict
        
        """
        self.start_time = ug.dt()
        self.logger.info(f"\n\nStarting dispatcher (i={self.disp_i}) | {self.ex_i} - Loading data...")

        try:
            image_dict = self.build_image_dict(data_loaders)

        except Exception as e:
            self.raise_error(e, errortype="loading") #, outdir_path=outputHandler.outdir_path)

        self.status = "loaded"
        self.logger.info(f"Example loaded successfully (elapsed time: {ug.dt()-self.start_time}).\n")
        return image_dict

    def process_example(self, pipeline, data, outputHandler):
        """
        Processes the example through the quantification pipeline.
        
        Args:
            pipeline: A Pipeline object that has a run() method accepting a data dictionary and configuration.
            data: image_dict
            outputHandler: OutputHandler object which manages writing/compiling data generated from multiple dispatchers
        
        Returns:
            A dictionary containing the updated state after processing.
        """
        self.logger.info(f"Processing dispatcher (i={self.disp_i}) | {self.ex_i} through pipeline stages...")

        try: # Run through the pipeline stages.

            data = pipeline.run(data, self.config)

            if data.get('EXIT_FLAG'):
                raise RuntimeError('EXIT_FLAG. See log for details.')

            self.status = "processed"

            # append this examples data to running aggregate
            outputHandler.place_outputs(data, self.config) 
            self.status = "complete."

        except Exception as e:
            self.raise_error(e, outdir_path=outputHandler.outdir_path)

        # log_str = rich_str(
        #     f"Processed dispatcher (i={self.disp_i}) | {self.ex_i} in {round(self.get_processing_time(), 3)} seconds.", 
        #     title='COMPLETED')
        log_str = str(f"Processed dispatcher (i={self.disp_i}) | {self.ex_i} in {round(self.get_processing_time(), 3)} seconds.\n\n")
        self.logger.info(log_str)
        
        gc.collect()
        return data

    def build_image_dict(self, data_loaders=None):
        """ 
        load data and metadata (exmd) from an example directory
            note: overrides metadata with processed version, i.e. image_dict['metadata'] = self.exmd
            note: exmd refers to attr self.exmd which is returned by MetadataParser.try_get_metadata(self.config.path_to_example)

        args:
            data_loaders: dict[str, callable], optional. 
                custom functions which adds entries to data (image_dict) passed between pipeline stages 
                must take as input 2 args; image_dict and exmd
                and return element which is to be inserted in image_dict under the key provided.
        """

        required_fns = self.config.get('required_fns', [])

        # load images
        #################################################################
        self.get_image_list = [item for sublist in self.config.FILE_MAP.values() for item in sublist]

        image_dict = MetadataParser.read_example(self.config.path_to_example, load_files=self.get_image_list, silent=True)

        # allow func pass through
        if data_loaders:
            for img_key, func in data_loaders.items():
                image_dict[img_key] = func(image_dict, self.exmd)
        
        # update exmd with current array shapes
        self.exmd['data_metadata']['data_shapes'] = {k: v.shape for k, v in image_dict.items() if hasattr(v, 'shape')}

        # override metadata with processed version
        image_dict['metadata'] = self.exmd

        self.check_required_data(required_fns, image_dict)

        # shape check + format/modify data
        image_dict = self.validate_data_formats(image_dict)
        image_dict = self.parse_annotations(image_dict)
        image_dict = self.parse_rois(image_dict)

        self.logger.info('\n'+rich_str(self.get_image_dict_info(image_dict), title='Loaded data objects'))
        
        return image_dict 
    
    def check_required_data(self, required_fns, image_dict):
        """ check if file stems (no suffix) are in image_dict, if not raise EXIT_FLAG """
        
        for fn in required_fns:
            if fn not in image_dict.keys():
                self.logger.error(f'required data key `{fn}` not found in image_dict')
                self.EXIT_FLAG = True

    def _standardize_and_sync(self, image_dict, key, config_param=None):
        """
        Helper to standardize array dims and sync metadata/config in one pass.
        """
        # Get current format and standardize
        current_fmt = self.get_current_format(key)
        data, new_fmt = uip.standardize_collapse(image_dict[key], current_fmt, STANDARD_FORMAT)
        
        # Update the image dictionary with the processed data
        image_dict[key] = data
        
        # Sync metadata (Formats and Shapes)
        self.exmd['data_metadata']['data_formats'][key] = new_fmt
        self.exmd['data_metadata']['data_shapes'][key] = data.shape
        
        # Sync specific config parameter (if requested)
        if config_param:
            self.config.params[config_param] = new_fmt
            
        return new_fmt
    
    def get_current_format(self, key):
        """
        Helper to get the current format of a data object via lookup in exmd or via shape inference.
        """
        if self.exmd['data_metadata']['data_formats'].get(key):
            return self.exmd['data_metadata']['data_formats'][key]
        else:
            return uip.estimate_format(
                self.exmd['data_metadata']['data_shapes'][key], 
                default_format=STANDARD_FORMAT, 
                channel_max=CHANNEL_MAX
            )

    def validate_data_formats(self, image_dict) -> dict:
        """ 
        check format of data objects match, collapse irrelevant dims, and parse roi types 
        """
        image_dict = self.validate_image_formats(image_dict)
        image_dict = self.validate_roi_formats(image_dict)
        return image_dict
    
    def validate_image_formats(self, image_dict) -> dict:
        """ check format of data objects match, collapse irrelevant dims """
        int_key = self.config.INTENSITY_IMAGE_NAME
        obj_key = self.config.OBJECTS_IMAGE_NAME

        # --- Phase 1: Process Main Images ---
        # Use the helper to process both images and sync their metadata
        img_fmt = self._standardize_and_sync(image_dict, int_key, config_param='img_fmt')
        obj_fmt = self._standardize_and_sync(image_dict, obj_key, config_param='obj_fmt')

        # Validate compatibility
        # TODO need to handle conversion of img -> obj format if they don't match
        if img_fmt != obj_fmt:
            raise ValueError(
                f"Formats do not match: {int_key} ({image_dict[int_key].shape}) {img_fmt} "
                f"!= {obj_key} ({image_dict[obj_key].shape}) {obj_fmt}"
            )        # update config & exmd with collapsed formats and shapes
        return image_dict
    
    def validate_roi_formats(self, image_dict) -> dict:
        """ 
        parse roi types

        TODO:  
            # TODO better incorporate formatting for rois, annotations, polygon rois, etc.
            # TODO maybe do this for annotated_files as well?
        
        """
        
        self.config.params['ROIS_FORMATS'] = []
        self.config.params['ROI_TYPES'] = []

        if self.config.FILE_MAP.get('ROIS'):
            for roi_i, fn in enumerate(self.config.FILE_MAP['ROIS']):
                
                data_key = ug.get_prefix(fn)
                if data_key not in image_dict:
                    raise ValueError(f'cannot validate roi format for data_key `{data_key}` - does not exist in image_dict (keys: {image_dict.keys()})')

                roi_type = self.parse_roi_types(image_dict, data_key)

                if roi_type == 'mask':
                    fmt = self._standardize_and_sync(image_dict, data_key)
                    
                elif roi_type == 'polygon':
                    fmt = 'YX'
                    # polys = image_dict[data_key]
                    # image_dict['polygons_per_label'] = {
                    #     i+1:[p.to_shapely()] for i,p in enumerate(polys.polygons)
                    # }

                self.config['ROIS_FORMATS'].append(fmt)
                self.config['ROI_TYPES'].append(roi_type)
        
        return image_dict
    
    def parse_roi_types(self, image_dict, data_key) -> str:
        """ 
        return roi object type.
            handled types:
            - 'mask' (np.array)
            - 'polygon' ('polycollection') 
        """
        
        is_arr = isinstance(image_dict[data_key], np.ndarray)
        if is_arr:
            return 'mask'


        from SynAPSeg.Plugins.ABBA.core_regionPoly import polyCollection
        is_polycollection = isinstance(image_dict[data_key], polyCollection)
        if is_polycollection:
            return 'polygon'

        else:
            raise ValueError(f'Unknown roi type in image_dict. Cannot parse data_key `{data_key}`')
        

    def get_annotated_predictions(self):
        """ 
        returns list of annotated filenames. discovered fns are loaded into image dict later.

        annotated files are label arrays which override data in specific channels (usually these are manually edited versions of predictions)
        note: this diverges from use in annotator where annotated files is anything created/modified by the user

        """
        
        file_patterns = self.config['annotated_file_patterns']
        fns = []
        for pattern in file_patterns:
            fullpaths = ug.get_contents(self.config.path_to_example, pattern, pattern=True, fail_on_empty=False, warn=False)
            filenames = [Path(el).name for el in fullpaths]
            fns.extend(filenames)
        
        self.config['FILE_MAP']['annotations'].extend(fns)
        return fns

    def _match_to_key(self, anFileStem, search_keys):
        """ helper func that ensures anFileStem matches to at most 1 of the search keys """
        m = [sk for sk in search_keys if sk in anFileStem]
        if len(m) == 0: 
            return None # no matches
        elif len(m) > 1:
            raise ValueError(f"Multiple matches found for {anFileStem}: {m}")
        return m[0]
    
    def _get_valid_annotations(self, image_dict) -> dict[Any, Any]:
        """ helper function to find and match annotations to parent object_image/rois. """
        self.config.params['parsed_annotated_predictions'] = []

        # filer roi data keys to include only if mask (array) type and has ch axis
        roi_data_keys = []
        for i,k in enumerate(self.config['FILE_MAP']['ROIS']):
            if (self.config.ROI_TYPES[i] == 'mask') and ('C' in self.config.ROIS_FORMATS[i]):
                roi_data_keys.append(k)

        # filter annotations to only insert if in self.config.OBJECTS_IMAGE_NAME
        search_keys = ([f"{self.config.OBJECTS_IMAGE_NAME}"] or []) + roi_data_keys
        search_keys = [] if len(search_keys) == 0 else [ug.get_prefix(k) for k in search_keys]

        get_annots = {} # dict mapping file stems to obj data keys they will insert into
        for anfn in self.config['FILE_MAP']['annotations']:
            # if anfn in self.config.FILE_MAP.get('ROIS', []): # ROIS need to be excluded from this
            #     continue
            anFileStem = ug.get_prefix(anfn)
            maybeMatch = self._match_to_key(anFileStem, search_keys)
            if maybeMatch is not None:

                insert_into_FileStem = maybeMatch
                insert_into_fmt = self.exmd['data_metadata']['data_formats'].get(insert_into_FileStem)

                if anFileStem not in image_dict.keys():
                    raise ValueError(
                        f"anFileStem `{anFileStem}` not in image_dict keys ({image_dict.keys()})"
                    )
                if insert_into_FileStem not in image_dict.keys():
                    raise ValueError(
                        f"anFileStem `{anFileStem}` matched to data obj key `{insert_into_FileStem}` not in image_dict keys ({image_dict.keys()})"
                    )
                if not insert_into_fmt:
                    raise ValueError(
                        f"data obj key `{insert_into_FileStem}` format not in exmd:{self.exmd['data_metadata']['data_formats']}"
                    )
                if "C" not in insert_into_fmt:
                    raise ValueError(
                        f"`C` not in fmt `{insert_into_fmt}` of data obj `{insert_into_FileStem}`"
                    )
                if image_dict[anFileStem].ndim != (image_dict[insert_into_FileStem].ndim - 1):
                    # annotated array should be same format but without C axis
                    raise ValueError(
                        f"{image_dict[anFileStem].ndim} != {image_dict[insert_into_FileStem].ndim - 1}"
                    )

                get_annots[anFileStem] = insert_into_FileStem

        self.config.params['parsed_annotated_predictions_to_insert'] = get_annots
        self.logger.info(f'<in parse_annotations> get_annots={get_annots}, search_keys={search_keys}')
        return get_annots

    def parse_annotations(self, image_dict):
        """ insert annotated versions of single channels within an object image, if any """
        get_annots = self._get_valid_annotations(image_dict)

        if len(get_annots) == 0:
            return image_dict

        for anFileStem, insert_into_FileStem in get_annots.items():
            ann_arr = image_dict[anFileStem]
            ann_arr_ch = MetadataParser.get_ch_from_str(anFileStem, as_int=True)

            insert_into_fmt = self.exmd['data_metadata']['data_formats'][insert_into_FileStem]
            ch_axis = insert_into_fmt.index('C')

            uip.insert_inplace(image_dict[insert_into_FileStem], ann_arr, ann_arr_ch, ch_axis)

            self.logger.info(
                f"inserted {anFileStem} in {insert_into_FileStem}\n"
                f"\tinsert shape:{ann_arr.shape}, @channel:{ann_arr_ch} | ch_axis:{ch_axis}.\n"
                f"\tinsert info:{uip.pai(ann_arr, asstr=True)}"
            )
            self.config.parsed_annotated_predictions.append(anFileStem)

        return image_dict 

    def parse_rois(self, image_dict):
        """ 
        reorg image_dict so roi arrays only exist inside rois key. only supports single roi for now.
        
        Args:
            image_dict: A dictionary containing the image data.
        Returns:
            A dictionary with the image data after parsing the rois.
        
        Config Args:
            FILE_MAP: A dictionary containing the file map.
        """
        filemap_rois = self.config.FILE_MAP['ROIS']
        if not filemap_rois:
            return image_dict
        
        # get rois
        roi_names = [ug.get_prefix(s) for s in filemap_rois]

        # verify roi in image dict
        if len(roi_names)>0:
            missing_keys = [k for k in roi_names if k not in image_dict]
            if len(missing_keys) > 0:
                raise ValueError(
                    f'roi_names `{missing_keys}` do not exist in image_dict (keys: {image_dict.keys()})\n'
                    f'these roi_names were infered from FILE_MAP ROI filenames: {filemap_rois}'
                )
            image_dict['rois'] = [image_dict.pop(k) for k in roi_names]

        return image_dict

    def get_processing_time(self):
        """
        Returns the total processing time if available, else sets it if start time is set.
        """
        if self.start_time and not self.end_time:
            self.end_time = ug.dt()
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

    def __str__(self):
        div = "="*60
        desc = f"ExampleDispatcher(index={self.ex_i}, status={self.status})\n{div}\n{self.description}\n" 
        return desc

    def __repr__(self):
        return self.__str__() 

    def get_image_dict_info(self, image_dict) -> str:
        """ returns str pprint-style info of image_dict like array data objs, fmts, and shapes"""

        # convience stuff for printing final image array data objs, fmts, and shapes
        self.image_dict_array_keys = [k for k, v in image_dict.items()
            if isinstance(v, np.ndarray) 
            or (isinstance(v, list) and all(isinstance(x, np.ndarray) for x in v))]
        _get_shape_fxn = lambda a: a.shape if isinstance(a, np.ndarray) else [aa.shape for aa in a]
        _get_size_fxn = lambda x: x.shape if hasattr(x, 'shape') else len(x) if hasattr(x, '__len__') else 'None'
        md_fmts = deepcopy(self.exmd['data_metadata']['data_formats'])
        md_fmts['rois'] = [md_fmts.get(ug.get_prefix(k)) for k in self.config.FILE_MAP.get('ROIS')]

        return (
            "\n".join(
                [f"{k}: <{md_fmts.get(k,'None')}> {_get_shape_fxn(image_dict[k])}" for k in self.image_dict_array_keys]+\
                ['-'*49] +\
                [f"{k}: <{type(v)}> size=({_get_size_fxn(v)})" for k,v in image_dict.items() if (k not in self.image_dict_array_keys)]
            )
        )
       

    def get(self, key: str, default = None):
        """Dictionary-style get method with default value"""
        if hasattr(self, key):
            return getattr(self, key)
        return default

    def raise_error(self, exc, **kwargs):
        """ re-raise a caught exception appending dispatcher info and the og traceback"""
        self.status = "error"

        tb = traceback.format_exc()
        emsg = f"{exc}\n\ntraceback:\n{tb}"
        self.logger.error(
            f"Error {kwargs.get('errortype', 'processing')} dispatcher (disp_i={self.disp_i}) | ex_i={self.ex_i}:\n{emsg}"
        )

        # append msg with re-run from this point info
        rerun_msg = (
            "To rerun from this point set:\n"
            f"\tdisp_slice = slice({self.disp_i}, None)\n"
            f"\toutputHandler.outdir_path = '{kwargs.get('outdir_path')}'"
        )

        # Re-raise the *original* exception with full traceback and added rerun msg
        new_exc = exc.__class__(f"{exc}\n\n{rerun_msg}")
        raise new_exc.with_traceback(exc.__traceback__)


class DispatcherCollection(DispatcherBase):
    """This class supports iteration, indexing, and filtering of examples.
    """
    def __init__(self, config: BaseConfig, project: Project, logger=None):
        
        self.attach_logger(logger)
        self.config = config
        self.project = project
        self.dispatchers = []
        
        self.examples_to_process = self._parse_config()
        self._create_dispatchers(self.examples_to_process)
        self._init_pipeline()
    
    def _parse_config(self):
        """ 
        check project example status and determine which examples/files to include in run
            looks through examples and pull out ones of interest. ['complete', or a list of example directory names (e.g. ['0000', ...])] 
        
        Args:
            project: A Project object.
        Config Args:
            FILE_MAP: A dictionary containing the file map.
            GET_EXS: A list of example directory names to include in the run.
        Returns:
            A list of example file paths to process.
        """
        # build file map
        FILE_MAP = self._build_FILE_MAP(self.config)
        # validate FILE_MAP structure
        self.validate_FILEMAP(FILE_MAP, "DispatcherCollection.config `FILE_MAP` validation error")
        self.config.params['FILE_MAP'] = FILE_MAP
                
        # filter annotated files by status
        example_filepath_list = self._filter_examples_status() 
    
        return example_filepath_list
    
    def _build_FILE_MAP(self, config):
        """ build FILE_MAP from config """

        FILE_MAP  = {
            "images": [config.params["INTENSITY_IMAGE_NAME"]],
            "labels": [config.params["OBJECTS_IMAGE_NAME"]],
            "ROIS": [],
            "annotations": [],
            "metadata": []
        }
        ROI_NAME = config.get("ROIS_NAME")
        if ROI_NAME is not None:
            FILE_MAP["ROIS"] = [ROI_NAME]
        
        # after filename with suffix is used to build filemap, remove suffix
        for k in ['INTENSITY_IMAGE_NAME', 'OBJECTS_IMAGE_NAME', 'ROIS_NAME']:
            val = config.params[k]
            
            if val is None:  # this is optional for ROIS_NAME, #TODO: may want to enforce others not being none at this stage
                continue

            updVal = ug.get_prefix(val)
            config.params[k] = [updVal] if k=='ROIS_NAME' else updVal
        
        return FILE_MAP
        
    
    def _filter_examples_status(self):
        """ parse user input to get examples that should be processed. GET_EXS must be one of ['complete', 'all', or a list of examples list(int or str)]"""

        emsg = f"No examples to process were found for Project `{self.project.name}`"
        
        # determine status of examples
        info_df, where_map, _, _, _, annotated_filenames_map = self.project.get_dir_progress(
            proj=self.project, 
            FILE_MAP=self.config.FILE_MAP
        )

        failed_filemap_check = info_df[info_df['filemap_check'] == False]
        files_not_found = set(ug.flatten_list(info_df['filemap_check_failures'].to_list()))
        
        # check if all failed bc of filemap check
        if len(info_df) == (info_df['filemap_check']==False).sum():
            raise ValueError(f"{emsg}\nFILE_MAP check failed to find files: {files_not_found}")
        
        if not failed_filemap_check.empty:
            from tabulate import tabulate
            self.logger.warning(
                f"some examples failed filemap check:\n{tabulate(failed_filemap_check, headers='keys')}\nFILE_MAP check failed to find files: {files_not_found}"
            )


        GET_EXS = self.config.GET_EXS
        
        if GET_EXS == 'complete':
            example_filepath_list = [Path(p).name for p in where_map['complete']]
            if len(example_filepath_list) == 0:
                raise ValueError(
                    f"{emsg}\nNo examples are marked complete"
                )

        elif GET_EXS == 'all':
            # get all that pass filemap check if applied
            example_filepath_list = info_df[info_df['filemap_check'] != False]['ex_i'].to_list()

        elif isinstance(GET_EXS, list):
            example_filepath_list = [str(el).zfill(4) for el in GET_EXS]
        else:
            raise ValueError(f"GET_EXS `{GET_EXS}` is not supported type")
        
        # remove specified exs
        EXCLUDE_EXAMPLES=self.config.get("EXCLUDE_EXAMPLES", [])
        if EXCLUDE_EXAMPLES:
            example_filepath_list = [el for el in example_filepath_list if el not in EXCLUDE_EXAMPLES]
            if len(example_filepath_list) == 0:
                raise ValueError(
                    f"{emsg}\nExcluding examples left no remaining examples to process\nEXCLUDE_EXAMPLES:{EXCLUDE_EXAMPLES}"
                )
                
        # check found
        if len(example_filepath_list) == 0:
            
            emsg = f"GET_EXS={GET_EXS}\nFILE_MAP={self.config.params['FILE_MAP']}\nexclude_exi_strs={exclude_exi_strs}\n\n{self.config}"
            print(f"No examples to process were found for Project `{self.project.name}` using parameters:\n{_params_str}")
        return example_filepath_list
    

    def _create_dispatchers(self, examples_to_process):
        """ initialize dispatcher objects """    
        errors_init_disps = []
        
        for disp_i, ex_i in enumerate(examples_to_process):
            # copy config + include annotation subsets (if present)
            cc = self.config.copy()
            cc['ex_i'] = ex_i

            try:
                disp = ExampleDispatcher(cc, disp_i, logger=self._shared_logger)
                self.dispatchers.append(disp)
            except Exception as e:
                tb = traceback.format_exc()
                errors_init_disps.append(f"Dispatcher (i={disp_i}, ex_i={ex_i}) raised error:\n{tb}")

        if len(errors_init_disps) > 0:
            raise ValueError("\n".join(errors_init_disps))
        
        self.logger.info('dispatchers initialized successfully')  
    
    def _init_pipeline(self):
        if not 'PIPELINE_STAGE_NAMES' in self.config:
            return None

        pass # TODO
          
    def __iter__(self):
        return iter(self.dispatchers)

    def __getitem__(self, index):
        return self.dispatchers[index]

    def __len__(self):
        return len(self.dispatchers)

    def filter(self, condition):
        """
        Returns a list of dispatchers that meet the given condition.
        
        Args:
        condition: A callable that takes an ExampleDispatcher and returns True/False.
        
        Returns:
        List of ExampleDispatcher objects.
        """
        return [d for d in self.dispatchers if condition(d)]


# if __name__ == '__main__': 
#     # Example usage: 
#     example_paths = [ "path/to/example1", "path/to/example2", "path/to/example3" ] 
     
#     # For demonstration purposes, we use a dummy configuration dictionary. 
#     config = BaseConfig('demo', params={"dummy_param": "dummy_value"})
#     collection = DispatcherCollection(example_paths, config) 
    
#     for dispatcher in collection: 
#         print(dispatcher)
