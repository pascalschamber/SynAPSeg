"""
this file handles the parsing of the config file, and setup required to run the pipeline
"""

from typing import Dict, List, Optional, Any
import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
import copy
import re
import sys

from SynAPSeg.utils import utils_general as ug
from SynAPSeg.IO.project import Project
from SynAPSeg.config import constants
from SynAPSeg.IO.BaseConfig import BaseConfig
from SynAPSeg.config.param_engine.interpreter import SchemaInterpreter


class ConfigParser:
    """ handles setting up config for run and sets values that do not need to be initialized in config"""
    def __init__(self, config, OUTPUT_DIR_PROJ=None, project_object=None):
        self.config = config
        self.OUTPUT_DIR_PROJ = OUTPUT_DIR_PROJ or config.OUTPUT_DIR_PROJ
        self.validated_image_order = False

        if project_object:
            self.project = project_object
            assert isinstance(self.project, Project)
        else:
            try:
                self.project = Project(self.OUTPUT_DIR_PROJ)
            except:
                self.project = None

        self.parse_config()
    
    def get_config(self):
        return self.config
    
    def parse_config(self):
        """ parse which img files to include in run using get_contents function and filter strings defined in config"""
        
        # auto set attributes
        self.config.IS_NEW_RUN = self._check_is_new_run() # TODO this ignores configs IS_NEW_RUN, making it useless
        
        # setup output directory (project folder)
        ug.verify_outputdir(self.config.OUTPUT_DIR_EXAMPLES, makedirs=True)
        
        # get image paths
        self.config.image_filepaths_to_process = self._find_image_files()

        # validate input order matches existing image filename order
        self.validated_image_order = self._validate_input_paths_order()

        # create md template from config
        self._build_md_template()

        # finish config setup - model outdirs and md_template
        self._setup_model_params()

    def _check_is_new_run(self):
        """ check if project already exists """
        # TODO: this could be done more sophistically by checking if any files exist in the project directory
        return self.project is None
    
    def _validate_input_paths_order(self):
        """ check if the order of CONFIG.image_filepaths_to_process matches the order of image paths for the examples 
            raises assertion error if not 
            raise ValueError if cannot determine matching due to e.g. non-unique filenames, this skips the check
        
        """
        if self.config.IS_NEW_RUN:
            return None
        if not self.config.VALIDATE_IMAGE_FILENAME_ORDER:
            return None
        
        try:
            # get existing paths from examples
            existing_image_paths = [ex.get_image_path() for ex in self.project.examples]
            existing_filenames = [Path(fn).name for fn in existing_image_paths]
            # get paths to process from self.config.image_filepaths_to_process
            input_image_filenames = [Path(fn).name for fn in self.config.image_filepaths_to_process]

            if len(set(existing_filenames)) != len(existing_filenames):
                raise ValueError('cannot determine order due to non unique exising image filenames. passing...')
            if len(set(input_image_filenames)) != len(input_image_filenames):
                raise ValueError('cannot determine order due to non unique filenames in config.image_filepaths_to_process. passing...')
            
            # handle case where run was interupted before all examples were generated
            if len(input_image_filenames) > len(existing_filenames):
                input_subset = input_image_filenames[:len(existing_filenames)]
                non_existing_subset = input_image_filenames[len(existing_filenames):]

                # check if these are in the same order if not, reorder them 
                if input_subset == existing_filenames:
                    return True
                
                reordered = self._correct_image_order(input_subset, existing_filenames)
                reordered = reordered + non_existing_subset
                self.config.image_filepaths_to_process = reordered
                print('successfully corrected image order')
                return True
            
            elif len(input_image_filenames) < len(existing_filenames): 
                # TODO process only subset of input files
                raise AssertionError(
                    "running on a subset of examples is not currently supported."
                    f"len(input_image_filenames) < len(existing_filenames): {len(input_image_filenames)} < {len(existing_filenames)}"
                )
                
            diffsym = set(existing_filenames).symmetric_difference(set(input_image_filenames))
            diff_og = set(existing_filenames) - set(input_image_filenames)
            diff_curr = set(input_image_filenames) - set(existing_filenames)
            assert len(diffsym) == 0, f"old not in new: {diff_og or None} | new not in old: {diff_curr or None}"
            assert len(existing_filenames) == len(input_image_filenames), f"{len(existing_filenames)} != {len(input_image_filenames)}"

        except ValueError as e:
            print(e)
            return None
        
        if not input_image_filenames == existing_filenames:
            reordered = self._correct_image_order(input_image_filenames, existing_filenames)
            self.config.image_filepaths_to_process = reordered
            print('successfully corrected image order')
        return True

    def _correct_image_order(self, input_image_filenames, existing_filenames):
        """ reorder the input list to match the existing order """
        all_match, reordered_filepaths = [], []
        for i, el in enumerate(existing_filenames):
            cfni = input_image_filenames.index(el)
            cfn = input_image_filenames[cfni]
            cpath = self.config.image_filepaths_to_process[cfni]
            reordered_filepaths.append(cpath)
            all_match.append(cfn==el)
        assert all(all_match)
        assert len(reordered_filepaths) == len(set(reordered_filepaths))
        assert all([os.path.exists(fp) for fp in reordered_filepaths])
        
        return reordered_filepaths
              

    def should_skip_file(self, example_metadata: Dict) -> bool:
        """Determine if a file should be skipped based on configuration
         TODO: where is the right place for this function, if still used??
        """
        if getattr(self.config, 'FORCE_NO_SKIP', False):
            return False
            
        if getattr(self.config, 'SKIP_COMPLETED', True):
            output_dir = example_metadata['output_dir']
            if os.path.exists(output_dir):
                if (Path(output_dir) / 'complete.txt').exists():
                    print(f"Skipping {Path(example_metadata['image_path']).stem} because it is marked complete.")
                    return True
        return False
    
    def _build_md_template(self):
        """Build the metadata template for the project"""
        channel_info = self.config.get('channel_info', {})
        input_image_format = self.config.get('input_image_format')
        self.config.params['MD_TEMPLATE'] = build_md_template(channel_info, input_image_format)

    def _setup_model_params(self) -> None:
        """Setup model parameters output directories"""
        # init model params template for each example
        for model_name, params in self.config.MODEL_PARAMS.items():
            if params is None:
                continue
           
            dir_name = params.get('output_dirname') or model_name
            params['output_dir'] = os.path.join(self.config.OUTPUT_DIR_EXAMPLES, dir_name)
            self.config.MD_TEMPLATE['segmentation_pipeline_config'][model_name] = {'model_params':params}
    
    def _find_image_files(self):
        """ search a directory for image files based on config parameters """
        image_filepaths_to_process = self.config.image_filepaths_to_process
        
        if not image_filepaths_to_process:
            image_filepaths_to_process = find_image_files(
                self.config.IMG_DIR,
                self.config.GET_CONTENTS_FUNCTION,
                self.config.GET_FILETYPE,
                self.config.GET_FILE_PATTERN
            )
        
        if self.config.get('SHUFFLE_IMAGE_PATHS', False):
            shuffle_list(image_filepaths_to_process, seed=self.config.get('SEED', 42))

        return image_filepaths_to_process


# functional
####################################################

def build_md_template(channel_info, input_image_format):

    MD_TEMPLATE = dict(
        data_metadata={
            "channel_info": channel_info,
            "data_shapes": {},
            "data_formats": {},
            "input_image_format": input_image_format,
        },
        image_path=None,
        image_metadata={},  # set by image_parser class
        annotation_metadata={"notes": "", "status": ""},
        COLOCALIZE_PARAMS={"colocalizations": []},
        segmentation_pipeline_config={},
    )
    return MD_TEMPLATE


def find_image_files(image_dir, search_function, filetype, pattern) -> list:
    """Find image files based on configuration"""

    
    # Get all files of specified type 
    dir_search_fxn = getattr(ug, search_function)
    
    image_filepaths_to_process = dir_search_fxn(image_dir, filetype=filetype)      
    
    # Apply pattern filtering if specified
    if pattern is not None:
        does_contain = not pattern.startswith('~')
        pattern = pattern[1:] if not does_contain else pattern
        image_filepaths_to_process = [
            f for f in image_filepaths_to_process 
            if (bool(re.search(pattern, Path(f).name)) == does_contain)
        ]
    
    return image_filepaths_to_process

def shuffle_list(alist, seed=42) -> None:
    """ shuffle the a list in place, e.g. for blinding by randomizing order of files """
    import random
    random.seed(seed)
    random.shuffle(alist)


# ui socket functions
#######################################################


def init_model_attributes(self):
    """ discover available models and load thier base configurations so they can be created """
    from SynAPSeg.models.factory import ModelPluginFactory
    # TODO generalize as 'plugins'
    available_models = ModelPluginFactory.PLUGINS
    
    model_config_specs = {}
    for m in available_models:
        model_config_specs[m] = ModelPluginFactory.get_plugin_default_parameters(m)
    model_config_specs = model_config_specs

    
    return available_models, model_config_specs

def interpret_run_config(PROJECT_NAME, default_params_path:Optional[str]=None, seg_config_path:Optional[str]=None):
    """ build seg config using schema interpreter """
    default_params_path = default_params_path or constants.SEG_DEFAULT_PARAMETERS_PATH
    seg_config_path = seg_config_path or constants.SEG_CONFIG_PATH
    # init config interpreter with fully spec'd params
    interp = get_schema_interpreter(default_params_path)
    
    #  update specs simulataneously with merged values
    SEG_CONFIG = BaseConfig(PROJECT_NAME, seg_config_path, default_params_path) # project params
    merged_values = get_merged_values(SEG_CONFIG)
    interp.update_schema(merged_values)
    
    # get config as dict 
    SEG_CONFIG.params = get_interpreter_run_config(interp)
    return SEG_CONFIG

def get_schema_interpreter(default_params_path:str):
    """ get schema interpreter from default params path """
    default_params_path = default_params_path or constants.SEG_DEFAULT_PARAMETERS_PATH
    interp = SchemaInterpreter.from_default_params_path(default_params_path, plugin_headings=['Model'])
    return interp

def get_merged_values(SEG_CONFIG:BaseConfig):
    """
    update dynamic parameters (e.g. MODEL_PARAMS) with param specs that have user's value if set
        these specs will get passed to widget config field
        strategy is to:
        1) load user values 
        2) use user model config values to get model param specs invidually, compile and pass 
            as current value of spec.model.model_params
    """

    # extract user's set values otherwise uses default value
    merged_values = SEG_CONFIG.get_configuration()

    # handle plugin's default values and rekey like param spec
    from SynAPSeg.models.factory import ModelPluginFactory    
    merged_values['Model']['MODEL_PARAMS'] = {
        'default_value': None, 
        'current_value': ModelPluginFactory.build_spec_from_user_config(SEG_CONFIG.MODEL_PARAMS, update_default_values=True)
    }
    
    return merged_values

def get_interpreter_run_config(interp: SchemaInterpreter) -> dict:
    """ coerce interpreter schema to run config dict """
    return interp.get_run_config(
        to_replace={
            'Run Configuration.':'', 'Data.':'', 'Metadata.':'', 'Model.':'', 'current_value.':'', 'root.':''
        }
    )
