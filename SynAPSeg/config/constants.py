import os
from pathlib import Path
import sys



# array dimension formats
###########################################################################################
STANDARD_FORMAT = "STCZYX"
DISPLAY_FORMAT = "CZYX"
SPATIAL_AXES = "ZYX"
CHANNEL_MAX = 4            # used for inferring C dimension format using array shape if format is unspecified

# units and conversion
###########################################################################################
PX_UNITS_CONVERSION_TO_UM = {
    'm': 1e6,
    'mm': 1e3,
    'um': 1,
    'nm': 1e-3,
}

# library specific constants
###########################################################################################
# set non-path constants these get set whenever IO.env.verify_and_set_env_dirs is called
NONPATH_ENV_VARS = {
    # 'NUMBA_NUM_THREADS': '8',
    'PYDANTIC_ERRORS_INCLUDE_URL': '0',
}   
TF_FORCE_GPU_ALLOW_GROWTH = 'true' # used in segmentation_script.py



# config file - default locations
###########################################################################################
SYNAPSEG_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
USER_SETTINGS_PATH = os.path.join(SYNAPSEG_BASE_DIR, 'config', 'user_settings.yaml')

# load from user settings if exists. otherwise create and fill with demo template
SEG_CONFIG_PATH = os.path.join(SYNAPSEG_BASE_DIR, 'config', 'segmentation_config.yaml')
SEG_DEFAULT_PARAMETERS_PATH = os.path.join(SYNAPSEG_BASE_DIR, 'config', 'segmentation_default_parameters.yaml')

QUANT_CONFIG_PATH = os.path.join(SYNAPSEG_BASE_DIR, 'config', 'quantification_config.yaml')
QUANT_DEFAULT_PARAMETERS_PATH = os.path.join(SYNAPSEG_BASE_DIR, 'config', 'quantification_default_parameters.yaml')




# USER SPECIFIC ENV VARS
###########################################################################################
# these are just here to make them explicit 
# they are set based on user_settings.yaml, after running IO.env.verify_and_set_env_dirs(), 
# and can be accessed through constants.user.ROOT_DIR, etc.
class user:
    keys = ['ROOT_DIR', 'PROJECTS_ROOT_DIR', 'MODELS_BASE_DIR']
    
    @property
    def ROOT_DIR(self):
        return os.getenv('ROOT_DIR')
    @property
    def PROJECTS_ROOT_DIR(self):
        return os.getenv('PROJECTS_ROOT_DIR')
    @property
    def MODELS_BASE_DIR(self):
        return os.getenv('MODELS_BASE_DIR')

USER = user()
