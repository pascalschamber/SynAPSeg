import os
from typing import Any, Dict, List

from SynAPSeg.config import constants
from SynAPSeg.Plugins.base import BasePluginFactory

PLUGINS_DIRS = [
    os.path.join(constants.SYNAPSEG_BASE_DIR, 'models', 'plugins'), # system plugins
    # os.path.join(constants.SYNAPSEG_BASE_DIR, 'Plugins') # user plugins
]
PLUGINS_DEFAULT_PARAMETERS_PATH = os.path.join(PLUGINS_DIRS[0], "model_default_parameters.yaml") # global default params that specifc modules may override
PLUGIN_BASE_CLASS = 'SegmentationModel'
REQUIRED_SIGNAL = {'__plugin_group__': 'model'}
CORE_PLUGINS = ['Stardist', 'Neurseg'] # display in this order
PLUGIN_PATTERN = '.*\.py$' # if filename.endswith(".py")

ModelPluginFactory = BasePluginFactory(
    PLUGINS_DIRS, 
    CORE_PLUGINS, 
    PLUGINS_DEFAULT_PARAMETERS_PATH, 
    PLUGIN_PATTERN, 
    PLUGIN_SIGNAL=REQUIRED_SIGNAL
)

def get_available_models() -> List[str]:
    """Returns a list of available model plugin names."""
    return list(ModelPluginFactory.PLUGINS.keys())
    

if __name__ == "__main__":
    # notes
    ######################################
    # name of each plugin will be filestem
    # e.g. for Stardist.py --> name = 'Stardist'
    
    # demo
    ######################################
    
    # example for listing available models
    for k, v in ModelPluginFactory.PLUGINS.items():
        print(k, v)


    # example for loading a model
    from SynAPSeg.IO.env import verify_and_set_env_dirs
    verify_and_set_env_dirs()

    plug = ModelPluginFactory.get_plugin(
        'Stardist',
        model_path=os.path.join(constants.USER.MODELS_BASE_DIR, 'synapsedist2D_v3.6.0.8_augAffine_smallerGrid_long'),
        in_dims_model='YX',
        out_dims_pipe='YX',
    )

