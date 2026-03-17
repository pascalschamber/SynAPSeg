#!/usr/bin/env python3
"""
registry.py

Load and discover UI plugins 

Usage:
    APP_MODULES = get_plugins()
    
"""
import os
from pathlib import Path
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.config import constants

UI_DIRNAME = "UI"      
PLUGINS_FOLDER = "plugins"   
CORE_PLUGINS = ['Segmentation', 'Annotation', 'Quantification'] # display in this order

def get_plugins() -> dict[str, str]:
    """ discover available plugins and return dictionary of [module name, module path]"""
    plugins_folder = os.path.join(constants.SYNAPSEG_BASE_DIR, UI_DIRNAME, PLUGINS_FOLDER)
    
    # treat all files in this folder as plugins in they end with .py and do not start with "__"
    pattern = r'^(?!__)[\w.-]+\.py$'
    plugins = ug.get_contents(plugins_folder, pattern, pattern=True)
    modules = {
        Path(script_path).stem: script_path for script_path in plugins
    }
    # sort so core plugins appear first and in order 
    return ug.sort_dict_by_list(modules, CORE_PLUGINS)

