"""
BasePluginFactory is base class for managing plugins. It handles:
    - initalizing plugin modules from their .py files 
    - loading default params from their yaml configs

Each class of plugins (e.g. models, quantification stages) should be managed by a factory which is the interface used by the pipeline/ui component to init and pass config
    - see QuantificationPluginFactory for example

each plugin should have the following attributes
    - __plugin__ var that names the class (e.g. 'ColocalizationStage') that will be instantiated
    - __parameters__ var that points to default params filename in same dir (e.g. 'stardist.yaml')
    
    # TODO: out of date, more attrs have been defined, need to update this docstring

"""
import os
import sys
from pathlib import Path
import importlib.util
from types import ModuleType
from typing import Any, Dict, List, Optional
import ast

from SynAPSeg.utils import utils_general as ug
from SynAPSeg.IO.BaseConfig import BaseConfig, read_config, merge_default_parameters # move to baseconfig class operations 

# TODO: DEPRECATE these functions once segmentation module has been converted 
def get_available_plugins(plugins_dir, module_pattern=r'^(?!__)[\w.-]+\.py$', sort_keys=[]) -> dict[str, str]:
    """get model module paths from subdirectory
    returns dict mapping model names to the path of its module (e.g. "stardist": "...SynAPSeg\models\plugins\stardist.py") 
    where each model can be loaded by key corresponding to the name of its .py file
    assume all files in this folder are models if they end with .py and do not start with "__"
    """
    plugin_module_paths = ug.get_contents(plugins_dir, module_pattern, pattern=True)
    modules = {
        Path(module_path).stem: module_path for module_path in plugin_module_paths
    }
    return ug.sort_dict_by_list(modules, sort_keys)

def get_plugin_module(plugin_module_name: str, plugin_path: str) -> ModuleType:
    """load a plugin module by importing it from a .py file path"""
    
    spec = importlib.util.spec_from_file_location(plugin_module_name, plugin_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module

# --- HELPER FUNCTIONS ---

def load_module_from_path(module_name: str, file_path: str) -> ModuleType:
    """Helper to load a module from a specific file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module # Optional: Cache in sys.modules
        spec.loader.exec_module(module)
        return module
    return None

def has_plugin_signal(file_path: str, signal_key: str, signal_value: Any) -> bool:
    """
    Parses a python file safely (without execution) to check for a global variable assignment.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Parse the source code into a syntax tree
            tree = ast.parse(f.read(), filename=file_path)
    except (SyntaxError, UnicodeDecodeError):
        # If the file isn't valid Python, it's not a valid plugin
        return False

    # Iterate ONLY over top-level nodes (global scope)
    for node in tree.body:
        # Check for standard assignment: name = value
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == signal_key:
                    if isinstance(node.value, ast.Constant): # Python 3.8+ for literals
                        if node.value.value == signal_value:
                            return True
        
        # Check for annotated assignment: name: type = value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == signal_key:
                 if node.value and isinstance(node.value, ast.Constant):
                     if node.value.value == signal_value:
                         return True
    return False

def discover_plugins(
    search_dirs: list[str], 
    plugin_signal: dict[str, Any],
    module_pattern: str = r'^(?!__)[\w.-]+\.py$'
) -> dict[str, str]:
    """
    Scans multiple directories for .py files, inspects them, 
    and returns those matching the specific signal.
    
    Args:
        search_dirs: List of folder paths to scan.
        plugin_signal: A dict of attributes the module must have. 
                       e.g. {'__plugin_group__': 'quantification'}
    """
    discovered_plugins = {}
    check_attr = '__plugin_group__' 
    assert check_attr in plugin_signal, f"plugin_signal must contain '{check_attr}'"
    expected_value = plugin_signal[check_attr]
    
    # 1. Iterate over all search paths
    for directory in search_dirs:
        if not os.path.exists(directory):
            continue
        # Get potential files
        plugin_paths = ug.get_contents_recursive(directory, pattern=module_pattern)
        
        for p_path in plugin_paths:
            try:
                path_obj = Path(p_path)
                module_name = path_obj.stem
                
                # 2. Inspect the module safely using AST
                if has_plugin_signal(p_path, check_attr, expected_value):
                    # Only NOW do we register it. 
                    # We do NOT import it yet. The Factory.load_module() will do that later when needed.
                    discovered_plugins[module_name] = p_path
                
            except Exception as e:
                print(f"Warning: Failed to inspect plugin at {p_path}: {e}")

    return discovered_plugins

def get_plugin_default_parameters_path(plugin_module, default_config_filename: str = "__parameters__"): # get_parameters
    """
    
    Args:
        plugin_module (ModuleType): loaded plugin module
        default_config_filename (str): this var is defined in respective module, is a filename of a .yaml file in the same folder as the model's .py"""
    
    assert hasattr(plugin_module, default_config_filename) # this var is defined in respective .py file, is a filename of a .yaml file in the same folder as the model's .py
    base_config_filename = getattr(plugin_module, default_config_filename)
    assert isinstance(base_config_filename, str)
    
    return os.path.join(Path(plugin_module.__file__).parent, base_config_filename)

def get_plugin_default_parameters(module, global_default_paramaters_path:str = None) -> Dict:
    
    global_cfg = [read_config(global_default_paramaters_path)] if global_default_paramaters_path else []
    module_cfg = [read_config(get_plugin_default_parameters_path(module))]
    return merge_default_parameters(global_cfg + module_cfg)


# TODO: make this the abstract base class that the segmentation and quantification modules use
class BasePluginFactory:

    def __init__(
        self,
        PLUGINS_DIRS,
        CORE_PLUGINS,
        PLUGINS_DEFAULT_PARAMETERS_PATH,
        PLUGIN_PATTERN,
        PLUGIN_SIGNAL: Dict[str, Any]   # e.g. {'__plugin_group__': 'quantification'}
    ):
        self.PLUGINS_DIRS = PLUGINS_DIRS            # dirs to search
        self.CORE_PLUGINS = CORE_PLUGINS            # plugins to load in order
        self.PLUGINS_DEFAULT_PARAMETERS_PATH = PLUGINS_DEFAULT_PARAMETERS_PATH # global default params that specifc modules may override
        self.PLUGIN_PATTERN = PLUGIN_PATTERN         # regex pattern to match plugin files
        self.PLUGIN_SIGNAL = PLUGIN_SIGNAL            # signal to match plugin files

        self.PLUGINS = discover_plugins(
            PLUGINS_DIRS, 
            plugin_signal=PLUGIN_SIGNAL, 
            module_pattern=PLUGIN_PATTERN
        )

        self._module_cache = {}

    def load_module(self, plugin_name):
        """check if already imported, otherwise look for plugin by name and load by .py path"""

        if plugin_name in self._module_cache:
            return self._module_cache[plugin_name]
        
        if plugin_name not in self.PLUGINS.keys():
            # handle model back-compat (used to be all lowercase but that interfers with imports)
            if plugin_name.lower().capitalize() in self.PLUGINS.keys():
                print(f"Warning: plugin of class `{plugin_name}` is not in the list of available plugins. Using `{plugin_name.lower().capitalize()}` instead.")
                plugin_name = plugin_name.lower().capitalize()
            else:
                raise KeyError(f"plugin of class ({plugin_name}) not found. Must be one of {self.PLUGINS.keys()}")

        module_path = self.PLUGINS[plugin_name]
        module = load_module_from_path(plugin_name, module_path)
        self._module_cache[plugin_name] = module
        return module

    def get_plugin(self, plugin_name: str, **kwargs):
        """Dynamically loads an available plugin"""

        # get plugin module
        module = self.load_module(plugin_name)

        # forward params if not set explicitly
        if 'name' not in kwargs:
            kwargs['name'] = plugin_name

        pluginObjectName = getattr(module, "__plugin__")
        assert isinstance(pluginObjectName, str)

        plugin_class = getattr(module, pluginObjectName)

        return plugin_class(**kwargs)

    def get_plugin_default_parameters(self, plugin_name: str) -> Dict:
        """ update global default cfg with plugins (if any) """
        module = self.load_module(plugin_name)
        return get_plugin_default_parameters(module, global_default_paramaters_path=self.PLUGINS_DEFAULT_PARAMETERS_PATH)

    def build_spec_from_user_config(self, param_values: Dict[str, Any], update_default_values=True):
        """ 
        builds fully spec'd param config from user's config values 
            this is invoked to build config interpreter with UI compatible model params
        
        Args:
            param_values: user's non-spec param values which provide pipeline structure (e.g. SEG_CONFIG.MODEL_PARAMS)
                example: {'stardist_3d': {'model_class': 'stardist',
                        'model_path': '...\\models/stardist3d_2025_0422_v2.420_aug_wRotate_32x128x128',
                        'in_dims_model': 'ZYX', 'out_dims_pipe': 'STCZYX'},}
            update_default_values: if True updates the returned param spec with the user-defined values
        
        Returns: a dict representing fully spec'd params 
            example: 
                {'stardist_3d': {'root': {
                    'model_path': {
                        'default_value': '',
                        'widget_type': 'path',
                        'tooltip': 'Path to directory containing the trained model',
                        'flags': ['required'],
                        'extra': {'path_type': 'directory'},
                        'group': 'root',
                        'current_value': '...\\models/stardist3d_2025_0422_v2.420_aug_wRotate_32x128x128'
                    },
                    'in_dims_model': ...
        """
        from SynAPSeg.IO.BaseConfig import update_header_spec_values
        update_default_values = True  # if model_config is just the widget spec don't update values, just return model_specs
        plugin_params_specs = {}

        for plugin_name, plugin_config in param_values.items():
            plugin_config['plugin_class'] = plugin_config.get('plugin_class') or plugin_name
            plugin_config['name'] = plugin_config.get('name') or plugin_name
            # plugin_config['output_dirname'] = plugin_config.get('output_dirname') or plugin_config['name']

            plugin_specs = self.get_plugin_default_parameters(plugin_config['plugin_class'])
            if update_default_values:
                plugin_specs = update_header_spec_values(plugin_config, plugin_specs)

            plugin_params_specs[plugin_name] = plugin_specs

        return plugin_params_specs
