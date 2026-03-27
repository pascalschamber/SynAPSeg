"""
abstract base class for a config object 
- functions to read a yaml file, extract the config for a specific project, and then can pass these to other modules
- each module will implement thier own specific version as needed


# TODO - add a schema validator
"""

import os
import sys
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Any
from collections.abc import MutableMapping
import re
import pprint
import copy 
import ast

from SynAPSeg.utils import utils_general as ug
from SynAPSeg.config import constants


# helper functions for reading and writing yaml files
################################################################################################
class TaggedValue:
    """ simple object to hold the tag value so the Dumper knows it's not a normal string """
    def __init__(self, value, tag):
        self.value = value
        self.tag = tag
    
    def __str__(self):
        return f"!{self.tag} {self.value}"
    
def read_config(config_path: str, raw=False) -> dict:
    class LocalLoader(yaml.SafeLoader):
        pass

    def raw_constructor(loader, tag_suffix, node):
        # Captures the raw text and the tag name (e.g., !ENV)
        value = loader.construct_scalar(node)
        return str(TaggedValue(value, tag_suffix))
    
    def python_object_constructor(loader, tag_suffix, node):
        # 1. Determine the type of data (dict, list, or string)
        if isinstance(node, yaml.SequenceNode):
            data = loader.construct_sequence(node)
        elif isinstance(node, yaml.MappingNode):
            data = loader.construct_mapping(node)
        else:
            data = loader.construct_scalar(node)
        return TaggedValue(data, tag_suffix)


    if raw:
        # We use add_multi_constructor with '!'. 
        # This catches tags like !ENV and passes 'ENV' as tag_suffix
        LocalLoader.add_multi_constructor('!', raw_constructor)
    else:
        LocalLoader.add_constructor('!ENV', BaseConfig._env_constructor)
        LocalLoader.add_constructor('tag:yaml.org,2002:python/tuple', BaseConfig._tup_constructor)
    
    # Register the multi-constructor for the specific python namespace
    LocalLoader.add_multi_constructor(
        'tag:yaml.org,2002:python/', 
        python_object_constructor
    )

    with open(config_path, 'r') as file:
        return yaml.load(file, Loader=LocalLoader)


def flow_list_representer(dumper, data):
    """Render lists in flow style - key: [a, b, c] """
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)


def tagged_representer(dumper, data):
    """ This tells PyYAML: "Take this TaggedValue and write it as !tag value" """
    return dumper.represent_scalar(f'!{data.tag}', data.value)

def auto_tagger(obj):
    """ Helper to find strings starting with '!' and wrap them as a TaggedValue """
    if isinstance(obj, dict):
        return {k: auto_tagger(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [auto_tagger(i) for i in obj]
    elif isinstance(obj, str) and obj.startswith('!'):
        # Pattern: matches '!TAG' then everything after the space
        match = re.match(r'!(\w+)\s+(.*)', obj)
        if match:
            tag_name = match.group(1)
            content = match.group(2)
            return TaggedValue(content, tag_name)
    return obj

def write_config(data: dict, output_path: str, width=None) -> None:
    
    class LocalDumper(yaml.SafeDumper):
        pass

    # Register the representer for our wrapper class
    LocalDumper.add_representer(TaggedValue, tagged_representer)
    LocalDumper.add_representer(list, flow_list_representer)

    tagged_data = auto_tagger(data)

    if width is None:
        from numpy import inf
        width = inf

    with open(output_path, 'w') as file:
        yaml.dump(tagged_data, file, Dumper=LocalDumper, default_flow_style=False, indent=2, width=inf, sort_keys=False)


def prepend_config_key(config_path, key, value):
    """ insert a key-value pair at the top of the config file, deletes existing key if present """
    config = read_config(config_path, raw=bool(1))
    if key in config.keys():
        del config[key]
    new_config = {key: value}
    new_config.update(config)
    
    write_config(new_config, config_path)





# config class 
################################################################################################
class BaseConfig(MutableMapping):
    def __init__(self, config_key, config_path=None, default_parameters_path=None, params=None, **kwargs):
        """Object for managing reading and parsing parameters from a .yaml file.
            supports object and dictionary-style syntax for attribute access.

        Args:
            config_key (str, or None): The key identifying which section of the config to load.
            config_path (str): Path to the configuration file. If None and params not directly provided, raises ValueError
            default_parameters_path (Optional[str]): path to .yaml file holding default parameters that will be set if not specified in config
                if None, this functionality is skipped.
                note if trying to init from default_params alone, ensure you pass params={'config_key': ''}, then read from that path
            params (dict): dictionary containing the configuration attributes 
                used for copying to set params directly without reloading 
            kwargs (dict): additional arguments which get set as attributes during init, 
                allows side-stepping the empty params check at the bottom.
        """
        self.params = params or {}
        self.config_key = config_key
        self.config_path = config_path             
        self.default_parameters_path = default_parameters_path            

        # set kwargs
        for k,v in kwargs.items():
            setattr(self, k, v)
        
        # define functions to parse resolvable parameters, # list of (pattern (str), replacement function (callable) and resolving function which takes patter, replacer, and value as input )
        # TODO extend to handle non string values e.g. expressions like size = ${customObject}(${height})
            # instead handling object parsing would be sufficient 'obj' = *{'customObject'}, 'obj_input' = value
        self.resolvers = [
            # Replace all pattern matches in the string with the value returned by the replacer # ex: "${basedir}\${filename}" --> self.basedir\self.filename
            # handle internal variable lookup/parsing: ${var}
            (re.compile(r'\$+\{([^}]+)\}'), self.string_var_replacer, lambda pattern, replacer, value: pattern.sub(replacer, value)), 
        ]

        if not self.params: # to prevent reloading on copy
            self._load_config()
        

    def try_get_env_var(self, env_var):
        if env_var in os.environ:
            return os.environ[env_var]
        return None
    

    def _load_config(self):
        """ 
        read config from file, resolve variable references, and unspecified default parameters
            if self.config_key is not empty only load params under that key, 
            else load everything 
        
        """
        if self.config_path is None:
            raise ValueError('config_path is None, init with path to quantification_config.yaml')
        if not os.path.exists(self.config_path):
            raise ValueError(f'config_path not found ({self.config_path}), init with path to quantification_config.yaml')
        
        configs = self._read_config(self.config_path)
        
        try:
            self.params = configs if not self.config_key else configs[self.config_key] # get the params for the specified project
        except KeyError as e:
            msg=f"{self.config_key} not found in config.\navailable keys:\n\t" + \
                "\n\t".join(list(configs.keys()))
            raise ValueError(msg) from e
        
        self.resolve_variable_references(self.params)
        self.resolve_unspecified_default_parameters(self.default_parameters_path, self.params)


    def _read_config(self, config_path) -> Dict:
        """ loads the entire config .yaml which contains params for mulitple projects (deprecated, need to update since general method now)"""
        # Register custom ENV tag constructor
        return read_config(config_path)

    def _write_config(self, config_path, config_dict=None):
        """ write config_dict to yaml file - if config_dict==None uses self.params"""
        config_dict = config_dict or self.params or {}
        write_config(config_dict, config_path)        

    def __getattribute__(self, key):
        if key in ('__dict__', '__class__', '__deepcopy__', '_load_config', '_read_config', 'resolve_variable_references'):
            return super().__getattribute__(key)

        params = super().__getattribute__('params')
        if key in params:
            return params[key]

        return super().__getattribute__(key)

    def copy(self):
        """ Create a blank instance without calling __init__"""
        cls = self.__class__
        # Manually copy all relevant attributes
        config_key = copy.deepcopy(self.config_key)
        config_path = copy.deepcopy(self.config_path)
        params = copy.deepcopy(self.params)

        return cls(config_key=config_key, config_path=config_path, params=params)
        

    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to mirror attribute access"""
        if key in self.params:
            return self.params[key]
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)
    

    def __setitem__(self, key: str, value: Any) -> None:
        """Enable dictionary-style assignment to mirror attribute assignment"""
        if key in self.params:
            self.params[key] = value
        else:
            setattr(self, key, value)

    def __setattr__(self, key, value):
        # Special case during initialization where 'params' isn't set yet
        if key == "params" or "params" not in self.__dict__:
            super().__setattr__(key, value)
        elif key in self.__dict__.get("params", {}):
            self.__dict__["params"][key] = value
        else:
            super().__setattr__(key, value)

    def __delitem__(self, key: str) -> None:
        """Enable dictionary-style deletion"""
        delattr(self, key)

    def __iter__(self):
        """Enable iteration over attributes excluding built-in and private ones"""
        return iter([attr for attr in self.params])

    def __len__(self) -> int:
        """Return number of public attributes"""
        return len([attr for attr in self.params])

    def get(self, key: str, default: Any = None) -> Any:
        """Dictionary-style get method with default value"""
        if key in self.params:
            return self.params[key]
        return getattr(self, key, default)
    
    def __str__(self):
        formatted_string = ''
        for k in sorted(list(self.params.keys())):
            formatted_string += (f'- {k} ({type(self.params[k]).__name__}) --> {self.params[k]}.\n')
        return formatted_string
    
    def __repr__(self):
        return self.__str__()
    
    def to_dict(self):
        """ note that this only return self.params and not other class attributes """
        return self.params
    
    def tabulate(self, p=True, tableformat='fancy_grid'):
        tbl = ug.dict_to_tabulated_rows(self.params, tableformat=tableformat)
        if p:
            print(tbl)
        else:
            return tbl
    
    @staticmethod
    def _env_constructor(loader, node):
        """Custom YAML constructor for environment variables"""
        value = loader.construct_scalar(node)
        pattern = r'\$\{([^}]+)\}'
        match = re.match(pattern, value)
        
        if match:
            env_var = match.group(1)
            env_value = os.getenv(env_var)
            if env_value is None:
                raise ValueError(f"Environment variable '{env_var}' is not set")
            return value.replace(f"${{{env_var}}}", env_value)
        raise ValueError(f"Could not parse the environment variable in '{value}'")

    @staticmethod
    def _tup_constructor(loader, node):
        """ Custom constructor for python tuples. node is a SequenceNode; convert it to a Python tuple"""
        seq = loader.construct_sequence(node)
        return tuple(seq)
        
    def is_resolvable_string(self, pattern, value):
        """ checks if string contains a resolvable pattern """
        return (isinstance(value, str) and re.search(pattern, value))
    
    
    def string_var_replacer(self, match):
        """ Replace all ${VAR} in the string with the attribute or param value """
        var_name = match.group(1)
        try:
            val = getattr(self, var_name)
            return str(val)
            # return val
        except AttributeError:
            raise KeyError(f"Variable '{var_name}' not found in config attributes.")
    

    def get_resolver_result(self, resolver, pattern, replacer, v):
        """ try ast.literal_eval on string returned by resolver """
        val = resolver(pattern, replacer, v)
        try: 
            return ast.literal_eval(val)
        except:
            return val

        
    def apply_resolvers(self, value):
        """ apply resolvers over a value, iterativly trying those defined in self.resolvers, 
                otherwise returning the input if not match the patterns """
        for (pattern, replacer, resolver) in self.resolvers:
            if self.is_resolvable_string(pattern, value):
                return self.get_resolver_result(resolver, pattern, replacer, value)
            elif isinstance(value, dict):
                new_dict = {}
                for k,v in value.items():
                    if self.is_resolvable_string(pattern, v):
                        new_dict[k] = self.get_resolver_result(resolver, pattern, replacer, v)
                    else:
                        new_dict[k] = v
                return new_dict
        return value
        
    
    def resolve_variable_references(self, config_dict):
        """Resolve variable references in the configuration parameters.

        This method scans all string values in `self.params` for variable
        references in the format `${VAR_NAME}` and replaces them with the
        corresponding attributes of the `BaseConfig` object. This supports
        nested references and is useful for templated configuration strings.

        Example:
            If `self.params` contains:
                {
                    "EXAMPLES_BASE_DIR": "/data/examples",
                    "EXAMPLE_PROJ": "my_project",
                    "EXAMPLE_PATH": "${EXAMPLES_BASE_DIR}/${EXAMPLE_PROJ}"
                }
            After calling this method:
                self.params["EXAMPLE_PATH"] == "/data/examples/my_project"
        """
        
        if self.resolvers:
            for key, value in config_dict.items():
                config_dict[key] = self.apply_resolvers(value)

    
    def resolve_unspecified_default_parameters(self, default_parameters_path, config_dict):
        """ loads and looks through a .yaml file containing default arguments, 
            these are set to thier default values if not already set
        """
        self.default_parameters_set = []
        if default_parameters_path is None:
            return None
        if not os.path.exists(default_parameters_path):
            return None
        
        conf = self._read_config(default_parameters_path)
        for k,v in conf.items():
            if (k not in config_dict) and isinstance(v, dict):
                self.default_parameters_set.append(k)
                try:
                    config_dict[k] = self.apply_resolvers(v['default_value'])
                except Exception as e:
                    raise KeyError(f"{e}\n{v}")


    def get_configuration(self):
        """ 
        returns self.params organized under headers defined in the param specs, 
            default values are replaced if present
            used for parsing config widget layout in UI.segmentation
        """
        
        configObj, config_dict = load_default_config(self.default_parameters_path, **self.params)
        byheading = group_by_heading(config_dict)
        
        new_dict = {}
        for heading, parameters in byheading.items():
            if heading not in new_dict:
                new_dict[heading] = {}
            for key, paramdict in parameters.items():
                new_dict[heading][key] = paramdict['default_value']
        return new_dict


def make_config_entry(default_parameters_path, **init_params) -> dict:
    params = {'config_key': 'template', **init_params}
    conf = BaseConfig(params.get('config_key'), None, default_parameters_path=default_parameters_path, params=params)
    config = conf._read_config(default_parameters_path)
    return config

def load_default_config(default_parameters_path, **init_params) -> tuple[BaseConfig, dict]:
    """ 
    init a config object from default params path only 
        may need to define env variables as in example below 
    Returns:
        conf - BaseConfig: resolved parameters holding default values
        config - dict: param specs
    """
        
    params = {'config_key': 'template', **init_params}
    conf = BaseConfig('template', None, default_parameters_path=default_parameters_path, params=params)
    
    config = conf._read_config(default_parameters_path)
    
    # TODO separate default_values from current_values <- curr should be updated vals from user config otherwise the default values
    # update config with overriden params
    for k,v in conf.params.items():
        if k in config: 
            config[k]['default_value'] = v
    # update conf with default config values
    for k,v in config.items():
        conf.params[k] = v['default_value']
    # now can resolve variable references
    conf.resolve_variable_references(conf.params)
    
    # now that values have been resolved, can re-insert resolved values back into config
    for k,v in conf.params.items():
        if k in config: config[k]['default_value'] = v

    return conf, config

def group_by_heading(config_dict):
    """ reorganize the dictionary by grouping items under their 'heading' """
    grouped = {}
    for key, value in config_dict.items():
        heading = value.get('heading', 'Uncategorized')
        is_param_spec = ('default_value' in value)
        if heading not in grouped:
            grouped[heading] = {}
        grouped[heading][key] = value
        if is_param_spec:
            if 'current_value' not in value:
                grouped[heading][key].update({"current_value": None})
            
    return grouped


def get_all_values(config_dict, p=False):
    """ view all key, default values """
    d = {}
    for key, attrs in config_dict.items():
        val = attrs['default_value'] if (isinstance(attrs, dict) and 'default_value' in attrs) else attrs
        d[key] = val
    
        if p:
            print(f"- {key} ({type(val).__name__}) --> {val}")
    return d


# UI socket
##################################################################
# used by UI config widgets to build widget fields 
# the UI uses these fxns to interact with the baseconfig object
# i have them here to show how this is done but they live in the UI module normally.

def merge_default_parameters(default_parameter_cfgs: List[Dict]):
    """ successivly update configuration based on order of passed BaseConfig objects and organize by header """
    byheading = {}
    for conf in default_parameter_cfgs:
        set_params(byheading, conf)
    return byheading
        
# def set_params(byheading, config_dict):
#     """ update default specs to be organized by header as header may only be provided as a param spec attribute and not actually under it's header
#         - only works with full spec params (has default value, widget_type, etc.)
#         - this fxn is not used for updating default params with user config values (that is done by update_header_spec_values)
#     """
#     # TODO this is where group gets converted into 'header' attr

#     for k,v in config_dict.items():
#         assert isinstance(v, dict), f"{k} {v} (type: {type(v)} not allowed)"
#         header = 'root'
#         _nested_categories = True if 'default_value' not in v else False
#         if _nested_categories:
#             header = k
#             for kk,vv in v.items():
#                 assert 'default_value' in vv, f"{kk}: {vv} format incorrect"
#                 if header not in byheading:
#                     byheading[header] = {}
#                 vv['group'] = header
#                 vv['current_value'] = None
#                 byheading[header][kk] = {}
#                 for kkk, vvv in vv.items():
#                     byheading[header][kk][kkk] = vvv
#         else:
#             if header not in byheading:
#                 byheading[header] = {}
#             v['group'] = header
#             vv['current_value'] = None
#             byheading[header][k] = {}
#             for kk, vv in v.items():
#                 byheading[header][k][kk] = vv
def set_params(byheading, config_dict):
    """
    Update `byheading` (dict) with fully-specified parameter specs from `config_dict`.

    - If an entry has no 'default_value', we treat it as a category (nested header)
      whose children are parameter specs.
    - If an entry has 'default_value', we treat it as a top-level parameter under 'root'.
    - Ensures each param gets 'group' (its header) and 'current_value' (default: None).
    """
    for k, v in config_dict.items():
        assert isinstance(v, dict), f"{k}: value must be dict, got {type(v)}"

        # If no 'default_value' at this level, it's a nested category (header)
        is_nested_category = ('default_value' not in v)

        if is_nested_category:
            header = k
            for kk, vv in v.items():
                _add_param(byheading, header, kk, vv)
        else:
            # Flat param under the implicit 'root' header
            _add_param(byheading, 'root', k, v)

def _add_param(byheading: dict, header: str, name: str, spec: dict):
    """ helper function for set_params """
    # Validate spec shape for a parameter
    assert isinstance(spec, dict), f"{header}.{name}: spec must be dict, got {type(spec)}"
    assert 'default_value' in spec, f"{header}.{name}: missing 'default_value' in spec"

    # Ensure header bucket exists
    if header not in byheading:
        byheading[header] = {}

    # Copy to avoid mutating caller's dict; fill missing fields
    out = dict(spec)
    out.setdefault('group', header)
    out.setdefault('current_value', None)

    byheading[header][name] = out


def set_headers(config_dict_vals):
    """ set headers for all values from dict with mixed headers
            this is used to convert user raw k,v params to header format - mainly putting free floating params under root header
    """
    byheading = {}
    for k,v in config_dict_vals.items():
        _is_header = True if isinstance(v, dict) else False
        header = 'root' if not _is_header else k
        if header not in byheading:
            byheading[header] = {}
        if _is_header:
            for kk,vv in v.items():
                byheading[header][kk] = vv
        else:
            byheading[header][k] = v
    return byheading

def is_param_spec(element, check_keys=['default_value', 'widget_type', 'tooltip']):
    """ returns true if element is a fully-spec'd param by matching all check_keys (e.g. has default value, widget_type, etc.) """
    if not element:
        return False
    return (
        isinstance(element, dict) and
        all(k in element for k in check_keys)
    )


def is_triple_nested_dict(obj, *, exact: bool = False) -> bool:
    # Top → dict; values → dict; their values → dict.
    # exact=False (default): allows deeper nesting beyond 3 layers.
    # exact=True: enforces exactly 3 dict layers (4th layer must be non-dict).
    return ug.is_nested_dict(obj, 3, exact=exact)

def is_dict_of_dicts(element):
    return isinstance(element, dict) and len(element)>0 and all(isinstance(v, dict) for v in element.values()) # w/o len check {} would return True here 

def is_header_spec(element, check_keys=['default_value', 'widget_type', 'tooltip']):
    """ check if element contains a dict of param specs """
    if not element:
        return False
    return (
        is_dict_of_dicts(element) and 
        all(k in v for k in check_keys for v in element.values())
    )
def validate_header_spec_format(config_dict, check_keys=['default_value', 'widget_type', 'tooltip', 'current_value'][:-1]):
    """ check e.g. model configuration is fully formatted with headers and fully-spec'd params """
    return all(is_header_spec(v, check_keys) for v in config_dict.values())

def update_header_spec_values(model_params, model_configuration_spec, update_value_key='current_value'):
    """ update values in fully spec'd config form 'raw' model_params (e.g. from seg_config.yaml)
        e.g. update value with those specified in user's config
        returns fully spec'd config with default values updated 
        previously update_value_key='default_value', so just updated the default value
    """
    # user input has mixed headings (no root header) --> need to convert to byheading fmt 
    byheadings_model_dict = set_headers(model_params)
        
    # update config values with user's values
    assert is_dict_of_dicts(byheadings_model_dict)

    try:
        for header, param_val_pairs in byheadings_model_dict.items():
            for param_name, param_val in param_val_pairs.items():
                # model_configuration_spec[header][param_name]['default_value'] = param_val 
                model_configuration_spec[header][param_name][update_value_key] = param_val
    except Exception as e:
        raise ValueError(
            f"Error: {e}\n"
            f"header: {header}\n param_name: {param_name}\n update_value_key: {update_value_key}\n param_val: {param_val}\n"
            "ensure parameter exists in plugin default parameters config" 
        )
        
    assert validate_header_spec_format(model_configuration_spec)
    updated_model_specs = model_configuration_spec
    return updated_model_specs 



# example - loading the default parameters from a config 
###################################################################################
if bool(0):
    from SynAPSeg.IO.env import verify_and_set_env_dirs
    from SynAPSeg.config import constants
    verify_and_set_env_dirs()

    C, conf = load_default_config(
        constants.SEG_DEFAULT_PARAMETERS_PATH,
    )
    byheading = group_by_heading(conf)
    dv = get_all_values(C.params, p=True)
    # print(ug.dict_to_tabulated_rows(dv))


# extra stuff to include eventually
# example schema validation using jsonschema
###################################################################################
if bool(0):

    import yaml
    from pathlib import Path
    from collections.abc import Mapping

    # filling of default params
    # Load default UI config
    ui_defaults_path = Path("ui_project_defaults.yaml")
    with open(ui_defaults_path) as f:
        ui_defaults = yaml.safe_load(f)

    def flatten_config_structure(config):
        """Flatten nested config for easier comparison, using parameter keys only."""
        flat = {}
        for section, params in config.items():
            for key, val in params.items():
                flat[key] = val
        return flat

    # Flatten the UI default structure to just parameter keys and default values
    flat_defaults = {
        key: val["default_value"] for key, val in flatten_config_structure(ui_defaults).items()
    }

    def autofill_user_config(user_config, defaults):
        """Autofill missing keys in user config with default values."""
        filled_config = defaults.copy()
        filled_config.update(user_config)  # user_config values take precedence
        return filled_config

    # Example usage:
    ##########################################################
    # Pretend the user only provided a few overrides
    example_user_config = {
        "EXAMPLES_BASE_DIR": "R:/UserData/Project",
        "WRITE_OUTPUT": True,
        "FILE_MAP": {
            "MIPS": ["my_image.tiff"]
        }
    }

    # Autofill it
    autofilled = autofill_user_config(example_user_config, flat_defaults)

    # Save to file
    autofilled_path = Path("autofilled_config.yaml")
    with open(autofilled_path, "w") as f:
        yaml.dump(autofilled, f)

    autofilled_path.name



    # schema validator using jsonschema
    ######################################################
    """
    other Python Libraries Often Used:
    jsonschema — great for YAML/JSON validation.
    pydantic — popular in Python apps (like FastAPI).
    Cerberus or voluptuous — simpler alternatives.
    """
    
    import jsonschema
    import json

    # Generate a JSON schema from the UI defaults
    def generate_json_schema(ui_config):
        schema = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False
        }

        for section, fields in ui_config.items():
            for key, info in fields.items():
                field_type = info["widget_type"]
                schema_type = {
                    "str": "string",
                    "int": "integer",
                    "float": "number",
                    "bool": "boolean",
                    "list": "array",
                    "dict": "object",
                    "directory": "string",  # assume path as string
                }.get(field_type, "string")

                schema["properties"][key] = {
                    "type": schema_type,
                    "description": info["tooltip"]
                }

                # Make required if default_value is not None
                if info["default_value"] is not None:
                    schema["required"].append(key)

        return schema

    # Load UI config from file again
    with open("ui_project_defaults.yaml") as f:
        ui_config = yaml.safe_load(f)

    # Generate schema
    generated_schema = generate_json_schema(ui_config)

    # Save to file
    schema_path = Path("project_config_schema.json")
    with open(schema_path, "w") as f:
        json.dump(generated_schema, f, indent=2)

    schema_path.name

    # example
    ########################
    import yaml
    import json
    from pathlib import Path

    # Re-load UI default config (after reset)
    ui_defaults_path = Path("ui_project_defaults.yaml")


    # Build the schema manually using UI defaults format
    def parse_widget_type_to_jsonschema(widget_type):
        return {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "directory": "string",
        }.get(widget_type, "string")

    # Simulate UI-config schema (just a subset of full one for simplicity)
    ui_config_schema = {
        "type": "object",
        "properties": {
            "EXAMPLES_BASE_DIR": {"type": "string", "description": "Path to examples."},
            "EXAMPLE_PROJ": {"type": "string", "description": "Load params with this variable."},
            "PX_SIZE_XY": {"type": "number", "description": "Define size of pixel in micrometers (μm)."},
            "WRITE_OUTPUT": {"type": "boolean", "description": "If false, files are not saved."},
            "OUTPUT_DIR_BASE": {"type": "string", "description": "Output directory — defaults to project directory if null."},
            "INTENSITY_IMAGE_NAME": {"type": "string", "description": "Name of image used for intensity measurements."},
            "OBJECTS_IMAGE_NAME": {"type": "string", "description": "Name of image used for object (label) extraction."},
            "OBJECTS_IMAGE_SIZE_RANGE": {"type": "array", "description": "Minimum and maximum allowed object size (in pixels)."},
            "GET_CHS": {"type": "array", "description": "Extract synapses from these channels only."},
            "GET_EXS": {"type": "string", "description": "Options: 'complete', 'all', or list of specific example folder names."},
            "EXCLUDE_EXAMPLES": {"type": "array", "description": "Skip example directories with these names."}
        },
        "required": ["EXAMPLES_BASE_DIR", "EXAMPLE_PROJ", "WRITE_OUTPUT"],
        "additionalProperties": False
    }

    # Save schema to file
    schema_path = Path("/mnt/data/project_config_schema.json")
    schema_path.write_text(json.dumps(ui_config_schema, indent=2))
    schema_path.name


    # Usage example
    from jsonschema import validate
    import yaml, json

    with open("user_config.yaml") as f:
        config = yaml.safe_load(f)

    with open("project_config_schema.json") as f:
        schema = json.load(f)

    validate(instance=config, schema=schema)
    # This will raise a jsonschema.exceptions.ValidationError if something is wrong — super helpful for catching bugs before processing!
