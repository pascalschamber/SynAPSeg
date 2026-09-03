"""
test loading param specs and merging with user config values for specific config or all configs

"""

from SynAPSeg.IO.BaseConfig import BaseConfig, read_config
from SynAPSeg.IO.project import Project, Example
from pathlib import Path
import pandas as pd
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_plotting as up
from SynAPSeg.Quantification.factory import QuantificationPluginFactory
from SynAPSeg.models.factory import ModelPluginFactory
from SynAPSeg.config import constants
from SynAPSeg.IO.env import verify_and_set_env_dirs
verify_and_set_env_dirs()

# obj template representing param manager (e.g. UI plugin) calling plugin factory for testing purposes
class obj:
    plugin_factory = None
    default_plugin_module_specs = None
    PLUGIN_PARAM_MAP = None
           
    def __init__(self):
        pass
    
    
if __name__ == '__main__':
    
    CONFIG_KEYS = ['all', ['Traf3_KO_PFC']][0]
    CONF_TYPE = ['seg', 'quant'][1]


    CONF_TYPE_IDX = {'seg':0, 'quant':1}[CONF_TYPE]
    CONFIG_LOG_PATH = [constants.SEG_CONFIG_PATH, constants.QUANT_CONFIG_PATH][CONF_TYPE_IDX]
    CONFIG_DEFAULT_PARAMS_PATH = [constants.SEG_DEFAULT_PARAMETERS_PATH, constants.QUANT_DEFAULT_PARAMETERS_PATH][CONF_TYPE_IDX]
    PLUGIN_FACTORY = [ModelPluginFactory, QuantificationPluginFactory][CONF_TYPE_IDX]

    allconfs = read_config(CONFIG_LOG_PATH)
    print(sorted(allconfs.keys()))
    self = obj()
    self.plugin_factory = PLUGIN_FACTORY
    self.PLUGIN_PARAM_MAP = self.plugin_factory.PLUGIN_PARAM_MAP

    self.default_plugin_module_specs = {
        m: self.plugin_factory.get_plugin_default_parameters(m) for m in self.plugin_factory.PLUGINS.keys()
    }

    if isinstance(CONFIG_KEYS, str) and CONFIG_KEYS == 'all':
        CONFIG_KEYS = list(allconfs.keys())

    success_count = 0
    exceptions = []
    for CONFIG_KEY in CONFIG_KEYS:
        try:
            CONFIG = BaseConfig(CONFIG_KEY, CONFIG_LOG_PATH, CONFIG_DEFAULT_PARAMS_PATH)

            # update spec'd values with values from user's config
            merged_values = CONFIG.get_configuration()
            for k,v in self.PLUGIN_PARAM_MAP.items():    
                merged_values[k][v] = {
                    'default_value': None, 
                    'current_value': self.plugin_factory.build_spec_from_user_config(CONFIG.params[v], update_default_values=True)
                }
        except Exception as e:
            exceptions.append((CONFIG_KEY, e))
            continue
        
        success_count += 1

    print(f"Successfully processed {success_count}/{len(CONFIG_KEYS)} configuration keys.")
    if exceptions:
        print(f"\nEncountered {len(exceptions)} errors:")
        for CONFIG_KEY, error in exceptions:
            print(f"  - {CONFIG_KEY}: {error}")
    else:
        print("All configurations processed successfully.")


# # debugging Plugin module's base factory's class method: build_spec_from_user_config 
# ########################################
# CONFIG = BaseConfig(CONFIG_KEY, CONFIG_LOG_PATH, CONFIG_DEFAULT_PARAMS_PATH)
# merged_values = CONFIG.get_configuration()
# plugin_param_key = self.PLUGIN_PARAM_MAP[['Model', 'Stages'][CONF_TYPE_IDX]]
# param_values = CONFIG.params[plugin_param_key]
# update_default_values = True  # if model_config is just the widget spec don't update values, just return model_specs

# from SynAPSeg.IO.BaseConfig import update_header_spec_values, set_headers
# plugin_params_specs = {}

# for plugin_name, pparams in param_values.items():
#     # print('\n' +plugin_name)
#     # print('____________________')
#     pparams['plugin_class'] = pparams.get('plugin_class') or plugin_name
#     pparams['name'] = pparams.get('name') or plugin_name
    
#     # plugin_specs is a dict where keys are headers and values is a dict[param_key, param_spec]
#     plugin_specs = self.plugin_factory.get_plugin_default_parameters(pparams['plugin_class'])
       
#     groupped_param_headers = self.plugin_factory.get_groupped_param_headers()             
#     print(groupped_param_headers)
    
        
    
#     if update_default_values:
#         # plugin_specs = update_header_spec_values(pparams, plugin_specs, update_value_key='current_value')
#         update_value_key = 'current_value'
        
#         # update_header_spec_values
#         ############################
        
#         # can this line be replaced by --- NO it globs everything under a root
#         # byheadings_model_dict = {'root': pparams}
#         # byheadings_model_dict = set_headers(pparams) 
        
#         # def set_headers(config_dict_vals, param_names:Optional[list[str]]=None):
#         # """ set headers for all values from dict with mixed headers
#         #         this is used to convert user raw k,v params to header format - mainly putting free floating params under root header
#         #     args
#         #         param_names:
#         #             optional list of keys that are not headers - this should be provided if a params value is a dict as it would other wise be interpreted as a header
#         #     notes:
#         #         user params keys are mix of grouped params (with header) and individual params (without header)
#         #         need to determine whether each key in params is a header or a param name
#         #         individual params need to then be put into a 'root' header
#         # """
#         # input args
#         config_dict_vals = pparams
#         param_headers = groupped_param_headers
#         # func logic
#         param_headers = param_headers or []
#         byheading = {'root':{}}
#         for k,v in config_dict_vals.items():

#             if k in param_headers and isinstance(v, dict): # is a known param header
#                 byheading[k] = v
#             else:
#                 byheading['root'][k] = v # other wise gets groupped under root
            
#             # # a dict is a header if it is not in param_names
#             # _is_header = True if (isinstance(v, dict) and k not in param_names) else False
            
#             # header = 'root' if not _is_header else k
#             # if header not in byheading:
#             #     byheading[header] = {}
#             # if _is_header:
#             #     for kk,vv in v.items():
#             #         byheading[header][kk] = vv
#             # else:
#             #     byheading[header][k] = v
#         # return byheading
#         byheadings_model_dict = byheading
        
        
#         for header, param_val_pairs in byheadings_model_dict.items():
#             print(header)
#             for param_name, param_val in param_val_pairs.items():
#                 try:
#                     if plugin_specs[header][param_name]['widget_type'] == 'dict':
#                         pass
#                         # in this case 
                    
#                     plugin_specs[header][param_name][update_value_key] = param_val
                    
#                 except Exception as e:
#                     print(
#                         f"Error: {e}\nheader: {header}\n param_name: {param_name}\n update_value_key: {update_value_key}\n param_val: {param_val}\nensure parameter exists in plugin default parameters config"
#                     )
#                     raise ValueError()

#     plugin_params_specs[plugin_name] = plugin_specs

