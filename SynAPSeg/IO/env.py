import os
import sys
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.config import constants

# TODO consider moving this whole scripts function to constants?
# as of now point is to get the first existing path listed in the contantants variable

def get_user_settings():
    """ fetch user settings from user_settings.yaml if exists, else make it using default settings """
    from SynAPSeg.IO.BaseConfig import read_config
        
    if os.path.exists(constants.USER_SETTINGS_PATH):
        return read_config(constants.USER_SETTINGS_PATH)
    else:
        from SynAPSeg.config import initial_setup
        initial_setup.create_user_settings()
        return read_config(constants.USER_SETTINGS_PATH)

    

def get_env_vars():
    """ 
    Gets dict mapping env vars defined in constants var (of same name) to list of base paths (options) pointing to locations of raw image data/projects/models. 
        Used with verify_and_set_env_dirs to fetech first existing dir 

    Returns: dict[str, str]. Has following keys, values
        SYNAPSEG_BASE_DIR: absolute path to base directory of SynAPSeg
        ROOT_DIR: absolute path to directory containing image raw data dir, used to dynamically fetch user settings in config syntax (e.g. !env ${ROOT_DIR}/data)
        PROJECTS_ROOT_DIR: absolute path to directory containing project directories
        MODELS_BASE_DIR: absolute path to directory containing model directories

    """
    user_settings = get_user_settings()
    return {
        'SYNAPSEG_BASE_DIR': constants.SYNAPSEG_BASE_DIR,
        'ROOT_DIR': user_settings['ROOT_DIR'],                     
        'PROJECTS_ROOT_DIR': user_settings['PROJECTS_ROOT_DIR'],   
        'MODELS_BASE_DIR': user_settings['MODELS_BASE_DIR'],   

    }
    

def verify_and_set_env_dirs(env_var_map=None, override=True, fail_on_error=True):
    """Verify required environment variables are set to valid paths.

    Args:
        env_var_map: dict mapping keys to paths
        override: bool. if false, skips var if already in os.environ.keys()
        fail_on_error: bool. if true, raises error if any required path does not exist. if false, continues to next var

    Raises:
        FileNotFoundError: If any required path does not exist.
    """
    env_var_map = env_var_map or {}
    BASE_MAP = get_env_vars() 
    BASE_MAP.update(env_var_map)
    
    msg = ""
    for var, path_const in BASE_MAP.items():
        if (var in os.environ.keys()) and not override: # if already defined and do not want to override
            continue
        
        resolved_path = ug.get_existant_path(path_const, fail_on_empty=False) # iterate through options, returning first existing option 
        if not resolved_path or not os.path.exists(resolved_path):
            emsg = (
                f"Environment variable {var} could not be set. "
                f"Path does not exist: {path_const}"
            )
            if fail_on_error: 
                raise FileNotFoundError(emsg)
            else: 
                print(emsg)
                continue

        os.environ[var] = resolved_path
        msg += f"\t\'{var}\': \'{resolved_path},\'\n"

    for k,v in constants.NONPATH_ENV_VARS.items():
        os.environ[k] = v

    print("✅ All environment variables verified and set.\n"+ msg)