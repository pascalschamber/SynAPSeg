import os
from SynAPSeg.IO.BaseConfig import read_config, write_config 
from SynAPSeg.config import constants
from SynAPSeg.utils import utils_general as ug

def create_user_settings():
    """ initialize default env vars and user settings file, if they don't exist, using the template configs """

    # build configs for segmentation and quantification to run the demos
    ROOT_DIR = os.path.dirname(constants.SYNAPSEG_BASE_DIR)
    os.environ['ROOT_DIR'] = constants.SYNAPSEG_BASE_DIR
    os.environ['PROJECTS_ROOT_DIR'] = ug.verify_outputdir(os.path.join(ROOT_DIR, "demo", "projects"), makedirs=True)
    os.environ['MODELS_BASE_DIR'] = ug.verify_outputdir(os.path.join(ROOT_DIR, "Models"))

    template = read_config(os.path.join(constants.SYNAPSEG_BASE_DIR, 'config', "config_templates.yaml"))
    for k,v in template.items():
        config_path = os.path.join(constants.SYNAPSEG_BASE_DIR, 'config', f"{k}.yaml")
        if os.path.exists(config_path):
            print('config already exists at', config_path)
            continue
        write_config(v, config_path)

    # template requires env vars to be set:
    if os.path.exists(constants.USER_SETTINGS_PATH):
        print('user settings already exists at', constants.USER_SETTINGS_PATH)
        user_settings = read_config(constants.USER_SETTINGS_PATH)
    else:   
        ROOT_DIR = os.path.dirname(constants.SYNAPSEG_BASE_DIR)
        user_settings = {
            'ROOT_DIR': [ROOT_DIR],                     
            'PROJECTS_ROOT_DIR': [ug.verify_outputdir(os.path.join(ROOT_DIR, "demo", "projects"), makedirs=True)],   
            'MODELS_BASE_DIR': [ug.verify_outputdir(os.path.join(ROOT_DIR, "Models"))],   
        }
    
        print(
            'created default user settings at', constants.USER_SETTINGS_PATH,
            '\nwith the following values:\n\t', user_settings
        )
        write_config(user_settings, constants.USER_SETTINGS_PATH)

    return user_settings



if __name__ == "__main__":
    create_user_settings()





