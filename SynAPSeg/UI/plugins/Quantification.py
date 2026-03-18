from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTextEdit, QLineEdit
import os
import sys
from pathlib import Path
from pprint import pformat

from SynAPSeg.config import constants
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.IO.BaseConfig import prepend_config_key, read_config, BaseConfig
from SynAPSeg.UI.plugins.__base import BaseApp


# TODO: pull project info to populate options in e.g. intensity image name, objects image name 
CONFIG_PATH = constants.QUANT_CONFIG_PATH
DEFAULT_PARAMETERS_PATH = constants.QUANT_DEFAULT_PARAMETERS_PATH
PLUGIN_PARAM_MAP = {'Stages': 'STAGE_PARAMS'} # dict mapping plugin headings to param key


class MainApp(BaseApp):
    def __init__(self, state_manager):
        super().__init__(state_manager)

        # Parameters
        self.app_name = "Quantification"
        self.default_config_path = DEFAULT_PARAMETERS_PATH
        self.CONFIG_PATH = CONFIG_PATH           # path to user config file which stores logged runs
        self.PLUGIN_PARAM_MAP = PLUGIN_PARAM_MAP # dict mapping plugin headings to param key
        self.PLUGIN_CLASS_KEY = 'plugin_class' # key storing plugin module reference used by plugin factory, 
        self.plugin_factory = None
        self.config_widget = None
        self.config_object = None
        self.available_plugins = {} # dict mapping model names to module paths
        self.default_plugin_module_specs = {} # dict mapping plugin module (stage classes) to default specs 
        
        # tracked states
        self._has_config = False
        self._has_built_config = False
        self.config_key_list = self.get_config_keys() # list of config keys

        # init data objects
        ##########################################
        # init the module plugin factory
        from SynAPSeg.Quantification.factory import QuantificationPluginFactory
        self.plugin_factory = QuantificationPluginFactory
        self.available_plugins = self.plugin_factory.PLUGINS

        # discover available plugins and load thier base configurations so they can be created
        self.default_plugin_module_specs = self.init_plugin_attributes()
       
        # initialize the schema interpreter
        self.interp = self.get_config_interpreter(raw=True)
       
        # make the config widget
        config_widget = self.make_config_widget(self.interp.get_ui_specs(unflatten=False))

        # run layout init
        self.init_layout()
        self.set_main_layout(config_widget)
        self.post_layout()
        self._on_select_project()
        
    def _on_switch_app(self):
        self._on_select_project()

    def _on_select_project(self):
        current_project = self.state_manager.get("selected_project", "")
        self.config_key_widget.setText(current_project)
        self.config_widget.set_widget_value("General Project Settings.EXAMPLE_PROJ", current_project)
        # self._on_select_config_key()
    
    def _on_select_config_key(self):
        # if self.get_config_key() in self.config_key_list:
        self.load_config_from_key()
    
    def _on_select_config_path(self):
        self.config_key_list = self.get_config_keys()
            


    # --- interp -> run_config ---
    def _run(self):
        """Executes segmentation process."""
        if not self.check_config_is_validated():
            return
        
        config_key = self.get_config_key()

        print(f'running in {self.app_name} with config key {config_key}...')
        
        built_config = self.get_built_config()

        # save config
        self.save_config(built_config)

        # run the quantification pipeline
        from SynAPSeg.quant_script import main
        result = main(
            config_key, 
            config_path=self.CONFIG_PATH, 
            default_parameters_path=self.default_config_path, 
            dispatchers_slice=None, 
            outdir_path=None
        )


    def get_config_key(self):
        """ get the config key from the config key widget """
        return self.config_key_widget.text()


    def get_config_keys(self):
        """ get the list of config keys from the config file """
        try: 
            return list(read_config(self.CONFIG_PATH).keys())
        except:
            return []
    
    
    def load_config_from_key(self):
        """ load config from config file using the config key """
        
        CONFIG_KEY = self.get_config_key()
        
        CONFIG = BaseConfig(CONFIG_KEY, self.CONFIG_PATH, self.default_config_path)
        
        # extract user's set values otherwise uses default value
        merged_values = CONFIG.get_configuration()

        # handle plugin's default values and rekey like param spec
        for k,v in self.PLUGIN_PARAM_MAP.items():    
            merged_values[k][v] = {
                'default_value': None, 
                'current_value': self.plugin_factory.build_spec_from_user_config(CONFIG.params[v], update_default_values=True)
            }

        interp = self.get_config_interpreter(raw=True)

        interp.update_schema(merged_values)

        self.set_config_interpreter(interp)

        self.config_widget.update_widget_layout(self.interp.get_ui_specs(unflatten=False))

        
    def save_config(self, built_config):
        """ write the built config to the config file, prepending the config key and overwriting existing key """
        prepend_config_key(self.CONFIG_PATH, self.get_config_key(), built_config)
        self.config_key_list.append(self.get_config_key())


    def get_built_config(self):
        """ extract run config input from interp, used after widget has updated values """
        return self.interp.to_run_config()
    
    
    def select_config_key_widget(self):
        """ make a widget for selecting/creating the config key """
        project_name = self.state_manager.get("selected_project", None)
        if project_name is None:
            project_name = ""

        the_date = ug.get_datetime()
        default_key = f"{project_name}"

        self.config_key_widget = QLineEdit(default_key)
        
        layout = QHBoxLayout()
        layout.addWidget(QLabel("Select/create config key:"))
        layout.addWidget(self.config_key_widget)

        layout2 = QVBoxLayout()
        layout2.addLayout(layout)
        load_key_button = QPushButton("Load Config Key")
        load_key_button.clicked.connect(self._on_select_config_key)
        layout2.addWidget(load_key_button)
        
        return layout2
        

    def set_main_layout(self, config_widget):
        """ initialize the this apps layout, including the ConfigWidget"""
        
        # add widget for selecting/creating the config key
        self.layout.addLayout(self.select_config_key_widget())
        
        # set the config widget as current layout
        self.config_layout = QVBoxLayout()
        self.set_config_widget(config_widget)        
        self.layout.addLayout(self.config_layout)
        


        

