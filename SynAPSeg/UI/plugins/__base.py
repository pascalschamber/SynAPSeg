from abc import ABC, abstractmethod
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox, QPushButton, 
    QCheckBox, QComboBox, QGroupBox, QFormLayout, QTabWidget, QTextEdit, QHBoxLayout, QFileDialog,
    QSizePolicy, 
    
)
from PyQt6.QtCore import Qt

import sys
import os
import yaml
from typing import Dict
from SynAPSeg.UI.widgets.config_widget import ParamConfigWidget
from SynAPSeg.UI.widgets.dialogs import warning_dialog
from SynAPSeg.config.param_engine.interpreter import SchemaInterpreter


class BaseApp(QWidget):
    """ base class for creating a plugin application """
    def __init__(self, state_manager):
        super().__init__()
        
        # Params set by each app
        self.app_name = ''
        
        # common args
        self.state_manager = state_manager

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.update_from_state()
        self.state_manager.settings_changed.connect(self.update_from_state)

        # params set by each app
        self.default_config_path = ''
        self.PLUGIN_PARAM_MAP = {}

        
        

    def run(self):
        """ calls child's _run() method"""
        print(f"Starting {self.app_name}...")
        self._run()
        print(f"{self.app_name} completed.")
    
    @abstractmethod
    def _run(self):
        """Each sub-app must implement this method to define its own behavior."""
        pass
    
    def init_layout(self):
        """ base class defined layout before child class implements custom layout """
        pass
    
    def post_layout(self):
        """ base class defined layout after child class implements custom layout """

        
        
        # Button to run app
        self.run_button = QPushButton(f"Run {self.app_name}")
        self.run_button.clicked.connect(self.run)
        self.layout.addWidget(self.run_button)

                
    def update_from_state(self):
        """Updates the UI when the main application's state changes."""
        # title label not used any more but kept to show how this maybe useful
        # self.title_label.setText(f"{self.get_app_name()} - Project: {self.state_manager.get('selected_project', '')}")

        pass

    
    def get_app_name(self):
        """Returns the name of the sub-app"""
        return self.app_name
    
                
    def get_examples_directory(self):
        """ returns path to project's examples data folder from state_manager, or raises warning dialog if invalid """
        project_root = self.state_manager.get("project_root_directory", None)
        project_name = self.state_manager.get("selected_project", None)

        # validation checks
        emsg = None

        if not project_root or not project_name:
            emsg = f'project root is invalid `{project_root}`' if project_root is None else f'project name is invalid `{project_name}`'
        
        else:
            path_to_examples = os.path.join(project_root, project_name, "examples")
            
            if not os.path.exists(path_to_examples):
                emsg = f'path does not exist `{path_to_examples}`'
        
        if emsg is not None:
            warning_dialog(self, "Invalid project path", emsg)
            return None


        return path_to_examples
        
    
    def on_switch_app(self):
        """ configure any ui updates when app is switched """
        print(f"on_switch_app: {self.app_name}")
        self._on_switch_app()
    
    @abstractmethod
    def _on_switch_app(self):
        pass
    
    def on_select_project(self):
        """ configure ui updates when a project is selected """
        print("selected_project: ", self.state_manager.get("selected_project"))
        self._on_select_project()
        
    # @abstractmethod
    def _on_select_project(self):
        pass

    def get_config_interpreter(self, raw=False):
        if not self.default_config_path:
            print('no default config path')
            return

        return SchemaInterpreter.from_default_params_path(
            self.default_config_path, plugin_headings=list(self.PLUGIN_PARAM_MAP.keys()), raw=raw
        )
    
    def set_config_interpreter(self, interp):
        self.interp = interp

    def init_plugin_attributes(self):
        """ discover available plugins and load thier base configurations so they can be created """

        default_plugin_module_specs = {
            m: self.plugin_factory.get_plugin_default_parameters(m) for m in self.available_plugins.keys()
        }
        
        return default_plugin_module_specs

        
    def set_config_widget(self, config_widget):
        """ set config widget object to the main layout and remove the old one """
        if not hasattr(self, 'config_layout'):
            return
        # Assume self.config_widget exists already and is in config_layout
        # track index if already exists and insert at same place 
        old_index = None
        if self.config_widget is not None:
            old_index = self.config_layout.indexOf(self.config_widget)
            self.config_layout.removeWidget(self.config_widget)
            self.config_widget.deleteLater()   # ensures cleanup
            self.config_widget = None

        self.config_widget = config_widget

        if old_index is None:
            self.config_layout.addWidget(self.config_widget)
        else:
            self.config_layout.insertWidget(old_index, self.config_widget)

    # --- interp -> config_widget ---
    def make_config_widget(self, param_specs: Dict) -> ParamConfigWidget:
        """ make a configwidget object 
            Args:
                param_specs: flattened scoped config param spec dict (e.g. {'Run Configuration.IMG_DIR': {'name': 'IMG_DIR',...}})
        """
        config_widget = ParamConfigWidget(
            param_specs,
            default_plugin_module_specs = self.default_plugin_module_specs,
            plugin_param_map = self.PLUGIN_PARAM_MAP,
            plugin_class_key = self.PLUGIN_CLASS_KEY,
            plugin_factory = self.plugin_factory,
            window_title = f"{self.app_name} Configuration",
        )
    
        config_widget.built_config.connect(self.config_is_built) # update internal state signalling run can proceed
        config_widget.config_changed.connect(self.config_is_changed)
        return config_widget
    
    def config_is_changed(self):
        self._has_built_config = False
        print('MainApp received signal config_is_changed')


    # --- config widget -> interp ---
    def config_is_built(self): # --- interp <= config_widget ---
        self._has_built_config = True
        print('MainApp received signal config_is_built. updating config interpreter values..')
        self.update_interp()
        

    def update_interp(self):
        """ update interp with widget config spec 
                invoked when config is built 
        """
        # updates specs simulataneously with merged values
        current_widget_config = self.get_config_widget_params()
        
        debugstr = '\n\n in update_interp current_widget_config:\n'
        
        for k,v in current_widget_config.items():
            debugstr += f"{k}: SPEC(current_value={v['current_value']}, ...)\n"
        print(debugstr)

        self.interp.update_schema(current_widget_config, flatten_input=False, clear_old=True)
    
    def get_config_widget_params(self):
        """ fetch config values from config_widget """
        return self.config_widget.get_config()
    
    def check_config_is_validated(self):
        if not self._has_built_config:
            warning_dialog(
                self, "Configuration Required", 
                "The configuration has not been validated.", 
                f"Please click 'Validate Configuration' before running {self.app_name}.")
            return False
        return True



    