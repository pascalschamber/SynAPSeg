from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QGroupBox, QFormLayout, QTabWidget, QTextEdit, QHBoxLayout, QPushButton, QFileDialog,
    QScrollArea, QToolButton
)
from PyQt6.QtCore import Qt, pyqtSignal
import os
import sys
from pathlib import Path
import copy
from typing import Dict, List, Tuple, Optional, Any
import ast
import traceback
import uuid
from pprint import pformat

from SynAPSeg.config import constants
from SynAPSeg.IO.env import verify_and_set_env_dirs
from SynAPSeg.IO.BaseConfig import BaseConfig, update_header_spec_values, prepend_config_key 
from SynAPSeg.Segmentation.config_parser import get_schema_interpreter, get_merged_values, get_interpreter_run_config
from SynAPSeg.utils import utils_general as ug

from SynAPSeg.UI.plugins.__base import BaseApp
from SynAPSeg.UI.widgets.thread_worker import run_with_progress


# internally used constants
SEG_CONFIG_PATH = constants.SEG_CONFIG_PATH
DEFAULT_PARAMETERS_PATH = constants.SEG_DEFAULT_PARAMETERS_PATH
PLUGIN_PARAM_MAP = {'Model': 'MODEL_PARAMS'} # dict mapping plugin headings to param key


class MainApp(BaseApp):
    
    def __init__(self, state_manager):
        super().__init__(state_manager)

        # Parameters / attributes
        self.app_name = "Segmentation"
        self.default_config_path = DEFAULT_PARAMETERS_PATH
        self.PLUGIN_PARAM_MAP = PLUGIN_PARAM_MAP # dict mapping plugin headings to param key
        self.PLUGIN_CLASS_KEY = 'plugin_class' # key storing plugin module reference used by plugin factory, 
        self.plugin_factory = None
        self.config_widget = None
        self.config_object = None
        self.available_plugins = {} # dict mapping model names to module paths
        self.default_plugin_module_specs = {} # dict mapping plugin module (model classes) to default  specs 
        
        # tracked states
        self._has_config = False
        self._has_built_config = False
        self._running_in_test_mode = False              # flag to indicate if run is being executed in test params mode
        self.current_worker = None                      # currently running worker thread (created in self._run)

        # init data objects
        ##########################################
        # init the module plugin factory
        from SynAPSeg.models.factory import ModelPluginFactory
        self.plugin_factory = ModelPluginFactory
        self.available_plugins = self.plugin_factory.PLUGINS
        self.default_plugin_module_specs = self.init_plugin_attributes()
        self.interp = self.get_config_interpreter(raw=True)
        config_widget = self.make_config_widget(self.interp.get_ui_specs(unflatten=False))

        # run layout init
        self.init_layout()
        self.set_main_layout(config_widget)
        self.add_test_mode_button()
        self.post_layout()   
    
    def _on_switch_app(self):
        pass
    
    def _on_select_project(self):
        self.load_project_seg_config()

    def _on_create_project(self):
        self.create_config_entry()
        # now that config entry exists can load, filling in defaults and setting up config param widget
        self.on_select_project()
    
    
    def set_main_layout(self, config_widget):
        """ initialize the ConfigWidget"""
        self.config_layout = QVBoxLayout()
        # self.config_layout.addLayout(self.config_log_selection_widget())    
        # self.config_layout.addLayout(self.make_project_params_layout())
        
        self.set_config_widget(config_widget)
        
        self.layout.addLayout(self.config_layout)
        
        # config widget with current project's parameters, if in loaded state
        self.load_project_seg_config()
    
    def add_test_mode_button(self):
        """ add test mode button to layout """
        self.test_mode_button = QPushButton("Run in Test Mode")
        self.test_mode_button.clicked.connect(self._run_test_mode)
        self.layout.addWidget(self.test_mode_button)
    
    def _run_test_mode(self):
        """Executes main run function in test mode"""
        self._running_in_test_mode = True
        self._run()
        
        
    
    # --- interp -> run_config ---       
    def _run(self):
        """Executes segmentation process."""
        if not self.check_config_is_validated():
            return

        built_config = self.get_built_config()

        if self._running_in_test_mode:
            error_msg = inject_test_mode_params(self, built_config)
            if error_msg:
                self._running_in_test_mode = False
                from SynAPSeg.UI.widgets.dialogs import warning_dialog
                warning_dialog(self, title="RUN IN TEST MODE FAILED", text=error_msg)
                return
            
            # handle cancel
            if not built_config['image_filepaths_to_process']:
                self._running_in_test_mode = False
                return

        
        print('\n\n running segmentation with built config')
        for k,v in built_config.items():
            print(f'{k}: {v}')  
            
        # fetch config key
        CONFIG_KEY = self.state_manager.get('selected_project')
        if not CONFIG_KEY: 
            raise ValueError(f"CONFIG_KEY must be set, but got: {CONFIG_KEY}")
        
        # update seg config log with current key, params
        seg_config_path = self.state_manager.get("segmentation_config_log_path")
        if not seg_config_path: 
            raise ValueError(f"seg_config_path must be set, but got: {seg_config_path}")
        if not os.path.exists(seg_config_path): 
            raise ValueError(f"seg_config_path does not exist, got: {seg_config_path}")
        
        prepend_config_key(seg_config_path, CONFIG_KEY, built_config)
        

        from SynAPSeg.segmentation_script import main as run_seg
        print('staring segmentation worker thread..')

        # TODO cancel button doesn't work
        self.current_worker = run_with_progress(
            run_seg, 
            parent=self,
            title="Running Segmentation",
            label="Preparing dispatchers...",
            
            # Arguments passed to segmentation.main
            CONFIG_KEY= CONFIG_KEY,
        )
        self.current_worker.finished_signal.connect(self.handle_run_worker_complete)
        self._running_in_test_mode = False
    

    def get_built_config(self) -> dict:
        """ extract run config input from interp, used after widget has updated values """
        validated_config = get_interpreter_run_config(self.interp)
        return validated_config
    
    def handle_run_worker_complete(self, result):
        """ 
        Handle the output from the run worker thread.
            When run in test mode, shows results (arr, predictions, ex_md) in a napari window.
        """
        
        if result is not None:
            # assumes pipeline was run in test mode
            print('run worker complete, handling non None result..')
            viewer = spawn_napari_segmentation_viewer(*result) # arr, predictions, ex_md = result

    

    
    
    
    

    
        
    
    # # config log selection - persists between instances
    # ##################################################################    
    
    # # handle applying and saveing project params
    # def make_project_params_layout(self):
    #     self.load_project_seg_config_button = QPushButton("Load seg params")
    #     self.load_project_seg_config_button.clicked.connect(self.load_project_seg_config)
    #     # self.apply_project_seg_config_button = QPushButton("Apply seg params")
    #     # self.apply_project_seg_config_button.clicked.connect(self.apply_project_seg_config)
    #     self.save_project_seg_config_button = QPushButton("Save seg params")
    #     self.save_project_seg_config_button.clicked.connect(self.save_project_seg_config)
    #     self.set_seg_config_status = QLineEdit()
        
    #     set_seg_config_layout = QHBoxLayout()
    #     set_seg_config_layout.addWidget(self.load_project_seg_config_button)
    #     # set_seg_config_layout.addWidget(self.apply_project_seg_config_button)
    #     set_seg_config_layout.addWidget(self.save_project_seg_config_button)
    #     set_seg_config_layout.addWidget(self.set_seg_config_status)
    #     return set_seg_config_layout
        
    # # segmentation config selection 
    # def config_log_selection_widget(self):
    #     """ create a widget for getting the segmentation params log. 
    #     note this is different than getting the default parameters, as these are parameters used by specific projects 
    #     """
    #     self.seg_params_label = QLabel("Segmentation Config Log:")
    #     self.seg_params_input = QLineEdit()
    #     self.seg_params_input.setText(self.state_manager.get("segmentation_config_log_path", ""))
    #     self.seg_params_input_button = QPushButton("Select Config Log")
    #     self.seg_params_input_button.clicked.connect(self.select_seg_params_input)
                
    #     seg_params_layout = QHBoxLayout()
    #     seg_params_layout.addWidget(self.seg_params_label)
    #     seg_params_layout.addWidget(self.seg_params_input)
    #     seg_params_layout.addWidget(self.seg_params_input_button)
    #     return seg_params_layout
        
    # def select_seg_params_input(self):
    #     """ opens a file diaglog to select the segmentation config log so params can be reused as templates for other runs """
    #     file_dialog = QFileDialog()
    #     file_path, _ = file_dialog.getOpenFileName(self, "Select YAML File", "", "YAML Files (*.yaml *.yml)")

    #     if file_path:
    #         self.seg_params_input.setText(f"{file_path}")
    #         self.state_manager.set("segmentation_config_log_path", file_path)
            
   
    # def save_project_seg_config(self): # TODO
    #     seg_params = self.get_built_config()
    #     log_path = self.state_manager.get("segmentation_config_log_path")
    #     log_config = self.state_manager.get("segmentation_config_log", None)
    #     if seg_params and log_path and os.path.exists(log_path) and log_config:
    #         pass
                   
    
    # handle I/O for project's params from config 
    ##########################################################################################
    
    def load_project_seg_config(self):
        """ load and apply the config settings for the current project  """
        
        PROJECT_NAME = self.state_manager.get('selected_project', "")
        seg_config_path = self.state_manager.get("segmentation_config_log_path")

        if PROJECT_NAME and seg_config_path:             
            try: 
                #  update specs simulataneously with merged values
                SEG_CONFIG = BaseConfig(PROJECT_NAME, seg_config_path, self.default_config_path) # project params
                merged_values = get_merged_values(SEG_CONFIG)

                interp = get_schema_interpreter(self.default_config_path)
                interp.update_schema(merged_values)

                # set new interp 
                self.set_project_seg_config(interp)
            except Exception as e:
                print(f'error loading current project key ({PROJECT_NAME})\n{e}\n')
        else:
            print(f'segmentation params not set:\n  PROJECT_NAME ({PROJECT_NAME}) or seg_config_path ({seg_config_path}) is empty')
    

    def set_project_seg_config(self, interp):
        """ update the config settings with the parameters from the current project """
        
        self.set_config_interpreter(interp)
        
        # update config widget UI
        self.set_config_widget(
            self.make_config_widget(self.interp.get_ui_specs(unflatten=False))
        )
        
        self.config_widget.update_widget_layout(self.interp.get_ui_specs(unflatten=False))
        self.state_manager.set("current_status", "Project config loaded successfully.")
    
    
    def create_config_entry(self):
        """ create minimal config entry for this project, other kwargs will be auto-filled with defaults once loaded """
        
        PROJECT_NAME = self.state_manager.get('selected_project', "")
        seg_config_path = self.state_manager.get("segmentation_config_log_path")
        
        print(f"creating project config entry: {PROJECT_NAME}")
        minimal_params = {
            'PROJECT_NAME': PROJECT_NAME, 
            'MODELS_BASE_DIR': '!ENV ${MODELS_BASE_DIR}',
            'ROOT_DIR': '!ENV ${ROOT_DIR}',
            'PROJECTS_ROOT_DIR': self.state_manager.get("project_root_directory"),
            'MODEL_PARAMS': {}
        }
        prepend_config_key(seg_config_path, PROJECT_NAME, minimal_params)



# code for handling running segmentation test mode
######################################################################################
def inject_test_mode_params(parent, validated_config):
    """if run is being executed in test params mode, inject required args"""
    
    file_paths, error_msg = fetch_image_filepath_selection(parent, validated_config)
    if error_msg:
        return error_msg

    test_params = {
        'WRITE_OUTPUT': False,
        'RETURN_OUTPUT': True,
        'VALIDATE_IMAGE_FILENAME_ORDER': False, # skips filename check, which would fail since running on subset is not currently supported.
        'image_filepaths_to_process': file_paths
    }
    validated_config.update(test_params)
    print(f'injected test mode params: {test_params}')

    return None


def fetch_image_filepath_selection(parent, config:dict):
    """ parse filepaths from config, create popup for filepath selection based on current run config
    """

    def _parse_available_filepaths(config) -> tuple[list[str], str]:
        """ parse filepaths from config """
        from SynAPSeg.Segmentation.config_parser import find_image_files

        try:
            image_filepaths_to_process = find_image_files(
                config['IMG_DIR'],
                config['GET_CONTENTS_FUNCTION'],
                config['GET_FILETYPE'],
                config['GET_FILE_PATTERN']
            )
        except Exception as e:
            return [], f"Error parsing filepaths: {e}" #errormsg

        return image_filepaths_to_process, ''

    def _display_filepath_selection_dialog(parent, filepaths: list[str]) -> list[str]:
        """ create a dialog to prompt user input for filepath selection based on current run config """
        from SynAPSeg.UI.widgets.config_fields import field_widget
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox, QLabel

        # 1. Create the Dialog Popup
        dialog = QDialog()
        dialog.setWindowTitle("Select Image to Process")
        dialog.setModal(True)
        dialog.resize(400, 100) # Set a reasonable default size

        # 2. Setup Layout
        layout = QVBoxLayout()
        
        # Optional: Add a label instruction
        layout.addWidget(QLabel("Please select the file you wish to process:"))

        # 3. Render selection widget
        # We assume field_widget returns a QWidget-based object
        selection_widget = field_widget({
            'widget_type': 'selection',
            'value_options': filepaths,
        })
        layout.addWidget(selection_widget.get_widget())

        # 4. Buttons for OK/Cancel
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        
        # Connect signals to the dialog's slots
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        
        layout.addWidget(button_box)
        dialog.setLayout(layout)

        # 5. Execute Dialog and Handle Result
        # dialog.exec_() blocks until the user closes the popup
        if dialog.exec_() == QDialog.Rejected:
            # User clicked Cancel or closed the window
            return []
        
        # User clicked OK
        selected_filepaths = selection_widget.get_value()

        # 6. Normalize Output
        if not selected_filepaths:
            return []
        if not isinstance(selected_filepaths, list):
            selected_filepaths = [selected_filepaths]
            
        return selected_filepaths
    
    print('running _parse_available_filepaths..')
    available_filepaths, parsing_error_msg = _parse_available_filepaths(config)
    print('parsed available filepaths: ', available_filepaths, parsing_error_msg)
    
    if parsing_error_msg:
        return [], parsing_error_msg

    selected_filepaths = _display_filepath_selection_dialog(parent, available_filepaths)

    return selected_filepaths, ''
    
def spawn_napari_segmentation_viewer(arr, predictions, ex_md):
    """ 
    show segmentation results in a napari window 
    
    """ 
    
    import napari
    import numpy as np
    import pprint
    from SynAPSeg.utils.utils_image_processing import pai
    from SynAPSeg.Annotation.annotation_core import preproc_images_for_display, display_images
    from SynAPSeg.config.constants import STANDARD_FORMAT
    
    
    # coerce prediction data to napari viewer format 
    ########################################################################
    
    # img dict (predictions) keys are file STEMS, values are arrays
    pred_keys = list(predictions.keys())
    predictions = {f"pred_{k}.tiff":v[0] for k,v in predictions.items()}
    FILE_MAP = {
        'images':['raw_img.tiff'],
        'labels':[],
        'annotations':[],
        'ROIS':[],
        'metadata':['metadata.yml']
    }
    for k,v in predictions.items():
        print(k, pai(v,asstr=True))
        if v.dtype == np.int32:
            FILE_MAP['labels'].append(f"{k}")
        else:
            FILE_MAP['images'].append(f"{k}")
    
    
    predictions['raw_img.tiff'] = arr

    ex_md['data_metadata'] = {
        'data_shapes': {k: arr.shape for k,arr in predictions.items()}, 
        'data_formats': {k:STANDARD_FORMAT for k,arr in predictions.items()}
    }
    ex_md['FILE_MAP'] = FILE_MAP
    
    predictions['metadata'] = ex_md

    pprint.pprint(ex_md)

    # display images in napari viewer
    ########################################################################
    get_image_list = ug.flatten_list(list(FILE_MAP.values()))
    images_to_display, viewer_kwargs = preproc_images_for_display(
        ex_md, FILE_MAP, predictions, 
        get_image_list
    )
    global viewer
    viewer = napari.Viewer(**viewer_kwargs)
    display_images(viewer, images_to_display, apply_colormaps=True, set_lbl_contours=0)
    return viewer
        
    
    
if bool(0):
    verify_and_set_env_dirs()
    PROJECT_NAME = 'demo3'
    seg_config_path = SEG_CONFIG_PATH
    default_config_path = DEFAULT_PARAMETERS_PATH
    
    SEG_CONFIG = BaseConfig(PROJECT_NAME, seg_config_path, default_config_path) # project params
    SEG_CONFIG.params['MODEL_PARAMS']={}
    merged_values = get_merged_values(SEG_CONFIG)

    interp = get_schema_interpreter(default_config_path)
    interp.update_schema(merged_values)

    # set new interp 
    self.set_project_seg_config(interp)
    
    from SynAPSeg.IO.BaseConfig import load_default_config
    def_conf, def_config = load_default_config(default_config_path, **{'config_key':'mykey'})
    
    
    CONFIG_KEY = 'mykey'
    minimal_params = {
        'PROJECT_NAME': CONFIG_KEY, 
        'MODELS_BASE_DIR': '!ENV ${MODELS_BASE_DIR}',
        'ROOT_DIR': '!ENV ${ROOT_DIR}',
        'PROJECTS_ROOT_DIR': '!ENV ${PROJECTS_ROOT_DIR}'
    }
    
    prepend_config_key(seg_config_path, CONFIG_KEY, minimal_params)
    
    SEG_CONFIG = BaseConfig(CONFIG_KEY, seg_config_path, default_config_path)
        
        

    
            
            




