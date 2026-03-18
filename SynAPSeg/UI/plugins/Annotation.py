from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QStackedWidget, QWidget, QVBoxLayout, QPushButton, 
    QFileDialog, QLabel, QLineEdit, QHBoxLayout, QComboBox, QTextEdit, QSizePolicy,QFormLayout,
)
from PyQt6.QtCore import Qt
import os
import sys
from pathlib import Path

from SynAPSeg.UI.plugins.__base import BaseApp
from SynAPSeg.UI.widgets.config_fields import field_widget
from SynAPSeg.IO.project import Project
from SynAPSeg.IO.metadata_handler import MetadataParser
from SynAPSeg.UI.widgets.dialogs import warning_dialog

class MainApp(BaseApp):
    def __init__(self, state_manager):
        super().__init__(state_manager)

        # Parameters
        self.app_name = "Annotation"
        
        # run layout init
        self.init_layout()
        
        # run module specific layout
        ############################
        
        # Dropdown for selecting an example the project directory
        self.example_folders_dropdown = QComboBox()
        self.example_folders_dropdown.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed))
        self.example_folders_dropdown.addItem("Select an example")
        self.example_folders_dropdown.currentIndexChanged.connect(self.update_selected_example)
        
        # select an example to load
        example_layout = QHBoxLayout()
        example_layout.addWidget(QLabel("Select an example"))#, alignment=Qt.AlignmentFlag.AlignTop)
        example_layout.addWidget(self.example_folders_dropdown)
        self.layout.addLayout(example_layout)

        # add annotation kwarg widgets
        self.run_kwarg_layout = QFormLayout()
        self.layout.addLayout(self.run_kwarg_layout)
        self.run_kwarg_widgets = {}
        # self.add_annotation_kwargs_widgets()
        
        # # Fetch examples
        # self.populate_example_folders()
        
        self.post_layout()

    def populate_example_folders(self):
        """Populates the dropdown with files from the selected project root directory."""
        self.example_folders_dropdown.clear()
        self.example_folders_dropdown.addItem("Select an example")
        exdir = self.get_examples_directory()
        if exdir and Path(exdir).exists():
            files = os.listdir(exdir)
            self.example_folders_dropdown.addItems(files)
            
    def update_selected_example(self):
        selected_example = self.example_folders_dropdown.currentText()
        if selected_example != 'Select an example':
            self.state_manager.set("selected_example", selected_example)
    
    def add_annotation_kwargs_widgets(self):
        dir_examples = self.get_examples_directory()
        PROJ_PATH = Path(dir_examples).parent if dir_examples else None
        
        # clear widgets
        while self.run_kwarg_layout.count():
            item = self.run_kwarg_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.run_kwarg_widgets = {}

        if PROJ_PATH:
            self.run_kwarg_widgets = build_annotation_params_widget(PROJ_PATH)
            for k, w in self.run_kwarg_widgets.items():
                self.run_kwarg_layout.addRow(QLabel(k), w.get_widget())
                w.value_changed.connect(lambda: print(f'value changed for {k}'))
        
            
    
    def _on_switch_app(self):
        self.populate_example_folders()
        self.add_annotation_kwargs_widgets()
    
    def _on_select_project(self):
        self.populate_example_folders()
        self.add_annotation_kwargs_widgets()


    def _run(self):
        """Executes Annotation process."""
        
        EXAMPLE_I = self.state_manager.get("selected_example", None)
        if not EXAMPLE_I:
            warning_dialog(self, "Invalid example", "please select an example first")
            return 
        

        dir_examples = self.get_examples_directory()
        PROJ_PATH = Path(dir_examples).parent if dir_examples else None
        
        if not PROJ_PATH:
            warning_dialog(self, "Invalid project", "please select a project first")
            return 

        # parse UI args
        run_kwargs = parse_annotation_params_widgets(self.run_kwarg_widgets)
        include_only = run_kwargs.get("include_only", None)
        exclude = run_kwargs.get("exclude", None)
        add_to_file_map = run_kwargs.get("add_to_file_map", None) #{'ROIS': ["dends_filt.tiff"]},
        fail_on_format_error = run_kwargs.get("fail_on_format_error", False)
        set_lbl_contours = run_kwargs.get("set_lbl_contours", 0) # if 1 will show lbls with 1px border

        
        from SynAPSeg.Annotation.annotation_IO import load_example_images
        from SynAPSeg.Annotation.annotation_core import create_napari_viewer

        project = Project(PROJ_PATH)
        ex = project.get_example(EXAMPLE_I)
        LABEL_INT_MAP, FILE_MAP, image_dict, get_image_list = load_example_images(
            ex,
            include_only=include_only,
            exclude=exclude,
            fail_on_format_error=fail_on_format_error,
            get_label_int_map=False, # currently has some issues with if raw_img format is not found. not-implemented/used
            use_prefix_as_key=False,
        )
        exmd, path_to_example = image_dict.pop('metadata'), ex.path_to_example
        # create napari viewer
        viewer, widget_objects = create_napari_viewer(
            exmd, 
            path_to_example, 
            FILE_MAP, 
            image_dict, 
            get_image_list=get_image_list,
            LABEL_INT_MAP=LABEL_INT_MAP,
            set_lbl_contours=set_lbl_contours,
        )



def build_annotation_params_widget(PROJ_PATH):
    """ build widgets for user input of parameters for annotation"""
    if not PROJ_PATH:
        return None, {}

    project = Project(PROJ_PATH)
    # ex = project.get_example(EXAMPLE_I)
    # exmd = ex.get_metadata()
    all_fns = sorted(list(project.get_all_unique_filenames('.*\.tiff?')))

    
    widgets = {
        'include_only': field_widget(dict(default_value=None, value_options=all_fns, widget_type='multi-selection', tooltip='',)),
        'exclude': field_widget(dict(default_value=None, value_options=all_fns, widget_type='multi-selection', tooltip='',)),
        # 'add_to_file_map': field_widget(dict(default_value=None, value_options=all_fns, widget_type='multi-selection', tooltip='',)),
        # 'fail_on_format_error': field_widget(dict(default_value=False, value_options=None, widget_type='checkbox', tooltip='',)),
    }
    
    
    return widgets

def parse_annotation_params_widgets(widgets):
    """ parse widgets for user input of parameters for annotation"""
    params = {}
    for k, w in widgets.items():
        params[k] = w.get_value()
    return params
    



