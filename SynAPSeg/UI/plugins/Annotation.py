from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QStackedWidget, QWidget, QVBoxLayout, QPushButton, 
    QFileDialog, QLabel, QLineEdit, QHBoxLayout, QComboBox, QTextEdit, QSizePolicy,QFormLayout, 
    QLayout,
)
from PyQt6.QtCore import Qt
import os
import sys
from pathlib import Path

from SynAPSeg.UI.plugins.__base import BaseApp
from SynAPSeg.UI.widgets.config_fields import field_widget
from SynAPSeg.IO.project import Project, Example
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
        self.example_folders_dropdown.currentIndexChanged.connect(self._on_select_example)

        # select an example to load
        example_layout = QHBoxLayout()
        example_layout.addWidget(QLabel("Select an example")) # , alignment=Qt.AlignmentFlag.AlignTop
        example_layout.addWidget(self.example_folders_dropdown)
        self.layout.addLayout(example_layout)

        # add annotation kwarg widgets
        self.run_kwarg_layout = QFormLayout()
        self.run_kwarg_layout.setSizeConstraint(QLayout.SizeConstraint.SetMinimumSize)
        self.layout.addLayout(self.run_kwarg_layout)
        self.run_kwarg_widgets = {}
        self.layout.addStretch() # align widgets to the top
        
        # attributes 
        self.previous_fns_selection = [] # ref to previous selection for persistance
        self.current_fns_selection = []

        self.post_layout()

    def populate_example_folders(self):
        """Populates the dropdown with files from the selected project root directory."""
        self.example_folders_dropdown.clear()
        self.example_folders_dropdown.addItem("Select an example")
        
        exdir = self.get_examples_directory()
        if exdir and Path(exdir).exists():
            self.example_folders_dropdown.addItems(os.listdir(exdir))

                
    def _on_select_example(self):
        self.previous_fns_selection = self.current_fns_selection
        self.selected_example = self.example_folders_dropdown.currentText()
    
        if self.selected_ex_is_valid(self.selected_example):
            self.state_manager.set("selected_example", self.selected_example)
            
            # if len(self.run_kwarg_widgets)==0: # parse kwargs if previously unset and build wigets
            self.add_annotation_kwargs_widgets()
                
    def selected_ex_is_valid(self, selected_example:str):
        return selected_example not in ['Select an example', '']

    def add_annotation_kwargs_widgets(self):
        
        dir_examples = self.get_examples_directory()
        PROJ_PATH = Path(dir_examples).parent if dir_examples else None

        # clear widgets
        while self.run_kwarg_layout.count():
            item = self.run_kwarg_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.run_kwarg_widgets = {}

        # only build if example is selected
        if PROJ_PATH and self.selected_ex_is_valid(self.selected_example):
            self.run_kwarg_widgets = self.build_annotation_params_widget(PROJ_PATH, self.selected_example)
            for k, w in self.run_kwarg_widgets.items():
                self.run_kwarg_layout.addRow(QLabel(k), w.get_widget())
                
    
    
        
    def build_annotation_params_widget(self, PROJ_PATH, selected_example, fn_pattern=".*\.tiff?"):
        """ build widgets for user input of parameters for annotation"""
        if not PROJ_PATH:
            return None, {}

        project = Project(PROJ_PATH)
        ex: Example = project.get_example(selected_example) 
        # exmd = ex.get_metadata()
        all_fns = sorted(list(ex.get_filenames(fn_pattern))) 
        
        set_selected_include_only = self.parse_include_fns_on_switch_example(all_fns)
        
        widgets = {
            'include_only': field_widget(dict(default_value=set_selected_include_only, value_options=all_fns, widget_type='multi-selection', tooltip='',)),
            # 'exclude': field_widget(dict(default_value=None, value_options=all_fns, widget_type='multi-selection', tooltip='',)),
            # 'add_to_file_map': field_widget(dict(default_value=None, value_options=all_fns, widget_type='multi-selection', tooltip='',)),
            # 'fail_on_format_error': field_widget(dict(default_value=False, value_options=None, widget_type='checkbox', tooltip='',)),
        }
        
        # extends the list so scroll bar isn't needed
        from PyQt6.QtWidgets import QAbstractScrollArea
        widgets['include_only'].list_widget.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        widgets['include_only'].value_changed.connect(self._on_select_includeOnly)
        # w.value_changed.connect(lambda: print(f'value changed for {k}'))
        
        return widgets
    
    def _on_select_includeOnly(self):
        self.current_fns_selection = self.run_kwarg_widgets['include_only'].get_value()   
        
    def parse_include_fns_on_switch_example(self, current_available):
        """ 
        allows selection to persist when example is changed 
        
        args:
            current_available: list of filenames in currently selected project
        """ 
        new_selection = [el for el in self.previous_fns_selection if el in current_available]
        return new_selection or None # return None if previous is empty 
    
    def reset_selections(self):
        self.previous_fns_selection = []
        self.current_fns_selection = []
    
    def refresh_params(self):
        self.reset_selections()
        self.populate_example_folders()
        self.add_annotation_kwargs_widgets()
        
    def _on_switch_app(self):
        self.refresh_params()
    def _on_select_project(self):
        self.refresh_params()
    
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
        exmd = image_dict.pop('metadata')
        
        # create napari viewer
        viewer, widget_objects = create_napari_viewer(
            exmd, 
            ex.path_to_example, 
            FILE_MAP, 
            image_dict, 
            get_image_list=get_image_list,
            LABEL_INT_MAP=LABEL_INT_MAP,
            set_lbl_contours=set_lbl_contours,
        )




def parse_annotation_params_widgets(widgets):
    """ parse widgets for user input of parameters for annotation"""
    params = {}
    for k, w in widgets.items():
        params[k] = w.get_value()
    return params
