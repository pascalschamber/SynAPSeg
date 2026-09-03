from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QFileDialog, QDialog
)
from PyQt6.QtCore import pyqtSignal
import os
from pathlib import Path
from SynAPSeg.UI.widgets.dialogs import browse_widget, dialog_ok_cancel_buttons



class ProjectManager:
    """ implements logic for getting available projects 
            communicates results to state_manager, which interfaces with UI objects
    """
    def __init__(self, state_manager):
        self.state_manager = state_manager
        self.project_root = ''
        self.project_files = []
        self.state_manager.set('available_projects', self.get_available_projects())
    
    def get_available_projects(self):
        """
        Populates the dropdown with projects available in the selected project root directory.
            Only counts projects that have >0 examples
        """
        self.project_root = self.state_manager.get('project_root_directory', '')

        if self.project_root and Path(self.project_root).exists():
            from SynAPSeg.IO.project import Project
            
            # handle user input error where project dir is selected instead of the project's root dir
            if Project.is_project_dir(self.project_root):
                self.project_root = str(Path(self.project_root).parent)
                print(f"WARNING: [SynAPSeg.UI.widgets.projectManager.get_available_projects] selected project root is a project directory, correcting input to use parent directory: {self.project_root}")
                # self.state_manager.set('project_root_directory', self.project_root)
                self.state_manager.mainwindow.project_selector.set_project_root_directory(self.project_root)
            
            # normal case: list all sub-folders and check if they are projects
            files = os.listdir(self.project_root)
            ffiles = [f for f in files if Project.is_project_dir(os.path.join(self.project_root, f))]
            if len(ffiles) == 0:
                print(f"ERROR: [SynAPSeg.UI.widgets.projectManager.get_available_projects] no projects exist at {self.project_root}")
            
            self.project_files = ffiles
        else: 
            print(f"ERROR: [SynAPSeg.UI.widgets.projectManager.get_available_projects] project_root directory does not exist at {self.project_root}")
            self.project_files = []
        
        return self.project_files
    
    def add_new_project(self, project_name:str):
        """ Adds a new project to the project root directory."""
        if self.project_root and Path(self.project_root).exists():
            new_project_path = os.path.join(self.project_root, project_name)
            os.makedirs(new_project_path, exist_ok=True)
            self.project_files.append(project_name)
            self.state_manager.set('available_projects', self.project_files)
            
        else:
            print(f"Project root directory is invalid, got {self.project_root}")


class ProjectSelectionDialog(QWidget):
    project_root_changed = pyqtSignal()
    project_updated = pyqtSignal()
    project_created = pyqtSignal()

    def __init__(self, state_manager):
        super().__init__()
        self.state_manager = state_manager
    
    
    def display_new_project(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("New Project")

        layout = QVBoxLayout()

        root_dir_layout, self.root_input = browse_widget(
            "Root Directory:", 
            self.state_manager.get('project_root_directory', ''), 
            self.browse_root_dir
        )

        # Project name selection
        project_layout = QHBoxLayout()
        project_label = QLabel("Project Name:")
        self.project_input = QLineEdit()
        project_layout.addWidget(project_label)
        project_layout.addWidget(self.project_input)

        # OK and Cancel buttons
        buttons_layout = dialog_ok_cancel_buttons(
            dialog,
            ok_callback=lambda: self.ok_select_project_clicked(
                self.project_input.text(), dialog, self.project_created
            ),
        )

        layout.addLayout(root_dir_layout)
        layout.addLayout(project_layout)
        layout.addLayout(buttons_layout)

        dialog.setLayout(layout)
        dialog.exec()
        

    def display_project_selection(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Project")

        layout = QVBoxLayout()

        # Root directory selection
        root_dir_layout, self.root_input = browse_widget(
            "Root Directory:", 
            self.state_manager.get('project_root_directory', ''), 
            lambda: self.browse_root_dir(trigger_update=True)
        )
        # Project dropdown selection
        project_layout = QHBoxLayout()
        project_label = QLabel("Select Project:")
        self.project_dropdown = QComboBox()
        self.update_project_dropdown()
        project_layout.addWidget(project_label)
        project_layout.addWidget(self.project_dropdown)

        # OK and Cancel buttons
        buttons_layout = dialog_ok_cancel_buttons(
            dialog,
            ok_callback=lambda: self.ok_select_project_clicked(
                self.project_dropdown.currentText(), dialog, self.project_updated
            ),
        )

        layout.addLayout(root_dir_layout)
        layout.addLayout(project_layout)
        layout.addLayout(buttons_layout)

        dialog.setLayout(layout)
        dialog.exec()

    def update_project_dropdown(self):
        projects = self.state_manager.get('available_projects', [])
        
        if len(projects) == 0 and self.state_manager.get('project_root_directory'):
            self.project_root_changed.emit()
            projects = self.state_manager.get('available_projects', [])
            
        current_project = self.state_manager.get("selected_project", "Select a project")
        self.project_dropdown.clear()
        self.project_dropdown.addItem("Select a project")
        self.project_dropdown.addItems(projects)
        self.project_dropdown.setCurrentText(current_project)

    def browse_root_dir(self, trigger_update=False):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Root Directory")
        if dir_path:
            if trigger_update: # only trigger update when selecting project, not when creating new project
                self.set_project_root_directory(dir_path)
            else:
                self.root_input.setText(dir_path)
    
    def get_root_dir_input(self):
        return self.root_input.text()
            
    def set_project_root_directory(self, dir_path:str):
        self.root_input.setText(dir_path)
        self.state_manager.set('project_root_directory', dir_path)
        self.project_root_changed.emit()  # updates state's available_projects
        self.update_project_dropdown()

    def ok_select_project_clicked(self, selected_project, dialog, signal):
        """ update state with project root and selected project name, then emit appropriate signal """

        project_root_directory = self.get_root_dir_input()
        self.state_manager.set_attributes({
            'project_root_directory': project_root_directory, 
            'selected_project': selected_project})
        dialog.accept()
        signal.emit()

    
