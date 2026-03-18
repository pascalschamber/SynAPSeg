from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QFileDialog, QDialog
)
from PyQt6.QtCore import pyqtSignal
import os
from pathlib import Path
from .dialogs import browse_widget, dialog_ok_cancel_buttons
from SynAPSeg.IO.project import Project

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
            files = os.listdir(self.project_root)
            ffiles = [f for f in files if Project.is_project_dir(os.path.join(self.project_root, f))]
            if len(files) == 0:
                print(f"cannot get_available_projects: os.listdir({self.project_root}) is empty")
            elif len(ffiles) == 0:
                print(f"cannot get_available_projects: Project.is_project_dir removed all possible folders")
                
            
            self.project_files = ffiles
        else: 
            print(f"cannot get_available_projects: {self.project_root} or {Path(self.project_root).exists()} is invalid")
        
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
            self.browse_root_dir
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

    def browse_root_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Root Directory")
        if dir_path:
            self.root_input.setText(dir_path)
            self.state_manager.set('project_root_directory', dir_path)
            print('in browse_root_dir, signal project_root_changed about to be emitted')
            self.project_root_changed.emit()  # updates state's available_projects
            self.update_project_dropdown()

    def ok_select_project_clicked(self, selected_project, dialog, signal):
        """ update state with project root and selected project name, then emit appropriate signal """

        project_root_directory = self.root_input.text()
        self.state_manager.set_attributes({
            'project_root_directory': project_root_directory, 
            'selected_project': selected_project})
        dialog.accept()
        signal.emit()

    
