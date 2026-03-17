from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QComboBox, QPushButton, QFileDialog, QDialog
)
from PyQt6.QtCore import pyqtSignal
import os
from pathlib import Path
from .dialogs import browse_widget, dialog_ok_cancel_buttons

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
        """Populates the dropdown with projects available in the selected project root directory."""
        self.project_root = self.state_manager.get('project_root_directory', '')

        if self.project_root and Path(self.project_root).exists():
            files = os.listdir(self.project_root)
            self.project_files = files
        
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
        buttons_layout = dialog_ok_cancel_buttons(dialog, 
            ok_callback=lambda: self.ok_select_project_clicked(
                self.project_dropdown.currentText(), dialog)
        )

        layout.addLayout(root_dir_layout)
        layout.addLayout(project_layout)
        layout.addLayout(buttons_layout)

        dialog.setLayout(layout)
        dialog.exec()

    def update_project_dropdown(self):
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
            self.project_root_changed.emit()
            self.update_project_dropdown()
                
    
    def ok_select_project_clicked(self, selected_project, dialog):
        project_root_directory = self.root_input.text()
        self.state_manager.set_attributes({
            'project_root_directory': project_root_directory, 
            'selected_project': selected_project})
        dialog.accept()
        self.project_updated.emit()

    
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
        def ok_callback(selected_project, dialog):
            self.ok_select_project_clicked(selected_project, dialog)
            self.project_created.emit()
            
        buttons_layout = dialog_ok_cancel_buttons(dialog, 
            ok_callback=lambda: ok_callback(self.project_input.text(), dialog), 
        )

        layout.addLayout(root_dir_layout)
        layout.addLayout(project_layout)
        layout.addLayout(buttons_layout)

        dialog.setLayout(layout)
        dialog.exec()