from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QMenuBar, QMenu, QToolBar,
    QFileDialog, QDialog, QLabel, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QMessageBox, QWidget, QStatusBar,
    QLineEdit, QListWidget, QFrame, QScrollArea
)
from PyQt6.QtGui import QIcon, QAction, QDesktopServices
from PyQt6.QtCore import Qt, QUrl

import yaml
import sys
import os

from SynAPSeg.IO.env import verify_and_set_env_dirs
from SynAPSeg.config import constants


class MenuBar(QMenuBar):
    def __init__(self, parent=None, callbacks=None):
        super().__init__(parent)
        self.callbacks = callbacks or {}

        self._create_file_menu()
        self._create_settings_menu()
        self._create_help_menu()

    def get_callback(self, key):
        return self.callbacks.get(key, lambda: None)

    def _create_file_menu(self):
        file_menu = self.addMenu("File")

        select_project_action = QAction("Select Project", self)
        select_project_action.triggered.connect(self.get_callback('select_project'))
        file_menu.addAction(select_project_action)

        new_project_action = QAction("New Project", self)
        new_project_action.triggered.connect(self.get_callback('new_project'))
        file_menu.addAction(new_project_action)

    def _create_settings_menu(self):
        settings_menu = self.addMenu("Settings")
        settings_action = QAction("Environment Settings", self)
        settings_action.triggered.connect(self.get_callback('settings'))
        settings_menu.addAction(settings_action)


    def _create_help_menu(self):
        help_menu = self.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.get_callback('about'))
        help_menu.addAction(about_action)

        overview_action = QAction("Overview", self)
        overview_action.triggered.connect(self.get_callback('overview'))
        help_menu.addAction(overview_action)

        doc_action = QAction("Documentation Website", self)
        doc_action.triggered.connect(self.get_callback('documentation'))
        help_menu.addAction(doc_action)

        reset_action = QAction("Reset", self)
        reset_action.triggered.connect(self.get_callback('reset'))
        help_menu.addAction(reset_action)

          
    def reset_warning(self):
        reset_callback = self.callbacks.get('reset')
        if reset_callback:
            reset_callback()


def show_settings_dialog(parent=None):
    # Define the config path
    config_path = os.path.join(constants.SYNAPSEG_BASE_DIR, "config", "user_settings.yaml")

    # Load the current YAML data
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            settings_data = yaml.safe_load(f) or {}
    else:
        settings_data = {}

    # Create Dialog
    dialog = QDialog(parent)
    dialog.setWindowTitle("User Settings")
    dialog.setMinimumWidth(500)
    
    main_layout = QVBoxLayout(dialog)
    
    # Scroll area in case the list of settings grows long
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    content_widget = QWidget()
    content_layout = QVBoxLayout(content_widget)
    
    # Track inputs to retrieve data on Save
    input_widgets = {} # Stores {key: widget_reference}

    def add_path_row(key, current_value):
        """Helper to create UI for a single file path."""
        layout = QVBoxLayout()
        label = QLabel(f"<b>{key}</b>")
        
        row = QHBoxLayout()
        line_edit = QLineEdit(str(current_value))
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(lambda: update_line_edit(line_edit))
        
        row.addWidget(line_edit)
        row.addWidget(browse_btn)
        
        layout.addWidget(label)
        layout.addLayout(row)
        content_layout.addLayout(layout)
        input_widgets[key] = line_edit

    def add_list_editor(key, current_list):
        """Helper to create UI for a list of paths with ordering controls."""
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"<b>{key} (Ordered List)</b>"))
        
        list_widget = QListWidget()
        list_widget.addItems(current_list or [])
        
        # Action Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("+ Add")
        rem_btn = QPushButton("- Remove")
        up_btn = QPushButton("↑ Up")
        down_btn = QPushButton("↓ Down")
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(rem_btn)
        btn_layout.addWidget(up_btn)
        btn_layout.addWidget(down_btn)
        
        layout.addWidget(list_widget)
        layout.addLayout(btn_layout)
        content_layout.addLayout(layout)
        
        # Logic for buttons
        add_btn.clicked.connect(lambda: add_to_list(list_widget))
        rem_btn.clicked.connect(lambda: list_widget.takeItem(list_widget.currentRow()))
        up_btn.clicked.connect(lambda: move_item(list_widget, -1))
        down_btn.clicked.connect(lambda: move_item(list_widget, 1))
        
        input_widgets[key] = list_widget

    # Helpers for the Editor
    def update_line_edit(edit):
        path = QFileDialog.getExistingDirectory(dialog, "Select Directory")
        if path: edit.setText(path)

    def add_to_list(lw):
        path = QFileDialog.getExistingDirectory(dialog, "Add Directory to List")
        if path: lw.addItem(path)

    def move_item(lw, direction):
        curr = lw.currentRow()
        if curr < 0: return
        target = curr + direction
        if 0 <= target < lw.count():
            item = lw.takeItem(curr)
            lw.insertItem(target, item)
            lw.setCurrentRow(target)

    # Build UI based on your specific keys
    # Handle Root Dirs (Lists)
    for list_key in constants.user.keys:
        add_list_editor(list_key, settings_data.get(list_key, []))

    # Handle Nested UI (Single Path)
    ui_settings = settings_data.get('UI', {})
    ui_settings_keys = ['segmentation_config_log_path']
    for key in ui_settings_keys:
        add_path_row(key, ui_settings.get(key, ''))

    # 5. Dialog Buttons (Save/Cancel)
    scroll.setWidget(content_widget)
    main_layout.addWidget(scroll)
    
    actions = QHBoxLayout()
    save_btn = QPushButton("Save Settings")
    save_btn.setStyleSheet("background-color: #2b5797; color: white; font-weight: bold;")
    cancel_btn = QPushButton("Cancel")
    
    actions.addWidget(cancel_btn)
    actions.addWidget(save_btn)
    main_layout.addLayout(actions)

    def save_and_close():
        # parse the UI inputs
        new_data = {
            'ROOT_DIR': [input_widgets['ROOT_DIR'].item(i).text() for i in range(input_widgets['ROOT_DIR'].count())],
            'PROJECTS_ROOT_DIR': [input_widgets['PROJECTS_ROOT_DIR'].item(i).text() for i in range(input_widgets['PROJECTS_ROOT_DIR'].count())],
            'MODELS_BASE_DIR': [input_widgets['MODELS_BASE_DIR'].item(i).text() for i in range(input_widgets['MODELS_BASE_DIR'].count())],
            'UI': {
                key: input_widgets[key].text() for key in ui_settings_keys
            }
        }

        # Update the original settings with new data
        for key in constants.user.keys:         # env vars
            settings_data[key] = new_data[key]  
        
        for k,v in new_data.get('UI', {}).items(): # UI settings
            settings_data['UI'][k] = v

        # Write back to YAML
        with open(config_path, 'w') as f:
            yaml.dump(settings_data, f, default_flow_style=False)
        
        # update env vars
        verify_and_set_env_dirs({key: path for key, path in settings_data.items() if key in constants.user.keys})
        dialog.accept()

    save_btn.clicked.connect(save_and_close)
    cancel_btn.clicked.connect(dialog.reject)

    dialog.exec()

def show_about_dialog(self):
    QMessageBox.about(self, "About", "Application Version: 1.0.0")

def show_overview(self):
    try:
        overview_path = os.path.join(constants.SYNAPSEG_BASE_DIR, 'UI', 'messages', 'help_overview.md')
        with open(overview_path, 'r') as file:
            content = file.read()
    except FileNotFoundError:
        content = "Overview file not found."

    dlg = QDialog(self)
    dlg.setWindowTitle("Application Overview")
    layout = QVBoxLayout()
    text_edit = QTextEdit()
    text_edit.setPlainText(content)
    text_edit.setReadOnly(True)
    layout.addWidget(text_edit)
    dlg.setLayout(layout)
    dlg.resize(500, 400)
    dlg.exec()

def open_documentation(self):
    try:
        doc_link_path = os.path.join(constants.SYNAPSEG_BASE_DIR, 'UI', 'messages', 'help_documentation_link.txt')
        with open(doc_link_path, 'r') as file:
            url = file.read().strip()
            QDesktopServices.openUrl(QUrl(url))
    except FileNotFoundError:
        QMessageBox.warning(self, "Error", "Documentation link file not found.")

