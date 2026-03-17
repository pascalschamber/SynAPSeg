from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QListWidget, QStackedWidget, QWidget, QVBoxLayout, QPushButton, 
    QFileDialog, QLabel, QLineEdit, QHBoxLayout, QComboBox, QTabWidget, 
)
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import QObject, pyqtSignal, Qt
from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QFont
from PyQt6.QtCore import Qt, QTimer, QVariantAnimation, QRectF

import importlib.util
import json
from pathlib import Path
import yaml
import os
import sys
import argparse

from SynAPSeg.utils import utils_general as ug
from SynAPSeg.config import constants
from SynAPSeg.IO.BaseConfig import read_config, write_config
from SynAPSeg.IO.env import verify_and_set_env_dirs
from SynAPSeg.UI.registry import get_plugins, UI_DIRNAME
from SynAPSeg.UI.widgets import mainMenu
from SynAPSeg.UI.widgets.toolbars import ToolBar, StatusBar
from SynAPSeg.UI.widgets.debugging import create_debug_console
from SynAPSeg.UI.widgets.projectManager import ProjectManager, ProjectSelectionDialog
from SynAPSeg.UI.widgets.control import handle_app_reset
from SynAPSeg.UI.widgets import style_sheets


# params
##############################################################################
SETTINGS_FILE = constants.USER_SETTINGS_PATH
APP_MODULES = get_plugins()



# classes
##############################################################################
class StateManager(QObject):
    settings_changed = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.settings = self.load_settings()
            
    def __str__(self):
        astr = "="*80 + "\n" + "Current state:\n" + "="*80 + "\n"
        astr += f"{yaml.dump(self.settings, default_flow_style=False)}\n\n"
        return astr
    
    def get(self, key, default=None):
        return self.settings.get(key, default)

    def set(self, key, value):
        self.settings[key] = value
        self.save_settings()
        self.settings_changed.emit()
    
    def set_attributes(self, attrs:dict):
        """ same as set but for mutliple values """
        for k, v in attrs.items():
            self.settings[k] = v
        self.save_settings()
        self.settings_changed.emit()

    def load_settings(self):
        if not Path(SETTINGS_FILE).exists(): 
            from SynAPSeg.config.initial_setup import create_user_settings
            create_user_settings()
            
        if Path(SETTINGS_FILE).exists():
            settings = read_config(SETTINGS_FILE)
            
            # set env vars
            env_var_map = {k:v for k,v in settings.items() if k in constants.user.keys}
            verify_and_set_env_dirs(env_var_map, override=True, fail_on_error=False)

            # check if these keys are in setting, if not, set default or load from user settings
            _parse_keys = {
                'segmentation_config_log_path':constants.SEG_CONFIG_PATH, 
                'project_root_directory':ug.get_existant_path(settings.get('PROJECTS_ROOT_DIR'))
            }

            _settings = settings.get('UI') or {}
            for k, v in _parse_keys.items():
                if k not in _settings and v is not None:
                    _settings[k] = v
            return _settings
        return {}

    def save_settings(self):
        _settings = read_config(SETTINGS_FILE)
        _settings['UI'] = self.settings
        write_config(_settings, SETTINGS_FILE)

    def display_state(self):
        """ print current state to console """
        print(self, flush=True)
    
    def reset_state(self):
        self.settings = self.load_settings()




class MainWindow(QMainWindow):
    def __init__(self, debug_mode=False):
        super().__init__()
        self.debug_mode = debug_mode
        self.init_window_attributes()

        # Initialize State Manager and project Managers
        self.state_manager = StateManager()
        self.project_manager = ProjectManager(self.state_manager)
                
        # emit only after full init
        self.state_manager.settings_changed.connect(self.update_ui_from_state)
                      
        # Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        

        # main window widgets
        #######################################################
        # Menubar
        self.menu_bar = self.build_menu()
        
        # TODO: implement toolbar functions in sub apps 
        # Toolbar
        # self.tool_bar = self.build_toolbar()
        
        # StatusBar
        self.status_bar = self.build_statusbar()
                
        # app tray
        self.app_tray, self.apps = self.build_app_tray()
        
        # state display 
        if self.debug_mode: 
            self.build_state_display_widgets()

            # debug console
            self.console, self.console_window = self.build_console_window()

        # update state
        self.update_ui_from_state()
        

    def build_menu(self):
        """Instantiate menu bar"""
        # menu objects 
        self.project_selector = ProjectSelectionDialog(self.state_manager)
        self.project_selector.project_root_changed.connect(self.update_available_projects)
        self.project_selector.project_updated.connect(self.update_selected_project)
        self.project_selector.project_created.connect(self.project_selector.update_project_dropdown)

        # create menu bar 
        menu_bar = mainMenu.MenuBar(callbacks={
            'select_project':   self.project_selector.display_project_selection,
            'new_project':      self.project_selector.display_new_project,
            'settings':         lambda: mainMenu.show_settings_dialog(self),
            'about':            lambda: mainMenu.show_about_dialog(self),
            'overview':         lambda: mainMenu.show_overview(self),
            'documentation':    lambda: mainMenu.open_documentation(self),
            'reset':            lambda: handle_app_reset(self),
        })
        self.setMenuBar(menu_bar)
        return menu_bar

    def build_toolbar(self):
        """Instantiate menu bar"""
        # Instantiate tool bar
        tool_bar = ToolBar(callbacks=[
            {"label": "select project", "fxn": lambda: print('select project clicked')},
            {"label": "apply params", "fxn": lambda: print('apply params clicked')},
        ])
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tool_bar)
        return tool_bar
    
    def build_statusbar(self):
        """Instantiate menu bar"""
        # Instantiate status bar
        status_bar = StatusBar(self)
        self.setStatusBar(status_bar)
        status_bar.set_status("Application started.")
        return status_bar
        
    
    def init_window_attributes(self):
        self.setWindowTitle("SynAPSeg")
         
        if self.debug_mode:
            self.setGeometry(-1079, -611, 862, 1022)  # my debug setup
            
        else:
            self.setGeometry(100, 100, 800, 600) # normal
        
        self.resize(862, 1022)

        

    def build_app_tray(self):
        """ Stacked Widget to hold sub-apps - for navigation"""
        self.app_tray = QTabWidget()
        self.app_tray.setObjectName("AppTray")
        self.app_tray.setStyleSheet(style_sheets.app_tray_tabs)
        self.app_tray.currentChanged.connect(self.switch_app)
        
        # Load sub-apps
        self.apps = {}
        self.load_sub_apps()
        
        # Add widgets to layout
        self.layout.addWidget(self.app_tray)
        
        # Show first app by default
        self.app_tray.setCurrentIndex(0)
        return self.app_tray, self.apps

    def build_console_window(self):
        # Create standalone IPython debug console window
        console = create_debug_console({
            "state_manager": self.state_manager,
            "window": self,
            "apps": self.apps,
        })

        console_window = QMainWindow(self)
        console_window.setWindowTitle("Debug Console")
        console_window.setCentralWidget(console)
        console_window.resize(862, 915)
        console_window.setGeometry(-1079, 443, 862, 915)
        console_window.show()
        return console, console_window

    def build_state_display_widgets(self):
        self.print_state_button = QPushButton("Print state")
        self.print_state_button.clicked.connect(self.state_manager.display_state)
        self.layout.addWidget(self.print_state_button)

        self.reset_state_button = QPushButton("Reset state")
        self.reset_state_button.clicked.connect(self.state_manager.reset_state)
        self.layout.addWidget(self.reset_state_button)

        
    def load_sub_apps(self):
        """Dynamically loads sub-apps from UI folder."""
        
        for name, script in APP_MODULES.items():
            app_path = Path(os.path.join(UI_DIRNAME, script))
            if app_path.exists():
                module_name = f"{UI_DIRNAME}.{script[:-3]}"  # Remove .py extension
                self.apps[name] = self.load_app_module(module_name, app_path)

        for name, app in self.apps.items():
            self.app_tray.addTab(app, name)
    
    def load_app_module(self, module_name, module_path):
        """Loads a sub-application dynamically."""
        from SynAPSeg.Plugins.base import load_module_from_path
        module = load_module_from_path(module_name, module_path)
        if module is not None:
            return module.MainApp(self.state_manager)
        else:
            raise ImportError(f"Could not load module {module_name} from {module_path}")

    def switch_app(self, index):
        """Switches to the selected sub-app."""
        if index >= 0 and index < len(self.apps):
            self.app_tray.setCurrentIndex(index)
            current_app = self.app_tray.currentWidget()
            current_app.on_switch_app()
    
    
    def update_available_projects(self):
        """ updates project selection drop down with project directories found in project_root_dir """
        self.state_manager.set('available_projects', self.project_manager.get_available_projects())
        
    def update_selected_project(self):
        """ Updates current app on project selection dialog completion """
        self.app_tray.currentWidget().on_select_project()
            
    def get_selected_project(self):
        return self.state_manager.get('selected_project', "")
    
    def update_ui_from_state(self):
        """Updates the UI when state changes."""        

        # update selected project
        selected_project = self.state_manager.get("selected_project", "Select a project")
        
        if hasattr(self, 'status_bar'):
            self.status_bar.update_current_project(selected_project)
        
        
class SynAPSegSplashScreen(QSplashScreen):
    def __init__(self, icon_path):
        pixmap = QPixmap(icon_path)
        super().__init__(pixmap)
        
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        
        self.rotation = 0
        self.current_opacity = 0.0

        # Fade-in Animation
        self.fade_anim = QVariantAnimation(self)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.setDuration(1000)
        self.fade_anim.valueChanged.connect(self._handle_fade)
        self.fade_anim.start()

        # Rotation Timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_spinner)
        self.timer.start(5)

    def _handle_fade(self, value):
        self.setWindowOpacity(value)

    def _update_spinner(self):
        self.rotation = (self.rotation + 10) % 360
        self.update() 

    def update_status(self, message):
        """Public method to change the loading text."""
        # Align center, bottom, white color, with a slight offset for the spinner
        self.showMessage(
            message, 
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, 
            Qt.GlobalColor.white
        )


def main():
    load_time_start = ug.dt()

    parser = argparse.ArgumentParser(description="SynAPSeg Application")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    icon_path = os.path.join(constants.SYNAPSEG_BASE_DIR, "UI", "icons", "128x128.png")

    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon(icon_path))

    # set global styles
    app.setStyleSheet(style_sheets.tooltip_style)
    
    
    # --- 1. Show Splash Screen ---
    splash = SynAPSegSplashScreen(icon_path)
    splash.show()
    splash.update_status("Loading ...")
    
    # --- 2. Initialize Main Window ---
    window = MainWindow(debug_mode=args.debug)
    window.setWindowIcon(QIcon(icon_path))

    # --- 3. Close Splash and Show Main Window ---
    window.show()
    splash.finish(window)
    print(f"Load time: {ug.dt() - load_time_start}")

    return app.exec()

if __name__ == "__main__":
    sys.exit(main())


