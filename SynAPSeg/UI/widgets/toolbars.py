from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QToolBar, QWidget, QStatusBar, QLabel
)
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtCore import Qt
import sys
import os


class ToolBar(QToolBar):
    def __init__(self, parent=None, callbacks=None):
        super().__init__(parent)
        self.callbacks = callbacks or {}

        self.setMovable(False)
        self._create_toolbar()

    def build_action(self, icon:QIcon = QIcon(), label:str = '', fxn = lambda: None):
        action = QAction(icon, label, self)
        action.triggered.connect(fxn)
        return action
    
    def _create_toolbar(self):
        for attrs in self.callbacks:
            assert isinstance(attrs, dict), f"{type(attrs)}"
            self.addAction(self.build_action(**attrs))


class StatusBar(QStatusBar):
    def __init__(self, parent):
        super().__init__(parent)
        
        # Add permanent widget to display current project on the right
        self.project_label = QLabel()
        self.addPermanentWidget(self.project_label)


    def set_status(self, message: str, timeout: int = 0):
        self.showMessage(message, timeout)
    
    def update_current_project(self, selected_project: str):
        """Update the permanent project label."""
        self.project_label.setText(f"Current Project: {selected_project}")
        self.project_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)



# Test visualization code
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = QMainWindow()
    main_win.setWindowTitle("Test GUI Components")
    main_win.resize(800, 600)

    # Instantiate tool bar
    tool_bar = ToolBar(callbacks=[
        {"icon":QIcon(), "label": "apply params", "fxn": lambda: print('apply params clicked')}
    ])
    main_win.addToolBar(Qt.ToolBarArea.TopToolBarArea, tool_bar)

    # Instantiate status bar
    status_bar = StatusBar(main_win)
    main_win.setStatusBar(status_bar)
    status_bar.set_status("Application started.")

    central_widget = QWidget()
    main_win.setCentralWidget(central_widget)

    main_win.show()
    sys.exit(app.exec())
