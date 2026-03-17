import sys
import os
import subprocess
from PyQt6.QtWidgets import QMessageBox, QApplication

def handle_app_reset(parent_widget=None):
    """
    Restarts the current program, handling paths with spaces correctly.
    """
    reply = QMessageBox.warning(
        parent_widget, 
        "Reset Application", 
        "This will restart the application. Proceed?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
    )

    if reply == QMessageBox.StandardButton.Yes:
        # 1. Get the current command line arguments
        # sys.executable is the python.exe path
        # sys.argv[0] is the script path (e.g., SynAPSeg/UI/main.py)
        args = [sys.executable] + sys.argv
        
        # 2. Launch the new process
        # This preserves all arguments, including '--debug'
        subprocess.Popen(args)
        
        # 3. Cleanly exit the current instance
        QApplication.quit()