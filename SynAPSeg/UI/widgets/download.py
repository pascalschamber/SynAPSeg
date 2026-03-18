import sys
import os
from pathlib import Path
import requests
import zipfile
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QDialog, QScrollArea, QHBoxLayout,
                             QPushButton, QProgressBar, QLabel)
from PyQt6.QtCore import QThread, pyqtSignal

from SynAPSeg.config import constants


class DownloadThread(QThread):
    # Signals to communicate with the GUI thread
    progress_changed = pyqtSignal(int)
    status_changed = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, url, output_path):
        """ extracts downloaded zip to output_path.Parent """
        super().__init__()
        self.url = url
        self.output_path = output_path
        self.extract_dir = Path(self.output_path).parent

    def run(self):
        try:
            self.status_changed.emit("Downloading..")
            response = requests.get(self.url, stream=True)
            
            # Get the total file size from headers
            total_size = int(response.headers.get('content-length', 0))
            
            if response.status_code == 200:
                downloaded = 0
                with open(self.output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Calculate percentage and emit signal
                            if total_size > 0:
                                percent = int((downloaded / total_size) * 100 * 10) # Multiply by 10 because our bar range is 0-1000
                                self.progress_changed.emit(percent)
                
                self.extract_and_cleanup()
            else:
                self.status_changed.emit(f"Error: Status {response.status_code}")
        except Exception as e:
            self.status_changed.emit(f"Error: {str(e)}")
        
        self.finished.emit()

    def extract_and_cleanup(self):
        
        if self.output_path.endswith('.zip'):
            self.status_changed.emit("Extracting files...")
                
            with zipfile.ZipFile(self.output_path, 'r') as zip_ref:
                zip_ref.extractall(self.extract_dir)
            
            os.remove(self.output_path)
        self.status_changed.emit(f"Done! Extracted to {self.extract_dir}")

class DownloadWindow(QDialog):
    def __init__(self, parent, url, save_path):
        super().__init__(parent)
        self.url = url
        self.save_path = save_path
        self.initUI()
        print('download window loaded.')

    def initUI(self):
        self.setWindowTitle("Download Models")
        self.setMinimumWidth(400)
                
        layout = QVBoxLayout()
        
        self.label = QLabel(f"download url: {self.url}\nsave to: {self.save_path}")
        layout.addWidget(self.label)

        self.pbar = QProgressBar(self)
        self.pbar.setRange(0, 1000) # 1000 steps = 0.1% increments
        self.pbar.setFormat("%p% (%v/%m)")
        layout.addWidget(self.pbar)

        self.btn = QPushButton("Start Download", self)
        self.btn.clicked.connect(self.start_download)
        layout.addWidget(self.btn)
        
        self.setLayout(layout)

    def start_download(self):
                
        self.btn.setEnabled(False)
        
        # Initialize the worker thread
        self.thread = DownloadThread(self.url, self.save_path)
        
        # Connect signals to UI slots
        self.thread.progress_changed.connect(self.pbar.setValue)
        self.thread.status_changed.connect(self.label.setText)
        self.thread.finished.connect(lambda: self.btn.setEnabled(True))
        
        self.thread.start()
