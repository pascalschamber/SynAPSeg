from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QProgressBar
)
from PyQt6.QtCore import Qt, QTimer

class ProgressWidget(QWidget):
    def __init__(self, label_text="Progress", parent=None):
        super().__init__(parent)

        # Layout and label
        self.layout = QVBoxLayout()
        self.label = QLabel(label_text)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        
        # self.progress_bar.setStyleSheet("""
        #     QProgressBar {
        #         border: 2px solid #ff00ff;
        #         border-radius: 6px;
        #         background-color: #0a0a0a;
        #         text-align: center;
        #         color: #ff00ff;
        #         font-weight: bold;
        #         font-family: 'Courier New', monospace;
        #         height: 20px;
        #     }

        #     QProgressBar::chunk {
        #         background: qlineargradient(
        #             x1: 0, y1: 0, x2: 1, y2: 1,
        #             stop: 0 #00ffff,
        #             stop: 0.5 #ff00ff,
        #             stop: 1 #ff0066
        #         );
        #         border-radius: 4px;
        #         margin: 1px;
        #         animation: pulse 1s infinite alternate;
        #     }

        #     @keyframes pulse {
        #         from { background-position: 0% 50%; }
        #         to   { background-position: 100% 50%; }
        #     }
        # """)

        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #2ecc71;
                border-radius: 8px;
                background-color: #1c1c1c;
                text-align: center;
                color: white;
                font-weight: bold;
                height: 20px;
            }

            QProgressBar::chunk {
                background-color: #2ecc71;
                border-radius: 6px;
                margin: 1px;
                animation: glow 1s infinite alternate;
            }

            @keyframes glow {
                from { box-shadow: 0 0 5px #2ecc71; }
                to   { box-shadow: 0 0 15px #2ecc71; }
            }
        """)
        # self.progress_bar.setStyleSheet("""
        #     QProgressBar {
        #         border: 2px solid #555;
        #         border-radius: 8px;
        #         background-color: #2b2b2b;
        #         text-align: center;
        #         color: white;
        #         font-weight: bold;
        #         height: 20px;
        #     }

        #     QProgressBar::chunk {
        #         background: qlineargradient(
        #             x1:0, y1:0, x2:1, y2:0,
        #             stop:0 #6aff95, stop:1 #26d0ce
        #         );
        #         border-radius: 6px;
        #         margin: 1px;
        #     }
        # """)
        self.layout.addWidget(self.progress_bar)
        



        self.setLayout(self.layout)

    def set_progress(self, percent: int):
        """Set progress (0–100)."""
        self.progress_bar.setValue(percent)

    def set_status_text(self, text: str):
        """Update the label above the progress bar."""
        self.label.setText(text)

    def mark_done(self):
        """Mark task as complete visually."""
        self.set_progress(100)
        self.set_status_text("✅ Done")
        # self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #4CAF50; }")


# ----- Main Widget -----
class MainWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Progress Footer Example")

        # Master layout
        self.layout = QVBoxLayout(self)

        # ----- Main content -----
        self.main_area = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_label = QLabel("Main Content Area")
        self.main_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.run_button = QPushButton("Run Task")
        self.run_button.clicked.connect(self.start_task)

        self.main_layout.addWidget(self.main_label)
        self.main_layout.addWidget(self.run_button)
        self.main_area.setLayout(self.main_layout)

        self.layout.addWidget(self.main_area)

        # ----- Footer progress bar -----
        self.progress_widget = ProgressWidget("Waiting to start...")
        self.layout.addWidget(self.progress_widget)

        self.progress = 0
        self.timer = QTimer(self)

    def start_task(self):
        self.progress = 0
        self.progress_widget.set_progress(0)
        self.progress_widget.set_status_text("Working...")
        # self.progress_widget.progress_bar.setStyleSheet("")  # Reset color
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(100)

    def update_progress(self):
        if self.progress >= 100:
            self.timer.stop()
            self.progress_widget.mark_done()
        else:
            self.progress += 5
            self.progress_widget.set_progress(self.progress)
            self.progress_widget.set_status_text(f"Progress: {self.progress}%")

# ----- Run App -----
if __name__ == "__main__":
    app = QApplication([])
    win = MainWidget()
    win.resize(400, 200)
    win.show()
    app.exec()
