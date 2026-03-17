from PyQt6.QtWidgets import QProgressDialog, QApplication, QMessageBox
from PyQt6.QtCore import QThread, pyqtSignal, Qt
import functools
import time
from PyQt6.QtCore import QElapsedTimer
from typing import Any
import traceback
import sys


class UniversalWorker(QThread):
    """Bridge between a standard Python function and the Qt Event Loop."""
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self._is_running = True
        self.progress_dialog = None
        self._finished = False

    def run(self) -> Any:
        # The 'Bridge' function we inject into your logic
        def ui_callback(percent, message):
            if not self._is_running:
                # This allows us to raise an exception inside the logic
                # if the user hits 'Cancel'
                raise InterruptedError("Process cancelled by user.")
            self.progress_signal.emit(percent, message)

        # Inject the callback into the kwargs
        self.kwargs['progress_callback'] = ui_callback
        
        try:
            result = self.func(*self.args, **self.kwargs)
            self.finished_signal.emit(result) # result is returned from the function in this signal
            self._is_running = False
            self._finished = True
            
        except InterruptedError:
            pass # Silent exit on manual cancel
        except Exception as e:
            # 1. Capture the full traceback as a string
            error_trace = traceback.format_exc()
            
            # 2. Print it to the console immediately (helpful for IDE debugging)
            print("Worker Thread Error:\n", error_trace, file=sys.stderr)
            
            # 3. Emit the full trace to the UI so the QMessageBox shows it
            self.error_signal.emit(error_trace)

    def stop(self):
        self._is_running = False

class SubProgress:
    """Helper to map a sub-function's 0-100% to a slice of the main progress bar."""
    def __init__(self, main_callback, start_pct, end_pct):
        self.main_callback = main_callback
        self.start = start_pct
        self.range = end_pct - start_pct

    def __call__(self, internal_pct, message):
        # Map internal 0-100 into the window (e.g., 20% to 50%)
        mapped_pct = int(self.start + (internal_pct / 100.0) * self.range)
        self.main_callback(mapped_pct, message)


def run_with_progress(func, parent=None, title="Processing", label="Initializing...", *args, **kwargs) -> UniversalWorker:
    progress_dialog = QProgressDialog(label, "Cancel", 0, 100, parent)
    progress_dialog.setWindowTitle(title)
    progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
    progress_dialog.setMinimumDuration(0)
    progress_dialog.setValue(0)
    
    start_time = time.time()

    # Explicit Slot to ensure the ProgressBar updates
    def update_ui(val, msg):
        # Force integer type and set value
        progress_val = int(val)
        progress_dialog.setValue(progress_val)
        
        # Timer Logic
        if progress_val > 2: # Wait for 2% for stability
            elapsed = time.time() - start_time
            remaining = (elapsed / (progress_val / 100.0)) - elapsed
            mins, secs = divmod(int(remaining), 60)
            time_str = f"{mins:02d}:{secs:02d} left"
        else:
            time_str = "Calculating remaining time..."
            
        progress_dialog.setLabelText(f"{msg}\n{time_str}")
    
    def cancel(worker):
        worker.stop()
        if not worker._finished:
            # worker.kwargs['progress_callback'](0, "Cancelling...")
            progress_dialog.setLabelText("Cancelling...")

    worker = UniversalWorker(func, *args, **kwargs)
    
    # Keep the dialog alive by attaching it to the worker
    worker.progress_dialog = progress_dialog

    # Connect signals
    worker.progress_signal.connect(update_ui)
    
    # Ensure dialog closes on finish or error
    worker.finished_signal.connect(progress_dialog.close)
    worker.error_signal.connect(lambda err: (progress_dialog.close(), QMessageBox.critical(parent, "Error", err)))

    # Handle Manual Cancel
    progress_dialog.canceled.connect(lambda: cancel(worker))

    progress_dialog.show()
    worker.start()
    
    return worker # Return the thread worker instead of the dialog