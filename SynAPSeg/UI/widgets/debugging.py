from qtconsole.rich_jupyter_widget import RichJupyterWidget
from qtconsole.inprocess import QtInProcessKernelManager

def create_debug_console(namespace: dict = None):
    kernel_manager = QtInProcessKernelManager()
    kernel_manager.start_kernel()
    kernel = kernel_manager.kernel
    kernel.gui = 'qt'

    kernel_client = kernel_manager.client()
    kernel_client.start_channels()

    console = RichJupyterWidget()
    console.kernel_manager = kernel_manager
    console.kernel_client = kernel_client

    # Push initial variables
    shell = kernel.shell
    shell.push(namespace or {})

    # Optional: bind shell for later use
    console.shell = shell
    console.push_vars = lambda ns: console.shell.push(ns)

    return console




# Cell 2
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel
from typing import Callable, Optional

_active_widgets = []

class TestUI(QWidget):
    """ The wrapper window that holds the test widget and test controls """
    def __init__(self, widget: QWidget, on_click: Optional[Callable] = None):
        super().__init__()
        self.tracked_widget = widget
        
        # Create a fresh layout for THIS window
        self.main_layout = QVBoxLayout(self)
        
        # 1. Add your custom widget
        # display the widget
        if hasattr(widget, 'get_widget'):
            w = self.tracked_widget.get_widget()
        else:
            w = self.tracked_widget
        self.main_layout.addWidget(w)
        
        # 2. Add the test utilities
        self.label = QLabel("Click the button to test connectivity")
        self.main_layout.addWidget(self.label)
        
        self.btn = QPushButton("Run Test Action")
        # Use a lambda to handle the default vs custom callback
        self.btn.clicked.connect(lambda: (on_click(self) if on_click else self.default_on_click()))
        self.main_layout.addWidget(self.btn)
        
    def default_on_click(self):
        """ Default logic: tries to call get_value() on the test widget """
        try:
            val = self.tracked_widget.get_value()
            self.label.setText(f"Widget Value: {val}")
        except Exception as e:
            self.label.setText(f"Error: {str(e)}")

def render_test_widget_safe(widget: QWidget, on_click: Optional[Callable] = None):
    global _active_widgets
    app = QApplication.instance() or QApplication(sys.argv)
    
    # Create the wrapper
    # This automatically parents 'widget' to 'test_window'
    test_window = TestUI(widget, on_click)
    test_window.setWindowTitle(f"Testing: {widget.__class__.__name__}")
    test_window.show()
        
    # Keep the window alive in memory
    _active_widgets.append(test_window)
    
    app.processEvents()
    # REMOVED app.exec() - Jupyter's %gui qt6 handles this for you!
    return test_window