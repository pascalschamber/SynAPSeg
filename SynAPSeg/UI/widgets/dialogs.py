from PyQt6.QtWidgets import QHBoxLayout, QLabel, QLineEdit, QPushButton, QWidget


def browse_widget(label_text, default_value, callback_fn):
    """
    Creates a horizontal layout containing a label, line edit, and browse button.
    
    Returns:
        layout: The QHBoxLayout containing the widgets.
        line_edit: The QLineEdit instance (so you can read its value later).
    """
    layout = QHBoxLayout()
    
    # 1. Create the Label
    label = QLabel(label_text)
    
    # 2. Create the Input Field
    line_edit = QLineEdit()
    line_edit.setText(default_value)
    
    # 3. Create the Button and connect the custom function
    browse_button = QPushButton("Browse")
    browse_button.clicked.connect(callback_fn)
    
    # Add to layout
    layout.addWidget(label)
    layout.addWidget(line_edit)
    layout.addWidget(browse_button)
    
    return layout, line_edit

def dialog_ok_cancel_buttons(dialog, ok_callback=None, cancel_callback=None):
    """
    Creates a horizontal layout containing OK and Cancel buttons.
    
    Returns:
        layout: The QHBoxLayout containing the widgets.
    """
    ok_callback = dialog.accept if ok_callback is None else ok_callback
    cancel_callback = dialog.reject if cancel_callback is None else cancel_callback

    buttons_layout = QHBoxLayout()
    ok_button = QPushButton("OK")
    cancel_button = QPushButton("Cancel")
    buttons_layout.addWidget(ok_button)
    buttons_layout.addWidget(cancel_button)

    ok_button.clicked.connect(
        ok_callback
    )
    cancel_button.clicked.connect(
        cancel_callback
    )
    return buttons_layout


def warning_dialog(parent, title=None, text=None, informative_text=None):
    """
    Creates a warning dialog pop up
    
    Returns:
        msg_box: The QMessageBox instance.
    """
    from PyQt6.QtWidgets import QMessageBox
    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Warning)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    msg_box.setInformativeText(informative_text)
    msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
    msg_box.exec()
    return msg_box




