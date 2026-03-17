import sys
import yaml
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
    QCheckBox, QComboBox, QGroupBox, QFormLayout, QTabWidget, QTextEdit, QPushButton,
    QHBoxLayout, QFileDialog, QSizePolicy, QScrollArea, QListWidget, QListWidgetItem
)
from PyQt6.QtGui import QPalette, QColor
import ast
import os
from pathlib import Path
from PyQt6.QtCore import pyqtSignal
from typing import Dict

from SynAPSeg.UI.widgets import style_sheets as Style
from SynAPSeg.UI.widgets.label import HoverLabel
from SynAPSeg.config.param_engine.flags import ParameterFlag


def field_widget(attributes):
    """ create a widget from a dictionary of attributes """
    widget_classes = {
        "int": IntConfigField,
        "float": FloatConfigField,
        "bool": BoolConfigField,
        "str": StringConfigField,
        "directory": DirectoryConfigField,
        "path": PathConfigField,
        "dict": DictConfigField,
        "list": ListConfigField,
        "tuple": TupleConfigField,
        "SegConfigModelsField": SegConfigModelsField,
        "selection": SelectionConfigField,
        "multi-selection": MultiSelectionConfigField,
        'NoneType': StringConfigField,
        'HiddenWidget': HiddenWidget,
        "dict_int_str": DictIntStrField,
        "dict_str_str": DictStrStrField,
        "dict_any_list": DictAnyListField,
        "list_list_int": ListListIntField,
        "extract_groups": ExtractGroupsField,
    }
    # check for widget_type
    wType = attributes.get("widget_type", "str")
    flags = attributes.get('flags', [])
    
    try:
        # can also handle passing a python widget_type by converting it to its string representation
        if isinstance(wType, type):
            wType = wType.__name__
        
        if flags:
            if ParameterFlag.HIDDEN in flags:
                wType = 'HiddenWidget'
        
        widget = widget_classes[wType](**attributes)

    except Exception as e:
        raise ValueError(f"error:{e}\nwType: {wType}\nattributes: {attributes}")

    return widget


# TODO would be nice to know when a widget's default value was overridden -> for config interpreter to simply parse default value

class BaseConfigField(QWidget):
    """Base widget class for all config fields, standardizing get/set behavior."""
    value_changed = pyqtSignal()
    widget: QWidget # each child class should use self.set_widget to init

    def __init__(self, default_value=None, widget_type='', tooltip='', heading='General', category='', flags=None):
        super().__init__()
        self.attribute_keys = ['default_value', 'widget_type', 'tooltip', 'heading', 'category', 'flags'] #, 'flag', 'uid']
        self.default_value = default_value
        self.tooltip = tooltip
        self.heading = heading
        self.category = category
        self.widget_type = widget_type
        self.tooltip = tooltip
        self.flags = flags

    def get_value(self):
        """Retrieve the current value from the widget."""
        raise NotImplementedError
    
    def set_value(self, value):
        """Set a new value in the widget."""
        raise NotImplementedError
    
    def set_widget(self, widget):
        """ initialize the field's widget and apply common formatting """    
        self.widget = widget
        if hasattr(self, "setSizePolicy"):
            self.widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self.widget.setToolTip(Style.format_tooltip(self.tooltip))
        
    def get_widget(self):
        if hasattr(self, 'widget'):
            return getattr(self, 'widget') # if it has a widget self class
        return self 

    def get_attributes(self):
        return {k: getattr(self, k) for k in self.attribute_keys}

    def to_param_spec(self):
        attrs = self.get_attributes()
        attrs['current_value'] = self.get_value()
        return attrs
    
    def set_valid(self, is_valid):
        color = "" if is_valid else Style.red_border(type(self.widget))  # light red on error
        # Style.update_stylesheet_property(self.widget, "border", color)
        self.widget.setStyleSheet(color)

    def connect_change_signal(self):
        """Connects the relevant signal from self.widget to self.value_changed"""
        
        if hasattr(self.widget, "textChanged"):
            self.widget.textChanged.connect(self.value_changed.emit)
        elif hasattr(self.widget, "valueChanged"):
            self.widget.valueChanged.connect(self.value_changed.emit)
        elif hasattr(self.widget, "currentIndexChanged"):
            self.widget.currentIndexChanged.connect(self.value_changed.emit)
        elif hasattr(self.widget, "currentTextChanged"):
            self.widget.currentTextChanged.connect(self.value_changed.emit)
        elif hasattr(self.widget, "stateChanged"):
            self.widget.stateChanged.connect(self.value_changed.emit)
        elif hasattr(self.widget, "clicked"):
            self.widget.clicked.connect(self.value_changed.emit)
        else:
            raise ValueError(f"⚠️ No known change signal for widget_type:{self.widget_type} -- {type(self.widget)}")
    
    def delete_self(self):
        """ Safely removes the widget from the UI """
        self.setParent(None)
        self.deleteLater()
    
    
        


class IntConfigField(BaseConfigField):
    def __init__(self, default_value=0, widget_type='', tooltip='', heading='General', category='', value_range=None, flags=None):
        super().__init__(default_value=default_value, widget_type=widget_type, tooltip=tooltip, heading=heading, category=category, flags=flags)
        self.set_widget(QSpinBox())
        value_range = value_range if value_range is not None else [-999999, 999999]
        assert isinstance(value_range, tuple) or isinstance(value_range, list), f"value range must be tuple or int but got: {type(value_range)}, value:{value_range}"
        self.widget.setRange(*value_range)
        self.set_value(default_value)
        self.connect_change_signal()

    def get_value(self):
        value = self.widget.value()
        return value if value != -1 else None

    def set_value(self, value):
        value = -1 if value is None else value
        self.widget.setValue(value)


class FloatConfigField(BaseConfigField):
    def __init__(self, default_value=0.0, widget_type='', tooltip='', heading='General', category='', value_range=None, flags=None):
        # default value cannot be None
        super().__init__(default_value=default_value, widget_type=widget_type, tooltip=tooltip, heading=heading, category=category, flags=flags)
        self.widget: QDoubleSpinBox
        self.set_widget(QDoubleSpinBox())
        value_range = value_range if value_range is not None else [-999999, 999999]
        assert isinstance(value_range, tuple) or isinstance(value_range, list), f"value range must be tuple or int but got: {type(value_range)}, value:{value_range}"
        self.widget.setSpecialValueText("None")
        self.widget.setRange(*value_range)
        self.set_value(default_value)
        self.connect_change_signal()

    def get_value(self):
        if self.widget.value() == self.widget.minimum():
            return None
        return self.widget.value()

    def set_value(self, value):
        if value is None:
            self.widget.setValue(self.widget.minimum())
        else:
            self.widget.setValue(value)


class BoolConfigField(BaseConfigField):
    def __init__(self, default_value=False, widget_type='', tooltip='', heading='General', category='', flags=None):
        super().__init__(default_value=default_value, widget_type=widget_type, tooltip=tooltip, heading=heading, category=category, flags=flags)
        self.set_widget(QCheckBox())
        self.widget.setChecked(default_value)
        self.connect_change_signal()

    def get_value(self):
        return self.widget.isChecked()

    def set_value(self, value):
        self.widget.setChecked(value)


class StringConfigField(BaseConfigField):
    def __init__(self, default_value="", widget_type='', tooltip='', heading='General', category='', flags=None):
        super().__init__(default_value=default_value, widget_type=widget_type, tooltip=tooltip, heading=heading, category=category, flags=flags)
        self.set_widget(QLineEdit())
        self.widget.setText(str(default_value))
        self.widget.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.widget.setMinimumWidth(50)
        self.widget.setMaximumWidth(500)
        self.connect_change_signal()

    def get_value(self):
        return self.widget.text()

    def set_value(self, value):
        self.widget.setText(str(value))


class DirectoryConfigField(BaseConfigField):
    """Custom widget for selecting directories."""
    def __init__(self, default_value="", widget_type='', tooltip='', heading='General', category='', flags=None):
        super().__init__(default_value=default_value, widget_type=widget_type, tooltip=tooltip, heading=heading, category=category, flags=flags)
        self.layout = QHBoxLayout()
        self.set_widget(QLineEdit())
        self.widget.setText(default_value)
        self.select_button = QPushButton("Browse")
        self.select_button.clicked.connect(self.open_file_dialog)
        self.layout.addWidget(self.select_button)
        self.layout.addWidget(self.widget)
        self.setLayout(self.layout)

        self.connect_change_signal()

    def open_file_dialog(self):
        """Open file dialog to select a directory."""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.widget.setText(directory)

    def get_value(self):
        return self.widget.text()

    def set_value(self, value):
        self.widget.setText(value)


class PathConfigField(BaseConfigField):
    """Custom widget for selecting a file or directory path."""
    def __init__(
        self,
        default_value="",
        widget_type="path",
        tooltip="",
        heading="General",
        category="",
        path_type=None,
        flags=None
    ):
        super().__init__(default_value=default_value, widget_type=widget_type, tooltip=tooltip, heading=heading, category=category, flags=flags)

        # force path type
        path_types = [path_type.lower()] if path_type else ["file", "directory"]
        
        # Save final composite widget to self.widget (for your system to use)
        widget = QWidget()  # <- this will contain everything

        self.path_type_selector = QComboBox()
        self.path_type_selector.addItems(path_types)

        self.path_label = HoverLabel(default_value, tooltip)
        self.select_button = QPushButton("Browse")
        self.select_button.clicked.connect(self.open_file_dialog)

        # Layout for path label + button + selector
        main_layout = QVBoxLayout()

        path_layout = QHBoxLayout()
        path_layout.addWidget(self.select_button)
        path_layout.addWidget(self.path_type_selector)
        path_layout.addStretch(1)

        path_label_layout = QHBoxLayout()
        path_label_layout.addWidget(self.path_label)

        main_layout.addLayout(path_label_layout)
        main_layout.addLayout(path_layout)
        main_layout.setContentsMargins(0, 0, 0, 0)
        widget.setLayout(main_layout)
        
        self.set_widget(widget)
        # self.connect_change_signal() emit directly in set value


    def open_file_dialog(self):
        """Open file or directory dialog based on current selection."""
        path_type = self.path_type_selector.currentText()

        if path_type == "directory":
            path = QFileDialog.getExistingDirectory(self, "Select Directory")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File")

        if path:
            self.path_label.setText(path)

    def get_value(self):
        return self.path_label.text()

    def set_value(self, value):
        self.path_label.setText(value)
        self.value_changed.emit()

    
    
class DictConfigField(BaseConfigField):
    def __init__(self, default_value="", widget_type='', tooltip='', heading='General', category='', flags=None):
        super().__init__(default_value=default_value, widget_type=widget_type, tooltip=tooltip, heading=heading, category=category, flags=flags)
        self.set_widget(QTextEdit())
        self.set_value(default_value)
        self.connect_change_signal()

    def get_value(self):
        return yaml.safe_load(self.widget.toPlainText())

    def set_value(self, value):
        self.widget.setPlainText(yaml.dump(value))

class ListConfigField(BaseConfigField):
    def __init__(self, default_value="", widget_type='', tooltip='', heading='General', category='', flags=None):
        super().__init__(default_value=default_value, widget_type=widget_type, tooltip=tooltip, heading=heading, category=category, flags=flags)
        self.set_widget(QComboBox())
        if isinstance(default_value, list) and len(default_value) > 0:
            self.widget.addItems([str(item) for item in default_value])
            self.widget.setCurrentIndex(0)
        self.connect_change_signal()
    
    def get_value(self):
        return self.widget.currentText()
    
    def set_value(self, value):
        if isinstance(value, list) and len(value) > 0:
            self.widget.addItems([str(item) for item in value])
            self.widget.setCurrentIndex(0)
        elif isinstance(value, str):
            self.widget.setCurrentText(value)
        else:
            pass

class SelectionConfigField(BaseConfigField):
    def __init__(self, default_value="", value_options=None, widget_type='',  tooltip='', heading='General', category='', flags=None):
        super().__init__(
            default_value=default_value,
            widget_type=widget_type,
            tooltip=tooltip,
            heading=heading,
            category=category, 
            flags=flags
        )
        self.options = value_options if value_options is not None else []
        combo = QComboBox()
        combo.addItems([str(opt) for opt in self.options])
        self.set_widget(combo)
        self.connect_change_signal()

        if default_value in self.options:
            combo.setCurrentText(str(default_value))
        elif self.options:
            combo.setCurrentIndex(0)

    def get_value(self):
        return self.widget.currentText()

    def set_value(self, value):
        if value in self.options:
            self.widget.setCurrentText(str(value))


class MultiSelectionConfigField(BaseConfigField):
    def __init__(self, default_value=None, value_options=None, widget_type='', tooltip='', heading='General', category='', flags=None):
        super().__init__(
            default_value=default_value or [],
            widget_type=widget_type,
            tooltip=tooltip,
            heading=heading,
            category=category,
            flags=flags
        )
        self.options = value_options if value_options is not None else []

        # TODO enable control flag
        self.extensible = True

        # setup selection list 
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self.list_widget.setStyleSheet(Style.list_widget_style)

        for opt in self.options:
            item = QListWidgetItem(str(opt))
            self.list_widget.addItem(item)
            if default_value and opt in default_value:
                item.setSelected(True)
        
        if self.extensible:
            widget = self.make_extensible()
        else:
            widget = self.list_widget

        self.set_widget(widget)
        # self.connect_change_signal()
        self.list_widget.itemSelectionChanged.connect(self.value_changed.emit)

    def get_value(self):
        return [item.text() for item in self.list_widget.selectedItems()]

    def set_value(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values]
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setSelected(item.text() in values)
    
    def make_extensible(self):
        # 1. Create a container and a vertical layout
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)  # Keep it flush with the parent

        # 3. Setup the custom text entry field
        self.custom_input = QLineEdit()
        self.custom_input.setPlaceholderText("Type a custom option and press Enter...")
        
        # Connect the Enter key to our custom method
        self.custom_input.returnPressed.connect(self.add_custom_option)

        # 4. Add widgets to the layout
        layout.addWidget(self.list_widget)
        layout.addWidget(self.custom_input)
        
        return container

    def add_custom_option(self):
        """Adds text from the QLineEdit to the list if it doesn't already exist."""
        text = self.custom_input.text().strip()
        
        if text:
            # Check for duplicates
            existing_items = [self.list_widget.item(i).text() for i in range(self.list_widget.count())]
            
            if text not in existing_items:
                item = QListWidgetItem(text)
                self.list_widget.addItem(item)
                self.options.append(text)  # Keep internal options list updated
                
                # Automatically select the newly added item
                item.setSelected(True)
                
                # Optional: If you need to alert the parent form that a change occurred
                # self.list_widget.itemSelectionChanged.emit() 

        # Clear the text entry field so it's ready for another input
        self.custom_input.clear()


class TupleConfigField(BaseConfigField):
    def __init__(self, default_value=(0, 0), widget_type='', tooltip='', heading='General', category='', flags=None):
        super().__init__(default_value=default_value, widget_type=widget_type, tooltip=tooltip, heading=heading, category=category, flags=flags)
        self.set_widget(QLineEdit())
        self.widget.setPlaceholderText("Enter comma separated integers")
        self.set_value(default_value)
                
        # Run initial validation
        self.validate_input()

        # Validate on change
        self.widget.textChanged.connect(self.validate_input)
        self.connect_change_signal()
    
    def validate_input(self):
        text = self.widget.text()
        if not text or text == 'None':
            self.set_valid(True)
        else:
            try:
                val = ast.literal_eval(text)
                if not isinstance(val, (tuple, list)):
                    raise ValueError("Not a tuple or list")
                self.set_valid(True)
            except Exception:
                self.set_valid(False)
    
    def get_value(self):
        """ Convert the comma-separated text into a tuple of ints. 
        Return None if no valid values are found 
        """
        try:
            return tuple(ast.literal_eval(self.widget.text()))
        except Exception:
            return None  # or raise, depending on how strict you want to be
        
    def set_value(self, value):
        if value is None:
            self.widget.clear()
        elif isinstance(value, tuple):
            self.widget.setText('(' + ', '.join(str(v) for v in value) + ')')
        else:
            # Fallback in case the value is not a tuple
            self.widget.setText(str(value))


class SegConfigModelsField(BaseConfigField):
    def __init__(self, default_value="", widget_type='', tooltip='', heading='General', category='', flags=None):
        super().__init__(default_value=default_value, widget_type=widget_type, tooltip=tooltip, heading=heading, category=category, flags=flags)
        self.widget = QTextEdit()
        self.set_value(default_value)
        self.connect_change_signal()

    def get_value(self):
        return yaml.safe_load(self.widget.toPlainText())

    def set_value(self, value):
        self.widget.setPlainText(yaml.dump(value))



class HiddenWidget:
    """ 
    A dummy widget that is not rendered but can still store and retrieve values. 
        It is still a valid config field. Purpose is to hold parameters user cannot set.
    """

    def __init__(self, default_value=None, **kwargs):
        self.widget = None
        self.default_value = default_value
        
        self.attribute_keys = []
        for k, v in kwargs.items():
            setattr(self, k, v)
            self.attribute_keys.append(k)

    def get_value(self):
        return self.default_value

    def set_value(self, value):
        self.default_value = value
        
    def get_attributes(self):
        return {k: getattr(self, k) for k in self.attribute_keys}

    def to_param_spec(self):
        attrs = self.get_attributes()
        attrs['current_value'] = self.get_value()
        return attrs
    
    def delete_self(self):
        pass
        
        
        
        

import re

class DictIntStrField(BaseConfigField):
    """Widget for dict[int, str] mapping."""
    def __init__(self, default_value=None, **kwargs):
        super().__init__(default_value=default_value or {}, **kwargs)
        container = QWidget()
        self.layout = QVBoxLayout(container)
        
        self.add_btn = QPushButton("+ Add Entry (Int -> Str)")
        self.add_btn.clicked.connect(lambda: self.add_row())
        self.layout.addWidget(self.add_btn)

        self.rows_layout = QVBoxLayout()
        self.layout.addLayout(self.rows_layout)
        
        self.set_widget(container)
        self.set_value(self.default_value)

    def add_row(self, k=0, v=""):
        row = QWidget()
        l = QHBoxLayout(row)
        l.setContentsMargins(0, 0, 0, 0)
        
        k_inp = QSpinBox()
        k_inp.setRange(-9999, 9999)
        k_inp.setValue(int(k))
        
        v_inp = QLineEdit(str(v))
        
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(20, 20)
        del_btn.clicked.connect(lambda: (self.rows_layout.removeWidget(row), row.deleteLater(), self.value_changed.emit()))

        l.addWidget(QLabel("Key (Int):"))
        l.addWidget(k_inp)
        l.addWidget(QLabel("Val (Str):"))
        l.addWidget(v_inp)
        l.addWidget(del_btn)
        
        self.rows_layout.addWidget(row)
        k_inp.valueChanged.connect(self.value_changed.emit)
        v_inp.textChanged.connect(self.value_changed.emit)

    def get_value(self):
        res = {}
        for i in range(self.rows_layout.count()):
            w = self.rows_layout.itemAt(i).widget()
            if w:
                k = w.findChild(QSpinBox).value()
                v = w.findChild(QLineEdit).text()
                res[k] = v
        return res

    def set_value(self, value):
        while self.rows_layout.count():
            child = self.rows_layout.takeAt(0).widget()
            if child: child.deleteLater()
        for k, v in value.items():
            self.add_row(k, v)

class DictAnyListField(BaseConfigField):
    """Widget for dict[str, list] where list is entered as comma-separated values."""
    def __init__(self, default_value=None, key_type='str', **kwargs):

        super().__init__(default_value=default_value or {}, **kwargs)
        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.add_btn = QPushButton(f"+ Add Entry ({key_type.capitalize()} -> List)")
        self.add_btn.clicked.connect(lambda: self.add_row())
        self.layout.addWidget(self.add_btn)
        self.rows_layout = QVBoxLayout()
        self.layout.addLayout(self.rows_layout)
        self.key_type = key_type
        self.set_widget(container)
        self.set_value(self.default_value)

    def add_row(self, k="", v=None):
        row = QWidget()
        l = QHBoxLayout(row)
        l.setContentsMargins(0, 0, 0, 0)
        k_inp = QLineEdit(str(k))
        v_inp = QLineEdit(", ".join(map(str, v)) if v else "")
        v_inp.setPlaceholderText("e.g. 1, 2, 3")
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(20, 20)
        del_btn.clicked.connect(lambda: (self.rows_layout.removeWidget(row), row.deleteLater(), self.value_changed.emit()))
        l.addWidget(QLabel("Key:"))
        l.addWidget(k_inp)
        l.addWidget(QLabel("List:"))
        l.addWidget(v_inp)
        l.addWidget(del_btn)
        self.rows_layout.addWidget(row)
        k_inp.textChanged.connect(self.value_changed.emit)
        v_inp.textChanged.connect(self.value_changed.emit)

    def get_value(self):
        res = {}
        for i in range(self.rows_layout.count()):
            w = self.rows_layout.itemAt(i).widget()
            if w:
                inputs = w.findChildren(QLineEdit)
                k = eval(f"{self.key_type}({inputs[0].text()})")

                v_text = inputs[1].text()
                res[k] = [x.strip() for x in v_text.split(",") if x.strip()]
        return res

    def set_value(self, value):
        while self.rows_layout.count():
            child = self.rows_layout.takeAt(0).widget()
            if child: child.deleteLater()
        for k, v in value.items():
            self.add_row(k, v)

class DictStrStrField(BaseConfigField):
    """Internal helper widget for dict[str, str] mapping."""
    def __init__(self, default_value=None, **kwargs):
        super().__init__(default_value=default_value or {}, **kwargs)
        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        self.add_btn = QPushButton("+ Add Regex Mapping")
        self.add_btn.clicked.connect(lambda: self.add_row())
        self.layout.addWidget(self.add_btn)

        self.rows_layout = QVBoxLayout()
        self.layout.addLayout(self.rows_layout)
        
        self.set_widget(container)
        self.set_value(self.default_value)

    def add_row(self, k="", v=""):
        row = QWidget()
        l = QHBoxLayout(row)
        l.setContentsMargins(0, 0, 0, 0)
        
        k_inp = QLineEdit(str(k))
        k_inp.setPlaceholderText("Regex Pattern")
        
        v_inp = QLineEdit(str(v))
        v_inp.setPlaceholderText("Category Label")
        
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(20, 20)
        del_btn.clicked.connect(lambda: (self.rows_layout.removeWidget(row), row.deleteLater(), self.value_changed.emit()))

        l.addWidget(k_inp)
        l.addWidget(QLabel("→"))
        l.addWidget(v_inp)
        l.addWidget(del_btn)
        
        self.rows_layout.addWidget(row)
        k_inp.textChanged.connect(self.value_changed.emit)
        v_inp.textChanged.connect(self.value_changed.emit)

    def get_value(self):
        res = {}
        for i in range(self.rows_layout.count()):
            w = self.rows_layout.itemAt(i).widget()
            if w:
                inputs = w.findChildren(QLineEdit)
                if len(inputs) >= 2:
                    k, v = inputs[0].text(), inputs[1].text()
                    if k: res[k] = v
        return res

    def set_value(self, value):
        while self.rows_layout.count():
            child = self.rows_layout.takeAt(0).widget()
            if child: child.deleteLater()
        for k, v in value.items():
            self.add_row(k, v)

class ListListIntField(BaseConfigField):
    """Widget for list[list[int]], entries are comma-separated ints."""
    def __init__(self, default_value=None, **kwargs):
        super().__init__(default_value=default_value or [], **kwargs)
        container = QWidget()
        self.layout = QVBoxLayout(container)
        self.add_btn = QPushButton("+ Add Int List")
        self.add_btn.clicked.connect(lambda: self.add_row())
        self.layout.addWidget(self.add_btn)
        self.rows_layout = QVBoxLayout()
        self.layout.addLayout(self.rows_layout)
        self.set_widget(container)
        self.set_value(self.default_value)

    def add_row(self, v=None):
        row = QWidget()
        l = QHBoxLayout(row)
        l.setContentsMargins(0, 0, 0, 0)
        v_inp = QLineEdit(", ".join(map(str, v)) if v else "")
        v_inp.setPlaceholderText("e.g. 0, 1")
        del_btn = QPushButton("✕")
        del_btn.setFixedSize(20, 20)
        del_btn.clicked.connect(lambda: (self.rows_layout.removeWidget(row), row.deleteLater(), self.value_changed.emit()))
        l.addWidget(v_inp)
        l.addWidget(del_btn)
        self.rows_layout.addWidget(row)
        v_inp.textChanged.connect(self.value_changed.emit)

    def get_value(self):
        res = []
        for i in range(self.rows_layout.count()):
            w = self.rows_layout.itemAt(i).widget()
            if w:
                txt = w.findChild(QLineEdit).text()
                try:
                    res.append([int(x.strip()) for x in txt.split(",") if x.strip()])
                except ValueError:
                    continue
        return res

    def set_value(self, value):
        while self.rows_layout.count():
            child = self.rows_layout.takeAt(0).widget()
            if child: child.deleteLater()
        for v in value:
            self.add_row(v)

class ExtractGroupsField(BaseConfigField):
    """Specialized widget for List[Dict[str, Any]] to handle metadata grouping."""
    def __init__(self, default_value=None, **kwargs):
        super().__init__(default_value=default_value or [], **kwargs)
        
        container = QWidget()
        self.layout = QVBoxLayout(container)

        self.add_btn = QPushButton("+ Add New Grouping Variable")
        self.add_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.add_btn.clicked.connect(lambda: self.add_group())
        
        self.groups_layout = QVBoxLayout()
        self.layout.addWidget(self.add_btn)
        self.layout.addLayout(self.groups_layout)
        
        # Track all active group UI components in a list for reliable data retrieval
        self.group_entries = [] 
        
        self.set_widget(container)
        self.set_value(self.default_value)

    def add_group(self, data=None):
        data = data or {"var": "", "attrib": "image_path", "mapping": {}}
        
        group_frame = QGroupBox("Grouping Configuration")
        form = QFormLayout(group_frame)

        var_name = QLineEdit(data.get("var", ""))
        attrib_name = QLineEdit(data.get("attrib", "image_path"))
        mapping_editor = DictStrStrField(default_value=data.get("mapping", {}))
        
        del_btn = QPushButton("Remove This Grouping Variable")
        del_btn.setStyleSheet("color: #e74c3c;")
        
        # Store references in a dictionary for this specific row
        entry_record = {
            "var_widget": var_name,
            "attrib_widget": attrib_name,
            "mapping_widget": mapping_editor,
            "frame": group_frame
        }
        self.group_entries.append(entry_record)

        # Cleanup logic
        del_btn.clicked.connect(lambda: self.remove_group(entry_record))
        
        form.addRow("Variable Name (var):", var_name)
        form.addRow("Metadata Attrib (attrib):", attrib_name)
        form.addRow("Regex Mappings:", mapping_editor.get_widget())
        form.addRow(del_btn)

        self.groups_layout.addWidget(group_frame)
        
        # Connect change signals
        var_name.textChanged.connect(self.value_changed.emit)
        attrib_name.textChanged.connect(self.value_changed.emit)
        mapping_editor.value_changed.connect(self.value_changed.emit)

    def remove_group(self, entry_record):
        self.groups_layout.removeWidget(entry_record['frame'])
        entry_record['frame'].deleteLater()
        if entry_record in self.group_entries:
            self.group_entries.remove(entry_record)
        self.value_changed.emit()

    def get_value(self):
        """Extracts values using direct references instead of findChild."""
        groups = []
        for entry in self.group_entries:
            groups.append({
                "var": entry["var_widget"].text(),
                "attrib": entry["attrib_widget"].text(),
                "mapping": entry["mapping_widget"].get_value()
            })
        return groups

    def set_value(self, value):
        # Clear existing entries properly
        for entry in list(self.group_entries):
            self.remove_group(entry)
            
        if isinstance(value, list):
            for group_data in value:
                self.add_group(group_data)