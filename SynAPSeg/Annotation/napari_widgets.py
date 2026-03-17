import os
import sys
import re
import pprint
from typing import Optional,  Dict, List
import yaml

import napari
from napari.layers import Labels
from napari.layers import Labels, Image
from napari.types import LayerDataTuple
from napari.viewer import Viewer
from magicgui import magicgui

import numpy as np
from skimage.morphology import binary_dilation, binary_erosion
from skimage.draw import draw
from scipy.ndimage import binary_fill_holes
from weakref import WeakKeyDictionary

from qtpy.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QPoint
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QListWidget, QSpinBox,
    QPushButton, QLabel, QRadioButton, QButtonGroup, QLineEdit,
    QMainWindow, QApplication, QToolBox, QMessageBox, 
    QFileDialog, QToolButton, QStyle, QListWidgetItem, QSizePolicy,
    QTextEdit, QDialog, QComboBox, QSlider,
    QScrollArea, QGroupBox,QTabWidget, QGraphicsOpacityEffect
)


from SynAPSeg.utils import utils_image_processing as uip
from SynAPSeg.utils import utils_general as ug
from SynAPSeg.utils import utils_colocalization as uc
from SynAPSeg.IO.metadata_handler import MetadataParser
from pathlib import Path




############################ widget container (tabs) class ############################
def _infer_widget_title(widget: QWidget) -> str:
    """Infer a human-readable title for a widget."""
    # 1) explicit name attr (good for magicgui too)
    name_attr = getattr(widget, "name", None)
    if isinstance(name_attr, str) and name_attr.strip():
        return name_attr.strip()

    # 2) Qt objectName
    obj_name = widget.objectName()
    if isinstance(obj_name, str) and obj_name.strip():
        return obj_name.strip()

    # 3) Class name -> "Class Name"
    cls_name = widget.__class__.__name__
    # split CamelCase into words: LabelEditWidget -> "Label Edit Widget"
    title = re.sub(r"(?<!^)(?=[A-Z])", " ", cls_name).strip()
    
    return title or str(widget) # think names must be unique or napari will be unhappy?

def _as_qwidget(widget) -> QWidget:
    """
    Normalize different widget types to a QWidget.

    Supports:
    - plain Qt widgets (QWidget subclasses)
    - magicgui widgets / FunctionGui with a `.native` QWidget
    """
    if isinstance(widget, QWidget):
        return widget

    # magicgui widgets usually have a .native attribute that is a QWidget
    native = getattr(widget, "native", None)
    if isinstance(native, QWidget):
        return native

    # If we get here, we don't know how to display this object
    raise TypeError(
        f"Cannot convert object of type {type(widget)!r} to QWidget; "
        f"expected QWidget or an object with .native QWidget."
    )


def make_tabbed_control_panel(
    tab_to_widgets: Dict[str, List[QWidget]],
) -> QWidget:
    """
    Create a single QWidget containing a QTabWidget where each tab is scrollable
    and holds a vertical list of widgets.

    Parameters
    ----------
    tab_to_widgets:
        Mapping from tab name (str) to a list of QWidget instances.
        Widgets can be standard Qt or magicgui widgets.

    Returns
    -------
    QWidget
        A container suitable for viewer.window.add_dock_widget(...).
    """
    # Outer container (what goes into napari's dock)
    container = QWidget()
    container_layout = QVBoxLayout(container)
    container_layout.setContentsMargins(0, 0, 0, 0)

    tabs = QTabWidget()
    container_layout.addWidget(tabs)

    for tab_name, widgets in tab_to_widgets.items():
        # Scroll area per tab
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        inner = QWidget()
        inner_layout = QVBoxLayout(inner)
        inner_layout.setContentsMargins(4, 4, 4, 4)
        inner_layout.setSpacing(4)

        for w in widgets:
            if w is None:
                continue

            title = _infer_widget_title(w)
            qwidget = _as_qwidget(w)

            # Wrap each widget in a group box so you can see what is what
            box = QGroupBox(title)
            box_layout = QVBoxLayout(box)
            box_layout.setContentsMargins(4, 10, 4, 4) # add a little space to avoid cropping the bottom of the label
            box_layout.addWidget(qwidget)

            inner_layout.addWidget(box)

        inner_layout.addStretch(1)
        scroll.setWidget(inner)

        tabs.addTab(scroll, tab_name)

    return container

# helpers/ generalized widget functions
####################################################################
class ToastLabel(QLabel):
    """
    A Widget used to convey temporary messages, like notifications.
    It auto-wraps text, centers itself, and fades out.
    """
    def __init__(self, parent, message, display_time_ms=1500):
        super().__init__(message, parent)
        
        # 1. Setup Text Handling
        self.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center text within the label
        self.setWordWrap(True) # Allow text to wrap to new lines
        
        # 2. Styling
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(76, 175, 80, 210); 
                color: white; 
                border-radius: 10px; 
                padding: 10px 15px;
                font-weight: bold;
                font-size: 12px;
            }
        """)
        
        # 3. Dynamic Sizing
        # We enforce that the toast creates a margin (e.g., 20% total margin)
        # so it never touches the absolute edge of the widget.
        parent_width = parent.width()
        max_width = int(parent_width * 0.8) # Max width is 80% of parent
        
        # We must set a minimum width to prevent very short words from looking squashed,
        # but ensure min_width isn't larger than max_width
        min_width = min(100, max_width) 
        
        self.setMinimumWidth(min_width)
        self.setMaximumWidth(max_width)
        
        # adjustSize() will now calculate height based on the width constraints
        self.adjustSize()
        
        # 4. Positioning (Centering)
        # Calculate position based on the final adjusted size
        x = (parent_width - self.width()) // 2
        y = (parent.height() - self.height()) // 2
        self.move(x, y)
        
        self.show()

        # 5. Auto-close Logic
        self.timer = QTimer()
        self.timer.singleShot(display_time_ms, self.fade_out)

    def fade_out(self):
        self.effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.effect)
        
        self.anim = QPropertyAnimation(self.effect, b"opacity")
        self.anim.setDuration(500)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        
        self.anim.finished.connect(self.close)
        self.anim.finished.connect(self.deleteLater)
        self.anim.start()


############################ widget zoo ############################

class FilterCollection:
    def __init__(self, viewer, **kwargs):
        self.viewer = viewer
        for k,v in kwargs.items():
            setattr(self, k, v)
        self.filters = {}
        self.filters_attrs = {}
        
    def add_filter(self, filter_key, labelImgFilter, filter_attrs={}):
        self.filters[filter_key] = labelImgFilter
        self.filters_attrs[filter_key] = {}
        for k,v in filter_attrs.items():
            self.filters_attrs[filter_key][k] = v
    
    def __getitem__(self, key):
        from Annotation.annotation_core import get_intimg_layer

        if isinstance(key, int):
            # If the key is an integer, get the filter by index
            key = list(self.filters.keys())[key]
        out = get_intimg_layer(key, self.filters)
        if out is None: 
            raise ValueError(f'{key} not in filters: {self.filters.keys()}')
        return out

    def setup(self, viewer):
        from Annotation.annotation_core import get_label_layer_properties
        from Annotation.annotation_core import get_intimg_layer

        for layer in viewer.layers:
            print(layer)
            continue_flag = False
            # only apply to labels
            if not isinstance(layer, napari.layers.labels.labels.Labels):
                continue_flag = True
                
            mapped_intimg_layername = get_intimg_layer(layer.name, self.LABEL_INT_MAP)
            if mapped_intimg_layername is None: 
                continue_flag = True
            
            if mapped_intimg_layername not in [el.name for el in viewer.layers]:
                print(f"mapped_intimg_layername ({mapped_intimg_layername}) not in viewer layer names")
                continue_flag = True
            
            if not continue_flag:
                
                if layer.data.shape != viewer.layers[mapped_intimg_layername].data.shape:
                    print(f'filter_widget shape mismatch between labels: {layer.name} and image: {viewer.layers[mapped_intimg_layername].name}. widget not available for this layer.')
                else:
                    rpdf = uc.get_rp_table(layer.data, viewer.layers[mapped_intimg_layername].data, ch_colocal_id=self.clc_map, prt_str='')
                    clc_id = 0
                    channel_key = get_intimg_layer(layer.name, self.channel_map)
                    properties = get_label_layer_properties(rpdf, clc_id)
                    properties['clc_id'] = clc_id
                    properties['ch_i'] = channel_key
                    layer.properties = properties
                    lbl_filterer = LabelImgFilter(layer.data, rpdf, self.clc_map)
                    self.add_filter(layer.name, lbl_filterer)
    
    def thresh_all_filters(self):
        pass
        # actually, don't want to thresh before hand since that will distrupt annotations
       
                
    def get_widget_init_vals(self):
        # just to save code in viewer loop
        min_intensity, max_intensity = [0, 99999]
        min_size, max_size = [0, 99999]
        min_ecc, max_ecc = [0, 99999]
        return min_intensity, max_intensity, min_size, max_size, min_ecc, max_ecc


class LabelImgFilter:
    def __init__(self, label_image, rpdf, clc_map, name=None):
        self.img = label_image 
        # for ch_i in range(self.img.shape[-1]): # relabel so every object has unique id
        #     self.img[..., ch_i] = ndi.label(self.img[..., ch_i])[0]
        self.rpdf = rpdf
        self.clc_map = clc_map
        self.name = name
    
    def filter_rpdf(self, threshold_dict=None):
        rpdf_filtered, fc = uc.filter_nuclei(threshold_dict, self.rpdf, self.clc_map.keys())
        uc.pretty_print_fcounts(fc)
        return rpdf_filtered
    
    def filter_label_image(self, threshold_dict, clc_id):
        # need to recalculate labels since napari expects input to map linearly from label i to prop val
        self.rpdf_filtered = self.filter_rpdf(threshold_dict)
        ch_i = self.clc_map[clc_id]
        og_lbls, filt_lbls = self.rpdf[self.rpdf['colocal_id']==clc_id]['label'].to_list(), self.rpdf_filtered[self.rpdf_filtered['colocal_id']==clc_id]['label'].to_list()
        filtered_out_lbls = list(set(og_lbls) - set(filt_lbls))
        kept_label_img = uip.filter_label_img(self.img[...,ch_i] if self.img.ndim>2 else self.img, filt_lbls)
        # removed_label_img = uip.filter_label_img(self.img[...,ch_i], filtered_out_lbls)
        return kept_label_img, self.rpdf_filtered #, removed_label_img


def get_filter_object(
    viewer, 
    LABEL_INT_MAP, 
    clc_map={0:0}, 
    channel_map=None, 
    filter_params_key=lambda x: 'neurseg' if 'neurseg' in x  else 'stardist'
    ) -> FilterCollection:
    
    if channel_map is None:
        channel_map = {'axons':0, 'dentrites':1, 'soma':2, 'stardist_ch0':0, 'stardist_ch1':1, 'stardist_ch2':2, 'stardist_ch3':3}
        
    filterObj = FilterCollection(
            viewer,
            LABEL_INT_MAP = LABEL_INT_MAP,
            clc_map = clc_map,
            channel_map = channel_map,
            filter_params_key = filter_params_key,
        )
    filterObj.setup(viewer)
    
    return filterObj


class ExportWidget:
    def __init__(self, basedir, exmd, logger=None):
        self.basedir = basedir 
        self.exmd = exmd or {} # allow this to work outside of example context - though some features won't work
        self.ROI_map = {}
        self.logger = logger
        
    def get_basedir(self):
        if self.basedir is None:
            raise ValueError("No base directory set. Cannot export.")
        if not os.path.exists(self.basedir):
            raise ValueError(f"Base directory does not exist/invalid. Cannot export.\n\tgot: `{self.basedir}`")
        return self.basedir

    def register_ROI(self, base_layer_name, roi_layer_name, base_layer_shape, roi_shape, dimension):
        
        # try: # to get format of the base layer this ROI was created on 
        #     base_layer_fmt = self.exmd['data_metadata']['data_formats'][base_layer_name]
        # except:
        #     base_layer_fmt = None
        
        # infer format from dimension
        if dimension == "2D":
            format = "YX"
        elif dimension == "3D":
            format = "ZYX"

        self.ROI_map[roi_layer_name] = {
            "base_layer_name":base_layer_name, "base_layer_shape":base_layer_shape, 
            "format":format, "shape":roi_shape, "dimension_selected":dimension,
        }
    
    def register_export(self, layer):

        # TODO: perhaps much of this code should be handled by ROI widget or a context manager
        name = layer.name
        shape = layer.data.shape

        # TODO: need to validate - many ways this could get screwed up 
        # e.g. you create an ROI delete it and rename a new one

        # update exmd with this object's shape 
        if 'data_metadata' not in self.exmd:
            self.exmd['data_metadata'] = {'data_shapes':{}, 'data_formats':{}}
        if 'data_shapes' not in self.exmd['data_metadata']:
            self.exmd['data_metadata']['data_shapes'] = {}
        if 'data_formats' not in self.exmd['data_metadata']:
            self.exmd['data_metadata']['data_formats'] = {}

        if 'ROI_exports' not in self.exmd['data_metadata']:
            self.exmd['data_metadata']['ROI_exports'] = {}
        
        # check if exporting a registered ROI
        registered = name in self.ROI_map 

        if registered:
            roi_info = self.ROI_map.get(name) or {}
            self.exmd['data_metadata']['ROI_exports'][name] = roi_info
            self.exmd['data_metadata']['data_shapes'][name] = shape 
            self.exmd['data_metadata']['data_formats'][name] = roi_info['format']
            ug.log_or_print(f'register_export of {name} as an ROI', self.logger)
        else:
            ug.log_or_print(f'register_export of {name} - not an ROI ({self.ROI_map})', self.logger)
            
    

    def write_exmd_update(self):
        MetadataParser.write_metadata(self.get_basedir(), self.exmd)


    def set_basedir(self, basedir):
        self.basedir = basedir


class AddROIWidget(QWidget):
    """
    Simple widget to create 2D/3D ROI layers on the currently selected layer
    in a napari viewer and register them via `exwidg.register_ROI(...)`.
    """

    def __init__(self, viewer: napari.Viewer, exwidg, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.name = 'Add ROI Widget'

        self.viewer = viewer
        self.exwidg = exwidg

        self._build_ui()

    def _build_ui(self):
        self.setLayout(QVBoxLayout())

        # Dimension selector row
        dim_row = QHBoxLayout()
        dim_label = QLabel("ROI dimension:")
        self.dimension_combo = QComboBox()
        self.dimension_combo.addItems(["2D", "3D"])
        self.dimension_combo.setCurrentText("2D")

        dim_row.addWidget(dim_label)
        dim_row.addWidget(self.dimension_combo)
        dim_row.addStretch()

        # Add ROI button
        self.add_roi_button = QPushButton("Add ROI")
        self.add_roi_button.clicked.connect(self.add_roi)

        # Assemble layout
        self.layout().addLayout(dim_row)
        self.layout().addWidget(self.add_roi_button)

    def add_roi(self):
        """Callback to create and register a new ROI layer."""
        current_layer = self.viewer.layers.selection.active
        if current_layer is None:
            QMessageBox.warning(
                self,
                "No Active Layer",
                "Please select a layer to create an ROI on.",
            )
            print("AddROIWidget: please select a layer to create an ROI on")
            return

        dimension = self.dimension_combo.currentText()
        current_layer_name = current_layer.name
        base_img_shape = current_layer.data.shape

        # Decide ROI shape based on dimension
        if dimension == "3D":
            roi_shape = list(base_img_shape)[-3:]  # (Z, Y, X)-like
        else:  # "2D"
            roi_shape = list(base_img_shape)[-2:]  # (Y, X)-like

        # Determine next ROI index based on existing ROI_* layers
        next_ROI_i = len([el for el in self.viewer.layers if el.name.startswith("ROI_")])
        roi_layer_name = f"ROI_{next_ROI_i}"

        # Create labels layer
        roi_data = np.zeros(roi_shape, dtype="int32")
        roi_layer = self.viewer.add_labels(roi_data, name=roi_layer_name)

        # Register new ROI with exwidg
        self.exwidg.register_ROI(
            base_layer_name=current_layer_name,
            roi_layer_name=roi_layer_name,
            base_layer_shape=base_img_shape,
            roi_shape=roi_shape,
            dimension=dimension,
        )

        print(f"AddROIWidget: added ROI layer '{roi_layer.name}'")


class MetadataWidget(QWidget):
    def __init__(self, viewer: napari.Viewer, exwidg):
        """ 
        class to create combined mark complete/skip and view metadata widget
            Usage: MetadataWidget(viewer, exwidg)

        """
        super().__init__()

        self.viewer = viewer
        self.exwidg = exwidg  # Object containing the metadata

        self.setLayout(QVBoxLayout())

        # Button to create 'complete.txt'
        self.complete_button = QPushButton("Mark as complete")
        self.complete_button.clicked.connect(self.create_complete_file)
        self.layout().addWidget(self.complete_button)

        # Button to create '__skip_this_example.txt'
        self.skip_button = QPushButton("Mark as skip")
        self.skip_button.clicked.connect(self.create_skip_file)
        self.layout().addWidget(self.skip_button)

        # Button to remove any marks that exist
        self.remove_marks_button = QPushButton("Remove marks")
        self.remove_marks_button.clicked.connect(self.remove_marks)
        self.layout().addWidget(self.remove_marks_button)


        # Button to view metadata
        self.view_metadata_button = QPushButton("View metadata")
        self.view_metadata_button.clicked.connect(self.show_metadata)
        self.layout().addWidget(self.view_metadata_button)

    def create_complete_file(self):
        """
        Create a 'complete.txt' file in the current Example directory.
        """
        file_path = os.path.join(self.exwidg.get_basedir(), 'complete.txt')
        if os.path.exists(file_path):
            message = f"complete.txt already exists at:\n  {file_path}."
        else:
            with open(file_path, 'w') as file:
                file.write("The Annotation is complete.")
            message = f"complete.txt created successfully at:\n  {file_path}."
        print(message)
        ToastLabel(self, message)

    def create_skip_file(self):
        """
        Create a '__skip_this_example.txt' file in the current Example directory.
        """
        file_path = os.path.join(self.exwidg.get_basedir(), '__skip_this_example.txt')
        if os.path.exists(file_path):
            message = f"__skip_this_example.txt already exists at:\n  {file_path}."
        else:
            with open(file_path, 'w') as file:
                file.write("The Annotation is marked as skipped.")
            message = f"__skip_this_example.txt created successfully at:\n  {file_path}."
        print(message)
        ToastLabel(self, message)

    def remove_marks(self):
        """
        Remove any marks that exist in the current Example directory.
        """
        success_deletes = 0
        for fn in ['complete.txt', '__skip_this_example.txt']:
            file_path = os.path.join(self.exwidg.get_basedir(), fn)
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"{fn} removed successfully at:\n  {file_path}.")
                success_deletes += 1
        if success_deletes == 0:
            message = "No marks found in the current directory."
        else:
            message = f"{success_deletes} marks removed successfully."
        print(message)
        ToastLabel(self, message)

    def show_metadata(self):
        # 1. Create the dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Metadata Viewer")
        dialog.setLayout(QVBoxLayout())

        # 2. Set Size to 800x800
        dialog.resize(800, 800)

        # 3. Center the Window on the Screen
        # We need to get the geometry of the screen that contains the main window
        screen = self.window().windowHandle().screen() 
        screen_geo = screen.geometry()
        
        # Calculate the center position
        x = (screen_geo.width() - dialog.width()) // 2
        y = (screen_geo.height() - dialog.height()) // 2
        
        dialog.move(screen_geo.x() + x, screen_geo.y() + y)

        # 4. Create text area
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        
        # Optional: Increase font size slightly for readability in a large window
        font = text_edit.font()
        font.setPointSize(10) 
        text_edit.setFont(font)

        # 5. content
        # We keep this left-aligned so the 'pprint' indentation stays correct
        metadata_str = pprint.pformat(self.exwidg.exmd, indent=2)
        text_edit.setPlainText(metadata_str)

        dialog.layout().addWidget(text_edit)
        dialog.exec()

   


class PatchNavigator(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.patch_coords = []
        self.current_index = -1
        self.patch_size = 64  # Default patch size

        # Create UI elements
        self.layout = QVBoxLayout()

        self.label = QLabel("Click 'Start', then 'add grid' to display patches.\nUse 'next' and 'previous' to navigate patches.")
        self.layout.addWidget(self.label)

        size_layout = QHBoxLayout()
        size_label = QLabel("Patch size:")
        self.size_input = QLineEdit(str(self.patch_size))
        self.size_input.setToolTip("Enter the patch size as an integer")
        size_layout.addWidget(size_label)
        size_layout.addWidget(self.size_input)
        self.layout.addLayout(size_layout)

        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start)
        self.layout.addWidget(self.start_button)

        self.grid_button = QPushButton("Add Grid")
        self.grid_button.clicked.connect(self.add_grid)
        self.layout.addWidget(self.grid_button)


        nav_layout = QHBoxLayout()
        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.next_patch)
        nav_layout.addWidget(self.next_button)

        self.prev_button = QPushButton("Previous")
        self.prev_button.clicked.connect(self.previous_patch)
        nav_layout.addWidget(self.prev_button)
        self.layout.addLayout(nav_layout)

        self.setLayout(self.layout)

    def get_focus_coords(self, patch_size=64):
        current_layer = self.viewer.layers.selection.active
        current_datashape = current_layer.data.shape
        if current_layer.data.ndim == 2:
            shape = current_datashape
        else:
            shape = current_datashape[-2:]

        patch_coords = uip.coord_patch_iterator(np.zeros(shape), "yx", patch_size=(patch_size, patch_size))
        patch_coords = [[el[0].start, el[0].stop, el[1].start, el[1].stop] for el in patch_coords]
        return patch_coords

    def start(self):
        try:
            self.patch_size = int(self.size_input.text())
        except ValueError:
            self.label.setText("Invalid patch size. Please enter an integer.")
            return

        self.patch_coords = self.get_focus_coords(patch_size=self.patch_size)
        self.current_index = 0
        self.label.setText(f"Loaded {len(self.patch_coords)} patches. Currently at patch 1.")
        if self.patch_coords:
            self.center_on_current_patch()

    def next_patch(self):
        if self.patch_coords and self.current_index < len(self.patch_coords) - 1:
            self.current_index += 1
            self.label.setText(f"Currently at patch {self.current_index + 1}.")
            self.center_on_current_patch()

    def previous_patch(self):
        if self.patch_coords and self.current_index > 0:
            self.current_index -= 1
            self.label.setText(f"Currently at patch {self.current_index + 1}.")
            self.center_on_current_patch()

    def center_on_current_patch(self):
        bbox = self.patch_coords[self.current_index]
        center_view_on_bbox(self.viewer, bbox)

    def add_grid(self):
        if not self.patch_coords:
            self.label.setText("No patches loaded. Click 'Start' first.")
            return

        rectangles = []
        for bbox in self.patch_coords:
            y_min, y_max, x_min, x_max = bbox
            rectangles.append([[y_min, x_min], [y_min, x_max], [y_max, x_max], [y_max, x_min]])

        self.viewer.add_shapes(
            rectangles,
            shape_type='rectangle',
            edge_color='red',
            face_color='transparent',
            name='Patch Grid'
        )
        self.label.setText("Grid added to viewer.")


def center_view_on_bbox(viewer: napari.Viewer, bbox: tuple):
    """
    Center the Napari viewer on a specified bounding box.

    Parameters:
    viewer (napari.Viewer): The Napari viewer object.
    bbox (tuple): The bounding box defined as (x_min, x_max, y_min, y_max).
    """
    x_min, x_max, y_min, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    # Update camera center
    viewer.camera.center = (x_center, y_center)

    # Calculate zoom level based on the bounding box and viewer dimensions
    width = x_max - x_min
    height = y_max - y_min

    # Get the viewer's canvas dimensions (in data space)
    canvas_size = viewer.window._qt_window.qt_viewer.view.canvas.size()
    canvas_width = canvas_size.width()
    canvas_height = canvas_size.height()

    # Compute zoom: the zoom level is proportional to the canvas size divided by the larger bounding box dimension
    viewer.camera.zoom = min(canvas_width / width, canvas_height / height)


class PatchNavigatorWindow(QMainWindow):
    """wrapper around PatchNavigator to make it a free floating dialog box"""
    def __init__(self, viewer):
        super().__init__()
        self.setWindowTitle("Patch Navigator")
        self.navigator_widget = PatchNavigator(viewer)
        self.setCentralWidget(self.navigator_widget)

def add_patch_navigator_widget(viewer, mode):
    """ add widget, mode = ['dock', 'window']"""
    if mode == 'dock':
        widgetPatchNavigator = PatchNavigator(viewer)
        viewer.window.add_dock_widget(widgetPatchNavigator, area='right')
        return widgetPatchNavigator
    elif mode == 'window':
        app = QApplication(sys.argv)
        patch_navigator_window = PatchNavigatorWindow(viewer)
        patch_navigator_window.show()
        return patch_navigator_window
    else:
        raise ValueError(mode)



class AutofillPlugin(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.autofill_enabled = True
        self.active_labels_layer = None

        # UI Elements
        self.layout = QVBoxLayout(self)
        self.autofill_checkbox = QCheckBox("Enable Autofill")
        self.autofill_checkbox.setToolTip("Automatically closes holes when drawing shapes.")
        self.autofill_checkbox.stateChanged.connect(self.toggle_autofill)
        self.layout.addWidget(self.autofill_checkbox)
        # self.toggle_autofill(2 if self.autofill_enabled else 0)
        self.autofill_checkbox.setChecked(self.autofill_enabled)

        self.viewer.layers.selection.events.changed.connect(self._on_layer_selection_change)

    def toggle_autofill(self, state):
        self.autofill_enabled = state == 2  # QCheckBox checked state
        self._update_autofill_callback()

    def _on_layer_selection_change(self, event=None):
        layer = self.viewer.layers.selection.active
        # print(f"Selected layer: {layer}")  # Debug
        
        if isinstance(layer, Labels):
            self.active_labels_layer = layer
            self._update_autofill_callback()
            
        # else: # not needed
        #     self._remove_autofill_callback()

    def _update_autofill_callback(self):
        if self.active_labels_layer and self.autofill_enabled:
            if self._fill_holes not in self.active_labels_layer.mouse_drag_callbacks:
                self.active_labels_layer.mouse_drag_callbacks.append(self._fill_holes)
                
        elif self.active_labels_layer:
            self._remove_autofill_callback()

    def _remove_autofill_callback(self):
        if self.active_labels_layer and self._fill_holes in self.active_labels_layer.mouse_drag_callbacks:
            self.active_labels_layer.mouse_drag_callbacks.remove(self._fill_holes)
           

    def _fill_holes(self, layer: Labels, event):
        """ v2 speed up by operating on bbox surrounding draw """
        if layer.mode != "paint":
            return

        try:
            # Which axes are currently displayed (2D view)
            dims_displayed = layer._slice_input.displayed

            # Start with the current displayed slice (all along displayed dims, 0 along others)
            slice_dims = tuple(
                slice(None) if i in dims_displayed else 0 for i in range(layer.ndim)
            )

            data_slice = layer.data[slice_dims]
            poly = []
            z_index = 0

            # Start listening for mouse events
            yield
            while event.type == "mouse_move":
                coord = layer.world_to_data(event.position)
                image_coord = tuple(int(np.rint(coord[i])) for i in dims_displayed)
                poly.append(image_coord)
                # assuming first axis is z
                z_index = int(np.rint(coord[0]))
                yield

            if not poly:
                return

            poly = np.array(poly, dtype=int)

            # ---- SPEED-UP: work only in a bounding box ROI ----
            rows = poly[:, 0]
            cols = poly[:, 1]

            r_min = max(rows.min(), 0)
            r_max = min(rows.max(), data_slice.shape[0] - 1)
            c_min = max(cols.min(), 0)
            c_max = min(cols.max(), data_slice.shape[1] - 1)

            # Optional padding so the fill can "see" the border properly
            pad = 1
            r_min = max(r_min - pad, 0)
            r_max = min(r_max + pad, data_slice.shape[0] - 1)
            c_min = max(c_min - pad, 0)
            c_max = min(c_max + pad, data_slice.shape[1] - 1)

            # Local coordinates inside the ROI
            poly_local_r = rows - r_min
            poly_local_c = cols - c_min

            roi_shape = (r_max - r_min + 1, c_max - c_min + 1)

            # Draw only in this small ROI
            current_draw_roi = np.zeros(roi_shape, dtype=bool)
            rr, cc = draw.polygon(poly_local_r, poly_local_c, roi_shape)
            current_draw_roi[rr, cc] = True

            # Fill holes in ROI
            filled_roi = binary_fill_holes(current_draw_roi)

            # Respect preserve_labels only inside ROI
            # Update slice_dims so other dims use z_index
            slice_dims = tuple(
                slice(None) if i in dims_displayed else z_index for i in range(layer.ndim)
            )

            data_slice = layer.data[slice_dims]  # view into layer.data

            if layer.preserve_labels:
                roi_labels = data_slice[r_min : r_max + 1, c_min : c_max + 1]
                filled_roi &= roi_labels == 0

            # Apply changes back to the layer in-place
            roi_labels = data_slice[r_min : r_max + 1, c_min : c_max + 1]
            roi_labels[filled_roi] = layer.selected_label

            # No need to reassign layer.data[slice_dims]; roi_labels is a view
            layer.refresh()

        except Exception as e:
            print(f"Error in _fill_holes: {e}")


def launch_autofill_plugin(viewer):
    plugin = AutofillPlugin(viewer)
    viewer.window.add_dock_widget(plugin, name="Autofill Plugin")
    return plugin





class LabelMorphologyWidget(QWidget):
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_label = None
        
        # Setup layout
        layout = QVBoxLayout()
        
        # Operation selection
        op_layout = QHBoxLayout()
        self.op_group = QButtonGroup()
        
        self.dilate_radio = QRadioButton("Dilate")
        self.erode_radio = QRadioButton("Erode")
        self.op_group.addButton(self.dilate_radio)
        self.op_group.addButton(self.erode_radio)
        self.dilate_radio.setChecked(True)
        
        op_layout.addWidget(QLabel("Operation:"))
        op_layout.addWidget(self.dilate_radio)
        op_layout.addWidget(self.erode_radio)
        layout.addLayout(op_layout)
        
        # Apply button
        apply_button = QPushButton("Apply Morphology")
        apply_button.clicked.connect(self.apply_morphology)
        layout.addWidget(apply_button)
        
        self.setLayout(layout)
        
        # Connect to mouse click event
        viewer.layers.events.inserted.connect(self._connect_label_layer)
    
    def _connect_label_layer(self, event):
        """Connect to label layer mouse click events"""
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                layer.mouse_double_click_callbacks.append(self._on_label_click)
    
    def _on_label_click(self, layer, event):
        """Capture the clicked label's value"""
        # Get coordinates of the click
        coordinates = tuple(map(int, layer.coordinates))
        
        # Get the label value at the clicked coordinates
        self.current_label = layer.data[coordinates]
        print(f"Selected label: {self.current_label}")
    
    def apply_morphology(self):
        """Apply dilation or erosion to the selected label"""
        if self.current_label is None:
            print("No label selected. Click on a label first.")
            return
        
        # Find all layers
        label_layers = [
            layer for layer in self.viewer.layers 
            if isinstance(layer, napari.layers.Labels)
        ]
        
        if not label_layers:
            print("No label layers found.")
            return
        
        # Use the first label layer if multiple exist
        label_layer = label_layers[0]
        labels = label_layer.data.copy()
        
        # Create a binary mask for the selected label
        mask = labels == self.current_label
        
        # Perform morphological operation
        if self.dilate_radio.isChecked():
            processed_mask = binary_dilation(mask)
        else:
            processed_mask = binary_erosion(mask)
        
        # Update the labels 
        labels[processed_mask] = self.current_label
        label_layer.data = labels

def add_label_morphology_widget(viewer):
    """Add the label morphology widget to the Napari viewer"""
    widget = LabelMorphologyWidget(viewer)
    viewer.window.add_dock_widget(widget, area='right')
    return widget


class LabelDeletionWidget(QWidget):
    """
    #TODO kind of works - need to connect events based on specific layer selected, also double click to select layer doesn't occur
    """
    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer
        self.current_label = None
        self.last_clicked_layer = None
        
        # Setup layout
        layout = QVBoxLayout()
        
        # Manual label selection
        manual_layout = QHBoxLayout()
        manual_layout.addWidget(QLabel("Label ID:"))
        self.label_spinbox = QSpinBox()
        self.label_spinbox.setMinimum(0)
        self.label_spinbox.setMaximum(999999)
        self.label_spinbox.valueChanged.connect(self._on_manual_label_change)
        manual_layout.addWidget(self.label_spinbox)
        layout.addLayout(manual_layout)
        
        # Label information
        self.label_info = QLabel("No label selected")
        self.label_info.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.label_info)
        
        # Delete button
        delete_button = QPushButton("Delete Selected Label")
        delete_button.clicked.connect(self.delete_label)
        layout.addWidget(delete_button)
        
        # Status message
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)
        
        self.setLayout(layout)
        
        # Connect to viewer events
        viewer.layers.events.inserted.connect(self._connect_layer_events)
        viewer.layers.events.removed.connect(self._connect_layer_events)
        
        # Connect existing layers
        self._connect_layer_events()
    
    def _connect_layer_events(self, event=None):
        """Connect to all label layer events"""
        for layer in self.viewer.layers:
            if isinstance(layer, napari.layers.Labels):
                if not hasattr(layer, '_delete_widget_connected'):
                    layer.mouse_double_click_callbacks.append(self._on_label_click)
                    # layer.events.selected.connect(self._on_layer_selection)
                    layer._delete_widget_connected = True
    
    def _on_layer_selection(self, event):
        """Track the currently selected layer"""
        layer = event.source
        if layer.selected:
            self.last_clicked_layer = layer
    
    def _on_manual_label_change(self, value):
        """Handle manual label selection"""
        self.current_label = value
        self._update_label_info()
    
    def _on_label_click(self, layer, event):
        """Handle label click events"""
        self.last_clicked_layer = layer
        coordinates = tuple(map(int, layer.coordinates))
        self.current_label = layer.data[coordinates]
        
        # Update the spinbox without triggering valueChanged
        self.label_spinbox.blockSignals(True)
        self.label_spinbox.setValue(self.current_label)
        self.label_spinbox.blockSignals(False)
        
        self._update_label_info()
    
    def _update_label_info(self):
        """Update the label information display"""
        if self.current_label is not None:
            self.label_info.setText(f"Selected label: {self.current_label}")
            # Clear any previous status
            self.status_label.setText("")
        else:
            self.label_info.setText("No label selected")
    
    def _get_active_layer(self):
        """Get the currently active label layer"""
        if self.last_clicked_layer is not None and self.last_clicked_layer in self.viewer.layers:
            return self.last_clicked_layer
        
        # Fallback to first available label layer
        label_layers = [layer for layer in self.viewer.layers 
                       if isinstance(layer, napari.layers.Labels)]
        return label_layers[0] if label_layers else None
    
    def delete_label(self):
        """Delete the currently selected label"""
        if self.current_label is None:
            self.status_label.setText("Error: No label selected")
            return
        
        layer = self._get_active_layer()
        if layer is None:
            self.status_label.setText("Error: No label layer found")
            return
        
        # Remove the selected label
        labels = layer.data.copy()
        mask = labels == self.current_label
        if not np.any(mask):
            self.status_label.setText(f"Error: Label {self.current_label} not found")
            return
            
        labels[mask] = 0
        layer.data = labels
        
        # Update status
        self.status_label.setText(f"Deleted label {self.current_label}")
        
        # Reset selection
        self.current_label = None
        self.label_spinbox.setValue(0)
        self._update_label_info()

def add_label_deletion_widget(viewer):
    """Add the label deletion widget to the Napari viewer"""
    widget = LabelDeletionWidget(viewer)
    viewer.window.add_dock_widget(
        widget, 
        area='right',
        add_vertical_stretch=True
    )
    return widget



def confirm_overwrite(parent: QWidget, paths) -> bool:
    """
    Show a warning dialog if any of the given paths already exist.
    Returns True if user clicks OK, False if user clicks Cancel.
    """
    # Ensure list of str / Path
    paths = [Path(p) for p in paths]
    existing = [p for p in paths if p.exists()]

    if not existing:
        return True  # nothing to warn about

    msg_box = QMessageBox(parent)
    msg_box.setIcon(QMessageBox.Icon.Warning)
    msg_box.setWindowTitle("Files already exist")

    # Main text
    msg_box.setText(f"{len(existing)} file(s) already exist at the destination with the same name.\n Are you sure you want to overwrite?")

    shown = "\n".join(str(p) for p in existing)
    msg_box.setInformativeText(shown)

    msg_box.setStandardButtons(
        QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel
    )
    msg_box.setDefaultButton(QMessageBox.StandardButton.Cancel)

    result = msg_box.exec()
    return result == QMessageBox.StandardButton.Ok


def layer_export(viewer, exwidg):

    container = QWidget()
    container.setObjectName("Layer export widget")
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)

    # --- Layer selection list ---
    layer_list = QListWidget()
    layer_list.setSelectionMode(QListWidget.MultiSelection)
    layer_list.setToolTip("Select one or more annotation layers to export.")
    layer_list.setMinimumHeight(120)
    layer_list.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    # --- Export button ---
    export_btn = QPushButton("Export Annotations")
    export_btn.setToolTip("Save selected annotation layers to TIFF in the export directory.")

    # --- Folder button ---
    folder_btn = QToolButton()
    folder_btn.setIcon(folder_btn.style().standardIcon(QStyle.SP_DirOpenIcon))
    folder_btn.setToolTip("Set export directory where annotations will be saved.")
    folder_btn.setFixedSize(24, 24)

    # --- Function to update exportable layers ---
    def update_layer_choices(event=None):
        existing_names = {layer_list.item(i).text() for i in range(layer_list.count())}
        current_layer_names = [layer.name for layer in viewer.layers if isinstance(layer.data, np.ndarray)]

        # Update the list without overwriting selection if possible
        layer_list.clear()
        for name in current_layer_names:
            item = QListWidgetItem(name)
            layer_list.addItem(item)

    # --- Name change event binding ---
    def connect_layer_name_event(layer):
        if hasattr(layer, 'events') and hasattr(layer.events, 'name'):
            layer.events.name.connect(update_layer_choices)

    # --- Export logic ---
    def export_annotations(container):
        selected_items = layer_list.selectedItems()
        if not selected_items:
            print("No layers selected.")
            return

        basedir = exwidg.get_basedir()

        successful_exports = 0
        _writes = [] # store paths, layernames to write
        _path_already_exists = []
        for item in selected_items:
            layer_name = item.text()
            filename = (
                f"{layer_name}.tiff" if layer_name.startswith('ROI_')
                else f"annotated_{layer_name}.tiff" if not layer_name.startswith('annotated_')
                else f"{layer_name}.tiff"
            )
            p = os.path.join(basedir, filename)
            layer = viewer.layers[layer_name]

            # if arrive here assumes good to write
            _writes.append([p, layer_name])
            if os.path.exists(p):
                _path_already_exists.append(p)
        
        # verify any overwrites are okay
        if len(_path_already_exists)>0:
            if not confirm_overwrite(viewer.window._qt_window, _path_already_exists):
                print("Export canceled by user.")
                return
        
        # do writing             
        for (p, layername) in _writes:
            layer = viewer.layers[layername]
            layer.save(p)
            exwidg.register_export(layer)
            print(f"Saved {filename} to {p}")
            successful_exports += 1

        if successful_exports > 0:
            exwidg.write_exmd_update()
            ToastLabel(
                container, f"Exported {successful_exports} layers.", display_time_ms=2000
            )

    export_btn.clicked.connect(lambda: export_annotations(container))
    
    # --- Folder picker logic ---
    def set_export_dir():
        directory = QFileDialog.getExistingDirectory(None, "Select Export Directory")
        if directory:
            exwidg.set_basedir(directory)
            print(f"Export directory set to: {directory}")

    folder_btn.clicked.connect(set_export_dir)

    # --- Layout: row for buttons ---
    row = QHBoxLayout()
    row.addWidget(export_btn)
    row.addWidget(folder_btn)

    # --- Final assembly ---
    layout.addWidget(layer_list)
    layout.addLayout(row)

    # --- Setup layer watchers ---
    viewer.layers.events.inserted.connect(lambda event: (
        connect_layer_name_event(event.value),
        update_layer_choices()
    ))
    viewer.layers.events.removed.connect(update_layer_choices)
    viewer.layers.events.reordered.connect(update_layer_choices)

    for layer in viewer.layers:
        connect_layer_name_event(layer)

    update_layer_choices()

    return container


_viewer_dirty_flags = WeakKeyDictionary()

def enable_exit_warning(viewer):
    """
    Warns user on exit if any layer has been added or modified in the viewer.
    """
    _viewer_dirty_flags[viewer] = False

    def mark_dirty(event=None):
        _viewer_dirty_flags[viewer] = True

    def connect_layer_events(layer):
        layer.events.data.connect(mark_dirty)
        layer.events.name.connect(mark_dirty)
        layer.events.metadata.connect(mark_dirty)

    viewer.layers.events.inserted.connect(lambda e: (
        connect_layer_events(e.value),
        mark_dirty()
    ))

    for layer in viewer.layers:
        connect_layer_events(layer)

    original_close_event = viewer.window._qt_window.closeEvent

    def custom_close_event(event):
        if _viewer_dirty_flags.get(viewer, False):
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Warning)
            msg.setText("You may have unsaved changes.")
            msg.setInformativeText("Do you want to exit without saving?")
            msg.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok)
            msg.setDefaultButton(QMessageBox.Cancel)
            response = msg.exec_()

            if response == QMessageBox.Ok:
                event.accept()
            else:
                event.ignore()
        else:
            original_close_event(event)

    viewer.window._qt_window.closeEvent = custom_close_event


class AddNoteWidget(QWidget):
    def __init__(self, exwidg):
        super().__init__()
        self.exwidg = exwidg
        layout = QVBoxLayout()
        layout.addWidget(QLabel("Add Note"))

        self.text = QTextEdit()
        # preload if desired
        annot_md =  self.exwidg.exmd.get('annotation_metadata', None)
        annot_md = annot_md if isinstance(annot_md, dict) else {'notes':'', 'status':''}
        exwidg.exmd['annotation_metadata'] = annot_md
        
        # get notes
        self.text.setPlainText(exwidg.exmd['annotation_metadata'].get('notes', ''))
        layout.addWidget(self.text)

        self.button = QPushButton("Save Note")
        self.button.clicked.connect(self.save_note)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def save_note(self):
        note = self.text.toPlainText()
        self.exwidg.exmd['annotation_metadata']['notes'] = note
        self.exwidg.write_exmd_update()
        message = f"Note saved:\n{note}"
        print(message)
        ToastLabel(
            self, message, display_time_ms=1500
        )
        


def create_image_filter_widget(viewer: napari.Viewer):
    import scipy.ndimage as ndi
    from magicgui import magicgui
    from qtpy.QtWidgets import QSizePolicy

    @magicgui(filter_type={'choices': ['Median', 'Gaussian', 'Minimum']},
                filter_size={'widget_type': 'Slider', 'min': 1, 'max': 999, 'step': 1},
                auto_call=False)
    def filter_widget(filter_type: str = 'Median', filter_size: int = 1):
        """
        Apply selected filter to the currently selected image layer in Napari.

        Parameters:
        - viewer: napari.Viewer instance.
        - filter_type: Type of filter to apply (Median, Gaussian, Minimum).
        - filter_size: Size of the filter kernel.
        """
        layer_name = viewer.layers.selection.active.name
        print(f'applying {filter_type} to layer: {layer_name}...')
        
        outmsg = '' # for holding messages during runtime
        layer = viewer.layers[layer_name] # TODO probably best to do this on a copy or have a check box
        
        # Apply the selected filter
        if filter_type == 'Median':
            if filter_size == 1: 
                outmsg += f"\tnote: using median filter of size 1 returns the input. size should be >= 3\n"
            filtered_data = ndi.median_filter(layer.data, size=filter_size)
        elif filter_type == 'Gaussian':
            filtered_data = ndi.gaussian_filter(layer.data, sigma=filter_size)
        elif filter_type == 'Minimum':
            filtered_data = ndi.minimum_filter(layer.data, size=filter_size)
        else:
            outmsg += f"\tunsupported filter_type ({filter_type}) selected"
            return  # If an unknown filter type is selected, do nothing.

        # Update the layer data
        layer.data = filtered_data
        # Refresh the viewer
        viewer.layers.events.changed()
        outmsg += (f'completed {filter_type} filter')
        print(outmsg)
    
    filter_widget.native.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

    return filter_widget





class LabelEditWidget(QWidget):
    """
    Simple widget to manipulate labels in a Labels layer.

    Features:
    - Remove labels: enter a comma/space-separated list of ints.
    - Merge labels: convert one label into another.
    - Uses active Labels layer by default, or a user-selected one.
    """

    def __init__(self, viewer):
        super().__init__()
        self.viewer = viewer

        # --- outer layout ---
        outer_layout = QVBoxLayout(self)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer_layout.addWidget(scroll)

        inner = QWidget()
        scroll.setWidget(inner)
        layout = QVBoxLayout(inner)

        # --- layer selection controls ---
        layer_box = QGroupBox("Target layer")
        layer_layout = QHBoxLayout(layer_box)

        self.use_active_chk = QCheckBox("Use active Labels layer")
        self.use_active_chk.setChecked(True)
        self.use_active_chk.stateChanged.connect(self._on_use_active_changed)

        self.layer_combo = QComboBox()
        self.layer_combo.setEnabled(False)

        layer_layout.addWidget(self.use_active_chk)
        layer_layout.addWidget(QLabel("or select:"))
        layer_layout.addWidget(self.layer_combo)

        layout.addWidget(layer_box)

        # keep combo synced with viewer
        self._rebuild_layer_combo()
        viewer.layers.events.inserted.connect(lambda e: self._rebuild_layer_combo())
        viewer.layers.events.removed.connect(lambda e: self._rebuild_layer_combo())
        viewer.layers.selection.events.changed.connect(self._sync_active_layer)

        # --- remove labels group ---
        rm_group = QGroupBox("Remove labels")
        rm_layout = QVBoxLayout(rm_group)

        rm_row = QHBoxLayout()
        rm_row.addWidget(QLabel("Labels to remove (e.g. 1,2,3):"))
        self.remove_edit = QLineEdit()
        rm_row.addWidget(self.remove_edit)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._on_remove_clicked)

        rm_layout.addLayout(rm_row)
        rm_layout.addWidget(self.remove_btn)

        layout.addWidget(rm_group)

        # --- merge labels group ---
        merge_group = QGroupBox("Merge labels")
        merge_layout = QVBoxLayout(merge_group)

        # from
        from_row = QHBoxLayout()
        from_row.addWidget(QLabel("Merge from:"))
        self.merge_from_spin = QSpinBox()
        self.merge_from_spin.setMinimum(0)
        self.merge_from_spin.setMaximum(10_000_000)
        from_row.addWidget(self.merge_from_spin)

        # into
        into_row = QHBoxLayout()
        into_row.addWidget(QLabel("into:"))
        self.merge_into_spin = QSpinBox()
        self.merge_into_spin.setMinimum(0)
        self.merge_into_spin.setMaximum(10_000_000)
        into_row.addWidget(self.merge_into_spin)

        self.merge_btn = QPushButton("Merge")
        self.merge_btn.clicked.connect(self._on_merge_clicked)

        merge_layout.addLayout(from_row)
        merge_layout.addLayout(into_row)
        merge_layout.addWidget(self.merge_btn)

        layout.addWidget(merge_group)

        # spacer at bottom so it looks nice when scrollable
        layout.addStretch(1)

    # -------------------------
    # Layer handling
    # -------------------------
    def _on_use_active_changed(self, state):
        use_active = bool(state)
        self.layer_combo.setEnabled(not use_active)

    def _rebuild_layer_combo(self):
        layers = [lyr for lyr in self.viewer.layers if isinstance(lyr, Labels)]
        current_name = self.layer_combo.currentText() if self.layer_combo.count() else None

        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        for lyr in layers:
            self.layer_combo.addItem(lyr.name)
        self.layer_combo.blockSignals(False)

        # try to keep selection stable
        if current_name and current_name in [l.name for l in layers]:
            self.layer_combo.setCurrentText(current_name)
        elif layers:
            self.layer_combo.setCurrentIndex(0)

    def _sync_active_layer(self, event=None):
        """If 'use active' is off, keep combo matched to active label layer."""
        if not self.use_active_chk.isChecked():
            active = self.viewer.layers.selection.active
            if isinstance(active, Labels):
                idx = self.layer_combo.findText(active.name)
                if idx >= 0:
                    self.layer_combo.setCurrentIndex(idx)

    def _get_target_layer(self) -> Labels | None:
        layer = None
        if self.use_active_chk.isChecked():
            candidate = self.viewer.layers.selection.active
            if isinstance(candidate, Labels):
                layer = candidate
        else:
            name = self.layer_combo.currentText()
            if name in self.viewer.layers:
                candidate = self.viewer.layers[name]
                if isinstance(candidate, Labels):
                    layer = candidate

        if layer is None:
            self.viewer.status = "No valid Labels layer selected."
        return layer

    # -------------------------
    # Operations
    # -------------------------
    def _on_remove_clicked(self):
        layer = self._get_target_layer()
        if layer is None:
            return

        text = self.remove_edit.text().strip()
        if not text:
            self.viewer.status = "No labels provided to remove."
            return

        # parse ints from comma/space separated text
        tokens = [t for t in text.replace(",", " ").split() if t]
        try:
            labels = sorted({int(t) for t in tokens if int(t) != 0})
        except ValueError:
            self.viewer.status = "Could not parse labels (use integers)."
            return

        if not labels:
            self.viewer.status = "No non-zero labels to remove."
            return

        data = layer.data
        for lbl in labels:
            # in-place modification to avoid reallocation
            mask = data == lbl
            if mask.any():
                data[mask] = 0

        layer.data = data  # trigger refresh
        self.viewer.status = f"Removed labels: {labels}"

    def _on_merge_clicked(self):
        layer = self._get_target_layer()
        if layer is None:
            return

        merge_from = int(self.merge_from_spin.value())
        merge_into = int(self.merge_into_spin.value())

        if merge_from == 0:
            self.viewer.status = "merge_from cannot be 0 (background)."
            return
        if merge_from == merge_into:
            self.viewer.status = "merge_from and merge_into are identical."
            return

        data = layer.data
        mask = data == merge_from
        if not mask.any():
            self.viewer.status = f"No voxels with label {merge_from} found."
            return

        data[mask] = merge_into
        layer.data = data
        self.viewer.status = f"Merged label {merge_from} into {merge_into}."





class BinaryThresholdWidget(QWidget):
    """
    Widget to apply a binary threshold to the currently selected image layer
    in a napari viewer.

    It creates/updates a labels layer named `<active_layer.name>_binary`.
    """

    def __init__(
        self,
        viewer: napari.Viewer,
        parent: Optional[QWidget] = None,
        max_value: int = 2**16,
    ):
        super().__init__(parent)

        self.viewer = viewer
        self.max_value = int(max_value)

        self._build_ui()

    def _build_ui(self):
        self.setLayout(QVBoxLayout())

        # --- Min threshold row ---
        min_row = QHBoxLayout()
        min_label = QLabel("Min Threshold:")
        self.min_slider = QSlider(Qt.Orientation.Horizontal)
        self.min_slider.setMinimum(0)
        self.min_slider.setMaximum(self.max_value)
        self.min_slider.setValue(0)

        self.min_spin = QSpinBox()
        self.min_spin.setMinimum(0)
        self.min_spin.setMaximum(self.max_value)
        self.min_spin.setValue(0)

        # Link slider and spinbox
        self.min_slider.valueChanged.connect(self.min_spin.setValue)
        self.min_spin.valueChanged.connect(self.min_slider.setValue)

        min_row.addWidget(min_label)
        min_row.addWidget(self.min_slider)
        min_row.addWidget(self.min_spin)

        # --- Max threshold row ---
        max_row = QHBoxLayout()
        max_label = QLabel("Max Threshold:")
        self.max_slider = QSlider(Qt.Orientation.Horizontal)
        self.max_slider.setMinimum(0)
        self.max_slider.setMaximum(self.max_value)
        self.max_slider.setValue(self.max_value)

        self.max_spin = QSpinBox()
        self.max_spin.setMinimum(0)
        self.max_spin.setMaximum(self.max_value)
        self.max_spin.setValue(self.max_value)

        # Link slider and spinbox
        self.max_slider.valueChanged.connect(self.max_spin.setValue)
        self.max_spin.valueChanged.connect(self.max_slider.setValue)

        max_row.addWidget(max_label)
        max_row.addWidget(self.max_slider)
        max_row.addWidget(self.max_spin)

        # --- Apply button (equivalent to auto_call = False) ---
        self.apply_button = QPushButton("Apply Binary Threshold")
        self.apply_button.clicked.connect(self.apply_threshold)

        # --- Assemble main layout ---
        self.layout().addLayout(min_row)
        self.layout().addLayout(max_row)
        self.layout().addWidget(self.apply_button)

    def apply_threshold(self):
        """Apply threshold to active layer and create/update a *_binary layer."""
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            QMessageBox.warning(
                self,
                "No Active Layer",
                "No active layer selected. Please select an image layer.",
            )
            print("BinaryThresholdWidget: No active layer selected.")
            return

        # Get thresholds
        min_threshold = int(self.min_spin.value())
        max_threshold = int(self.max_spin.value())

        if min_threshold > max_threshold:
            QMessageBox.warning(
                self,
                "Invalid Threshold Range",
                "Min threshold must be less than or equal to max threshold.",
            )
            print(
                f"BinaryThresholdWidget: invalid range min={min_threshold}, max={max_threshold}"
            )
            return

        data = active_layer.data

        # Apply threshold
        binary_data = (data >= min_threshold) & (data <= max_threshold)

        # Create or update binary layer
        binary_layer_name = f"{active_layer.name}_binary"
        if binary_layer_name in self.viewer.layers:
            self.viewer.layers[binary_layer_name].data = binary_data
        else:
            # Cast to int32 for labels-like behavior (0/1)
            self.viewer.add_labels(
                binary_data.astype(np.int32),
                name=binary_layer_name,
            )

        print(
            f"BinaryThresholdWidget: Binary threshold applied "
            f"with min={min_threshold}, max={max_threshold} "
            f"to layer '{active_layer.name}'."
        )


class LabelErosionWidget(QWidget):
    """Napari widget for eroding labels based on intensity."""
    # TODO refactor as qwidget
    
    def __init__(self, viewer):
        self.viewer = viewer
        self._create_widget()
    
    def _create_widget(self):
        """Create the erosion widget."""
                
        @magicgui(
            call_button="Apply Erosion",
            layout='vertical',
            label_layer={'label': 'Label Layer:'},
            intensity_layer={'label': 'Intensity Layer:'},
            threshold={'label': 'Intensity Threshold:', 'min': 0.0, 'max': 65535, 'step': 0.01},
            iterations={'label': 'Max Iterations:', 'min': 1, 'max': 10},
            min_object_size={'label': 'Min Object Size:', 'min': 0, 'max': 999999},
            target_label={'label': 'Target Label (0=all):', 'min': 0, 'max': 999999},
            output_name={'label': 'Output Name:'}
        )
        def erosion_widget(
            label_layer: Labels,
            intensity_layer: Image,
            threshold: float = 0.5,
            iterations: int = 1,
            min_object_size: int = 5,
            target_label: int = 0,
            output_name: str = 'eroded_labels'
        ) -> LayerDataTuple:
            
            if label_layer is None or intensity_layer is None:
                print("Please select both label and intensity layers")
                return None
            
            # Get data
            label_image = label_layer.data
            intensity_image = intensity_layer.data

            if label_image.shape != intensity_image.shape:
                print(f'Selected layers must have the same shape. Got label_image.shape:{label_image.shape}, intensity_image.shape:{intensity_image.shape}.')
                return None

            
            # Apply erosion to whole image or single label
            target = None if target_label == 0 else target_label

            if target is None: # whole image
                result = uip.shrink_object_perimeters(label_image, intensity_image, threshold, iter_max=iterations)
            else:
                from skimage.measure import regionprops
                result = label_image.copy() # init output
                
                mask = (label_image == target)*target

                bbox_slices, shrunk_mask = uip._shrink_single_label(
                    label_image,
                    intensity_image,
                    target,
                    regionprops(mask)[0].bbox,
                    threshold,
                    connectivity=1,
                    iter_max=iterations,
                    find_boundaries_mode='inner',
                    remove_islands = True,
                    restore_background_labels = True,
                )
                result[bbox_slices] = shrunk_mask
            
            # Remove small objects
            if min_object_size > 0:
                from skimage.morphology import remove_small_objects
                result = remove_small_objects(result, min_object_size)
            
            print(f"Erosion complete. Output: {output_name}")
            
            # Return as LayerDataTuple
            return (result, {'name': output_name}, 'labels')
        

        self.widget = erosion_widget
        self.widget.native.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    
    def add_to_viewer(self):
        """Add widget to viewer."""
        self.viewer.window.add_dock_widget(self.widget, area='right', name='Label Erosion')
        return self.widget


def create_erosion_widget(viewer):
    """
    Convenience function to create and add erosion widget to viewer.
    
    Parameters
    ----------
    viewer : napari.Viewer
        
    Returns
    -------
    LabelErosionWidget
        Widget instance
    
    """
    widget = LabelErosionWidget(viewer)
    widget.add_to_viewer()
    return widget





def labels_to_polygons(label_image: np.ndarray, level: float = 0.5):
    """
    Convert a 2D labels image into polygons suitable for napari Shapes.
        note: ensure all objects have unique labels, otherwise largest contour is returned 

    Parameters
    ----------
    label_image : (H, W) ndarray of ints
        Each connected region has a unique integer ID; 0 is treated as background.
    level : float, optional
        The contour value passed to skimage.measure.find_contours (default=0.5).

    Returns
    -------
    polygons : List[np.ndarray]
        List of length-N arrays of shape (Mi, 2), each a polygon in (row, col).
    label_ids : List[int]
        The integer label corresponding to each polygon in `polygons`.
    """
    
    from skimage.measure import find_contours

    polygons = []
    label_ids = []
    for lab in np.unique(label_image):
        if lab == 0:
            continue
        mask = (label_image == lab).astype(np.uint8)
        contours = find_contours(mask, level)
        if not contours:
            continue
        # pick the longest contour (most points)
        contour = max(contours, key=lambda c: c.shape[0])
        polygons.append(contour)
        label_ids.append(int(lab))
    return polygons, label_ids


def add_label_shapes_to_viewer(
    viewer: Viewer,
    label_image: np.ndarray,
    *,
    shape_type: str = 'polygon',
    edge_width: float = 1.0,
    edge_color: str = 'coral',
    face_color: str = 'royalblue',
):
    """
    Extract polygons from `label_image` and add them as a Shapes layer.
        note: ensure all objects have unique labels, otherwise largest contour is returned 

    Returns the created Shapes layer.
    """
    polygons, label_ids = labels_to_polygons(label_image)

    shapes = viewer.add_shapes(
        data=polygons,
        shape_type=shape_type,
        edge_width=edge_width,
        edge_color=edge_color,
        face_color=face_color,
        properties={'label_id': label_ids},
    )
    # show the label_id next to each shape
    shapes.text = 'label_id'
    return shapes


def make_add_label_shapes_widget(viewer: Viewer):
    """
    Create a dockable widget that calls:
    add_label_shapes_to_viewer(viewer, labs, edge_width, edge_color, face_color)
    on a selected Labels layer.
    """

    @magicgui(
        call_button="Add Shapes",
        layout="vertical",
        labels_layer={"label": "Labels layer:"},
        edge_width={"label": "Edge width:", "min": 0.0, "max": 10.0, "step": 0.1},
        edge_color={"label": "Edge color:"},
        face_color={"label": "Face color:"},
    )
    def _widget(
        labels_layer: Labels,
        edge_width: float = 0.1,
        edge_color: str = "red",
        face_color: str = "transparent",
    ):
        if labels_layer is None:
            print("Please select a Labels layer.")
            return

        # Extract the label image (numpy array) from the layer
        labs = labels_layer.data

        # Call your function exactly as requested, with the chosen params
        add_label_shapes_to_viewer(
            viewer,
            labs,
            edge_width=edge_width,
            edge_color=edge_color,
            face_color=face_color,
        )

    _widget.native.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

    viewer.window.add_dock_widget(_widget, area="right")

    return _widget

