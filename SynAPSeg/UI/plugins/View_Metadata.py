from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QGridLayout, QTreeWidget, QTreeWidgetItem, QMainWindow, QMessageBox,
    QSizePolicy, QFrame, QSpinBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage, QFont

import os
import yaml
import numpy as np
from pathlib import Path

from SynAPSeg.IO.project import Project, Example
from SynAPSeg.UI.plugins.__base import BaseApp


# ==============================================================================
# PDF Viewer Base & Specific Implementation
# ==============================================================================

class ZoomPanGraphicsView(QGraphicsView):
    """Base graphics view allowing mouse drag panning and scroll wheel zooming."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setRenderHint(self.renderHints()) # Base hinting
        
    def wheelEvent(self, event):
        """Handle zoom on scroll wheel."""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1 / zoom_factor, 1 / zoom_factor)

class PDFViewerBase(QMainWindow):
    """Base window class for displaying pan/zoomable images and PDF renders."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(1000, 800)
        
        self.scene = QGraphicsScene(self)
        self.view = ZoomPanGraphicsView(self.scene)
        self.setCentralWidget(self.view)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

    def set_image_array(self, array: np.ndarray):
        """Converts an RGB/RGBA numpy array to a QPixmap and displays it."""
        h, w = array.shape[:2]
        ch = array.shape[2] if len(array.shape) == 3 else 1
        
        if ch == 4:
            fmt = QImage.Format.Format_RGBA8888
        elif ch == 3:
            fmt = QImage.Format.Format_RGB888
        else:
            fmt = QImage.Format.Format_Grayscale8
            
        bytes_per_line = ch * w
        q_img = QImage(array.data, w, h, bytes_per_line, fmt)
        self.pixmap_item.setPixmap(QPixmap.fromImage(q_img))
        self.scene.setSceneRect(self.pixmap_item.boundingRect())


class SegmentationResultsViewer(PDFViewerBase):
    """Specific implementation to navigate and view init_segmentation_results."""
    def __init__(self, project: Project, start_index: int, parent=None):
        super().__init__(parent)
        self.project = project
        self.current_index = start_index

        # Add top navigation bar
        nav_toolbar = self.addToolBar("Navigation")
        nav_toolbar.setMovable(False)

        self.btn_prev = QPushButton("◄ Previous Example")
        self.btn_next = QPushButton("Next Example ►")
        self.status_label = QLabel()
        self.status_label.setStyleSheet("padding: 0 15px; font-weight: bold;")

        self.btn_prev.clicked.connect(self.load_previous)
        self.btn_next.clicked.connect(self.load_next)

        nav_toolbar.addWidget(self.btn_prev)
        nav_toolbar.addWidget(self.status_label)
        nav_toolbar.addWidget(self.btn_next)

        self.load_example(self.current_index)

    def load_previous(self):
        self._navigate(-1)

    def load_next(self):
        self._navigate(1)

    def _navigate(self, direction):
        # Scan for the next example that actually has the file
        idx = self.current_index + direction
        while 0 <= idx < len(self.project.examples):
            ex = self.project.examples[idx]
            if len(ex.get_filenames(pattern='init_segmentation_results.*')) > 0:
                self.load_example(idx)
                return
            idx += direction
        
        QMessageBox.information(self, "End of List", "No more examples with segmentation results in this direction.")

    def load_example(self, index):
        self.current_index = index
        example = self.project.examples[index]
        
        self.setWindowTitle(f"Segmentation Results - {example.name}")
        self.status_label.setText(f"Viewing: {example.name}")

        try:
            # Loads PDF or SVG into numpy array
            # Using dpi 150 for snappier initial loading; can increase to 300 if needed
            arr = example.get_segmentation_results(dpi=150)
            self.set_image_array(arr)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load PDF for {example.name}:\n{str(e)}")


# ==============================================================================
# Metadata Tree & Viewers
# ==============================================================================

class NestedDictEditor(QTreeWidget):
    """Reusable widget for viewing and editing arbitrarily nested dictionaries."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Key", "Value"])
        self.setAlternatingRowColors(True)
        self.setColumnWidth(0, 300)
        self.itemChanged.connect(self._on_item_changed)

    def load_dict(self, data: dict):
        self.clear()
        self.blockSignals(True)
        self._populate_tree(self.invisibleRootItem(), data)
        self.blockSignals(False)
        self.expandToDepth(1)

    def _populate_tree(self, parent_item, data):
        if isinstance(data, dict):
            for key, value in data.items():
                item = QTreeWidgetItem(parent_item, [str(key), ""])
                self._populate_tree(item, value)
        elif isinstance(data, list):
            for i, value in enumerate(data):
                item = QTreeWidgetItem(parent_item, [f"[{i}]", ""])
                self._populate_tree(item, value)
        else:
            val_str = "" if data is None else str(data)
            parent_item.setText(1, val_str)
            parent_item.setFlags(parent_item.flags() | Qt.ItemFlag.ItemIsEditable)

    def get_dict(self):
        return self._reconstruct_data(self.invisibleRootItem())

    def _reconstruct_data(self, item):
        if item.childCount() > 0:
            first_key = item.child(0).text(0)
            if first_key.startswith("[") and first_key.endswith("]"):
                result = []
                for i in range(item.childCount()):
                    result.append(self._reconstruct_data(item.child(i)))
                return result
            else:
                result = {}
                for i in range(item.childCount()):
                    child = item.child(i)
                    result[child.text(0)] = self._reconstruct_data(child)
                return result
        else:
            val_str = item.text(1)
            try:
                return yaml.safe_load(val_str) if val_str != "" else None
            except Exception:
                return val_str

    def _on_item_changed(self, item, column):
        if column == 1:
            self.resizeColumnToContents(1)


class MetadataWindow(QMainWindow):
    """Detached window displaying the metadata nested dict editor."""
    def __init__(self, example: Example, font_size: int, parent=None):
        super().__init__(parent)
        self.example = example
        self.setWindowTitle(f"Metadata Viewer - {self.example.name}")
        self.resize(800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Header
        header_layout = QHBoxLayout()
        header_label = QLabel(f"<b>Example:</b> {self.example.name} | <b>Project:</b> {self.example.project_name}")
        header_label.setStyleSheet(f"font-size: {font_size}px;")
        
        # Expand/Collapse Controls
        btn_expand = QPushButton("Expand All")
        btn_collapse = QPushButton("Collapse All")
        btn_expand.setStyleSheet(f"font-size: {font_size}px;")
        btn_collapse.setStyleSheet(f"font-size: {font_size}px;")
        
        header_layout.addWidget(header_label)
        header_layout.addStretch()
        header_layout.addWidget(btn_expand)
        header_layout.addWidget(btn_collapse)
        layout.addLayout(header_layout)

        # Nested Dictionary Editor
        self.editor = NestedDictEditor()
        self.editor.setStyleSheet(f"font-size: {font_size}px;")
        self.editor.load_dict(self.example.exmd)
        layout.addWidget(self.editor)

        btn_expand.clicked.connect(self.editor.expandAll)
        btn_collapse.clicked.connect(self.editor.collapseAll)

        # Save Button
        self.save_btn = QPushButton("Save Metadata")
        self.save_btn.setStyleSheet(f"font-size: {font_size}px; font-weight: bold;")
        self.save_btn.clicked.connect(self.save_metadata)
        layout.addWidget(self.save_btn, alignment=Qt.AlignmentFlag.AlignRight)

    def save_metadata(self):
        try:
            self.example.exmd = self.editor.get_dict()
            self.example.write_metadata(archive_previous=True)
            QMessageBox.information(self, "Success", f"Metadata for {self.example.name} saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save metadata:\n{str(e)}")


class ExampleCard(QFrame):
    """Visual card representing an individual example."""
    clicked = pyqtSignal(object)
    view_pdf_clicked = pyqtSignal(int)

    def __init__(self, example: Example, index: int, font_size: int, parent=None):
        super().__init__(parent)
        self.example = example
        self.index = index
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setLineWidth(2)
        
        # Check for Segmentation Results PDF/SVG
        self.has_pdf = len(self.example.get_filenames(pattern='init_segmentation_results.*')) > 0

        self.setStyleSheet("""
            ExampleCard {
                background-color: #2b2b2b;
                border-radius: 8px;
                padding: 10px;
                margin: 5px;
            }
            ExampleCard:hover {
                background-color: #3b3b3b;
                border: 1px solid #5c5c5c;
            }
        """)

        layout = QHBoxLayout(self)

        # Thumbnail setup
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(80, 80)
        self.thumb_label.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        self.thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._load_thumbnail()
        layout.addWidget(self.thumb_label)

        # Details Layout
        details_layout = QVBoxLayout()
        
        # Header (Name + Status)
        header_layout = QHBoxLayout()
        name_label = QLabel(f"<b>{self.example.name}</b>")
        name_label.setStyleSheet(f"font-size: {font_size + 2}px;")
        
        status_indicator = QLabel()
        status_indicator.setFixedSize(12, 12)
        status_indicator.setStyleSheet(self._get_status_stylesheet())
        status_indicator.setToolTip(self._get_status_tooltip())

        header_layout.addWidget(name_label)
        header_layout.addStretch()
        header_layout.addWidget(status_indicator)
        details_layout.addLayout(header_layout)

        # Source Path
        img_path = self.example.get_image_path() or "No image path defined"
        path_label = QLabel(img_path)
        path_label.setStyleSheet(f"color: #888; font-size: {max(8, font_size - 4)}px;")
        path_label.setWordWrap(True)
        details_layout.addWidget(path_label)

        # Actions Row
        actions_layout = QHBoxLayout()
        if self.has_pdf:
            btn_view_pdf = QPushButton("View Segmentation Results")
            btn_view_pdf.setStyleSheet(f"font-size: {max(9, font_size - 2)}px;")
            btn_view_pdf.clicked.connect(self._on_view_pdf)
            actions_layout.addWidget(btn_view_pdf)
            
        actions_layout.addStretch()
        details_layout.addLayout(actions_layout)

        layout.addLayout(details_layout)

    def _get_status_stylesheet(self):
        if self.example.is_skipped:
            color = "#ff4c4c" # Red
        elif self.example.is_complete:
            color = "#4caf50" # Green
        else:
            color = "#ffeb3b" # Yellow
        return f"background-color: {color}; border-radius: 6px;"

    def _get_status_tooltip(self):
        if self.example.is_skipped: return "Skipped"
        if self.example.is_complete: return "Complete"
        return "Incomplete"

    def _load_thumbnail(self):
        try:
            thumb_path = self.example.get_path("thumbnail.png")
            pixmap = QPixmap(thumb_path).scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatioByExpanding)
            self.thumb_label.setPixmap(pixmap)
        except Exception:
            self.thumb_label.setText("No\nThumb")

    def mousePressEvent(self, event):
        # Only emit click if we aren't clicking a button
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.example)
        super().mousePressEvent(event)

    def _on_view_pdf(self):
        self.view_pdf_clicked.emit(self.index)


# ==============================================================================
# Main Plugin Application
# ==============================================================================

class MainApp(BaseApp):
    def __init__(self, state_manager):
        super().__init__(state_manager)

        self.app_name = "View Metadata"
        self.active_windows = []  

        self.init_layout()

        # Top Control Bar (Font Settings)
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Global Text Size:"))
        
        self.font_spinner = QSpinBox()
        self.font_spinner.setRange(8, 36)
        
        # Load user preference from state manager
        initial_font_size = self.state_manager.get('metadata_viewer_fontsize', 12)
        self.font_spinner.setValue(initial_font_size)
        self.font_spinner.valueChanged.connect(self._on_fontsize_changed)
        
        controls_layout.addWidget(self.font_spinner)
        controls_layout.addStretch()
        self.layout.addLayout(controls_layout)

        # Scroll area for the cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        self.cards_widget = QWidget()
        self.cards_layout = QGridLayout(self.cards_widget)
        self.cards_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        self.scroll_area.setWidget(self.cards_widget)
        self.layout.addWidget(self.scroll_area)

        self.post_layout()

    def _on_fontsize_changed(self, size: int):
        """Saves preference and dynamically refreshes the cards."""
        self.state_manager.set('metadata_viewer_fontsize', size)
        self.refresh_cards()

    def _on_switch_app(self):
        self.refresh_cards()

    def _on_select_project(self):
        self.refresh_cards()

    def refresh_cards(self):
        while self.cards_layout.count():
            item = self.cards_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        dir_examples = self.get_examples_directory(warn_diag=False)
        if not dir_examples:
            return

        PROJ_PATH = Path(dir_examples).parent
        self.project = Project(PROJ_PATH)

        if not self.project.examples:
            self.cards_layout.addWidget(QLabel("No examples found in this project."))
            return

        current_font_size = self.font_spinner.value()
        columns = 3
        
        for index, ex in enumerate(self.project.examples):
            card = ExampleCard(ex, index=index, font_size=current_font_size)
            card.clicked.connect(self.open_metadata_window)
            card.view_pdf_clicked.connect(self.open_pdf_viewer)
            
            row = index // columns
            col = index % columns
            self.cards_layout.addWidget(card, row, col)

    def open_metadata_window(self, example: Example):
        current_font_size = self.font_spinner.value()
        win = MetadataWindow(example, font_size=current_font_size, parent=self)
        self.active_windows.append(win)
        win.show()

    def open_pdf_viewer(self, index: int):
        win = SegmentationResultsViewer(project=self.project, start_index=index, parent=self)
        self.active_windows.append(win)
        win.show()

    def _run(self):
        self.refresh_cards()