




"""
This module provides a custom QWidget class (GTWidget) for use with the napari viewer.
It includes various functionalities to display synapse images, edit and create point and
label annotations, interconvert between points and labels, and save data in as .zarr.

implementation based on: https://github.com/ucsdmanorlab/cochlea-synapseg/blob/main/src/napari_cochlea_synapse_seg/_widget.py

# TODO use this class as a container for my widgets
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QTabWidget, QScrollArea, QSizePolicy, QVBoxLayout, QWidget
from qtpy.QtCore import Qt

from ._GTwidget import GTWidget
from ._predwidget import PredWidget
from ._preprocesswidget import PreProcessWidget


if TYPE_CHECKING:
    import napari
                
class SynapSegWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.init_ui()
    
    def init_ui(self):
        tab_widget = QTabWidget()
        tab0 = self._init_scroll(PreProcessWidget(viewer=self.viewer))
        tab1 = self._init_scroll(GTWidget(viewer=self.viewer))
        tab2 = self._init_scroll(PredWidget(viewer=self.viewer))
        
        tab_widget.addTab(tab0, "Preprocess")
        tab_widget.addTab(tab1, "Ground Truth")
        tab_widget.addTab(tab2, "Predict")
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(tab_widget)

    def _init_scroll(self, widget):
        scrollarea = QScrollArea(self)
        scrollarea.setWidgetResizable(True)

        scrollContent = QWidget(scrollarea)
        scrollLayout = QVBoxLayout(scrollContent)
        scrollContent.setLayout(scrollLayout)
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        scrollLayout.setAlignment(Qt.AlignTop)

        scrollLayout.addWidget(widget)
        scrollarea.setWidget(scrollContent)

        return scrollarea