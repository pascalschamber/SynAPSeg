from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QHBoxLayout,  QScrollArea, QToolButton, QPushButton
)
from PyQt6.QtCore import Qt

class ScrollableContainerWidget(QWidget):
    """ Scrollable container for holding multiple CollapsibleWidgets """
    def __init__(self, model_param_forms, parent=None, collapsable_widget_kwargs=None):
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.add_model_params(model_param_forms, collapsable_widget_kwargs=collapsable_widget_kwargs) 
        
        scroll.setWidget(self.scroll_content)
        self.scroll = scroll
        self.layout().addWidget(scroll)
        
    
    def add_model_params(self, model_param_forms, collapsable_widget_kwargs=None):
        """ add collapsable widget over e.g. QFormLayout() """
        scroll_layout = QVBoxLayout(self.scroll_content)
        scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        for model_name, param_form in model_param_forms.items():
            model_widget = QWidget()
            model_widget.setLayout(param_form)
            collapsible = CollapsibleWidget(model_name, model_widget, **collapsable_widget_kwargs or {})
            scroll_layout.addWidget(collapsible)
        
        self.scroll_content.setLayout(scroll_layout)


class CollapsibleWidget(QWidget):
    def __init__(
        self, 
        title, 
        content_widget, 
        parent=None, 
        delete_callback=None, 
        deletable=True,
        is_visible=False,
        ):
        """ 
            widget layout that can be shown or hidden 
            
            args:
                title (str): title for the widget
                content_widget (QWidget): widget to be displayed when expanded
                parent (QWidget): parent widget
                delete_callback (callable): callback function to be called when widget is deleted
                deletable (bool): whether the widget can be deleted
        """
        super().__init__(parent)
        self.setLayout(QVBoxLayout())
        
        # Create a header widget with a horizontal layout
        self.header_widget = QWidget()
        header_layout = QHBoxLayout(self.header_widget)
        
        self.title_label = QLabel(title)
        # self.title_label.setStyleSheet("color: black;")
        
        # button for dropdown display of content_widget
        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        # Initially show right arrow indicating a collapsed state
        self.toggle_button.setArrowType(Qt.ArrowType.RightArrow)
        self.toggle_button.setStyleSheet("background-color: #D9D9D9; color: black;")
        self.toggle_button.clicked.connect(self.toggle)

        header_layout.addWidget(self.toggle_button)
        header_layout.addWidget(self.title_label)
        header_layout.addStretch()

        self.delete_callback = delete_callback
        self.deleteable = deletable
        self.title = title

        if deletable: # --- NEW: Delete Button ---
            self.delete_button = QPushButton("✕")
            self.delete_button.setFixedSize(20, 20)
            self.delete_button.setStyleSheet("""
                QPushButton { 
                    border: none; 
                    color: gray; 
                    font-weight: bold; 
                }
                QPushButton:hover { 
                    color: red; 
                }
            """)
            self.delete_button.clicked.connect(self.delete_self)
            header_layout.addWidget(self.delete_button) # Add it to the right
            
        # --------------------------

        
        self.content_area = QWidget()
        self.content_area.setLayout(QVBoxLayout())
        self.content_area.layout().addWidget(content_widget)
        self.content_area.setVisible(False)
        
        self.layout().addWidget(self.header_widget)
        self.layout().addWidget(self.content_area)

        if is_visible:
            self.toggle_button.setChecked(True)
            self.toggle()

    
    def toggle(self):
        # Toggle visibility of the content area
        checked = self.toggle_button.isChecked()
        self.content_area.setVisible(checked)
        # Change the arrow direction to indicate state
        self.toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )
    

    def delete_self(self):
        """ Safely removes the widget from the UI """
        self.setParent(None)
        self.deleteLater()

        if self.delete_callback is not None:
            self.delete_callback(self.title)






