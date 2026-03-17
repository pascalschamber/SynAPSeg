from PyQt6.QtWidgets import QLabel, QApplication
from PyQt6.QtCore import Qt
from SynAPSeg.UI.widgets.style_sheets import format_tooltip

class HoverLabel(QLabel):
    """A QLabel that highlights on hover with a formatted tooltip."""
    def __init__(self, text="", tooltip_text="", parent=None):
        super().__init__(text, parent)
        self.original_text = text
        self.tooltip_text = tooltip_text
        
        # Enable rich text tooltips and set the content
        self.setToolTip(format_tooltip(tooltip_text))
        
        # Set a default style (non-hovered)
        self.set_hover_style(False)

    def set_hover_style(self, is_hovered):
        """Update the stylesheet based on hover state."""
        if is_hovered:
            # Highlighted: Blue color
            self.setStyleSheet("color: #3498db; font-weight: normal; font-size: 12px;")
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            # Default: Standard weight and color
            self.setStyleSheet("color: #ffffff; font-weight: normal; font-size: 12px;")
            self.unsetCursor()

    def enterEvent(self, event):
        self.set_hover_style(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.set_hover_style(False)
        super().leaveEvent(event)

