import sys
from PyQt6.QtWidgets import (QApplication, QGraphicsView, QGraphicsScene, 
                             QGraphicsRectItem, QGraphicsPathItem, QMainWindow, 
                             QPushButton, QVBoxLayout, QWidget, QDialog, 
                             QLabel, QLineEdit, QFormLayout)
from PyQt6.QtCore import Qt, QPointF, QLineF
from PyQt6.QtGui import QPen, QPainterPath, QColor, QBrush, QPainter

class ParameterDialog(QDialog):
    def __init__(self, node_name, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Parameters: {node_name}")
        layout = QFormLayout(self)
        self.param1 = QLineEdit()
        layout.addRow("Setting Value:", self.param1)
        
        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.accept)
        layout.addRow(save_btn)

class ConnectionItem(QGraphicsPathItem):
    """A line connecting two nodes."""
    def __init__(self, start_item, end_item):
        super().__init__()
        self.start_item = start_item
        self.end_item = end_item
        self.setPen(QPen(QColor("#7f8c8d"), 3))
        self.update_path()

    def update_path(self):
        """Redraws the line when nodes move."""
        path = QPainterPath()
        start_pos = self.start_item.sceneBoundingRect().center()
        end_pos = self.end_item.sceneBoundingRect().center()
        path.moveTo(start_pos)
        path.lineTo(end_pos)
        self.setPath(path)

class NodeItem(QGraphicsRectItem):
    def __init__(self, name, is_deletable=True):
        super().__init__(0, 0, 100, 50)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemSendsScenePositionChanges)
        
        self.setBrush(QBrush(QColor("#3498db") if is_deletable else QColor("#e74c3c")))
        self.name = name
        self.is_deletable = is_deletable
        self.connections = []

        # Correct way to add text: Create the item directly
        self.text_item = QGraphicsScene().addText(name) 
        # Actually, let's use the simpler constructor to avoid the deletion error:
        from PyQt6.QtWidgets import QGraphicsTextItem
        self.label = QGraphicsTextItem(name, self)
        self.label.setPos(5, 10)

    def itemChange(self, change, value):
        if change == QGraphicsRectItem.GraphicsItemChange.ItemPositionHasChanged:
            for conn in self.connections:
                conn.update_path()
        return super().itemChange(change, value)

    def mouseDoubleClickEvent(self, event):
        dialog = ParameterDialog(self.name)
        dialog.exec()

class PipelineCanvas(QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QGraphicsScene(0, 0, 2000, 2000)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        self.first_selected_node = None

        # Add the non-deletable Input Node
        self.input_node = NodeItem("Input", is_deletable=False)
        self.input_node.setPos(50, 250)
        self.scene.addItem(self.input_node)

    def add_node(self):
        new_node = NodeItem(f"Stage {len(self.scene.items())}")
        new_node.setPos(200, 250)
        self.scene.addItem(new_node)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if isinstance(item, NodeItem):
            if self.first_selected_node is None:
                self.first_selected_node = item
            elif self.first_selected_node != item:
                # Create connection
                conn = ConnectionItem(self.first_selected_node, item)
                self.scene.addItem(conn)
                self.first_selected_node.connections.append(conn)
                item.connections.append(conn)
                self.first_selected_node = None
            else:
                self.first_selected_node = None
        super().mousePressEvent(event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Interactive Pipeline Builder")
        self.resize(1000, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.add_btn = QPushButton("Add Pipeline Stage")
        self.add_btn.clicked.connect(self.add_node_to_canvas)
        layout.addWidget(self.add_btn)

        self.canvas = PipelineCanvas()
        layout.addWidget(self.canvas)

    def add_node_to_canvas(self):
        self.canvas.add_node()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())