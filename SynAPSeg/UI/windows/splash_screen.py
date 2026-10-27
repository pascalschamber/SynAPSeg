from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QTimer, QVariantAnimation

class SplashScreen(QSplashScreen):
    def __init__(self, icon_path):
        pixmap = QPixmap(icon_path)
        super().__init__(pixmap)
        
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.FramelessWindowHint)
        
        self.rotation = 0
        self.current_opacity = 0.0

        # Fade-in Animation
        self.fade_anim = QVariantAnimation(self)
        self.fade_anim.setStartValue(0.0)
        self.fade_anim.setEndValue(1.0)
        self.fade_anim.setDuration(1000)
        self.fade_anim.valueChanged.connect(self._handle_fade)
        self.fade_anim.start()

        # Rotation Timer
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self._update_spinner)
        # self.timer.start(5)

    def _handle_fade(self, value):
        self.setWindowOpacity(value)

    def _update_spinner(self):
        self.rotation = (self.rotation + 10) % 360
        self.update() 

    def update_status(self, message):
        """Public method to change the loading text."""
        # Align center, bottom, white color, with a slight offset for the spinner
        self.showMessage(
            message, 
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom, 
            Qt.GlobalColor.white
        )