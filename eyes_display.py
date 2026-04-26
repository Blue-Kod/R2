#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Eye Display using PyQt5 with QtWebEngine.
Optimized for Orange Pi 4 Pro (Debian ARM).

REQUIRED APT PACKAGES (Debian/ARM):
    sudo apt-get update
    sudo apt-get install -y python3-pyqt5 python3-pyqt5.qtwebengine \
        libqt5webengine5 libqt5webenginecore5 libqt5webenginewidgets5 \
        qtwebengine5-dev-tools

Optionally for OpenGL ES support:
    sudo apt-get install -y libgles2 libgles2-mesa-dev

INSTALLATION:
    pip install PyQt5 PyQtWebEngine

INTEGRATION WITH MAIN APPLICATION:

    from eyes_display import RobotEyes

    eyes = RobotEyes()
    eyes.start()  # starts the display (non-blocking if called from main thread)
    # Later, call bridge methods:
    eyes.bridge.update_emote("happy")
    eyes.bridge.update_eyes_position(0.5, -0.3)
    eyes.bridge.trigger_blink()

    # To stop:
    eyes.stop()

ARCHITECTURE:
    - RobotEyes (QMainWindow) creates a QWebEngineView and loads screen.html.
    - EyeBridge (QObject) provides signals and slots for Python-JavaScript communication.
    - QWebChannel is used to expose the bridge to JavaScript.
    - Fullscreen, frameless, hidden cursor for kiosk mode.
"""

import os
import sys
import signal
import logging
import base64
from pathlib import Path
from PyQt5.QtCore import (
    Qt, QUrl, QObject, pyqtSignal, pyqtSlot, QVariant
)
from PyQt5.QtGui import QCursor
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWebChannel import QWebChannel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger(__name__)

# Determine script directory for relative asset loading
SCRIPT_DIR = Path(__file__).parent.absolute()
HTML_PATH = SCRIPT_DIR / "templates" / "screen.html"

# Obfuscated API_KEY (decode first, then reverse)
# Example: original key "MY_SECRET_API_KEY" -> base64 -> reverse string
# Replace the obfuscated string with your own if needed.
_OBFUSCATED_API_KEY = "UkVMQUNFX1dJVEhfWU9VUl9BUElfS0VZ"  # base64 of "REPLACE_WITH_YOUR_API_KEY" (not reversed yet)
def get_api_key() -> str:
    """Return the decoded API key."""
    # Reverse the obfuscated string first, then base64 decode
    reversed_key = _OBFUSCATED_API_KEY[::-1]
    try:
        key = base64.b64decode(reversed_key).decode('utf-8')
    except Exception:
        key = "invalid_key"
    return key


class EyeBridge(QObject):
    """
    Bridge object exposed to JavaScript via QWebChannel.
    Provides signals that JavaScript can connect to, and slots callable from Python.
    """
    emoteChanged = pyqtSignal(str)
    eyesPositionChanged = pyqtSignal(float, float)
    blinkTriggered = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_emote = "normal"
        self.current_x = 0.0
        self.current_y = 0.0

    # --- Methods to push updates from Python to JavaScript ---
    def update_emote(self, emote_name: str):
        """Called from Python backend to update emote."""
        self.current_emote = emote_name
        self.emoteChanged.emit(emote_name)
        log.info(f"[EyeBridge] Emote updated: {emote_name}")

    def update_eyes_position(self, x: float, y: float):
        """Called from Python backend to update eye position."""
        x = max(-1.0, min(1.0, float(x)))
        y = max(-1.0, min(1.0, float(y)))
        self.current_x = x
        self.current_y = y
        self.eyesPositionChanged.emit(x, y)
        log.info(f"[EyeBridge] Eyes position updated: x={x}, y={y}")

    def trigger_blink(self):
        """Called from Python backend to trigger blink."""
        self.blinkTriggered.emit()
        log.info("[EyeBridge] Blink triggered")

    # --- Slots that JavaScript can call via QWebChannel ---
    @pyqtSlot(result=QVariant)
    def getEmote(self):
        """Return current emote to JavaScript."""
        return {"emote": self.current_emote}

    @pyqtSlot(result=QVariant)
    def getEyesPosition(self):
        """Return current eye position to JavaScript."""
        return {"x": self.current_x, "y": self.current_y}

    @pyqtSlot(result=QVariant)
    def getIp(self):
        """Return IP address information."""
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
        except Exception:
            ip = "127.0.0.1"
        return {"ip": ip}

    @pyqtSlot(result=QVariant)
    def shutdown(self):
        """Handle shutdown request from UI."""
        log.info("[EyeBridge] Shutdown requested from UI")
        # Emit a signal or call a method to close the window
        # The RobotEyes instance will handle this via a custom signal if needed.
        return {"status": "shutting_down"}


class RobotEyes(QMainWindow):
    """
    Main window for the robot eye display.
    Uses QWebEngineView to render the HTML/CSS/JS eye animation.
    """

    def __init__(self):
        super().__init__()
        self.bridge = EyeBridge()
        self.view = QWebEngineView()
        self.channel = QWebChannel()
        self.channel.registerObject("eyeBridge", self.bridge)
        self.view.page().setWebChannel(self.channel)

        self._setup_ui()
        self._load_html()

    def _setup_ui(self):
        """Configure the main window."""
        # Fullscreen, frameless, no window decorations
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setCursor(QCursor(Qt.BlankCursor))
        self.setCentralWidget(self.view)

    def _load_html(self):
        """Load the HTML template."""
        if not HTML_PATH.exists():
            log.error(f"[RobotEyes] HTML file not found: {HTML_PATH}")
            sys.exit(1)
        log.info(f"[RobotEyes] Loading HTML from: {HTML_PATH}")
        url = QUrl.fromLocalFile(str(HTML_PATH))
        self.view.load(url)

    def start(self):
        """Show the window fullscreen."""
        self.showFullScreen()
        log.info("[RobotEyes] Display started fullscreen")

    def stop(self):
        """Close the display."""
        self.close()
        log.info("[RobotEyes] Display stopped")


def optimize_for_arm():
    """
    Apply optimizations for ARM-based Debian systems (Orange Pi 4 Pro).
    Must be called before QApplication is created.
    """
    from PyQt5.QtCore import Qt
    # Enable high DPI scaling
    if hasattr(Qt, 'AA_EnableHighDpiScaling'):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    # Force OpenGL ES for hardware acceleration (important on ARM)
    if hasattr(Qt, 'AA_UseOpenGLES'):
        QApplication.setAttribute(Qt.AA_UseOpenGLES, True)
    log.info("[Optimizer] Attributes set for ARM/Debian")


def main():
    """Main entry point."""
    log.info("=" * 40)
    log.info("  R2 Eye Display (PyQt5)")
    log.info("=" * 40)

    # Apply ARM optimizations before creating QApplication
    optimize_for_arm()

    app = QApplication(sys.argv)

    # Optional: set application name/properties
    app.setApplicationName("R2 Eyes")
    app.setOrganizationName("R2")

    # Create and start the display
    eyes = RobotEyes()
    eyes.start()

    # Set up signal handlers for clean shutdown
    def signal_handler(sig, frame):
        log.info(f"[Main] Received signal {sig}, shutting down...")
        eyes.stop()
        app.quit()

    import signal as sig_module
    sig_module.signal(sig_module.SIGINT, signal_handler)
    sig_module.signal(sig_module.SIGTERM, signal_handler)

    # Start the Qt event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()