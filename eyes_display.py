#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robot Eye Display using pywebview.
Replaces Chromium kiosk mode with a lightweight native window.
Optimized for Orange Pi 4 Pro (Debian ARM).

INSTALLATION & SETUP (Debian/ARM):

1. Install system dependencies (GTK3 for pywebview):
   sudo apt-get update
   sudo apt-get install -y python3-gi python3-gi-cairo gir1.2-gtk-3.0 \
       libwebkit2gtk-4.0-dev libglib2.0-dev libgtk-3-dev

2. Install Python dependencies:
   pip install pywebview[gtk]

   Or if using requirements.txt:
   pip install -r requirements.txt

3. Environment variables (set before running):
   export DISPLAY=:0
   export XAUTHORITY=/home/orangepi/.Xauthority  # if needed

4. Run the display:
   python3 eyes_display.py

INTEGRATION WITH MAIN APPLICATION:

The EyeAPI class provides methods to push updates from your Python backend:

    from eyes_display import EyeDisplay

    display = EyeDisplay()

    # In a separate thread or async loop:
    display.api.update_emote("happy")
    display.api.update_eyes_position(0.5, -0.3)  # x, y normalized -1..1
    display.api.trigger_blink()

    # Start the display (blocks until window is closed):
    display.start()

For integration with existing main.py, you can run the display in a separate
thread or process, then call the API methods via a shared reference or IPC.

DEBIAN/ARM OPTIMIZATION NOTES:

- GTK backend is explicitly used (best support on Debian ARM)
- Fullscreen mode with no window decorations for kiosk feel
- Context menu and text selection disabled
- Signal handlers (SIGINT, SIGTERM) for clean shutdown
- DISPLAY=:0 should be set in systemd service or autostart script

Example systemd service:
    [Unit]
    Description=R2 Eye Display
    After=graphical.target

    [Service]
    User=orangepi
    Environment=DISPLAY=:0
    Environment=XAUTHORITY=/home/orangepi/.Xauthority
    ExecStart=/usr/bin/python3 /path/to/eyes_display.py
    Restart=on-failure

    [Install]
    WantedBy=graphical.target

"""

import os
import sys
import time
import signal
import logging
from pathlib import Path

try:
    import webview
except ImportError:
    print("[!] pywebview not installed. Install with: pip install pywebview[gtk]")
    sys.exit(1)

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


class EyeAPI:
    """
    JavaScript-Python bridge for the eye display system.
    This class exposes methods that can be called from JavaScript,
    and provides methods to push data to the JavaScript layer.
    """

    def __init__(self):
        self._window = None
        self.current_emote = "normal"
        self.current_x = 0
        self.current_y = 0

    def set_window(self, window):
        """Set the pywebview window reference for JS evaluation."""
        self._window = window

    # Methods callable from JavaScript
    def getEmote(self):
        """Return current emote to JavaScript."""
        return {"emote": self.current_emote}

    def getEyesPosition(self):
        """Return current eye position to JavaScript."""
        return {"x": self.current_x, "y": self.current_y}

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

    def shutdown(self):
        """Handle shutdown request from UI."""
        log.info("[EyeAPI] Shutdown requested from UI")
        if self._window:
            self._window.destroy()
        return {"status": "shutting_down"}

    # Methods to push data from Python to JavaScript
    def update_emote(self, emote_name):
        """
        Update emote from Python backend and push to JavaScript.
        Call this method from your main application logic.
        """
        self.current_emote = emote_name
        if self._window:
            try:
                self._window.evaluate_js(f'applyEmote("{emote_name}")')
                log.info(f"[EyeAPI] Emote updated: {emote_name}")
            except Exception as e:
                log.error(f"[EyeAPI] Failed to update emote: {e}")

    def update_eyes_position(self, x, y):
        """
        Update eye position from Python backend and push to JavaScript.
        x, y should be normalized values between -1 and 1.
        """
        self.current_x = max(-1, min(1, float(x)))
        self.current_y = max(-1, min(1, float(y)))
        if self._window:
            try:
                self._window.evaluate_js(f'applyEyesPosition({self.current_x}, {self.current_y})')
                log.info(f"[EyeAPI] Eyes position updated: x={self.current_x}, y={self.current_y}")
            except Exception as e:
                log.error(f"[EyeAPI] Failed to update eyes position: {e}")

    def trigger_blink(self):
        """Trigger a blink animation from Python."""
        if self._window:
            try:
                self._window.evaluate_js('blink()')
            except Exception as e:
                log.error(f"[EyeAPI] Failed to trigger blink: {e}")


class EyeDisplay:
    """
    Main display controller for the robot eye system.
    Manages the pywebview window lifecycle.
    """

    def __init__(self):
        self.api = EyeAPI()
        self.window = None
        self._shutdown_requested = False
        self._ready_event = threading.Event()
        self._window_created_event = threading.Event()

    @property
    def ready(self):
        """Return True if the page is fully loaded."""
        return self._ready_event.is_set()

    def wait_until_ready(self, timeout=None):
        """Wait until the page is loaded and ready. Returns True if ready, False if timeout."""
        return self._ready_event.wait(timeout)

    def wait_until_window_created(self, timeout=None):
        """Wait until the window is created. Returns True if created, False if timeout."""
        return self._window_created_event.wait(timeout)

    def _on_closed(self):
        """Handle window close event."""
        log.info("[EyeDisplay] Window closed")
        self._shutdown_requested = True

    def _on_loaded(self):
        """Handle page loaded event."""
        log.info("[EyeDisplay] Page loaded successfully")
        self._ready_event.set()

    def start(self):
        """
        Initialize and start the eye display.
        This method blocks until the window is closed.
        """
        if not HTML_PATH.exists():
            log.error(f"[EyeDisplay] HTML file not found: {HTML_PATH}")
            sys.exit(1)

        log.info(f"[EyeDisplay] Loading HTML from: {HTML_PATH}")

        # Configure pywebview for optimal performance on Debian/ARM
        # Using GTK backend is recommended for Debian-based systems
        webview.WEBVIEW_GTK = True

        # Create window with kiosk-like settings
        self.window = webview.create_window(
            title="R2 Eyes",
            url=str(HTML_PATH),
            fullscreen=True,
            frameless=True,      # Remove window decorations
            easy_drag=False,     # Disable dragging
            focus=True,
            js_api=self.api,     # Register our JS bridge
            text_select=False,   # Disable text selection
            context_menu=False,  # Disable right-click menu
        )

        # Set window reference in API
        self.api.set_window(self.window)

        # Register event handlers
        self.window.events.closed += self._on_closed
        self.window.events.loaded += self._on_loaded

        log.info("[EyeDisplay] Window created, starting main loop...")

        # Set up signal handlers for clean shutdown
        def signal_handler(sig, frame):
            log.info(f"[EyeDisplay] Received signal {sig}, shutting down...")
            self._shutdown_requested = True
            if self.window:
                self.window.destroy()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Start the pywebview event loop
        # This blocks until the window is closed
        try:
            webview.start(
                gui=webview.GTK,  # Explicitly use GTK backend for Debian
                debug=False       # Set to True for debugging
            )
        except KeyboardInterrupt:
            log.info("[EyeDisplay] Keyboard interrupt received")
        except Exception as e:
            log.error(f"[EyeDisplay] Error in main loop: {e}")
        finally:
            log.info("[EyeDisplay] Display stopped")

    def stop(self):
        """Programmatically stop the display."""
        self._shutdown_requested = True
        if self.window:
            self.window.destroy()


def optimize_for_arm():
    """
    Apply optimizations for ARM-based Debian systems (Orange Pi 4 Pro).
    Call this before starting the display if needed.
    """
    # Set environment variables for optimal GUI performance
    os.environ.setdefault('DISPLAY', ':0')

    # Force GTK backend for pywebview on Debian
    os.environ['WEBVIEW_GUI'] = 'gtk'

    # Disable compositing if running on minimal X setup
    # os.environ['XLIB_SKIP_ARGB_VISUALS'] = '1'

    log.info("[Optimizer] Environment optimized for ARM Debian")


def main():
    """Main entry point."""
    log.info("=" * 40)
    log.info("  R2 Eye Display (pywebview)")
    log.info("=" * 40)

    # Apply ARM optimizations
    optimize_for_arm()

    # Create and start display
    display = EyeDisplay()

    try:
        display.start()
    except Exception as e:
        log.error(f"Fatal error: {e}")
        sys.exit(1)

    log.info("Application terminated cleanly")
    sys.exit(0)


if __name__ == "__main__":
    main()