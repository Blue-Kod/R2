import os
import subprocess
import threading
from dataclasses import dataclass

from .config import AppConfig
from .logging_utils import AppLogger
from .services.emote_service import EmoteService
from .services.gemini_service import GeminiGateway
from .services.hardware_service import HardwareService
from .services.shell_service import ShellService
from .services.system_service import SystemService
from .web import create_app


@dataclass
class ServiceContainer:
    config: AppConfig
    logger: AppLogger
    system: SystemService
    shell: ShellService
    hardware: HardwareService
    gemini: GeminiGateway
    emote: EmoteService


def build_services() -> ServiceContainer:
    config = AppConfig()
    logger = AppLogger(buffer_size=500)
    system = SystemService()
    shell = ShellService(logger=logger)
    hardware = HardwareService(config=config, logger=logger)
    gemini = GeminiGateway(api_key=os.getenv("GEMINI_API_KEY"), model=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"))
    emote = EmoteService()
    return ServiceContainer(
        config=config,
        logger=logger,
        system=system,
        shell=shell,
        hardware=hardware,
        gemini=gemini,
        emote=emote,
    )


def open_browser(url: str) -> None:
    """Attempt to open the given URL in the default browser."""
    import webbrowser
    import shutil
    
    # Try webbrowser module first
    try:
        webbrowser.open(url)
        return
    except Exception:
        pass
    
    # Fallback: try common browser commands
    browsers = [
        "chromium-browser",
        "chromium",
        "firefox-esr",
        "firefox",
        "google-chrome",
        "xdg-open",
    ]
    for browser in browsers:
        if shutil.which(browser):
            try:
                subprocess.Popen([browser, url])
                return
            except Exception:
                continue


def run() -> None:
    services = build_services()
    services.logger.log("Запуск веб-сервера R2 (новая архитектура)")

    services.hardware.initialize()
    services.shell.start()

    app = create_app(services=services, logger=services.logger)
    
    # Schedule browser opening after a short delay
    def launch_browser():
        import time
        time.sleep(2)  # Wait for server to start
        url = f"http://localhost:{services.config.http_port}"
        services.logger.log(f"Открытие браузера: {url}")
        open_browser(url)
    
    threading.Thread(target=launch_browser, daemon=True).start()
    
    app.run(host=services.config.host, port=services.config.http_port, debug=False, threaded=True)

