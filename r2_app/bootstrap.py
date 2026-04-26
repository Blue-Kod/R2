import os
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


def run() -> None:
    services = build_services()
    services.logger.log("Запуск веб-сервера R2 (новая архитектура)")

    services.hardware.initialize()
    services.shell.start()

    app = create_app(services=services, logger=services.logger)
    app.run(host=services.config.host, port=services.config.http_port, debug=False, threaded=True)

