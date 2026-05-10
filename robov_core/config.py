from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class AppConfig:
    host: str = "0.0.0.0"
    http_port: int = 80
    camera_source: int = 0
    camera_params_file: str = "cam_params.json"
    launcher_script: str = "launcher.py"
    tracking_timeout_seconds: float = 10.0
    eyes_scale_factor: float = 1.3
    default_servo_angles: Dict[int, int] = field(
        default_factory=lambda: {
            0: 90,  # Шея
            1: 135,  # Правое плечо
            2: 135,  # Левое плечо
            3: 90,  # Наклон головы
            4: 45,  # Поворот правого плеча
            5: 45,  # Поворот левого плеча
            6: 135,  # Правый локоть
            7: 135  # Левый локоть
        }
    )

    @property
    def root_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def camera_config_path(self) -> Path:
        return self.root_dir / self.camera_params_file

    @property
    def launcher_path(self) -> Path:
        return self.root_dir / self.launcher_script

