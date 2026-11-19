"""ШАБЛОН кинематики R2.

Новая механика пока описана только для оси плеча (4 сервы через редуктор
3:1, см. ``servo.py``). Остальная геометрия (звенья, оси, IK, камера) ещё не
задана, поэтому здесь оставлены интерфейсы, которые вызывают
``high_level`` и веб-панели, с безопасными заглушками.

Когда появится новая кинематика — заполнить ``fk``/``ik_solve``/``limits``
и размеры звеньев; остальной код менять не придётся.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

BASE_X = 0.0
L1 = 0.0
L2 = 0.0

HEAD = np.array([0.0, 100.0, 0.0])
HEAD_LENS_Z = 60.0
HEAD_REST = 0.0
NECK_CHANNEL = None
TILT_CHANNEL = None

HEAD_SAFE_C = np.array([0.0, 100.0, 30.0])
HEAD_SAFE_R = 0.0

IK_ERR_OK = 3.0
IK_TOLERANCE_MM = 50.0
IK_BRANCH = "template"
IK_BRANCHES = ("template",)

JOINT_NAMES = ("shoulder",)
ARM_CHANNELS: Dict[str, Dict[str, int]] = {
    "right": {"shoulder": 0},
    "left": {},
}

TOOL_MM: Dict[str, np.ndarray] = {
    "right": np.zeros(3),
    "left": np.zeros(3),
}

MESSAGE = "кинематика переделывается: IK — шаблон"


def load_tool_offsets(path=None) -> None:
    return None


def set_tool_offset(dx: float, dy: float, dz: float, left: bool = False) -> None:
    TOOL_MM["left" if left else "right"] = np.array([dx, dy, dz], float)


def tool_offset(left: bool = False) -> np.ndarray:
    return TOOL_MM["left" if left else "right"]


def ee_for_grasp(grasp_mm: Sequence[float], left: bool = False) -> np.ndarray:
    return np.asarray(grasp_mm, float) - tool_offset(left)


def grasp_point(theta: Sequence[float], left: bool = False) -> np.ndarray:
    return fk(theta, left)["EE"] + tool_offset(left)


def camera_to_body(p_mm: Sequence[float], neck: float = HEAD_REST,
                   tilt: float = HEAD_REST) -> np.ndarray:
    """Камера -> корпус. Голова в шаблоне неподвижна (без pan/tilt)."""
    return HEAD + np.asarray(p_mm, dtype=float) + np.array([0.0, 0.0, HEAD_LENS_Z])


def rest_angles(left: bool = False) -> Dict[int, int]:
    return {} if left else {0: 0}


def servo_ranges(left: bool = False) -> Dict[int, Tuple[int, int]]:
    return {} if left else {0: (0, 90)}


def limits(left: bool = False) -> Dict[str, Tuple[float, float]]:
    return {} if left else {"shoulder": (0.0, 90.0)}


def theta_from_commands(commands: Dict[int, float],
                        left: bool = False) -> Tuple[float, ...]:
    return () if left else (float(commands.get(0, 0)),)


def to_servo_commands(theta: Sequence[float],
                      left: bool = False) -> Dict[int, int]:
    if left or not theta:
        return {}
    return {0: int(round(min(90.0, max(0.0, float(theta[0])))))}


def fk(theta: Sequence[float], left: bool = False) -> Dict[str, np.ndarray]:
    """Заглушка: реальные точки появятся вместе с новой геометрией."""
    return {"S": np.zeros(3), "E": np.zeros(3), "EE": np.zeros(3)}


def head_clearance(theta: Sequence[float], left: bool = False) -> float:
    return float("inf")


def ik_solve(x: float, y: float, z: float, left: bool = False,
             start: Optional[Sequence[float]] = None) -> dict:
    return {"theta": None, "status": "template", "message": MESSAGE,
            "err_mm": None, "left": bool(left), "servo": None,
            "ee": None, "ok": False,
            "wanted": [float(x), float(y), float(z)]}


def browser_config() -> dict:
    return {
        "branch": IK_BRANCH,
        "base_x": BASE_X,
        "l1": L1,
        "l2": L2,
        "channels": {side: dict(chans)
                     for side, chans in ARM_CHANNELS.items()},
        "rest": {"0": 0},
        "torso": {"half_width": 70.0, "half_height": 95.0, "half_depth": 45.0},
        "head": {"y": float(HEAD[1]), "radius": 45.0},
        "start": {"0": 0},
        "template": True,
    }


def set_branch(name: str) -> str:
    return IK_BRANCH
