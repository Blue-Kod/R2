"""FK/IK for the mirrored R2 arms in camera coordinates.

Coordinates are millimetres: X is camera-right, Y is up, Z is forward.
Logical servo commands are used throughout; ServoController applies the
physical inversion for right-side channels.
"""

import math
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from robov_core.servo import DEFAULT_COMMAND_LIMITS, DEFAULT_POSE


BASE_X = 115.0
L1 = 330.0
L2 = 220.0

TORSO = np.array([0.0, 0.0, 0.0])
HEAD = np.array([0.0, 100.0, 0.0])
TORSO_HW = 70.0
TORSO_HD = 45.0
TORSO_HH = 95.0
HEAD_R = 45.0
ARM_RADIUS = 30.0

JOINT_NAMES = ("shoulder_z", "shoulder_x", "elbow_x")
ARM_CHANNELS = {
    "right": {"shoulder_z": 4, "shoulder_x": 1, "elbow_x": 6},
    "left": {"shoulder_z": 5, "shoulder_x": 2, "elbow_x": 7},
}

GRID_STEPS = ((8.0, None), (2.0, 10.0), (0.4, 2.0), (0.08, 0.5))
IK_ERR_OK = 3.0
IK_TOLERANCE_MM = 10.0
# IK-проверки ведут на половину диапазона пана (45..225), чтобы открыть
# природную ветку с локтём вниз (shoulder_z < 0). Отступ от туловища/головы
# при выборе позы смягчается: жёсткий ARM_RADIUS отсекает правильные позы,
# где локоть проходит вплотную к торсу. Физический контроль остаётся в
# arm_clearance() (margin=ARM_RADIUS) для диагностики.
IK_CLEARANCE_MARGIN = 0.0
# Штраф за «вывернутый локоть»: локоть выше линии плечо→кисть. Малый вес
# отдаёт предпочтение природной ветке, но не блокирует точные решения на
# границе рабочей зоны (где природная ветка недостижима).
W_NAT = 0.3


def _side(left: bool) -> str:
    return "left" if left else "right"


def _channels(left: bool) -> Dict[str, int]:
    return ARM_CHANNELS[_side(left)]


def rest_angles(left: bool = False) -> Dict[int, int]:
    """Logical servo commands for the arm-down pose."""
    return {ch: int(DEFAULT_POSE[ch]) for ch in _channels(left).values()}


def servo_ranges(left: bool = False) -> Dict[int, Tuple[int, int]]:
    return {ch: tuple(DEFAULT_COMMAND_LIMITS[ch])
            for ch in _channels(left).values()}


def limits(left: bool = False, ik_only: bool = False) -> Dict[str, Tuple[float, float]]:
    rest = rest_angles(left)
    ranges = servo_ranges(left)
    result = {}
    for name, ch in _channels(left).items():
        lo, hi = ranges[ch]
        if name == "elbow_x":
            result[name] = (float(rest[ch] - hi), float(rest[ch] - lo))
        else:
            result[name] = (float(lo - rest[ch]), float(hi - rest[ch]))
    if ik_only:
        # Pan command 45..225: the safe half-turn used by IK only.
        # Full range (-45..225) opens the natural branch with elbow down;
        # the twisted branch is penalised at selection time (W_NAT).
        result["shoulder_z"] = (-45.0, 225.0)
    return result


def base(left: bool = False) -> np.ndarray:
    return np.array([-BASE_X if left else BASE_X, 0.0, 0.0])


def theta_from_commands(commands: Dict[int, float], left: bool = False) -> Tuple[float, float, float]:
    rest = rest_angles(left)
    ch = _channels(left)
    theta = []
    for name in JOINT_NAMES:
        cmd = commands.get(ch[name], rest[ch[name]]) - rest[ch[name]]
        if name == "elbow_x":
            cmd = -cmd
        elif name == "shoulder_z" and not left:
            # Правая поворотная физически зеркальна модели (команда убывает
            # с ростом theta) — см. to_servo_commands.
            cmd = -cmd
        theta.append(float(cmd))
    return tuple(theta)


def to_servo_commands(theta: Sequence[float], left: bool = False) -> Dict[int, int]:
    rest = rest_angles(left)
    ranges = servo_ranges(left)
    result = {}
    for name, value in zip(JOINT_NAMES, theta):
        ch = _channels(left)[name]
        lo, hi = ranges[ch]
        if name == "elbow_x":
            command = rest[ch] - float(value)
        elif name == "shoulder_z" and not left:
            # Правая поворотная (ch4) физически зеркальна: theta растёт ->
            # команда убывает. Иначе боковые цели уходили бы во внутреннюю
            # сторону (инверсия «только в IK»). Rest не меняется: theta=0 ->
            # cmd=rest. Парится с theta_from_commands и armFk в webxr.html.
            command = rest[ch] - float(value)
        else:
            command = rest[ch] + float(value)
        result[ch] = int(round(max(lo, min(hi, command))))
    return result


def fk(theta: Sequence[float], left: bool = False) -> Dict[str, np.ndarray]:
    """Forward kinematics for an arm.

    Positive shoulder/elbow pitch reaches camera-forward (+Z). The left arm
    is the X-mirror of the verified right-arm model.
    """
    t1, t2, t3 = (math.radians(float(v)) for v in theta)
    c1, s1 = math.cos(t1), math.sin(t1)
    c2, s2 = math.cos(t2), math.sin(t2)
    c23, s23 = math.cos(t2 + t3), math.sin(t2 + t3)
    side_x = -1.0 if left else 1.0
    shoulder = base(left)
    upper = np.array([side_x * s1 * c2, -c1 * c2, s2])
    lower = np.array([side_x * s1 * c23, -c1 * c23, s23])
    elbow = shoulder + L1 * upper
    ee = elbow + L2 * lower
    return {"S": shoulder, "E": elbow, "EE": ee}


def _point_zone_clearance(points: np.ndarray, margin: float = 0.0) -> np.ndarray:
    lo = TORSO - np.array([TORSO_HW, TORSO_HH, TORSO_HD]) - margin
    hi = TORSO + np.array([TORSO_HW, TORSO_HH, TORSO_HD]) + margin
    d_box = np.linalg.norm(np.maximum(np.maximum(lo - points, points - hi), 0.0),
                           axis=-1)
    d_head = np.linalg.norm(points - HEAD, axis=-1) - (HEAD_R + margin)
    return np.minimum(d_box, d_head)


def target_in_zones(point: Sequence[float], margin: float = 0.0) -> bool:
    points = np.asarray(point, dtype=float)[None, :]
    return bool(float(np.min(_point_zone_clearance(points, margin))) <= 0.0)


def arm_clearance(theta: Sequence[float], left: bool = False,
                  margin: float = ARM_RADIUS) -> float:
    pose = fk(theta, left)
    samples = []
    for start, end in ((pose["S"], pose["E"]), (pose["E"], pose["EE"])):
        count = max(2, int(np.linalg.norm(end - start) // 15.0) + 1)
        ratio = np.linspace(0.0, 1.0, count)
        samples.append(start[None, :] + (end - start)[None, :] * ratio[:, None])
    return float(np.min(_point_zone_clearance(np.vstack(samples), margin)))


def _elbow_above_line(theta: Sequence[float], left: bool = False) -> float:
    """How far (mm) the elbow sits above the shoulder→hand line.

    The perpendicular offset of E relative to the S→EE line; only the
    camera-vertical (Y) part counts. Negative/near-zero is the natural
    elbow-below pose, positive is the twisted elbow-above pose.
    """
    pose = fk(theta, left)
    s, e, ee = pose["S"], pose["E"], pose["EE"]
    v = ee - s
    length_sq = float(np.dot(v, v))
    if length_sq < 1e-6:
        return 0.0
    perp = e - s - v * (float(np.dot(e - s, v)) / length_sq)
    return float(perp[1])


def _natural_penalty(theta: Sequence[float], left: bool = False) -> float:
    return W_NAT * max(0.0, _elbow_above_line(theta, left))


def _natural_penalty_series(ranges: Tuple[np.ndarray, ...],
                            left: bool) -> np.ndarray:
    """Vectorised natural penalty over a coarse FK grid (N1,N2,N3)."""
    t1, t2, t3 = ranges
    side_x = -1.0 if left else 1.0
    shoulder = base(left)
    c1, s1 = np.cos(np.radians(t1)), np.sin(np.radians(t1))
    c2, s2 = np.cos(np.radians(t2)), np.sin(np.radians(t2))
    c3, s3 = np.cos(np.radians(t3)), np.sin(np.radians(t3))
    c23 = c2[:, None] * c3[None, :] - s2[:, None] * s3[None, :]
    s23 = s2[:, None] * c3[None, :] + c2[:, None] * s3[None, :]
    shape = (t1.size, t2.size, t3.size)
    c1, s1 = np.broadcast_to(c1[:, None, None], shape), np.broadcast_to(s1[:, None, None], shape)
    c2, s2 = np.broadcast_to(c2[None, :, None], shape), np.broadcast_to(s2[None, :, None], shape)
    c23, s23 = np.broadcast_to(c23[None, :, :], shape), np.broadcast_to(s23[None, :, :], shape)
    upper = np.stack([side_x * s1 * c2, -c1 * c2, s2], axis=-1)
    lower = np.stack([side_x * s1 * c23, -c1 * c23, s23], axis=-1)
    elbow = shoulder + L1 * upper
    ee = elbow + L2 * lower
    v = ee - shoulder
    len_sq = np.sum(v * v, axis=-1, keepdims=True)
    dot = np.sum((elbow - shoulder) * v, axis=-1, keepdims=True)
    scale = np.divide(dot, len_sq,
                      where=len_sq > 1e-6,
                      out=np.zeros_like(len_sq))
    perp = elbow - shoulder - v * scale
    return W_NAT * np.maximum(0.0, perp[..., 1])


def _fk_grid_positions(t1: np.ndarray, t2: np.ndarray, t3: np.ndarray,
                       left: bool) -> np.ndarray:
    c1, s1 = np.cos(np.radians(t1)), np.sin(np.radians(t1))
    c2, s2 = np.cos(np.radians(t2)), np.sin(np.radians(t2))
    c3, s3 = np.cos(np.radians(t3)), np.sin(np.radians(t3))
    c23 = c2[:, None] * c3[None, :] - s2[:, None] * s3[None, :]
    s23 = s2[:, None] * c3[None, :] + c2[:, None] * s3[None, :]
    planar = L1 * c2[:, None] + L2 * c23
    z_off = L1 * s2[:, None] + L2 * s23
    side_x = -1.0 if left else 1.0
    shoulder = base(left)
    x = shoulder[0] + side_x * planar[None, :, :] * s1[:, None, None]
    y = shoulder[1] - planar[None, :, :] * c1[:, None, None]
    z = np.broadcast_to(shoulder[2] + z_off[None, :, :], x.shape)
    return np.stack([x, y, z], axis=-1)


def _fk_grid_elbows(t1: np.ndarray, t2: np.ndarray, left: bool) -> np.ndarray:
    """Elbow positions over a (t1, t2) sub-grid, for refinement clearance."""
    c1, s1 = np.cos(np.radians(t1)), np.sin(np.radians(t1))
    c2, s2 = np.cos(np.radians(t2)), np.sin(np.radians(t2))
    side_x = -1.0 if left else 1.0
    shoulder = base(left)
    planar = L1 * c2
    x = shoulder[0] + side_x * planar[None, :] * s1[:, None]
    y = shoulder[1] - planar[None, :] * c1[:, None]
    z = np.broadcast_to(shoulder[2], x.shape)
    return np.stack([x, y, z], axis=-1)


def _ranges(left: bool, step: float, window: Optional[float],
            center: Optional[Sequence[float]] = None) -> Tuple[np.ndarray, ...]:
    out = []
    model_limits = limits(left, ik_only=True)
    for name, current in zip(JOINT_NAMES, center or (None,) * 3):
        lo, hi = model_limits[name]
        if window is not None and current is not None:
            lo, hi = max(lo, current - window), min(hi, current + window)
        grid = np.arange(lo, hi + step * 0.5, step)
        if grid.size == 0:
            grid = np.array([lo], dtype=float)
        out.append(grid)
    return tuple(out)


def _best_on_grid(ranges: Tuple[np.ndarray, ...], target: np.ndarray,
                  left: bool) -> Tuple[Tuple[float, float, float], float]:
    positions = _fk_grid_positions(*ranges, left)
    error = np.linalg.norm(positions - target, axis=-1)
    penalty = np.where(_point_zone_clearance(positions, IK_CLEARANCE_MARGIN) <= 0.0,
                       1e6, 0.0)
    # Локоть тоже не должен пролегать сквозь туловище/голову: иначе
    # refinement может увести позу в коллизию, которую потом отклонит
    # финальный arm_clearance (и решение потеряется).
    elbow = _fk_grid_elbows(ranges[0], ranges[1], left)
    elbow_penalty = np.where(_point_zone_clearance(elbow, IK_CLEARANCE_MARGIN) <= 0.0,
                             1e6, 0.0)
    penalty += np.broadcast_to(elbow_penalty[..., None], penalty.shape)
    index = int(np.argmin(error + penalty))
    i1, i2, i3 = np.unravel_index(index, error.shape)
    theta = (float(ranges[0][i1]), float(ranges[1][i2]), float(ranges[2][i3]))
    return theta, float(error[i1, i2, i3])


def _result(theta: Optional[Tuple[float, float, float]], status: str,
            message: str, left: bool, err_mm: Optional[float] = None) -> dict:
    result = {"theta": theta, "status": status, "message": message,
              "err_mm": err_mm, "left": bool(left), "servo": None,
              "ee": None, "ok": False}
    if theta is not None:
        result["servo"] = to_servo_commands(theta, left)
        result["ee"] = [float(v) for v in fk(theta, left)["EE"]]
    result["ok"] = err_mm is not None and err_mm <= IK_TOLERANCE_MM
    return result


def ik_solve(x: float, y: float, z: float, left: bool = False,
             start: Optional[Sequence[float]] = None) -> dict:
    """Find a collision-free arm pose nearest to a camera-frame target."""
    target = np.array([float(x), float(y), float(z)], dtype=float)
    if target_in_zones(target):
        return _result(None, "blocked", "Цель внутри туловища или головы", left)

    coarse = _ranges(left, *GRID_STEPS[0])
    positions = _fk_grid_positions(*coarse, left)
    errors = np.linalg.norm(positions - target, axis=-1)
    clearance = _point_zone_clearance(positions, IK_CLEARANCE_MARGIN)
    errors += np.where(clearance <= 0.0, 1e6, 0.0)
    # Природная поза (локоть ниже линии плечо→кисть) получает фору при
    # выборе стартов грубой сетки, чтобы вывернутая ветка не доминировала.
    errors += _natural_penalty_series(coarse, left)
    count = min(6, errors.size)

    def _pick_start(scores):
        picked = []
        for index in np.argpartition(scores.ravel(), count - 1)[:count]:
            i1, i2, i3 = np.unravel_index(index, scores.shape)
            theta = (float(coarse[0][i1]), float(coarse[1][i2]),
                     float(coarse[2][i3]))
            if all(max(abs(a - b) for a, b in zip(theta, old)) > 10.0
                   for old in picked):
                picked.append(theta)
        return picked

    # Кандидаты стартов — несколько веток IK по чистой ошибке,
    # чтобы глобальный минимум не терялся из-за локальных минимумов.
    starts = _pick_start(errors)

    if start is not None:
        # Текущая поза — всегда кандидат: удерживаем ветку при непрерывном
        # движении цели, чтобы рука не «перелётывала» между ветками IK.
        starts.append(tuple(float(v) for v in start))

    results = []
    for theta in starts:
        current = theta
        for step, window in GRID_STEPS[1:]:
            current, _ = _best_on_grid(_ranges(left, step, window, current),
                                       target, left)
        error = float(np.linalg.norm(fk(current, left)["EE"] - target))
        if arm_clearance(current, left, IK_CLEARANCE_MARGIN) >= 0.0:
            results.append((error, _natural_penalty(current, left), current))

    if not results:
        return _result(None, "blocked",
                       "Цель достижима только с касанием туловища или головы",
                       left)

    # Natural-штраф ограничен сверху: он лишь «добивает» вывернутую позу
    # среди равноточных решений, но не вытесняет точное решение в пользу
    # недостижимой природной (край рабочей зоны).
    weights = (1.0, 0.5, 0.25)
    angle_weight = 0.2
    max_angle_penalty = 100.0
    natural_cap = IK_TOLERANCE_MM

    def score(error, natural, theta):
        total = error + min(natural, natural_cap)
        if start is not None:
            angle_cost = sum(w * (a - b) ** 2
                             for w, a, b in zip(weights, theta, start))
            total += min(angle_weight * angle_cost, max_angle_penalty)
        return total

    # Непрерывность: текущая поза всегда в кандидатах; если её погрешность
    # в пределах бюджета, оставляем ветку — рука не дёргается при плавном
    # движении цели через переходную зону. Бюджет чуть шире teleop-deadband
    # (20 мм): при медленном движении руки ошибка до ~25 мм не заметна,
    # зато нет резкого «перелёта» между ветками. Если ветка сильно отстала
    # от цели — переключаемся на точное решение.
    continuity_budget = 25.0

    def _pick_continuity():
        if start is None:
            return None
        candidates = [(score(error, natural, theta), error, theta)
                      for error, natural, theta in results]
        if not candidates:
            return None
        best = min(candidates, key=lambda item: item[0])
        return best[1], best[2]

    continuity = _pick_continuity()
    if continuity is not None and continuity[0] <= continuity_budget:
        best_error, best_theta = continuity
    else:
        accurate = [item for item in results if item[0] <= IK_TOLERANCE_MM]
        pool = accurate if accurate else results
        best = min(pool, key=lambda item: score(*item))
        best_error, _, best_theta = best

    if best_error <= IK_ERR_OK:
        status = "ok"
    elif best_error <= IK_TOLERANCE_MM:
        status = "limits"
    else:
        status = "unreachable"
    return _result(best_theta, status,
                   f"FK-поиск: |ee−цель|={best_error:.1f} мм",
                   left, best_error)


def browser_config() -> dict:
    """Small JSON-safe configuration for the browser renderer."""
    return {
        "base_x": BASE_X,
        "l1": L1,
        "l2": L2,
        "rest": {str(ch): value for left in (False, True)
                 for ch, value in rest_angles(left).items()},
        "channels": {
            side: {name: ch for name, ch in channels.items()}
            for side, channels in ARM_CHANNELS.items()
        },
        "torso": {"half_width": TORSO_HW, "half_height": TORSO_HH,
                  "half_depth": TORSO_HD},
        "head": {"y": float(HEAD[1]), "radius": HEAD_R},
    }


if __name__ == "__main__":
    right = (42.0, 35.0, -70.0)
    mirrored = fk(right, left=True)["EE"]
    expected = fk(right, left=False)["EE"].copy()
    expected[0] *= -1.0
    assert np.allclose(mirrored, expected), "left arm must mirror right arm"
    for is_left in (False, True):
        point = fk((20.0, 30.0, -60.0), is_left)["EE"]
        result = ik_solve(*point, left=is_left)
        assert result["ok"], result
    assert not ik_solve(0.0, 0.0, 0.0)["ok"]

    # Природная поза: для цели вперёд-вбок shoulder_z должен уходить в
    # отрицательную зону (локоть вниз), а не выворачиваться на +145°.
    for is_left in (False, True):
        sx = -1 if is_left else 1
        result = ik_solve(sx * 150.0, 50.0, 300.0, left=is_left)
        assert result["ok"], result
        theta = result["theta"]
        assert theta[0] < 0.0, f"shoulder_z должен быть природным, got {theta}"
        assert _elbow_above_line(theta, is_left) <= 0.0, \
            f"локоть вывернут вверх: {theta}"
    # Край рабочей зоны (x=200) — точная природная ветка недостижима;
    # допустим вывернутый fallback, но решение обязано существовать.
    edge = ik_solve(200.0, 50.0, 250.0)
    assert edge["ok"], edge
    print("arm_kinematics: PASS")
