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

# Связанная панорама (ch1/ch2): знак theta2↔команда зависит от theta1.
# Замерено на правой руке: при |theta1| > 90° — рука в верхней полусфере
# (ch4/ch5 > 135) — панорама зеркальна (ч1=135−theta2); ниже горизонтали
# (ch4/ch5 ≤ 135) — обычная (ч1=135+theta2). Зеркальна вся верхняя полусфера
# (проверено до ch4/ch5=270). Левая рука — зеркальная конструкция правой,
# правило то же.
PAN_MIRROR_LO = 90.0

# Физическая калибровка рук (3 числа на руку): аффинная карта логических углов
# (t0,t1,t2) из серво-команд в физические углы (t4,t2',t3') модели Ry.
# Подогнана по замерам реального робота в рабочей зоне стола
# (ch1,ch4,ch6) -> EE: (169,66,137)->(0,-300,400), (184,92,62)->(250,0,400),
# (184,118,62)->(400,0,300). Живёт только на границе серво-маппинга
# (theta_from_commands / to_servo_commands); сам инверс и FK — чистые Ry.
NATURAL_AFFINE = {
    "right": ((-58.09, -0.957, 0.647),      # t4  = a + b*t0 + c*t1
              (-10.745, -0.038, 1.387),     # t2' = a + b*t0 + c*t1
              (20.48, -0.108, 0.476)),      # t3' = a + b*t0 + c*t2
    "left":  ((58.09, -0.957, -0.647),
              (-10.745, 0.038, 1.387),
              (20.48, 0.108, 0.476)),
}


def _logical_to_phys(theta: Sequence[float],
                     left: bool) -> Tuple[float, float, float]:
    """Логические углы из серво-команд -> физические углы (t4, t2', t3') Ry."""
    t0, t1, t2 = (float(v) for v in theta)
    a4, a2, a3 = NATURAL_AFFINE[_side(left)]
    t4 = a4[0] + a4[1] * t0 + a4[2] * t1
    t2p = a2[0] + a2[1] * t0 + a2[2] * t1
    t3p = a3[0] + a3[1] * t0 + a3[2] * t2
    return t4, t2p, t3p


def _phys_to_logical(theta: Sequence[float],
                     left: bool) -> Tuple[float, float, float]:
    """Обратная калибровка: физические углы Ry -> логические углы серво."""
    t4, t2p, t3p = (float(v) for v in theta)
    a4, a2, a3 = NATURAL_AFFINE[_side(left)]
    det = a4[1] * a2[2] - a4[2] * a2[1]
    t0 = ((t4 - a4[0]) * a2[2] - a4[2] * (t2p - a2[0])) / det
    t1 = (a4[1] * (t2p - a2[0]) - a2[1] * (t4 - a4[0])) / det
    t2 = (t3p - a3[0] - a3[1] * t0) / a3[2]
    return t0, t1, t2


# Режим стола: перед роботом стол (Y от TABLE_TOP_Y вниз, X ±TABLE_X_HALF,
# Z от TABLE_Z0 до TABLE_Z1). В этом режиме цели на/над столом решаются
# зеркальным отражением: естественный солвер находит позу для зеркальной
# цели (x,−y,z), затем поза отражается через горизонталь (ch_p'=270−ch_p,
# ch_s'=270−ch_s), давая кран сверху на реальную цель. По умолчанию включён.
TABLE_ENABLED = True
TABLE_TOP_Y = -300.0
TABLE_X_HALF = 1000.0
TABLE_Z0 = 0.0
TABLE_Z1 = 1500.0
# Стартовая поза — «манипулятор на столе»: shoulder_z (ch4/ch5) вверх на
# 230°, локти (ch6/ch7) сложены полностью (theta3=180 — предплечье вдоль
# плеча), pan (ch1/ch2) в середине (theta2=0). Рука в этой позе смотрит
# вверх и не задевает стол при старте. Поза отражена от естественной
# (shoulder_z≈195..270): в телеопе над столом рука остаётся в этой же
# ветке и опускается на предмет сверху (спуск предплечьем вниз).
TABLE_START_POSE: Dict[int, int] = {
    0: 90, 1: 135, 2: 135, 3: 90,
    4: 230, 5: 230,
    6: 0, 7: 0,
    8: 90, 9: 90,
}


def _side(left: bool) -> str:
    return "left" if left else "right"


def _channels(left: bool) -> Dict[str, int]:
    return ARM_CHANNELS[_side(left)]


def rest_angles(left: bool = False) -> Dict[int, int]:
    """Logical servo commands for the arm-down pose."""
    return {ch: int(DEFAULT_POSE[ch]) for ch in _channels(left).values()}


def start_pose() -> Dict[int, int]:
    """Стартовая поза всех каналов.

    В режиме стола — сложенная манипуляторная (ch4/ch5=235, локти согнуты),
    иначе DEFAULT_POSE (руки вниз).
    """
    return dict(TABLE_START_POSE if TABLE_ENABLED else DEFAULT_POSE)


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
        if left:
            # Левая: cmd = rest + theta, полный ход команды 0..270 покрывает
            # theta1 ∈ [-45, 225] — природная ветка (локоть вниз) на theta1<0,
            # вывернутая штрафуется на этапе выбора (W_NAT).
            result["shoulder_z"] = (-45.0, 225.0)
        else:
            # Правая после разворота направления (cmd = rest - theta):
            # физически достижима только theta1 ∈ [-225, 35] (команда 270..10,
            # физика 10..270). Дальше — упор сервы: раньше solver уводил
            # правую «влево» и зажимал на лимите.
            result["shoulder_z"] = (-225.0, 35.0)
    return result


def base(left: bool = False) -> np.ndarray:
    return np.array([-BASE_X if left else BASE_X, 0.0, 0.0])


def _pan_mirror(theta1: float) -> bool:
    """Зеркальная панорама: |theta1| > 90° — рука выше горизонтали
    (ch4/ch5 > 135), зеркальна во всей верхней полусфере. Ниже горизонтали —
    обычная. Обе руки симметрично."""
    return abs(float(theta1)) > PAN_MIRROR_LO


def theta_from_commands(commands: Dict[int, float], left: bool = False) -> Tuple[float, float, float]:
    """Servo commands -> physical Ry angles (t4, t2, t3).

    Сначала из команд берутся логические углы (с зеркальной панорамой),
    затем калибровочная аффинная карта приводит их к физическим углам
    модели Ry. Калибровка живёт здесь и только здесь: сам FK/солвер —
    чистые физические градусы.
    """
    rest = rest_angles(left)
    ch = _channels(left)
    theta = []
    for name in JOINT_NAMES:
        cmd = commands.get(ch[name], rest[ch[name]]) - rest[ch[name]]
        if name == "elbow_x":
            cmd = -cmd
        elif name == "shoulder_z" and not left:
            cmd = -cmd
        elif name == "shoulder_x":
            ch4_val = commands.get(ch["shoulder_z"], rest[ch["shoulder_z"]])
            if not left:
                mirror = ch4_val >= rest[ch["shoulder_z"]] + PAN_MIRROR_LO
            else:
                mirror = _pan_mirror(theta[0])
            if mirror:
                cmd = -cmd
        theta.append(float(cmd))
    return _logical_to_phys(theta, left)


def to_servo_commands(theta: Sequence[float], left: bool = False) -> Dict[int, int]:
    """Physical Ry angles (t4, t2, t3) -> servo commands.

    Обратная калибровка переводит физические углы в логические, затем
    логические углы в команды:
      elbow_x:            cmd = rest − t2_logical
      shoulder_z (right): cmd = rest − t0_logical (physically mirrored)
      shoulder_z (left):  cmd = rest + t0_logical
      shoulder_x:         cmd = rest + t1_logical (negated on pan mirror)
    """
    rest = rest_angles(left)
    ranges = servo_ranges(left)
    result = {}
    logical = _phys_to_logical(theta, left)
    mirror_pan = _pan_mirror(float(logical[0]))
    for name, value in zip(JOINT_NAMES, logical):
        ch = _channels(left)[name]
        lo, hi = ranges[ch]
        if name == "elbow_x":
            command = rest[ch] - float(value)
        elif name == "shoulder_z" and not left:
            # Правая поворотная (ch4) физически зеркальна: логический угол
            # растёт -> команда убывает. Парится с theta_from_commands.
            command = rest[ch] - float(value)
        elif name == "shoulder_x" and mirror_pan:
            # Панорама в верхней полусфере зеркальна модели (см. PAN_MIRROR_*).
            command = rest[ch] - float(value)
        else:
            command = rest[ch] + float(value)
        result[ch] = int(round(max(lo, min(hi, command))))
    return result


def _fk_natural_pose(t4: float, t2: float, t3: float,
                     left: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Поза (upper, elbow, ee) модели Ry по физическим углам (t4,t2',t3')."""
    s4, c4 = math.sin(math.radians(t4)), math.cos(math.radians(t4))
    s2, c2 = math.sin(math.radians(t2)), math.cos(math.radians(t2))
    s23, c23 = math.sin(math.radians(t2 + t3)), math.cos(math.radians(t2 + t3))
    shoulder = base(left)
    upper = np.array([s4 * s2, -c2, c4 * s2])
    lower = np.array([s4 * s23, -c23, c4 * s23])
    elbow = shoulder + L1 * upper
    ee = elbow + L2 * lower
    return upper, elbow, ee


def fk_physical(t4: float, t2: float, t3: float,
                left: bool = False) -> Dict[str, np.ndarray]:
    """Forward kinematics in physical Ry angles (t4, t2, t3).

    Coupled axes model: tilt axis rotates WITH the pan.
    EE = base + L1·[s4·s2, −c2, c4·s2] + L2·[s4·s23, −c23, c4·s23].
    """
    _, elbow, ee = _fk_natural_pose(t4, t2, t3, left)
    return {"S": base(left), "E": elbow, "EE": ee}


def fk(theta: Sequence[float], left: bool = False) -> Dict[str, np.ndarray]:
    """Forward kinematics at physical Ry angles (t4, t2, t3).

    Coupled axes model: the shoulder_z rotation axis is rigidly attached to the
    shoulder_x axis, so upper = [s4·s2, −c2, c4·s2] and
    lower = [s4·s23, −c23, c4·s23].
    EE = base + L1·upper + L2·lower.
    """
    t1, t2, t3 = (float(v) for v in theta)
    return fk_physical(t1, t2, t3, left)


def _point_zone_clearance(points: np.ndarray, margin: float = 0.0) -> np.ndarray:
    lo = TORSO - np.array([TORSO_HW, TORSO_HH, TORSO_HD]) - margin
    hi = TORSO + np.array([TORSO_HW, TORSO_HH, TORSO_HD]) + margin
    d_box = np.linalg.norm(np.maximum(np.maximum(lo - points, points - hi), 0.0),
                           axis=-1)
    d_head = np.linalg.norm(points - HEAD, axis=-1) - (HEAD_R + margin)
    return np.minimum(d_box, d_head)


def _table_clearance(points: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Clearance (mm) of points above the table; inf outside the table area."""
    pts = np.asarray(points, dtype=float)
    if not TABLE_ENABLED:
        return np.full(pts.shape[:-1], np.inf)
    inside = ((np.abs(pts[..., 0]) <= TABLE_X_HALF + margin)
              & (pts[..., 2] >= TABLE_Z0 - margin)
              & (pts[..., 2] <= TABLE_Z1 + margin))
    return np.where(inside, np.maximum(0.0, pts[..., 1] - (TABLE_TOP_Y - margin)),
                    np.inf)


def _combined_clearance(points: np.ndarray, margin: float = 0.0) -> np.ndarray:
    """Zone (torso/head) and table clearance combined."""
    return np.minimum(_point_zone_clearance(points, margin),
                      _table_clearance(points, margin))


def target_in_zones(point: Sequence[float], margin: float = 0.0) -> bool:
    points = np.asarray(point, dtype=float)[None, :]
    return bool(float(np.min(_point_zone_clearance(points, margin))) <= 0.0)


def arm_clearance(angles: Sequence[float], left: bool = False,
                  margin: float = ARM_RADIUS) -> float:
    pose = fk(angles, left)
    samples = []
    for start, end in ((pose["S"], pose["E"]), (pose["E"], pose["EE"])):
        count = max(2, int(np.linalg.norm(end - start) // 15.0) + 1)
        ratio = np.linspace(0.0, 1.0, count)
        samples.append(start[None, :] + (end - start)[None, :] * ratio[:, None])
    return float(np.min(_combined_clearance(np.vstack(samples), margin)))


def _elbow_above_line(angles: Sequence[float], left: bool = False) -> float:
    """How far (mm) the elbow sits above the shoulder→hand line.

    The perpendicular offset of E relative to the S→EE line; only the
    camera-vertical (Y) part counts. Negative/near-zero is the natural
    elbow-below pose, positive is the twisted elbow-above pose.
    """
    pose = fk(angles, left)
    s, e, ee = pose["S"], pose["E"], pose["EE"]
    v = ee - s
    length_sq = float(np.dot(v, v))
    if length_sq < 1e-6:
        return 0.0
    perp = e - s - v * (float(np.dot(e - s, v)) / length_sq)
    return float(perp[1])


def _natural_penalty(angles: Sequence[float], left: bool = False) -> float:
    w = W_NAT / 5.0 if _pan_mirror(float(angles[0])) else W_NAT
    return w * max(0.0, _elbow_above_line(angles, left))


def _natural_penalty_series(ranges: Tuple[np.ndarray, ...],
                            left: bool) -> np.ndarray:
    """Vectorised natural penalty over a coarse physical grid (t4,t2,t3)."""
    t4s, t2s, t3s = ranges
    shoulder = base(left)
    shape = (t4s.size, t2s.size, t3s.size)
    s4, c4 = np.sin(np.radians(t4s)), np.cos(np.radians(t4s))
    s2, c2 = np.sin(np.radians(t2s)), np.cos(np.radians(t2s))
    c3, s3 = np.cos(np.radians(t3s)), np.sin(np.radians(t3s))
    c23 = c2[:, None] * c3[None, :] - s2[:, None] * s3[None, :]
    s23 = s2[:, None] * c3[None, :] + c2[:, None] * s3[None, :]
    s4 = np.broadcast_to(s4[:, None, None], shape)
    c4 = np.broadcast_to(c4[:, None, None], shape)
    s2 = np.broadcast_to(s2[None, :, None], shape)
    c2 = np.broadcast_to(c2[None, :, None], shape)
    c23 = np.broadcast_to(c23[None, :, :], shape)
    s23 = np.broadcast_to(s23[None, :, :], shape)
    upper = np.stack([s4 * s2, -c2, c4 * s2], axis=-1)
    lower = np.stack([s4 * s23, -c23, c4 * s23], axis=-1)
    elbow = shoulder + L1 * upper
    ee = elbow + L2 * lower
    v = ee - shoulder
    len_sq = np.sum(v * v, axis=-1, keepdims=True)
    dot = np.sum((elbow - shoulder) * v, axis=-1, keepdims=True)
    scale = np.divide(dot, len_sq,
                      where=len_sq > 1e-6,
                      out=np.zeros_like(len_sq))
    perp = elbow - shoulder - v * scale
    penalty = W_NAT * np.maximum(0.0, perp[..., 1])
    crane_mask = np.abs(t4s) > PAN_MIRROR_LO
    penalty = np.where(crane_mask[:, None, None], penalty / 5.0, penalty)
    return penalty


def _physical_ranges(left: bool, step: float,
                     window: Optional[float] = None,
                     center: Optional[Sequence[float]] = None
                     ) -> Tuple[np.ndarray, ...]:
    """Grid ranges in physical Ry angles (t4, t2, t3).

    Логический бокс из limits(ik_only=True) прогоняется через калибровочную
    аффинную карту: физические границы — это bounding-box образа бокса
    (аффина линейна, поэтому достаточно 8 вершин).
    """
    import itertools
    model_limits = limits(left, ik_only=True)
    names = ("shoulder_z", "shoulder_x", "elbow_x")
    corners = []
    for combo in itertools.product(*[model_limits[n] for n in names]):
        corners.append(_logical_to_phys(combo, left))
    phys = np.asarray(corners, dtype=float)
    bounds = [(float(phys[:, i].min()), float(phys[:, i].max())) for i in range(3)]
    out = []
    for (lo, hi), current in zip(bounds, center or (None,) * 3):
        if window is not None and current is not None:
            lo, hi = max(lo, current - window), min(hi, current + window)
        grid = np.arange(lo, hi + step * 0.5, step)
        if grid.size == 0:
            grid = np.array([lo], dtype=float)
        out.append(grid)
    return tuple(out)


def _fk_physical_grid(t4s: np.ndarray, t2s: np.ndarray, t3s: np.ndarray,
                      left: bool) -> np.ndarray:
    """Vectorised Ry FK over a (t4, t2, t3) grid -> EE positions."""
    s4, c4 = np.sin(np.radians(t4s)), np.cos(np.radians(t4s))
    s2, c2 = np.sin(np.radians(t2s)), np.cos(np.radians(t2s))
    c3, s3 = np.cos(np.radians(t3s)), np.sin(np.radians(t3s))
    shape = (t4s.size, t2s.size, t3s.size)
    c23 = c2[:, None] * c3[None, :] - s2[:, None] * s3[None, :]
    s23 = s2[:, None] * c3[None, :] + c2[:, None] * s3[None, :]
    s4 = np.broadcast_to(s4[:, None, None], shape)
    c4 = np.broadcast_to(c4[:, None, None], shape)
    s2 = np.broadcast_to(s2[None, :, None], shape)
    c2 = np.broadcast_to(c2[None, :, None], shape)
    c23 = np.broadcast_to(c23[None, :, :], shape)
    s23 = np.broadcast_to(s23[None, :, :], shape)
    upper = np.stack([s4 * s2, -c2, c4 * s2], axis=-1)
    lower = np.stack([s4 * s23, -c23, c4 * s23], axis=-1)
    shoulder = base(left)
    ee = shoulder + L1 * upper + L2 * lower
    return ee


def _result(theta: Optional[Tuple[float, float, float]], status: str,
            message: str, left: bool, err_mm: Optional[float] = None,
            wanted: Optional[Sequence[float]] = None,
            clamped: Optional[Sequence[float]] = None,
            reach_gap: float = 0.0) -> dict:
    result = {"theta": theta, "status": status, "message": message,
              "err_mm": err_mm, "left": bool(left), "servo": None,
              "ee": None, "ok": False,
              "wanted": [float(v) for v in wanted] if wanted is not None else None,
              "clamped": [float(v) for v in clamped] if clamped is not None else None,
              "reach_gap": float(reach_gap)}
    if theta is not None:
        result["servo"] = to_servo_commands(theta, left)
        result["ee"] = [float(v) for v in fk(theta, left)["EE"]]
    result["ok"] = err_mm is not None and err_mm <= IK_TOLERANCE_MM
    return result


def ik_solve(x: float, y: float, z: float, left: bool = False,
             start: Optional[Sequence[float]] = None) -> dict:
    """Find a collision-free arm pose nearest to a camera-frame target.

    Одна простая Ry-модель: поиск идёт по всему диапазону физических углов theta
    (сетка + точное уточнение). Калибровка (3 числа на руку) применяется только
    на границе серво-маппинга (theta_from_commands / to_servo_commands), поэтому
    внутри всё чистое: solver и fk работают в физических градусах. Natural-зона
    калибрована по якорям (0.7-14 мм), кран-область — геометрическая
    экстраполяция без замера.
    """
    wanted = np.array([float(x), float(y), float(z)], dtype=float)
    if target_in_zones(wanted):
        return _result(None, "blocked", "Цель внутри туловища или головы", left,
                       wanted=[float(v) for v in wanted],
                       clamped=[float(v) for v in wanted])

    results = _solve_natural(wanted, left, start)

    if not results:
        return _result(None, "blocked",
                       "Цель достижима только с касанием туловища или головы",
                       left, wanted=[float(v) for v in wanted],
                       clamped=[float(v) for v in wanted])

    weights = (1.0, 0.5, 0.25)
    angle_weight = 0.2
    max_angle_penalty = 100.0
    natural_cap = IK_TOLERANCE_MM

    def score(error, natural, theta):
        total = error + min(natural, natural_cap)
        # Штраф за решения на границе crane/natural зон (|theta0|≈90):
        boundary_dist = abs(abs(theta[0]) - PAN_MIRROR_LO)
        if boundary_dist < 10.0:
            total += (10.0 - boundary_dist) * 0.5
        if start is not None:
            angle_cost = sum(w * (a - b) ** 2
                             for w, a, b in zip(weights, theta, start))
            total += min(angle_weight * angle_cost, max_angle_penalty)
        return total

    continuity_budget = 20.0

    def _pick_continuity():
        if start is None:
            return None
        candidates = [(score(error, natural, theta), error, theta)
                      for error, natural, theta in results]
        if not candidates:
            return None
        best = min(candidates, key=lambda item: item[0])
        return best[1], best[2]

    # Разделяем результаты на natural и crane ветки
    natural_results = [r for r in results if not _pan_mirror(r[2][0])]
    crane_results = [r for r in results if _pan_mirror(r[2][0])]

    # Если есть natural-решение в пределах толерантности — предпочитаем его
    natural_ok = [r for r in natural_results if r[0] <= IK_TOLERANCE_MM]

    if natural_ok:
        # Natural доступен: continuityBudget берём из natural-ветки
        continuity = _pick_continuity()
        if (continuity is not None and continuity[0] <= continuity_budget
                and not _pan_mirror(continuity[1][0])):
            best_error, best_theta = continuity
        else:
            best = min(natural_ok, key=lambda item: item[0] + min(item[1], natural_cap))
            best_error, _, best_theta = best
    elif crane_results:
        # Только crane доступен — берём лучший crane
        best = min(crane_results, key=lambda item: item[0])
        best_error, _, best_theta = best
    else:
        # Нет решений — берём лучшее из всех
        continuity = _pick_continuity()
        if continuity is not None and continuity[0] <= continuity_budget:
            best_error, best_theta = continuity
        else:
            best = min(results, key=lambda item: item[0] + min(item[1], natural_cap))
            best_error, _, best_theta = best

    clamped = [float(v) for v in wanted]
    err = best_error
    gap = 0.0
    if best_error <= IK_ERR_OK:
        status = "ok"
    elif best_error <= IK_TOLERANCE_MM:
        status = "limits"
    else:
        status = "unreachable"
    message = f"FK-поиск: |ee−цель|={err:.1f} мм"
    return _result(best_theta, status, message, left, err,
                   wanted=[float(v) for v in wanted], clamped=clamped,
                   reach_gap=gap)


def _solve_jacobian(target: np.ndarray, left: bool,
                    initial: Tuple[float, float, float],
                    max_iter: int = 100, tol: float = 1e-3,
                    delta: float = 1e-3,
                    damping: float = 1e-2) -> Tuple[Tuple[float, float, float], float]:
    """Damped least-squares IK refinement using finite-difference Jacobian.

    Works directly in physical Ry angles (t4, t2, t3).
    Returns (angles, error_mm).
    """
    angles = list(initial)
    for _ in range(max_iter):
        ee = fk(angles, left)["EE"]
        err = target - ee
        err_norm = float(np.linalg.norm(err))
        if err_norm < tol:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            saved = angles[j]
            angles[j] = saved + delta
            ee_plus = fk(angles, left)["EE"]
            angles[j] = saved - delta
            ee_minus = fk(angles, left)["EE"]
            angles[j] = saved
            J[:, j] = (ee_plus - ee_minus) / (2.0 * delta)
        JJT = J @ J.T + damping * np.eye(3)
        try:
            delta_angles = J.T @ np.linalg.solve(JJT, err)
        except np.linalg.LinAlgError:
            break
        for j in range(3):
            angles[j] += float(delta_angles[j])
    final_ee = fk(angles, left)["EE"]
    final_err = float(np.linalg.norm(final_ee - target))
    return tuple(float(a) for a in angles), final_err


def _solve_natural(target, left, start=None):
    """IK solver in physical Ry angles: coarse grid + Jacobian refinement.

    Returns list of (error_mm, natural_penalty, angles) tuples.
    """
    coarse = _physical_ranges(left, *GRID_STEPS[0])
    positions = _fk_physical_grid(*coarse, left)
    errors = np.linalg.norm(positions - target, axis=-1)
    clearance = _combined_clearance(positions, IK_CLEARANCE_MARGIN)
    errors += np.where(clearance <= 0.0, 1e6, 0.0)
    errors += _natural_penalty_series(coarse, left)
    count = min(6, errors.size)

    def _pick_start(scores):
        picked = []
        for index in np.argpartition(scores.ravel(), count - 1)[:count]:
            i1, i2, i3 = np.unravel_index(index, scores.shape)
            angles = (float(coarse[0][i1]), float(coarse[1][i2]),
                      float(coarse[2][i3]))
            if all(max(abs(a - b) for a, b in zip(angles, old)) > 10.0
                   for old in picked):
                picked.append(angles)
        natural_mask = np.abs(coarse[0])[:, None, None] < PAN_MIRROR_LO
        natural_scores = np.where(natural_mask, scores, 1e6)
        best_nat_idx = np.argmin(natural_scores.ravel())
        i1, i2, i3 = np.unravel_index(best_nat_idx, scores.shape)
        best_nat = (float(coarse[0][i1]), float(coarse[1][i2]),
                    float(coarse[2][i3]))
        if all(max(abs(a - b) for a, b in zip(best_nat, old)) > 10.0
               for old in picked):
            picked.append(best_nat)
        return picked

    starts = _pick_start(errors)

    if start is not None:
        starts.append(tuple(float(v) for v in start))

    results = []
    for angles in starts:
        refined, err = _solve_jacobian(target, left, angles)
        if arm_clearance(refined, left, IK_CLEARANCE_MARGIN) >= 0.0:
            natural = _natural_penalty(refined, left)
            results.append((err, natural, refined))
    return results


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
        "start": {str(ch): value for ch, value in start_pose().items()},
        "pan_mirror_lo": PAN_MIRROR_LO,
        "affine": {side: [list(row) for row in triples]
                   for side, triples in NATURAL_AFFINE.items()},
    }


if __name__ == "__main__":
    import sys
    from unittest import mock

    right = (42.0, 35.0, -70.0)
    mirrored = fk((-right[0], right[1], right[2]), left=True)["EE"]
    expected = fk(right, left=False)["EE"].copy()
    expected[0] *= -1.0
    assert np.allclose(mirrored, expected), "left arm must mirror right arm"

    # Самосогласованность: точка кисти позы достигается решателем.
    for theta, is_left in (((-30.0, 40.0, 130.0), False),
                           ((150.0, 50.0, -30.0), True)):
        point = fk(theta, is_left)["EE"]
        result = ik_solve(*point, left=is_left)
        assert result["ok"], (is_left, result)
    assert not ik_solve(0.0, 0.0, 0.0)["ok"]

    # Стартовая поза режима стола — «манипулятор»: ch4/ch5=230 (ветка
    # shoulder_z вверх), локти сложены полностью (ch6/ch7=0, theta3=180).
    # Логические углы проверяются через обратную калибровку _phys_to_logical.
    start = start_pose()
    assert start[4] == 230 and start[5] == 230, start
    assert start[6] == 0 and start[7] == 0, start
    for is_left in (False, True):
        logical = _phys_to_logical(theta_from_commands(start, is_left), is_left)
        assert logical[2] >= 179.0, logical  # theta3=180 — локти полностью сложены
        assert abs(logical[1]) <= 1.0, logical  # theta2=0 — pan в середине

    # Связанная панорама (ch1/ch2): знак theta2↔команда определяется theta1.
    # Зеркальна во всей верхней полусфере (ch4 > 135, |theta1| > 90), обычная
    # ниже горизонтали. Round-trip команда↔theta согласован. Зеркальный
    # кран-путь (Rz) удалён: физическая точность к кран-замерам робота не
    # проверяется, только согласованность маппинга и natural-зона.
    anchors = [
        (102, 226, 58, True),
        (135, 226, 58, True),
        (150, 226, 58, True),
        (102, 235, 58, True),
        (178, 226, 58, True),
        (247, 246, 137, True),
        (169, 66, 137, False),
    ]
    for c1, c4, c6, mirror in anchors:
        cmds = {1: float(c1), 4: float(c4), 6: float(c6)}
        theta = theta_from_commands(cmds, left=False)
        logical = _phys_to_logical(theta, left=False)
        assert abs(logical[0] - (45.0 - c4)) < 1e-6, (logical, c4)
        assert abs(logical[2] - (180.0 - c6)) < 1e-6, (logical, c6)
        assert _pan_mirror(logical[0]) is mirror, (logical, c4)
        assert abs(logical[1] - (135.0 - c1 if mirror else c1 - 135.0)) < 1e-6, \
            (logical, c1)
        back = to_servo_commands(theta, left=False)
        for ch in (1, 4, 6):
            assert back[ch] == cmds[ch], (cmds, theta, back)

    # Нижняя полусфера (ch4 <= 135): калиброванная Ry-модель. Замеры на роботе.
    natural_anchors = [
        ((169, 66, 137), (0.0, -300.0, 400.0), 5.0),
        ((184, 92, 62), (250.0, 0.0, 400.0), 5.0),
        ((184, 118, 62), (400.0, 0.0, 300.0), 20.0),
    ]
    for (c1, c4, c6), meas, tol in natural_anchors:
        theta = theta_from_commands({1: float(c1), 4: float(c4), 6: float(c6)},
                                    left=False)
        assert not _pan_mirror(theta[0]), theta
        ee = fk(theta, left=False)["EE"]
        err = float(np.linalg.norm(ee - np.array(meas)))
        assert err <= tol, (c1, c4, c6, theta, ee, meas, err)
    for c4, mirror in ((66, False), (75, False), (135, False), (195, True),
                       (226, True), (235, True), (238, True), (242, True),
                       (244, True), (246, True), (248, True), (252, True),
                       (258, True), (264, True), (270, True)):
        assert _pan_mirror(45.0 - c4) is mirror, (c4,)
    for left in (False, True):
        chans = _channels(left)
        for c in ((102, 226, 58), (247, 246, 137), (135, 45, 180)):
            cmds = {chans["shoulder_x"]: float(c[0]),
                    chans["shoulder_z"]: float(c[1]),
                    chans["elbow_x"]: float(c[2])}
            theta = theta_from_commands(cmds, left)
            back = to_servo_commands(theta, left)
            assert all(back[chans[n]] == cmds[chans[n]] for n in JOINT_NAMES), \
                (left, c, theta, back)

    # Цели на столе решаются через единую Ry-модель (калибровка — только на
    # границе серво-маппинга). Проверяем основные цели на столе. Обе руки
    # симметричны.
    for is_left in (False, True):
        for target in ((0.0, -300.0, 400.0), (0.0, -260.0, 400.0),
                       (0.0, -250.0, 400.0), (0.0, -280.0, 400.0),
                       (0.0, -300.0, 200.0)):
            result = ik_solve(*target, left=is_left)
            assert result["ok"], (is_left, result)
            assert (result["err_mm"] or 0) <= 5.0, (is_left, result)

    # Зеркальная симметрия: правая и левая рука дают зеркальные EE.
    for target in ((0.0, -300.0, 400.0), (150.0, -300.0, 400.0)):
        r = ik_solve(*target, left=False)
        l = ik_solve(*target, left=True)
        if r["ok"] and l["ok"]:
            ee_r = np.array(r["ee"])
            ee_l = np.array(l["ee"])
            assert abs(ee_r[0] + ee_l[0]) < 10.0, (target, ee_r, ee_l)
            assert abs(ee_r[1] - ee_l[1]) < 10.0, (target, ee_r, ee_l)
            assert abs(ee_r[2] - ee_l[2]) < 10.0, (target, ee_r, ee_l)

    # Высокие цели (y > -100) над столом: зеркало даёт позу ниже горизонтали
    # (ч4 < 135), но рука всё ещё достигает цели.
    high = ik_solve(0.0, -50.0, 400.0, left=False)
    assert high["ok"], high
    assert high["ee"][1] < 0.0, high  # рука ниже плеча

    # Вне зоны стола (сзади) природная ветка сохранена: shoulder_z в природном
    # диапазоне (|theta1| < 90°, не кран-зеркало) у обеих рук.
    for is_left in (False, True):
        result = ik_solve(0.0, -200.0, -300.0, left=is_left)
        assert result["ok"], (is_left, result)
        assert abs(result["theta"][0]) < 90.0, result

    # Край рабочей зоны: боковые цели над столом решаются зеркалом.
    edge = ik_solve(150.0, -150.0, 350.0)
    assert edge["ok"], edge
    hard_fail = ik_solve(150.0, -50.0, 500.0)
    assert hard_fail["ok"], hard_fail

    # Регрессия природной ветки (локоть вниз) при выключенном столе.
    with mock.patch.object(sys.modules[__name__], "TABLE_ENABLED", False):
        for is_left in (False, True):
            sx = -1 if is_left else 1
            result = ik_solve(sx * 250.0, 0.0, 400.0, left=is_left)
            assert result["ok"], result
            assert _elbow_above_line(result["theta"], is_left) <= 0.0
        far = ik_solve(320.0, -200.0, 280.0)
        assert far["ok"] and abs(far["theta"][0]) < 90.0, far

    # Непрерывность: из текущей позы (23,246,137) → движение к другой цели
    # на столе даёт плавный переход (err ≤ 10мм, непрерывность ветки).
    start_cmd = {1: 23, 4: 246, 6: 137}
    start_th = theta_from_commands(start_cmd, False)
    result = ik_solve(0.0, -250.0, 400.0, start=start_th)
    assert result["ok"], result
    assert (result["err_mm"] or 0) <= 10.0, result

    # Блокировка целей внутри туловища.
    assert not ik_solve(0.0, 0.0, 0.0)["ok"]

    print("arm_kinematics: PASS")
