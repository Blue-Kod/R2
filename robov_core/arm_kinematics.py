"""FK/IK for the mirrored R2 arms in camera coordinates.

Coordinates are millimetres: X is camera-right, Y is up, Z is forward.
Logical servo commands are used throughout; ServoController applies the
physical inversion for right-side channels.

Physical model: serial X-Z-X chain (shoulder_x, shoulder_z, elbow_x) built
on the vendored tinyik library.  Joint angles of the chain are calibrated
per channel with a diagonal map (scale + offset), no cross-coupling:
    t = s_i*(cmd_i - rest_i) + o_i
The fit was done against natural-zone anchors measured on the right arm
((169,66,137)->(0,-300,400), (184,92,62)->(250,0,400), (184,118,62)->(400,0,300)):
rms = 30.9 mm, every anchor < 50 mm.  The crane zone (shoulder_z high) is a
geometric extrapolation without measurement (~240 mm rms) and is documented
as approximate.
"""

import math
from functools import lru_cache
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from robov_core.servo import DEFAULT_COMMAND_LIMITS, DEFAULT_POSE
from robov_core.tinyik import Actuator


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
IK_TOLERANCE_MM = 50.0
IK_CLEARANCE_MARGIN = 0.0
W_NAT = 0.3

# Порог «высокой» ветки shoulder_z (|theta_sz| > CRANE_LO). Кран-зона — это
# геометрическая экстраполяция без замера на роботе: значения подходят по
# порядку, но не калиброваны. Граница natural/кран условна (имеет смысл
# только для выбора непрерывности в high_level.move_ik_detail).
CRANE_LO = 90.0

# Диагональная калибровка суставов цепи X-Z-X: t = s*(cmd-rest)+o.
# Подобрана по natural-якорям правой руки (rms 30.9 мм). Левая рука пока не
# установлена: предположение — зеркальная конструкция с той же картой.
CAL = {
    "right": {"shoulder_z": (1.234, -38.6),
              "shoulder_x": (1.147, -109.4),
              "elbow_x": (1.661, 114.0)},
    "left": {"shoulder_z": (1.234, -38.6),
             "shoulder_x": (1.147, -109.4),
             "elbow_x": (1.661, 114.0)},
}


# Режим стола: перед роботом стол (Y от TABLE_TOP_Y вниз, X ±TABLE_X_HALF,
# Z от TABLE_Z0 до TABLE_Z1).
TABLE_ENABLED = True
TABLE_TOP_Y = -300.0
TABLE_X_HALF = 1000.0
TABLE_Z0 = 0.0
TABLE_Z1 = 1500.0
# Стартовая поза — «манипулятор на столе»: shoulder_z (ch4/ch5) вверх на
# 230°, локти (ch6/ch7) сложены полностью, pan (ch1/ch2) в середине.
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
    """Стартовая поза всех каналов (манипуляторная в режиме стола)."""
    return dict(TABLE_START_POSE if TABLE_ENABLED else DEFAULT_POSE)


def servo_ranges(left: bool = False) -> Dict[int, Tuple[int, int]]:
    return {ch: tuple(DEFAULT_COMMAND_LIMITS[ch])
            for ch in _channels(left).values()}


def limits(left: bool = False, ik_only: bool = False) -> Dict[str, Tuple[float, float]]:
    """Диапазоны физических углов суставов (t_sz, t_sx, t_eb), градусы.

    Берутся из командных диапазонов каналов через диагональную калибровку.
    Для обеих рук диапазоны одинаковы (зеркальная конструкция).
    """
    rest = rest_angles(left)
    ranges = servo_ranges(left)
    result = {}
    for name in JOINT_NAMES:
        s, o = CAL[_side(left)][name]
        lo, hi = ranges[_channels(left)[name]]
        result[name] = (float(s * (lo - rest[_channels(left)[name]]) + o),
                        float(s * (hi - rest[_channels(left)[name]]) + o))
    return result


def base(left: bool = False) -> np.ndarray:
    return np.array([-BASE_X if left else BASE_X, 0.0, 0.0])


def _crane_zone(theta: Sequence[float]) -> bool:
    """Высокая ветка shoulder_z (|t_sz| > 90°) — кран-зона без калибровки."""
    return abs(float(theta[0])) > CRANE_LO


def theta_from_commands(commands: Dict[int, float], left: bool = False) -> Tuple[float, float, float]:
    """Servo commands -> chain angles (t_sz, t_sx, t_eb), degrees.

    Диагональная калибровка: t_i = s_i*(cmd - rest) + o_i. Никакой связки
    между суставами. Калибровка живёт здесь и только здесь; FK/солвер —
    чистые физические градусы.
    """
    side = _side(left)
    rest = rest_angles(left)
    ch = _channels(left)
    theta = []
    for name in JOINT_NAMES:
        s, o = CAL[side][name]
        cmd = commands.get(ch[name], rest[ch[name]])
        theta.append(float(s * (cmd - rest[ch[name]]) + o))
    return (theta[0], theta[1], theta[2])


def to_servo_commands(theta: Sequence[float], left: bool = False) -> Dict[int, int]:
    """Chain angles (t_sz, t_sx, t_eb), degrees -> servo commands.

    Обратная калибровка: cmd = rest + (t - o)/s.
    """
    side = _side(left)
    rest = rest_angles(left)
    ranges = servo_ranges(left)
    ch = _channels(left)
    result = {}
    for name, value in zip(JOINT_NAMES, theta):
        s, o = CAL[side][name]
        c = rest[ch[name]] + (float(value) - o) / s
        lo, hi = ranges[ch[name]]
        result[ch[name]] = int(round(max(lo, min(hi, c))))
    return result


@lru_cache(maxsize=2)
def _actuator(left: bool) -> Actuator:
    """tinyik X-Z-X chain: base, shoulder_x (x), shoulder_z (z), forearm."""
    bx = -BASE_X if left else BASE_X
    return Actuator([[bx, 0.0, 0.0], "x", "z", [0.0, -L1, 0.0],
                     "x", [0.0, -L2, 0.0]])


def _chain_angles(theta: Sequence[float], left: bool = False) -> np.ndarray:
    """Physical degrees (t_sz, t_sx, t_eb) -> radians in chain order (x,z,x).

    Left arm is a mirror construction of the right one (X→−X). The X-rotations
    survive the mirror, the Z-rotation (shoulder_z) flips sign:
    R_z(a) under diag(-1,1,1) conjugation becomes R_z(-a).
    """
    t_sz, t_sx, t_eb = (float(v) for v in theta)
    if left:
        t_sz = -t_sz
    return np.radians([t_sx, t_sz, t_eb])


def fk(theta: Sequence[float], left: bool = False) -> Dict[str, np.ndarray]:
    """Forward kinematics at chain angles (t_sz, t_sx, t_eb), degrees.

    Serial X-Z-X chain via tinyik:
      EE = base + L1·Rx(sx)·Rz(sz)·(0,-1,0)
                + L2·Rx(sx)·Rz(sz)·Rx(eb)·(0,-1,0)
    The left arm is a mirror construction: same angles on base -115.
    """
    act = _actuator(left)
    ang = _chain_angles(theta, left)
    elbow = np.asarray(act.fk.solve(ang, index=3), dtype=float)
    ee = np.asarray(act.fk.solve(ang, index=5), dtype=float)
    return {"S": base(left), "E": elbow, "EE": ee}


def fk_physical(t_sz: float, t_sx: float, t_eb: float,
                left: bool = False) -> Dict[str, np.ndarray]:
    """Backwards-compatible alias of fk() with explicit joint angles."""
    return fk((t_sz, t_sx, t_eb), left)


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
    """How far (mm) the elbow sits above the shoulder->hand line.

    The perpendicular offset of E relative to the S->EE line; only the
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
    w = W_NAT / 5.0 if _crane_zone(angles) else W_NAT
    return w * max(0.0, _elbow_above_line(angles, left))


def _physical_ranges(left: bool, step: float,
                     window: Optional[float] = None,
                     center: Optional[Sequence[float]] = None
                     ) -> Tuple[np.ndarray, ...]:
    """Grid ranges in physical chain angles (t_sz, t_sx, t_eb), degrees."""
    model_limits = limits(left, ik_only=True)
    out = []
    for name in JOINT_NAMES:
        lo, hi = model_limits[name]
        current = center[JOINT_NAMES.index(name)] if center is not None else None
        if window is not None and current is not None:
            lo = max(lo, current - window)
            hi = min(hi, current + window)
        grid = np.arange(lo, hi + step * 0.5, step)
        if grid.size == 0:
            grid = np.array([lo], dtype=float)
        out.append(grid)
    return tuple(out)


def _fk_physical_grid(t_szs: np.ndarray, t_sxs: np.ndarray, t_ebs: np.ndarray,
                      left: bool) -> np.ndarray:
    """Vectorised X-Z-X FK over a (t_sz, t_sx, t_eb) grid -> EE positions.

    Matches tinyik chain exactly:
      v1 = Rx(sx)·Rz(sz)·(0,-1,0),  v2 = Rx(sx)·Rz(sz)·Rx(eb)·(0,-1,0)
      EE = base + L1·v1 + L2·v2
    """
    sz = np.radians(t_szs)
    if left:
        sz = -sz
    sx, eb = np.radians(t_sxs), np.radians(t_ebs)
    SZ, SX, EB = np.meshgrid(sz, sx, eb, indexing='ij')
    s_sz, c_sz = np.sin(SZ), np.cos(SZ)
    s_sx, c_sx = np.sin(SX), np.cos(SX)
    s_eb, c_eb = np.sin(EB), np.cos(EB)

    v1 = np.stack([s_sz, -c_sx * c_sz, -s_sx * c_sz], axis=-1)
    v2 = np.stack([(s_sz * c_eb),
                   (-c_sx * c_sz * c_eb + s_sx * s_eb),
                   (-s_sx * c_sz * c_eb - c_sx * s_eb)], axis=-1)
    shoulder = base(left)
    ee = shoulder + L1 * v1 + L2 * v2
    return ee


def _natural_penalty_series(t_szs: np.ndarray, t_sxs: np.ndarray,
                            t_ebs: np.ndarray, left: bool) -> np.ndarray:
    """Vectorised elbow-above-line penalty over a physical grid."""
    sz = np.radians(t_szs)
    if left:
        sz = -sz
    sx, eb = np.radians(t_sxs), np.radians(t_ebs)
    SZ, SX, EB = np.meshgrid(sz, sx, eb, indexing='ij')
    s_sz, c_sz = np.sin(SZ), np.cos(SZ)
    s_sx, c_sx = np.sin(SX), np.cos(SX)
    s_eb, c_eb = np.sin(EB), np.cos(EB)

    v1 = np.stack([s_sz, -c_sx * c_sz, -s_sx * c_sz], axis=-1)
    v2 = np.stack([(s_sz * c_eb),
                   (-c_sx * c_sz * c_eb + s_sx * s_eb),
                   (-s_sx * c_sz * c_eb - c_sx * s_eb)], axis=-1)
    shoulder = base(left)
    elbow = shoulder + L1 * v1
    ee = elbow + L2 * v2
    v = ee - shoulder
    len_sq = np.sum(v * v, axis=-1, keepdims=True)
    dot = np.sum((elbow - shoulder) * v, axis=-1, keepdims=True)
    scale = np.divide(dot, len_sq,
                      where=len_sq > 1e-6,
                      out=np.zeros_like(len_sq))
    perp = elbow - shoulder - v * scale
    crane_mask = np.abs(t_szs)[:, None, None] > CRANE_LO
    w = np.where(crane_mask, W_NAT / 5.0, W_NAT)
    return w * np.maximum(0.0, perp[..., 1])


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

    Единая серийная X-Z-X модель: поиск по всей области физических углов
    theta (сетка + jacobian-уточнение). Калибровка — только на границе
    серво-маппинга; внутри всё чистое. Natural-зона калибрована по якорям
    (rms 30.9 мм), кран-область — геометрическая экстраполяция (~240 мм).
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
    continuity_budget = 20.0

    def score(error, natural, theta):
        total = error + min(natural, natural_cap)
        if start is not None:
            angle_cost = sum(w * (a - b) ** 2
                             for w, a, b in zip(weights, theta, start))
            total += min(angle_weight * angle_cost, max_angle_penalty)
        return total

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
    if (continuity is not None and continuity[0] <= continuity_budget):
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

    Works directly in physical chain angles (t_sz, t_sx, t_eb), degrees.
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
    """IK solver in physical chain angles: coarse grid + Jacobian refinement.

    Returns list of (error_mm, natural_penalty, angles) tuples.
    """
    coarse = _physical_ranges(left, *GRID_STEPS[0])
    positions = _fk_physical_grid(*coarse, left)
    errors = np.linalg.norm(positions - target, axis=-1)
    clearance = _combined_clearance(positions, IK_CLEARANCE_MARGIN)
    errors += np.where(clearance <= 0.0, 1e6, 0.0)
    errors += _natural_penalty_series(*coarse, left)
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
        natural_scores = np.where(
            np.abs(coarse[0])[:, None, None] <= CRANE_LO, scores, 1e6)
        best_nat_idx = np.argmin(natural_scores.ravel())
        i1, i2, i3 = np.unravel_index(best_nat_idx, natural_scores.shape)
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
        "cal": {side: {name: [s, o] for name, (s, o) in cal.items()}
                for side, cal in CAL.items()},
        "crane_lo": CRANE_LO,
    }


if __name__ == "__main__":
    import sys
    from unittest import mock

    right = (20.0, 35.0, -40.0)
    mirrored = fk(right, left=True)["EE"]
    expected = fk(right, left=False)["EE"].copy()
    expected[0] *= -1.0
    assert np.allclose(mirrored, expected), "left arm must mirror right arm"

    # Самосогласованность: точка кисти позы достигается решателем.
    for theta, is_left in (((20.0, 35.0, -40.0), False),
                           ((20.0, 35.0, -40.0), True)):
        point = fk(theta, is_left)["EE"]
        result = ik_solve(*point, left=is_left)
        assert result["ok"], (is_left, result)
    assert not ik_solve(0.0, 0.0, 0.0)["ok"]

    # Стартовая поза режима стола — «манипулятор».
    start = start_pose()
    assert start[4] == 230 and start[5] == 230, start
    assert start[6] == 0 and start[7] == 0, start

    # Round-trip команда<->theta для обеих рук.
    for left in (False, True):
        chans = _channels(left)
        for c in ((102, 226, 58), (247, 246, 137), (135, 45, 180)):
            cmds = {chans["shoulder_x"]: float(c[0]),
                    chans["shoulder_z"]: float(c[1]),
                    chans["elbow_x"]: float(c[2])}
            theta = theta_from_commands(cmds, left)
            back = to_servo_commands(theta, left)
            assert all(back[chans[n]] == int(cmds[chans[n]])
                       for n in JOINT_NAMES), (left, c, theta, back)

    # Natural-якоря правой руки: FK по калибровке точнее 50 мм на каждый.
    natural_anchors = [
        ((169, 66, 137), (0.0, -300.0, 400.0), 50.0),
        ((184, 92, 62), (250.0, 0.0, 400.0), 50.0),
        ((184, 118, 62), (400.0, 0.0, 300.0), 50.0),
    ]
    for (c1, c4, c6), meas, tol in natural_anchors:
        theta = theta_from_commands({1: float(c1), 4: float(c4), 6: float(c6)},
                                    left=False)
        ee = fk(theta, left=False)["EE"]
        err = float(np.linalg.norm(ee - np.array(meas)))
        assert err <= tol, (c1, c4, c6, theta, ee, meas, err)
        assert not _crane_zone(theta), theta

    # Natural-цели на столе решаются и достигаются.
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
        l = ik_solve(-target[0], target[1], target[2], left=True)
        if r["ok"] and l["ok"]:
            ee_r = np.array(r["ee"])
            ee_l = np.array(l["ee"])
            assert abs(ee_r[0] + ee_l[0]) < 10.0, (target, ee_r, ee_l)
            assert abs(ee_r[1] - ee_l[1]) < 10.0, (target, ee_r, ee_l)
            assert abs(ee_r[2] - ee_l[2]) < 10.0, (target, ee_r, ee_l)

    # Непрерывность: движение к другой цели на столе из текущей позы.
    start_cmd = {1: 23, 4: 246, 6: 137}
    start_th = theta_from_commands(start_cmd, False)
    result = ik_solve(0.0, -250.0, 400.0, start=start_th)
    assert result["ok"], result
    assert (result["err_mm"] or 0) <= 10.0, result

    # Регрессия природной ветки (локоть вниз) при выключенном столе.
    with mock.patch.object(sys.modules[__name__], "TABLE_ENABLED", False):
        for is_left in (False, True):
            sx = -1 if is_left else 1
            result = ik_solve(sx * 250.0, 0.0, 400.0, left=is_left)
            assert result["ok"], result
            assert _elbow_above_line(result["theta"], is_left) <= 0.0

    # Блокировка целей внутри туловища.
    assert not ik_solve(0.0, 0.0, 0.0)["ok"]

    print("arm_kinematics: PASS")