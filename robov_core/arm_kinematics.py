"""FK/IK для рук R2 — чистая серийная цепь X–Z–X (идеальная геометрия).

=====================================================================
ФИЗИЧЕСКАЯ СТРУКТУРА РУКИ (подтверждено владельцем, сент 2026)
=====================================================================

Правая рука — последовательная серийная цепь из трёх вращательных
соединений (по аналогии с UR-манипулятором):

    база ──[ч1]──/плечо L1=330мм/──[ч4]──/предплечье L2=220мм/──[ч6]── кисть

      ч1 : shoulder_x  — ОСЬ X в системе координат базы.
      ч4 : shoulder_z  — ось вращения НЕ жёстко Z: это следующее (серверное)
          соединение после ч1, его фактическая ось в пространстве зависит
          от поворота ч1. В базовой позе ось ч4 смотрит по Z, но при повороте
          ч1 в 225° (90 от базового угла) эта же ось разворачивается в сторону оси Y (по правилам
          серийной кинематики: матрица поворота ч1 умножается на остаток цепи).
          В терминах tinyik это соединение — 'z' в локальной цепочке
          ['x','z','x'] (см. ниже).
      ч6 : elbow_x     — ОСЬ X, тоже в локальной системе (следует за ч1, ч4).

    База правой руки:  X = +115 мм (вправо от центра туловища).
    База левой руки:   X = -115 мм (зеркально). ЛЕВАЯ РУКА ПОКА НЕ СТОИТ.

Соединения по номеру канала (логические команды серво, см. servo.py):
    правая:  shoulder_x=ч1, shoulder_z=ч4, elbow_x=ч6
    левая:   shoulder_x=ч2, shoulder_z=ч5, elbow_x=ч7
    (у правой/левой по своим каналам — зеркальная конструкция).

ВАЖНО про порядок: в каналах чтобы ч4 был "как второй сустав", а не
независимый — это серийная цепь x-z-x, а НЕ три независимые оси.
tinyik-цепочка (внешние координаты, мм):
    Actuator([[±115,0,0], 'x', 'z', [0,-330,0], 'x', [0,-220,0]])
    углы tinyik (радианы) = (sx, sz, eb)  [порядок суставов цепи x,z,x].

ПРАВИЛО ЗЕРКАЛА (левая рука): база -115, а угол shoulder_z для левой
берётся с противоположным знаком (R_z(a) при зеркале diag(-1,1,1)
преобразуется в R_z(-a)); ч1/ч6 (оси X) знак НЕ меняют.

=====================================================================
УГЛЫ / ДИАПАЗОНЫ / НАПРАВЛЕНИЯ
=====================================================================

theta = (t_sz, t_sx, t_eb) — физические углы звеньев, градусы, порядок
ГЕОМЕТРИЧЕСКИЙ (sz, sx, eb), а в tinyik передаём как (sx, sz, eb).

Диапазоны команд каналов (из servo.DEFAULT_COMMAND_LIMITS):
    ч1/ч2/ч4/ч5        : 0..270° (0.102..0.540 мс пульс)
    ч6/ч7 (локоть)     : 0..180° (0.102..0.540 мс пульс)
    Физически серво ч4 (правое shoulder_z) ИНВЕРТИРОВАНО на уровне PWM
    (INVERTED_CHANNELS={2,4,8} в servo.py), но arm_kinematics работает
    с логическими командами — инверсию применяет ServoController.

Маппинг theta <-> логические команды (у левой shoulder_z зеркальный):
    правая:  sz = cmd4 - 45;   sx = cmd1 - 135;   eb = 180 - cmd6
    левая:   sz = 45 - cmd5;   sx = cmd2 - 135;   eb = 180 - cmd7
    (локоть и левое shoulder_z: cmd = rest - theta; остальное cmd = rest + theta).

Направления (в системе координат камера: X вправо, Y вверх, Z вперёд):
    shoulder_z (ч4) : поворот кисти вбок относительно корпуса.
    shoulder_x (ч1) : наклон руки вперёд/назад (от -падения).
    elbow_x    (ч6) : сгиб локтя (0 = выпрямлено, 180 = согнуто).

ВНИМАНИЕ: это ИДЕАЛЬНАЯ геометрическая модель (серийная цепь X–Z–X
по описанию выше), БЕЗ эмпирической калибровки под замеры. Она НЕ
сходится с приведёнными ниже якорными замерами (расхождение до ~800 мм)
из-за монтажных смещений осей серво на железе. Соответствие роботу
достигается оффсетами каналов (set_servo_offset) после постановки руки.
Реальные направления уточняются по arm_test.py.

=====================================================================
ЯКОРНЫЕ ТОЧКИ (measurements, см. коммит до переделки и arm_test.py)
=====================================================================
Измерено на ПРАВОЙ руке (логические команды ч1,ч4,ч6 -> EE в мм
в системе камеры):

  natural (ниже горизонтали, ч4<=135):
    (169,66,137)  -> (0,-300,400)
    (184,92,62)   -> (250,0,400)
    (184,118,62)  -> (400,0,300)

  crane (верхняя полусфера, ч4>135):
    (102,226,58)  -> (120,30,300)
    (135,226,58)  -> (120,200,240)
    (150,226,58)  -> (120,280,150)
    (178,226,58)  -> (120,290,0)
    (247,246,137) -> (99,-42,-511)
    (102,235,58)  -> (115,30,280)

ЛЕВАЯ РУКА ПОКА НЕ УСТАНОВЛЕНА: зеркально база -115 и t_sz с минусом.
=====================================================================
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

IK_ERR_OK = 3.0
IK_TOLERANCE_MM = 50.0
IK_CLEARANCE_MARGIN = 0.0
W_NAT = 0.3

# Сетка поиска IK: (шаг, радиус окна вокруг текущей точки).
# None-радиус — первая грубая решётка по всему диапазону.
GRID_STEPS = ((8.0, None), (2.0, 10.0), (0.4, 2.0), (0.08, 0.5))

# Режим стола (поза старта; поиск по столу — вне скоупа, см. план).
TABLE_ENABLED = True
TABLE_TOP_Y = -300.0
TABLE_X_HALF = 1000.0
TABLE_Z0 = 0.0
TABLE_Z1 = 1500.0
# Коллайдер «стол»: рука (локоть и кисть) никогда не опускается ниже
# плоскости TABLE_MIN_Y (*Y* — вертикаль модели, +Y вверх; база на y=0,
# стол на 300 мм ниже плеча). TABLE_MARGIN = 0 — касание поверхности
# разрешено; TABLE_COLLISION_PENALTY добавляется к ошибке коллизийных
# клеток сетки, чтобы они проигрывали любой валидной позе.
TABLE_MARGIN = 0.0
TABLE_MIN_Y = TABLE_TOP_Y + TABLE_MARGIN
TABLE_COLLISION_PENALTY = 1e6
# Кран-предпочтение (работает, пока TABLE_ENABLED=True — ВСЕГДА в режиме
# стола): среди поз, достающих одну и ту же цель, выбирается ветка с
# высоким локтем (elbow_y заметно выше кисти), «кран-ветка», а не
# провисшая натуральная («через низ»). Байас ДОМИНАНТНЫЙ
# (TABLE_CRANE_BIAS_MAX = 1e6 > любой ошибки достижения): если кран-ветка
# существует — выбирается всегда, даже если ей не хватает точности до цели.
# TABLE_CRANE_LIFT — целевой «лифт» elbow_y − ee_y (мм); за каждый мм
# недобора добавляется TABLE_CRANE_WEIGHT мм-ошибки (до BIAS_MAX).
TABLE_CRANE_LIFT = 150.0
TABLE_CRANE_WEIGHT = 5.0
TABLE_CRANE_BIAS_MAX = 1e6
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
    """Логические серво-команды позы «рука вниз» (из DEFAULT_POSE)."""
    return {ch: int(DEFAULT_POSE[ch]) for ch in _channels(left).values()}


def start_pose() -> Dict[int, int]:
    """Стартовая поза всех каналов (стол/обычная)."""
    return dict(TABLE_START_POSE if TABLE_ENABLED else DEFAULT_POSE)


def servo_ranges(left: bool = False) -> Dict[int, Tuple[int, int]]:
    return {ch: tuple(DEFAULT_COMMAND_LIMITS[ch])
            for ch in _channels(left).values()}


def limits(left: bool = False, ik_only: bool = False) -> Dict[str, Tuple[float, float]]:
    """Диапазоны theta (sz, sx, eb) для одной руки.

    Вычисляются из командных лимитов и правила маппинга theta↔команды:
    right sz→cmd=rest+θ, left sz→cmd=rest−θ, elbow→cmd=rest−θ, sx→cmd=rest+θ.
    Для идеальной модели ik_only == полный диапазон (эмпирического сужения,
    как в прежней двухзонной реализации, больше нет).
    """
    rest = rest_angles(left)
    ranges = servo_ranges(left)
    result: Dict[str, Tuple[float, float]] = {}
    for name, ch in _channels(left).items():
        lo, hi = ranges[ch]
        if name == "elbow_x" or (name == "shoulder_z" and left):
            result[name] = (float(rest[ch] - hi), float(rest[ch] - lo))
        else:
            result[name] = (float(lo - rest[ch]), float(hi - rest[ch]))
    return result


def base(left: bool = False) -> np.ndarray:
    return np.array([-BASE_X if left else BASE_X, 0.0, 0.0])


def theta_from_commands(commands: Dict[int, float], left: bool = False) -> Tuple[float, float, float]:
    """Логические команды -> геометрический theta (sz, sx, eb) в градусах."""
    rest = rest_angles(left)
    ch = _channels(left)
    theta = []
    for name in JOINT_NAMES:
        cmd = commands.get(ch[name], rest[ch[name]])
        if name == "elbow_x" or (name == "shoulder_z" and left):
            theta.append(float(rest[ch[name]] - cmd))
        else:
            theta.append(float(cmd - rest[ch[name]]))
    return tuple(theta)


def to_servo_commands(theta: Sequence[float], left: bool = False) -> Dict[int, int]:
    """Геометрический theta (sz, sx, eb) -> логические команды серво (int)."""
    rest = rest_angles(left)
    ranges = servo_ranges(left)
    result = {}
    for name, value in zip(JOINT_NAMES, theta):
        ch = _channels(left)[name]
        lo, hi = ranges[ch]
        if name == "elbow_x" or (name == "shoulder_z" and left):
            command = rest[ch] - float(value)
        else:
            command = rest[ch] + float(value)
        result[ch] = int(round(max(lo, min(hi, command))))
    return result


def fk(theta: Sequence[float], left: bool = False) -> Dict[str, np.ndarray]:
    """Forward kinematics идеальной серийной цепи X–Z–X.

    theta геометрический = (sz, sx, eb). Цепь:
        R = Rx(sx)·Rz(sz)·Rx(eb),
        E   = base + L1 · R|шарнир sz.. ·(0,-1,0),
        EE  = E   + L2 · R · (0,-1,0).
    Закрытая форма (s1=sin sz и т.д.; для левой — зеркало по X):
        upper = (s1, -cs·cz, -ss·cz)
        lower = (s1·ce, -cs·cz·ce + ss·se, -ss·cz·ce - cs·se)
    Возвращает {"S": плечо, "E": локоть, "EE": кисть} в мм (система камеры).
    """
    t_sz, t_sx, t_eb = (float(v) for v in theta)
    sz, sx, eb = (math.radians(v) for v in (t_sz, t_sx, t_eb))
    s1, c1 = math.sin(sz), math.cos(sz)
    s2, c2 = math.sin(sx), math.cos(sx)
    s3, c3 = math.sin(eb), math.cos(eb)
    if left:
        s1 = -s1

    upper = np.array([s1, -c2 * c1, -s2 * c1])
    lower = np.array([s1 * c3, -c2 * c1 * c3 + s2 * s3, -s2 * c1 * c3 - c2 * s3])
    shoulder = base(left)
    elbow = shoulder + L1 * upper
    ee = elbow + L2 * lower
    return {"S": shoulder, "E": elbow, "EE": ee}


def _fk_grid_positions(t_sz: Sequence[float], t_sx: Sequence[float],
                       t_eb: Sequence[float], left: bool
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Векторизованная FK по декартовой сетке углов.

    Углы — градусы; сетка полная (каждая комбинация всех трёх осей).
    Возвращает (EE, локоть) — позиции (…,3) кисти и локтя (для коллайдера).
    """
    T1, T2, T3 = np.meshgrid(np.asarray(t_sz), np.asarray(t_sx),
                             np.asarray(t_eb), indexing="ij")
    c1 = np.cos(np.radians(T1))
    s1 = np.sin(np.radians(T1))
    c2 = np.cos(np.radians(T2))
    s2 = np.sin(np.radians(T2))
    c3 = np.cos(np.radians(T3))
    s3 = np.sin(np.radians(T3))
    if left:
        s1 = -s1

    bx = -BASE_X if left else BASE_X
    upper = np.stack([s1, -c2 * c1, -s2 * c1], axis=-1)
    lower = np.stack([s1 * c3, -c2 * c1 * c3 + s2 * s3,
                      -s2 * c1 * c3 - c2 * s3], axis=-1)
    shoulder = np.array([bx, 0.0, 0.0])
    elbow = shoulder + L1 * upper
    ee = elbow + L2 * lower
    return ee, elbow


def _collides(theta: Sequence[float], left: bool) -> bool:
    """Попала ли какая-либо часть руки ниже плоскости стола."""
    if not TABLE_ENABLED:
        return False
    f = fk(theta, left)
    return bool(f["E"][1] < TABLE_MIN_Y or f["EE"][1] < TABLE_MIN_Y)


def _crane_lift_bias(elbow_y, ee_y):
    """Добавочная стоимость за «низкий» локоть (кран-предпочтение).

    Работает с массивами (сетки) и скалярами; 0, когда TABLE_ENABLED=False
    или лифт elbow_y − ee_y уже достиг целевого TABLE_CRANE_LIFT.
    """
    if not TABLE_ENABLED:
        return np.zeros_like(np.asarray(elbow_y, dtype=float))
    return np.minimum(TABLE_CRANE_BIAS_MAX, TABLE_CRANE_WEIGHT * np.maximum(
        0.0, TABLE_CRANE_LIFT - (np.asarray(elbow_y, dtype=float)
                                 - np.asarray(ee_y, dtype=float))))


def _ranges(left: bool, step: float, window: Optional[float],
            center: Optional[Sequence[float]] = None
            ) -> Tuple[np.ndarray, ...]:
    lim = limits(left, ik_only=True)
    out = []
    for name, current in zip(JOINT_NAMES, center or (None,) * 3):
        lo, hi = lim[name]
        if window is not None and current is not None:
            lo, hi = max(lo, current - window), min(hi, current + window)
        grid = np.arange(lo, hi + step * 0.5, step)
        if grid.size == 0:
            grid = np.array([lo], dtype=float)
        out.append(grid)
    return tuple(out)


def _best_on_grid(ranges: Tuple[np.ndarray, ...], target: np.ndarray,
                  left: bool, prefer_crane: bool = False
                  ) -> Tuple[Tuple[float, float, float], float]:
    positions, elbow = _fk_grid_positions(*ranges, left)
    error = np.linalg.norm(positions - target, axis=-1)
    if TABLE_ENABLED:
        below = (elbow[..., 1] < TABLE_MIN_Y) | (positions[..., 1] < TABLE_MIN_Y)
        error = error + np.where(below, TABLE_COLLISION_PENALTY, 0.0)
        if prefer_crane:
            error = error + _crane_lift_bias(elbow[..., 1], positions[..., 1])
    index = int(np.argmin(error))
    i1, i2, i3 = np.unravel_index(index, error.shape)
    theta = (float(ranges[0][i1]), float(ranges[1][i2]), float(ranges[2][i3]))
    return theta, float(error[i1, i2, i3])


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
    """Inverse kinematics идеальной цепи X–Z–X (сетка + уточнение).

    Мультистарт по грубой решётке (8°) -> до count бассейнов, каждый
    уточняется каскадом окон (2°/0.4°/0.08°). Выбор ветки — по ошибке
    |ee−цель|, при равенстве — по близости к start (непрерывность движения).

    Возвращает dict вида:
        {theta, status, message, err_mm, left, servo, ee, ok,
         wanted, clamped, reach_gap}
    со всеми значениями в нативных типах Python (JSON-совместимо).
    status: "ok" (err<=IK_ERR_OK), "limits" (err<=IK_TOLERANCE_MM),
            "unreachable" (дальше лимита; servo всё равно указывает на
            ближайшую достижимую позу).
    """
    z = -z
    wanted = np.array([float(x), float(y), float(z)], dtype=float)
    prefer_crane = TABLE_ENABLED
    coarse = _ranges(left, *GRID_STEPS[0])
    positions, elbow = _fk_grid_positions(*coarse, left)
    errors = np.linalg.norm(positions - wanted, axis=-1)
    if TABLE_ENABLED:
        below = (elbow[..., 1] < TABLE_MIN_Y) | (positions[..., 1] < TABLE_MIN_Y)
        errors = errors + np.where(below, TABLE_COLLISION_PENALTY, 0.0)
    # Стартовые бассейны сеются по ДВУМ ценам: чистой (коллайдер, без крана)
    # и кран-предпочтительной. Это гарантирует, что в старты попадут и
    # узкие натуральные бассейны, и кран-бассейны; ветку выбирает финальный
    # score ниже (кран-байас работает в каскаде и при выборе).
    if prefer_crane:
        errors_crane = errors + _crane_lift_bias(elbow[..., 1], positions[..., 1])
    else:
        errors_crane = errors

    count = min(8, errors.size)
    starts = []
    for scored in (errors_crane, errors):
        for index in np.argpartition(scored.ravel(), count - 1)[:count]:
            i1, i2, i3 = np.unravel_index(index, scored.shape)
            theta = (float(coarse[0][i1]), float(coarse[1][i2]),
                     float(coarse[2][i3]))
            if all(max(abs(a - b) for a, b in zip(theta, picked)) > 10.0
                   for picked in starts):
                starts.append(theta)

    if start is not None:
        model = limits(left, ik_only=True)
        clamped_start = tuple(min(max(float(v), model[n][0]), model[n][1])
                              for n, v in zip(JOINT_NAMES, start))
        starts.append(clamped_start)

    results = []
    for theta in starts:
        current = theta
        for step, window in GRID_STEPS[1:]:
            current, _ = _best_on_grid(_ranges(left, step, window, current),
                                       wanted, left, prefer_crane=prefer_crane)
        ee = fk(current, left)["EE"]
        error = float(np.linalg.norm(ee - wanted))
        results.append((error, current))

    def score(error, theta):
        total = error
        if prefer_crane:
            f = fk(theta, left)
            total += float(_crane_lift_bias(f["E"][1], f["EE"][1]))
        if start is not None:
            weights = (1.0, 0.5, 0.25)
            angle_cost = sum(w * (a - b) ** 2
                             for w, a, b in zip(weights, theta, start))
            total += min(0.2 * angle_cost, 100.0)
        return total

    best_error, best_theta = min(results, key=lambda item: score(*item))

    if best_error <= IK_ERR_OK:
        status = "ok"
    elif best_error <= IK_TOLERANCE_MM:
        status = "limits"
    else:
        status = "unreachable"
    message = f"FK-поиск: |ee−цель|={best_error:.1f} мм"
    if TABLE_ENABLED:
        if wanted[1] < TABLE_MIN_Y:
            message = ("Цель ниже стола — кисть зажата к его поверхности "
                       f"(|ee−цель|={best_error:.1f} мм)")
        elif _collides(best_theta, left):
            message += " · рука у самой кромки стола"
    return _result(best_theta, status, message, left, best_error,
                   wanted=[float(v) for v in wanted],
                   clamped=[float(v) for v in wanted])


def browser_config() -> dict:
    """Конфиг для браузера: только базовые параметры для ориентации,
    никакой локальной FK-кинематики в браузере больше нет."""
    return {
        "base_x": BASE_X,
        "l1": L1,
        "l2": L2,
        "channels": {
            side: {name: ch for name, ch in channels.items()}
            for side, channels in ARM_CHANNELS.items()
        },
        "rest": {str(ch): value for left in (False, True)
                 for ch, value in rest_angles(left).items()},
        "torso": {"half_width": TORSO_HW, "half_height": TORSO_HH,
                  "half_depth": TORSO_HD},
        "head": {"y": float(HEAD[1]), "radius": HEAD_R},
        "start": {str(ch): value for ch, value in start_pose().items()},
    }


if __name__ == "__main__":
    # Короткий самотест: round-trip команды<->theta, зеркало, пара FK/IK.
    print(f"rest right: {rest_angles(False)}  left: {rest_angles(True)}")
    print("limits right:", limits(False), " left:", limits(True))
    for cmds, left in (((169, 66, 137), False), ((178, 226, 58), False),
                       ((60, 100, 150), True)):
        c1, c4, c6 = cmds
        if left:
            commands = {1: 60, 2: c4, 4: 60, 5: c4, 6: c6, 7: c6}
        else:
            commands = {1: c1, 4: c4, 6: c6}
        th = theta_from_commands(commands, left)
        back = to_servo_commands(th, left)
        ch = _channels(left)
        rt_ok = all(abs(back[ch[n]] - commands[ch[n]]) <= 2
                    for n in JOINT_NAMES)
        ee = fk(th, left)["EE"]
        print(f"cmds={cmds} left={int(left)} rr_ok={rt_ok} th=({th[0]:.1f},{th[1]:.1f},{th[2]:.1f}) EE=({ee[0]:.0f},{ee[1]:.0f},{ee[2]:.0f})")
    right = theta_from_commands({1: 135, 4: 45, 6: 180}, False)
    m = fk(right, True)["EE"]; r = fk(right, False)["EE"]
    print(f"mirror: right EE x={r[0]:.1f} vs left=-right? {abs(m[0] + r[0]) < 1e-6 and abs(m[1]-r[1])<1e-6 and abs(m[2]-r[2])<1e-6}")
    for x, y, z in ((0, -300, 400), (120, 290, 0), (250, 0, 400), (0, -50, 400)):
        res = ik_solve(x, y, z)
        print(f"ik_solve({x},{y},{z}) -> {res['status']:9} err={res['err_mm']:5.1f} th={tuple(round(v,2) for v in res['theta'])} servo={res['servo']}")

    print("table collider checks:")
    for x, y, z, left in ((250, -600, 400, False), (-250, -650, 300, True),
                          (200, -300, 450, False), (0, -400, 500, False)):
        res = ik_solve(x, y, z, left=left)
        f = fk(res["theta"], left)
        ee_y, el_y = f["EE"][1], f["E"][1]
        lift = el_y - ee_y
        clean = ee_y >= TABLE_MIN_Y - 1e-9 and el_y >= TABLE_MIN_Y - 1e-9
        ch4 = res["servo"][_channels(left)["shoulder_z"]]
        print(f"  ({x},{y},{z}) l={int(left)} -> ch4={ch4:3} lift={lift:6.1f} "
              f"ee_y={ee_y:6.1f} el_y={el_y:6.1f} min={TABLE_MIN_Y:6.1f} "
              f"clean={clean} craned={lift >= TABLE_CRANE_LIFT} "
              f"st={res['status']:9} {res['message']}")