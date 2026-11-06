"""Стерео-глубина R2 по калибровке ChArUco (dev/charuco_calib.py).

Глубина считается по выпрямленной паре:

    Z = K / (d + c)

где d — диспаратность SGBM в полном разрешении глаза (1280x720), K и c — из
калибровки. c нужен потому, что доска, снятая в центре кадра, не разделяет
поворот камер друг относительно друга и их базу: лишний поворот даёт почти
постоянный сдвиг диспаратности, и без поправки глубина врёт в 1.5-2 раза.
Поправка снимается по известным дистанциям (`charuco_calib.py --depth-fit`)
и лежит в calib.json полями depth_K, depth_c, depth_f_ref.

Дыры (гладкие поверхности без текстуры) заполняются значением ближайшего
валидного пикселя, поэтому стены и стол выходят сплошными.

Считается только по запросу (high_level.depth / coords) — не в потоке.
Датчиков нет: вход — только изображение, ничего больше не нужно.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np


class Vector3(NamedTuple):
    """Точка в системе координат камеры: X вправо, Y вверх, Z вперёд, мм.

    Та же система и единицы, что у `ik_detail` / `move_ik_detail`: результат
    `coords()` можно сразу отдавать в IK.
    """
    x: float
    y: float
    z: float


# Полное разрешение: диспаратность вдвое больше, ошибка глубины вдвое меньше.
# Цена — ~100 мс на запрос (считаем только по запросу, не в потоке).
# numDisparities=256 при базе 60 мм даёт ближний предел ~13 см.
DS = 1.0
ROI_HALF = 20     # полуразмер квадрата в центре кадра для чтения дистанции

# Быстрый путь: считаем SGBM только в полосе строк вокруг запрошенной точки.
# Параметры SGBM при этом НЕ меняются: сужение numDisparities до 128 проверено —
# на периодической текстуре (одеяло) даёт ошибки до 10 px, то есть врёт; полоса
# строк с теми же параметрами даёт те же значения (Δd медиана 0.125 px).
# По ширине кропать нельзя: SGBM при ненулевом x0 теряет соответствие
# (проверено: отдаёт 64 px вместо 34).
FAST_BAND = 160                      # полувысота полосы строк вокруг точки

SGBM_PARAMS = dict(
    minDisparity=0, numDisparities=256, blockSize=5,
    P1=8 * 3 * 25, P2=32 * 3 * 25, disp12MaxDiff=2, uniquenessRatio=5,
    speckleWindowSize=100, speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
)


def split_stereo(raw: np.ndarray):
    """Кадр -> (левый, правый). Если кадр одиночный (w < 1.8h) — правый = None."""
    w, h = raw.shape[1], raw.shape[0]
    if w < 1.8 * h:
        return raw, None
    half = w // 2
    return raw[:, :half], raw[:, half:half * 2]


def fit_depth(samples):
    """Z = K/(d + c) по точкам (Z, d): линейно по (K, c), т.к. Z*d = K - c*Z."""
    if len(samples) < 2:
        print("нужно минимум 2 точки")
        return None, None
    A = np.array([[1.0, -z] for z, _ in samples])
    b = np.array([z * dd for z, dd in samples])
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    K, c = float(sol[0]), float(sol[1])
    if not np.isfinite(K) or K <= 0:
        print("вырожденная привязка (точки на одной дистанции)")
        return None, None
    err = [K / (dd + c) - z for z, dd in samples]
    print(f"  привязка: K {K:.1f} м·px, сдвиг {c:+.1f} px")
    for (z, dd), e in zip(samples, err):
        print(f"    {100*z:.0f} см -> {100*K/(dd+c):.1f} см (ошибка {100*e:+.1f} см)")
    if max(abs(e) for e in err) > 0.05:
        print("  ВНИМАНИЕ: точки не ложатся на модель — перемеряйте, "
              "сдвиг диспаратности не постоянный")
    return K, c


class StereoDepth:
    """Ректификация + SGBM + перевод диспаратности в метры."""

    def __init__(self, calib=None):
        if calib is None:                      # по умолчанию — корень проекта
            calib = Path(__file__).resolve().parent.parent / "calib.json"
        self.path = None if isinstance(calib, dict) else Path(calib)
        if self.path is not None and not self.path.exists():
            raise FileNotFoundError(
                f"калибровка не найдена: {self.path}. Снимите доску "
                f"(dev/charuco_calib.py --shoot) — calib.json кладётся в корень "
                f"проекта и коммитится; на роботе должен быть после git pull")
        d = (calib if isinstance(calib, dict)
             else json.loads(Path(calib).read_text(encoding="utf-8")))
        if d.get("mode") != "stereo":
            raise ValueError("нужна стерео-калибровка (mode=stereo)")
        self.info = d
        self.K = (np.array(d["K_left"], float), np.array(d["K_right"], float))
        self.D = (np.array(d["D_left"], float).reshape(-1, 1),
                  np.array(d["D_right"], float).reshape(-1, 1))
        self.R = np.array(d["R"], float)
        self.T = np.array(d["T_m"], float).reshape(3, 1)
        self.baseline = float(d.get("baseline_m") or np.linalg.norm(self.T))
        self.K_emp = d.get("depth_K")
        self.c_emp = float(d.get("depth_c") or 0.0)
        self.f_ref = float(d.get("depth_f_ref") or 0.0)
        self.sgbm = cv2.StereoSGBM_create(**SGBM_PARAMS)
        self._maps = None
        self._size = None
        self.f = 0.0
        size = d.get("image_size")      # готовим ректификацию заранее: нужен f
        if size:
            try:
                self._prepare((int(size[0]), int(size[1])))
            except cv2.error:
                pass

    # -- ректификация --------------------------------------------------------
    def _prepare(self, size) -> None:
        h, w = size[1], size[0]
        # balance=1: выпрямленный кадр покрывает исходный (у fisheye края иначе
        # оказываются за кадром и глубина там теряется). f падает 578 -> 478,
        # но это компенсируется привязкой (она нормируется на f_ref).
        R1r, R2r, P1r, P2r, _ = cv2.fisheye.stereoRectify(
            self.K[0], self.D[0], self.K[1], self.D[1], (w, h), self.R, self.T,
            flags=cv2.CALIB_ZERO_DISPARITY, balance=1.0)
        m1 = cv2.fisheye.initUndistortRectifyMap(self.K[0], self.D[0], R1r, P1r,
                                                 (w, h), cv2.CV_32FC1)
        m2 = cv2.fisheye.initUndistortRectifyMap(self.K[1], self.D[1], R2r, P2r,
                                                 (w, h), cv2.CV_32FC1)
        self._maps, self._size = (m1, m2), (w, h)
        self._Rrect = (np.asarray(R1r)[:3, :3], np.asarray(R2r)[:3, :3])
        self._P = (np.asarray(P1r)[:3, :3], np.asarray(P2r)[:3, :3])
        self.f = float(P1r[0, 0]) * DS

    @staticmethod
    def _gray(img: np.ndarray) -> np.ndarray:
        """Кадр в градациях серого: SGBM цвета не нужен (и декод серого в 8x быстрее)."""
        return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    @property
    def size(self) -> tuple:
        """Размер одного глаза, в котором считается глубина, (w, h)."""
        return self._size

    def rectify(self, left, right):
        h, w = left.shape[:2]
        if self._maps is None or self._size != (w, h):
            self._prepare((w, h))
        r1 = cv2.remap(left, self._maps[0][0], self._maps[0][1], cv2.INTER_LINEAR)
        r2 = cv2.remap(right, self._maps[1][0], self._maps[1][1], cv2.INTER_LINEAR)
        return r1, r2

    def rectify_crop(self, left, right, x0, y0, x1, y1):
        """Ректификация только окна (x0:y1, y0:y1), без remap всего кадра."""
        h, w = left.shape[:2]
        if self._maps is None or self._size != (w, h):
            self._prepare((w, h))
        m1 = (np.ascontiguousarray(self._maps[0][0][y0:y1, x0:x1]),
              np.ascontiguousarray(self._maps[0][1][y0:y1, x0:x1]))
        m2 = (np.ascontiguousarray(self._maps[1][0][y0:y1, x0:x1]),
              np.ascontiguousarray(self._maps[1][1][y0:y1, x0:x1]))
        return (cv2.remap(left, m1[0], m1[1], cv2.INTER_LINEAR),
                cv2.remap(right, m2[0], m2[1], cv2.INTER_LINEAR))

    def disp_point(self, left, right, x, y, eye="left", window=2):
        """Диспаратность в одной точке (быстро): полоса строк + узкий диапазон."""
        u, v = self._pixel(x, y, eye)
        if not (np.isfinite(u) and np.isfinite(v)):
            return float("nan")
        h, w = left.shape[:2]
        y0 = int(max(0, min(v - FAST_BAND, h - 2 * FAST_BAND)))
        y1 = int(min(h, y0 + 2 * FAST_BAND))
        r1, r2 = self.rectify_crop(left, right, 0, y0, w, y1)
        d = self.sgbm.compute(self._gray(r1), self._gray(r2))
        d = d.astype(np.float32) / 16.0
        tx, ty = int(round(u)), int(round(v)) - y0
        value = self._median_at(d, tx, ty, window)
        if eye != "left" and np.isfinite(value):
            # карта индексируется левой сеткой: для правого глаза берём пиксель,
            # отстоящий на его собственную диспаратность
            shifted = self._median_at(d, tx + value / DS, ty, window)
            if np.isfinite(shifted):
                value = shifted
        return value

    @staticmethod
    def _median_at(d: np.ndarray, tx: float, ty: int, window: int) -> float:
        """Медиана валидных диспаратностей вокруг (tx, ty), окно расширяется.

        Так же закрывает дыры, как fill() на полной карте, но только вокруг
        точки: distanceTransform на весь кадр в запросе не нужен.
        """
        ah, aw = d.shape
        ix = int(round(tx))
        for half in (max(2, 2 * window), 10, 20, max(ah, aw)):
            p = d[max(0, ty - half):ty + half + 1, max(0, ix - half):ix + half + 1]
            p = p[p > 0.5]
            if p.size:
                return float(np.median(p))
        return float("nan")

    def depth_point(self, left, right, x, y, eye="left", window=2) -> float:
        """Глубина в точке кадра, м. Считается локально — в разы быстрее карты."""
        d = self.disp_point(left, right, x, y, eye, window)
        if not np.isfinite(d):
            return float("nan")
        dref = d * (self.f_ref / self.f if self.f_ref > 0 else 1.0)
        if self.K_emp is not None:
            return float(float(self.K_emp) / max(dref + self.c_emp, 1e-3))
        return float(self.f * self.baseline / max(dref, 1e-3))

    def point_at(self, left, right, x, y, eye="left", window=2) -> Vector3:
        """Точка пикселя в системе ЛЕВОЙ камеры (мм), по локальному замеру."""
        z = self.depth_point(left, right, x, y, eye, window)
        if not np.isfinite(z):
            return Vector3(float("nan"), float("nan"), float("nan"))
        u, v = self._pixel(x, y, eye)
        i = 0 if eye == "left" else 1
        P = self._P[i]
        p = np.array([(u - P[0, 2]) / P[0, 0] * z,
                      (v - P[1, 2]) / P[1, 1] * z, z])
        p = self._Rrect[i].T @ p
        if eye != "left":
            p = self.R.T @ (p - self.T.ravel())
        return Vector3(float(p[0] * 1000), float(-p[1] * 1000), float(p[2] * 1000))

    # -- диспаратность и глубина --------------------------------------------
    def disparity(self, left, right) -> np.ndarray:
        """Диспаратность в половинном разрешении (SGBM, 16x уже снят)."""
        r1, r2 = self.rectify(left, right)
        h, w = r1.shape[:2]
        small = (int(w * DS), int(h * DS))
        s1 = cv2.resize(self._gray(r1), small, interpolation=cv2.INTER_AREA)
        s2 = cv2.resize(self._gray(r2), small, interpolation=cv2.INTER_AREA)
        return self.sgbm.compute(s1, s2).astype(np.float32) / 16.0

    @staticmethod
    def fill(disp: np.ndarray, smooth: int = 5) -> np.ndarray:
        """Дыры — значением ближайшего валидного пикселя, затем медиана.

        SGBM не находит сопоставление на гладком (стена, стол, белый объект).
        Ближайший сосед не выдумывает геометрию, а продлевает поверхность,
        поэтому стена получается сплошной; медиана убирает одиночный шум.
        """
        valid = disp > 0.5
        if not valid.any():
            return disp
        out = disp
        if (~valid).any():
            # метки — это номера компонент (1..N), а не индексы: строим таблицу
            # «метка -> плоский индекс валидного пикселя»
            _, labels = cv2.distanceTransformWithLabels(
                (~valid).astype(np.uint8), cv2.DIST_L2, 5,
                labelType=cv2.DIST_LABEL_PIXEL)
            lab = labels.ravel()
            flat = np.flatnonzero(valid.ravel())
            lut = np.zeros(int(lab.max()) + 1, np.int32)
            lut[lab[flat]] = flat          # у валидного пикселя метка своя
            out = disp.ravel()[lut[lab]].reshape(disp.shape)
        return cv2.medianBlur(out, smooth) if smooth > 0 else out

    def depth(self, left, right=None):
        """(Z в метрах, диспаратность). Z заполнена по соседям, дыр нет."""
        if right is None:
            left, right = split_stereo(left)
            if right is None:
                raise ValueError("нужен стереокадр (два глаза), пришёл одиночный")
        disp = self.disparity(left, right)
        if not (disp > 0.5).any():
            return np.full_like(disp, np.nan), disp
        dfill = self.fill(disp)
        dref = dfill * (self.f_ref / self.f if self.f_ref > 0 else 1.0)
        if self.K_emp is not None:
            z = float(self.K_emp) / np.maximum(dref + self.c_emp, 1e-3)
        else:
            z = self.f * self.baseline / np.maximum(dfill, 1e-3)
        return z.astype(np.float32), disp

    # -- чтение в центре кадра ----------------------------------------------
    @staticmethod
    def _center(a: np.ndarray, half: int = ROI_HALF) -> np.ndarray:
        h, w = a.shape[:2]
        cy, cx = h // 2, w // 2
        return a[max(0, cy - half):cy + half, max(0, cx - half):cx + half]

    def distance(self, z: np.ndarray, half: int = ROI_HALF) -> float:
        roi = self._center(z, half)
        roi = roi[np.isfinite(roi)]
        return float(np.median(roi)) if roi.size else float("nan")

    def roi_disparity(self, disp: np.ndarray, half: int = ROI_HALF) -> float:
        roi = self._center(disp, half)
        roi = roi[roi > 0.5]
        return float(np.median(roi)) if roi.size else float("nan")

    # -- пиксель -> дистанция -> точка --------------------------------------
    def _pixel(self, x, y, eye="left") -> tuple:
        """Пиксель глаза -> выпрямленный пиксель (дисторсия + ректификация)."""
        i = 0 if eye == "left" else 1
        q = cv2.fisheye.undistortPoints(
            np.array([[[float(x), float(y)]]], np.float64),
            self.K[i], self.D[i], R=self._Rrect[i], P=self._P[i])
        return float(q[0, 0, 0]), float(q[0, 0, 1])

    def _patch(self, z, u, v, window) -> float:
        if not (np.isfinite(u) and np.isfinite(v)):
            return float("nan")
        h, w = z.shape[:2]
        # прижимаем к карте: внешние ~5% fisheye-кадра в выпрямленный не влезают
        iu = min(max(int(round(u * DS)), 0), w - 1)
        iv = min(max(int(round(v * DS)), 0), h - 1)
        p = z[max(0, iv - window):iv + window + 1,
              max(0, iu - window):iu + window + 1]
        p = p[np.isfinite(p)]
        return float(np.median(p)) if p.size else float("nan")

    def disp_from_depth(self, z: float) -> float:
        """Диспаратность (половинное разрешение), соответствующая глубине z, м."""
        if self.K_emp is not None:
            d = float(self.K_emp) / z - self.c_emp
            return d * (self.f / self.f_ref) if self.f_ref > 0 else d
        return self.f * self.baseline / z

    def depth_at(self, z: np.ndarray, x: float, y: float,
                 eye: str = "left", window: int = 2) -> float:
        """Глубина в пикселе (x, y) глаза, м. NaN — если сопоставления нет.

        Карта глубины индексируется левой сеткой, поэтому для правого глаза
        точку сперва сдвигаем на её собственную диспаратность.
        """
        u, v = self._pixel(x, y, eye)
        d = self._patch(z, u, v, window)
        if eye != "left" and np.isfinite(d):
            shifted = self._patch(z, u + self.disp_from_depth(d) / DS, v, window)
            if np.isfinite(shifted):
                d = shifted
        return d

    def point(self, z: np.ndarray, x: float, y: float,
              eye: str = "left", window: int = 2) -> Vector3:
        """Точка пикселя в системе ЛЕВОЙ камеры: X вправо, Y вверх, Z вперёд, мм."""
        d = self.depth_at(z, x, y, eye, window)
        if not np.isfinite(d):
            return Vector3(float("nan"), float("nan"), float("nan"))
        u, v = self._pixel(x, y, eye)
        i = 0 if eye == "left" else 1
        P = self._P[i]
        p = np.array([(u - P[0, 2]) / P[0, 0] * d,
                      (v - P[1, 2]) / P[1, 1] * d, d])
        p = self._Rrect[i].T @ p                  # выпрямленный -> исходный кадр
        if eye != "left":
            p = self.R.T @ (p - self.T.ravel())   # правый глаз -> левый
        return Vector3(float(p[0] * 1000), float(-p[1] * 1000), float(p[2] * 1000))

    # -- привязка по известным дистанциям -----------------------------------
    def fit(self, samples, cm=None, save: bool = True):
        """samples: [(Z м, d px)]. Пишет поправку в свою калибровку."""
        K, c = fit_depth(samples)
        if K is None:
            return None, None
        self.K_emp, self.c_emp, self.f_ref = K, c, self.f
        if save and self.path is not None:
            d = json.loads(self.path.read_text(encoding="utf-8"))
            d["depth_K"], d["depth_c"], d["depth_f_ref"] = K, c, self.f
            d["depth_fit_cm"] = [float(x) for x in (cm or [])]
            d["depth_fit_samples"] = [[round(100.0 * z, 1), round(dd, 2)]
                                      for z, dd in samples]
            self.path.write_text(json.dumps(d, ensure_ascii=False, indent=1),
                                 encoding="utf-8")
            self.info = d
            print(f"  сохранено в {self.path}")
        return K, c
