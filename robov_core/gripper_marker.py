"""ArUco-маркер на кисти: надёжная детекция и точка захвата.

Геометрия (замер владельца, фото левого захвата): маркер 50x50 мм, dict 4x4,
ID 0 — правая рука, ID 1 — левая. Он наклеен на задне-верхнюю площадку кисти,
поэтому в системе кисти (X — сторона неподвижного пальца, Y — вверх, Z — подход):

    M = G + 10 мм · Y − 100 мм · Z          (M — центр маркера, G — точка захвата)

Ориентация: X_m = X кисти, Y_m = −Z кисти, нормаль Z_m = +Y кисти (вверх).
Отсюда точка захвата в СИСТЕМЕ МАРКЕРА:

    G = M − 100 мм · Y_m − 10 мм · Z_m      →  (0, −100, −10) мм

Детекция «сильная»: несколько наборов параметров, поиск на 1x/1.5x/2x (мелкий
маркер на дистанции), субпиксельная доводка углов, IPPE-поза + LM-уточнение,
проверка по ошибке репроекции и площади. Берётся детекция с наибольшей площадью.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

MARKER_SIZE_M = 0.050
MARKER_DICT = "DICT_4X4_50"
MARKER_ID = {False: 0, True: 1}          # правая 0, левая 1
# Точка захвата в системе МАРКЕРА, мм (100 назад по Y, 10 ниже по Z).
GRASP_FROM_MARKER_MM = np.array([0.0, -100.0, -10.0])
# Порог ошибки репроекции позы, px: хуже — считаем детекцию ненадёжной.
MAX_REPROJ_PX = 2.0
MIN_MARKER_PX = 16                       # меньше — на таком размере поза мусор

_DICT = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT))
_DETECTORS: list = []


def _params(kind: str) -> cv2.aruco.DetectorParameters:
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 50
    p.cornerRefinementMinAccuracy = 0.01
    if kind == "fine":            # мелкие маркеры: мягкие пороги
        p.adaptiveThreshWinSizeMin = 3
        p.adaptiveThreshWinSizeMax = 23
        p.adaptiveThreshWinSizeStep = 10
        p.minMarkerPerimeterRate = 0.01
        p.maxMarkerPerimeterRate = 4.0
        p.polygonalApproxAccuracyRate = 0.05
        p.minCornerDistanceRate = 0.01
        p.minDistanceToBorder = 1
        p.perspectiveRemovePixelPerCell = 2
        p.maxErroneousBitsInBorderRate = 0.5
    else:                          # "wide": крупный контраст, другие окна порога
        p.adaptiveThreshWinSizeMin = 7
        p.adaptiveThreshWinSizeMax = 41
        p.adaptiveThreshWinSizeStep = 4
        p.minMarkerPerimeterRate = 0.02
        p.polygonalApproxAccuracyRate = 0.03
        p.minCornerDistanceRate = 0.02
        p.minDistanceToBorder = 1
    return p


def _detectors() -> list:
    global _DETECTORS
    if not _DETECTORS:
        _DETECTORS = [cv2.aruco.ArucoDetector(_DICT, _params("fine")),
                      cv2.aruco.ArucoDetector(_DICT, _params("wide"))]
    return _DETECTORS


def _solve(corners_px: np.ndarray, P: np.ndarray):
    half = MARKER_SIZE_M / 2.0
    obj = np.array([[-half, half, 0.0], [half, half, 0.0],
                    [half, -half, 0.0], [-half, -half, 0.0]], np.float32)
    ok, rvec, tvec = cv2.solvePnP(obj, corners_px.astype(np.float32), P, None,
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    try:                                  # LM-уточнение позы
        rvec, tvec = cv2.solvePnPRefineLM(
            obj, corners_px.astype(np.float32), P, None, rvec, tvec)
    except Exception:                     # noqa: BLE001 — старые OpenCV
        pass
    proj, _ = cv2.projectPoints(obj, rvec, tvec, P, None)
    err = float(np.sqrt(np.mean((proj.reshape(-1, 2) - corners_px) ** 2)))
    return rvec.reshape(3), tvec.reshape(3), err


def detect(frame: np.ndarray, K: np.ndarray, D: np.ndarray,
           Rrect: np.ndarray, P: np.ndarray,
           ids: Optional[Tuple[int, ...]] = None,
           roi: Optional[Tuple[int, int, int, int]] = None) -> Dict[int, dict]:
    """Найти маркеры кисти. {id: {center, rvec, tvec, corners, reproj}}.

    ids — какие ID вообще интересуют (по умолчанию оба, 0 и 1).
    Кадр может быть цветным или серым; углы переводятся в выпрямленную
    систему своего глаза (undistortPoints с Rrect/P), поза решается по P.
    roi — (x0, y0, x1, y1) в пикселях кадра: искать только в этом окне (в
    ~50 раз быстрее для слежения за маркером при спуске).
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    want = set(MARKER_ID.values()) if ids is None else set(ids)
    best: Dict[int, dict] = {}
    ox, oy = 0, 0
    if roi is not None:
        x0, y0, x1, y1 = [int(v) for v in roi]
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(gray.shape[1], x1), min(gray.shape[0], y1)
        if x1 - x0 < 24 or y1 - y0 < 24:
            return best
        ox, oy = x0, y0
        gray = gray[y0:y1, x0:x1]
    # масштабы: 1x, 1.5x (если кадр большой), 2x — для далёкого/мелкого маркера
    scales = [1.0, 2.0] if roi is not None else (
        [1.0, 1.5, 2.0] if gray.size <= 1280 * 720 * 1.1 else [1.0])
    for scale in scales:
        img = gray if scale == 1.0 else cv2.resize(
            gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        for detector in _detectors():
            corners, marker_ids, _ = detector.detectMarkers(img)
            if marker_ids is None:
                continue
            for quad, marker_id in zip(corners, marker_ids.ravel()):
                marker_id = int(marker_id)
                if marker_id not in want:
                    continue
                pts = quad.reshape(-1, 2).astype(np.float64) / scale
                pts[:, 0] += ox
                pts[:, 1] += oy
                if _area(pts) < (MIN_MARKER_PX * scale) ** 2:
                    continue
                q = cv2.fisheye.undistortPoints(
                    pts.reshape(-1, 1, 2), K, D, R=Rrect, P=P).reshape(-1, 2)
                solved = _solve(q, P)
                if solved is None:
                    continue
                rvec, tvec, err = solved
                if err > MAX_REPROJ_PX:
                    continue
                area = _area(q)
                prev = best.get(marker_id)
                if prev is not None and prev["area"] >= area:
                    continue
                best[marker_id] = {"center": q.mean(axis=0), "rvec": rvec,
                                   "tvec": tvec, "corners": q,
                                   "reproj": err, "area": area}
    return best


def _area(pts: np.ndarray) -> float:
    return float(abs(cv2.contourArea(pts.reshape(-1, 1, 2).astype(np.float32))))


def grasp_from_marker(marker: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(точка захвата, м, система камеры с Y ВВЕРХ; центр маркера, м, так же)."""
    R, _ = cv2.Rodrigues(np.asarray(marker["rvec"], float))
    t = np.asarray(marker["tvec"], float).reshape(3)
    g = t + R @ (GRASP_FROM_MARKER_MM / 1000.0)
    return np.array([g[0], -g[1], g[2]]), np.array([t[0], -t[1], t[2]])
