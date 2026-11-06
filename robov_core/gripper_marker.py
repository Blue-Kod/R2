"""ArUco-маркер на кисти: детекция и точка захвата.

Геометрия (замер владельца, см. фото): маркер 50x50 мм, dict 4x4, ID 0 —
правая рука, ID 1 — левая. Он наклеен на задне-верхнюю площадку кисти, поэтому
в системе кисти (X — сторона неподвижного пальца, Y — вверх, Z — подход):

    M = G + 10 мм · Y − 100 мм · Z          (M — центр маркера, G — точка захвата)

Ориентация маркера: X_m = X кисти, Y_m = −Z кисти (вниз по картинке), нормаль
Z_m = +Y кисти (смотрит вверх). Отсюда точка захвата в СИСТЕМЕ МАРКЕРА:

    G = M − 100 мм · Y_m − 10 мм · Z_m      →  (0, −100, −10) мм

Этого вектора достаточно: `G = t + R @ GRASP_FROM_MARKER`. Никаких других
калибровок маркера не нужно — поворот R приходит из solvePnP.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

MARKER_SIZE_M = 0.050
MARKER_DICT = "DICT_4X4_50"
MARKER_ID = {False: 0, True: 1}          # правая 0, левая 1
# Точка захвата в системе МАРКЕРА, мм.
GRASP_FROM_MARKER_MM = np.array([0.0, -100.0, -10.0])

_DETECTOR: Optional[cv2.aruco.ArucoDetector] = None


def _detector() -> cv2.aruco.ArucoDetector:
    global _DETECTOR
    if _DETECTOR is None:
        params = cv2.aruco.DetectorParameters()
        # те же настройки, что вытащили мелкие маркеры доски (см. charuco_calib)
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 23
        params.adaptiveThreshWinSizeStep = 10
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 4.0
        params.polygonalApproxAccuracyRate = 0.05
        params.minCornerDistanceRate = 0.01
        params.minDistanceToBorder = 1
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.cornerRefinementWinSize = 5
        _DETECTOR = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, MARKER_DICT)),
            params)
    return _DETECTOR


def detect(frame: np.ndarray, K: np.ndarray, D: np.ndarray,
           Rrect: np.ndarray, P: np.ndarray) -> Dict[int, dict]:
    """Найти маркеры кисти: {id: {center, rvec, tvec, corners}}.

    Углы переводятся в ВЫПРЯМЛЕННУЮ систему глаза (undistortPoints с Rrect/P),
    поза решается по P (пинхол) — так же, как считаются все остальные точки.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector().detectMarkers(gray)
    out: Dict[int, dict] = {}
    if ids is None:
        return out
    half = MARKER_SIZE_M / 2.0
    # порядок углов ArUco: TL, TR, BR, BL; в системе маркера Y — вверх
    obj = np.array([[-half, half, 0.0], [half, half, 0.0],
                    [half, -half, 0.0], [-half, -half, 0.0]], np.float32)
    for marker_corners, marker_id in zip(corners, ids.ravel()):
        pts = marker_corners.reshape(-1, 2).astype(np.float64)
        q = cv2.fisheye.undistortPoints(pts.reshape(-1, 1, 2), K, D,
                                        R=Rrect, P=P).reshape(-1, 2)
        ok, rvec, tvec = cv2.solvePnP(obj, q.astype(np.float32), P, None,
                                      flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            continue
        out[int(marker_id)] = {
            "center": q.mean(axis=0),
            "rvec": rvec.reshape(3),
            "tvec": tvec.reshape(3),
            "corners": q,
        }
    return out


def grasp_from_marker(marker: dict) -> Tuple[np.ndarray, np.ndarray]:
    """(точка захвата в системе камеры, м; центр маркера в системе камеры, м)."""
    R, _ = cv2.Rodrigues(np.asarray(marker["rvec"], float))
    t = np.asarray(marker["tvec"], float).reshape(3)
    return t + R @ (GRASP_FROM_MARKER_MM / 1000.0), t
