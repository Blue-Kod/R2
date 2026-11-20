#!/usr/bin/env python3
"""Проверка Orbbec Astra без панели: цвет, глубина, интринсики.

Запуск: .venv/bin/python dev/test_orbbec_camera.py
Камера не обязательна: проверка карты глубины в JPEG идёт на синтетике.
"""
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robov_core.camera import (StereoCamera, DEPTH_STATUS_OK,  # noqa: E402
                               DEPTH_STATUS_NO_DATA, DEPTH_STATUS_UNKNOWN,
                               DEPTH_STATUS_UNKNOWN_NEAR)
from robov_core.orbbec_camera import (OrbbecCamera, depth_jpeg,  # noqa: E402
                                      fill_depth_holes)


def check_statuses() -> None:
    depth = np.full((120, 160), 1200, np.uint16)
    depth[:, :80] = 600
    depth[60:70, 60:75] = 0          # дыра на ближней половине
    depth[60:70, 120:140] = 0        # дыра на дальней половине
    cam = StereoCamera("cam_params.json")
    cam.orbbec = object()
    cam._latest_depth = depth
    cam._depth_seq = 1
    assert cam.probe_depth_mm(0.25, 0.5, get_status=True)[1] == DEPTH_STATUS_OK
    assert cam.probe_depth_mm(0.42, 0.54, get_status=True)[1] == \
        DEPTH_STATUS_UNKNOWN_NEAR
    assert cam.probe_depth_mm(0.81, 0.54, get_status=True)[1] == \
        DEPTH_STATUS_UNKNOWN
    cam.orbbec = None
    assert cam.probe_depth_mm(0.5, 0.5, get_status=True)[1] == \
        DEPTH_STATUS_NO_DATA
    print("depth statuses: ok")


def check_fill() -> None:
    depth = np.full((120, 160), 1000, np.uint16)
    depth[40:80, 50:110] = 0
    filled = fill_depth_holes(depth)
    assert filled.shape == depth.shape
    assert (filled > 0).all(), "в заполненной карте остались дыры"
    assert abs(int(filled[60, 80]) - 1000) < 150
    print("fill_depth_holes: ok")


def check_depth_jpeg() -> None:
    depth = np.zeros((480, 640), np.uint16)
    depth[100:200, 100:200] = 500     # 0.5 м
    depth[300:400, 300:400] = 1500    # 1.5 м
    assert depth_jpeg(depth), "depth_jpeg вернул пусто"
    assert depth_jpeg(None) is None
    print("depth_jpeg: ok")


def check_camera() -> None:
    try:
        running = subprocess.run(
            ["systemctl", "is-active", "--quiet", "r2-robot"]).returncode == 0
    except (OSError, subprocess.SubprocessError):
        running = False
    if running:
        print("r2-robot запущен — живой тест пропущен (остановите службу)")
        return
    cam = OrbbecCamera()
    if not cam.open():
        print("камера не найдена — пропускаю живой тест")
        return
    try:
        assert cam.intrinsics is not None, "нет интринсиков цвета"
        deadline = time.time() + 5.0
        color = depth = None
        while time.time() < deadline:
            color, depth = cam.read()
            if color is not None and depth is not None:
                break
        if color is None or depth is None:
            print("камера занята (r2-robot) или нет кадров — пропускаю живой тест")
            return
        assert color.shape == (480, 640, 3), color.shape
        assert depth.shape == (480, 640), depth.shape
        assert depth.max() > 0, "глубина вся нулевая (закрыт объектив?)"
        print(f"цвет: {color.shape}, глубина: {depth.shape}, "
              f"макс {int(depth.max())} мм, FOV {cam.fov_h:.1f}°")
    finally:
        cam.close()


if __name__ == "__main__":
    check_fill()
    check_statuses()
    check_depth_jpeg()
    check_camera()
    print("ok")
