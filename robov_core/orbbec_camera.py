"""Orbbec Astra Mini S: цвет и глубина через C API Orbbec SDK.

SDK — набор разделяемых библиотек (libOrbbecSDK, libob_usb, liblive555),
поэтому ctypes, без pybind11 и сборки под конкретный Python: на Orange Pi
достаточно положить arm64-библиотеки из vendor/orbbec/arm64.

Глубина приходит в миллиметрах, выровнена по цвету (SW D2C): пиксель (u,v)
карты глубины соответствует пикселю (u,v) цветного кадра. Это позволяет
считать 3D прямо по интринсикам цветной камеры.

Устройство отдаёт Y16, 640x480@30 (Astra Mini S). Цвет — RGB 640x480@30;
SDK сам распаковывает поток.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import platform
import threading
from pathlib import Path
from typing import NamedTuple, Optional, Tuple

import cv2
import numpy as np

# Форматы и типы из libobsensor/h/ObTypes.h
OB_FORMAT_UYVY = 2
OB_FORMAT_MJPG = 5
OB_FORMAT_Y16 = 8
OB_FORMAT_Y11 = 11
OB_FORMAT_RGB = 22

OB_SENSOR_COLOR = 2
OB_SENSOR_DEPTH = 3
ALIGN_D2C_SW = 2
OB_PROP_LASER_BOOL = 3

# Орббек пишет info/warning в stdout на каждый кадр/остановку; для панели
# оставляем только ошибки.
OB_LOG_SEVERITY_ERROR = 4


class _Intrinsic(ctypes.Structure):
    _fields_ = [("fx", ctypes.c_float), ("fy", ctypes.c_float),
                ("cx", ctypes.c_float), ("cy", ctypes.c_float),
                ("width", ctypes.c_int16), ("height", ctypes.c_int16)]


class _Distortion(ctypes.Structure):
    _fields_ = [(name, ctypes.c_float) for name in
                ("k1", "k2", "k3", "k4", "k5", "k6", "p1", "p2")]


class _Extrinsic(ctypes.Structure):
    _fields_ = [("rot", ctypes.c_float * 9), ("trans", ctypes.c_float * 3)]


class _CameraParam(ctypes.Structure):
    _fields_ = [("depthIntrinsic", _Intrinsic), ("rgbIntrinsic", _Intrinsic),
                ("depthDistortion", _Distortion), ("rgbDistortion", _Distortion),
                ("transform", _Extrinsic), ("isMirrored", ctypes.c_bool)]


class Intrinsics(NamedTuple):
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def _arch_dir() -> str:
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64"):
        return "x64"
    if machine in ("aarch64", "arm64"):
        return "arm64"
    return machine


def _lib_dir() -> Optional[Path]:
    """Каталог с libOrbbecSDK и зависимостями.

    R2_ORBBEC_LIB может указывать на каталог или на сам .so — удобно для
    проверки другой версии SDK без правки репозитория.
    """
    env = os.environ.get("R2_ORBBEC_LIB")
    if env:
        path = Path(env).expanduser()
        return path.parent if path.is_file() else path
    base = Path(__file__).resolve().parent / "vendor" / "orbbec" / _arch_dir()
    return base if base.is_dir() else None


def _tune_usbfs() -> None:
    """Orbbec рекомендует usbfs_memory_mb=128; на роботе процесс root."""
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    path = Path("/sys/module/usbcore/parameters/usbfs_memory_mb")
    try:
        if int(path.read_text().strip()) < 128:
            path.write_text("128\n")
    except Exception:
        pass


def _frame_jpeg(frame: np.ndarray, quality: int = 70) -> Optional[bytes]:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None


def fill_depth_holes(depth_mm: Optional[np.ndarray],
                     scale: int = 2) -> Optional[np.ndarray]:
    """Заменить дыры (0) глубиной ближайшего валидного пикселя.

    Ближняя граница Astra (~0.5 м) даёт нули на объектах вплотную к камере;
    продление соседней поверхности убирает чёрные провалы. Значение в дыре —
    приблизительное (ближайший валидный пиксель), это помечается в API.

    Заполнение считается на уменьшенной копии (scale=2): на 640x480
    distanceTransform дороже кадра, а карта глубины низкочастотная.
    """
    if depth_mm is None or depth_mm.size == 0:
        return depth_mm
    h, w = depth_mm.shape[:2]
    if scale > 1 and min(h, w) >= 2 * scale:
        small = cv2.resize(depth_mm, (w // scale, h // scale),
                           interpolation=cv2.INTER_NEAREST)
        filled = _fill_nearest(small)
        return cv2.resize(filled, (w, h), interpolation=cv2.INTER_NEAREST)
    return _fill_nearest(depth_mm)


def _fill_nearest(depth_mm: np.ndarray) -> np.ndarray:
    valid = depth_mm > 0
    if valid.all() or not valid.any():
        return depth_mm
    _, labels = cv2.distanceTransformWithLabels(
        (~valid).astype(np.uint8), cv2.DIST_L2, 5,
        labelType=cv2.DIST_LABEL_PIXEL)
    flat = np.flatnonzero(valid.ravel())
    lut = np.zeros(int(labels.max()) + 1, np.int32)
    lut[labels.ravel()[flat]] = flat
    return depth_mm.ravel()[lut[labels.ravel()]].reshape(depth_mm.shape)


def depth_jpeg(depth_mm: Optional[np.ndarray], near_mm: int = 200,
               far_mm: int = 2000, quality: int = 70) -> Optional[bytes]:
    """Карта глубины -> монохромный JPEG: ближе светлее, дырки чёрные."""
    if depth_mm is None or depth_mm.size == 0:
        return None
    d = depth_mm.astype(np.float32)
    valid = d > 0
    if not valid.any():
        img = np.zeros((*depth_mm.shape, 3), np.uint8)
    else:
        t = np.clip((d - near_mm) / float(far_mm - near_mm), 0.0, 1.0)
        vis = (255.0 * (1.0 - t)).astype(np.uint8)   # near=255, far=0
        img = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        img[~valid] = (0, 0, 0)
    return _frame_jpeg(img, quality)


class OrbbecCamera:
    """Один поток: open() -> read() в цикле -> close()."""

    _lib = None
    _lib_lock = threading.Lock()

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.name = "Orbbec"
        self.intrinsics: Optional[Intrinsics] = None
        self._ctx = None
        self._dev = None
        self._pipe = None
        self._config = None
        self._io_lock = threading.Lock()

    # -- загрузка SDK -------------------------------------------------------
    @classmethod
    def _load_library(cls):
        if cls._lib is not None:
            return cls._lib
        with cls._lib_lock:
            if cls._lib is not None:
                return cls._lib
            libdir = _lib_dir()
            candidates = []
            if libdir is not None:
                # зависимости должны быть в RTLD_GLOBAL, иначе libOrbbecSDK
                # не найдёт liblive555/libob_usb при загрузке по полному пути
                for dep in ("liblive555.so", "libob_usb.so"):
                    path = libdir / dep
                    if path.exists():
                        ctypes.CDLL(str(path),
                                    mode=getattr(ctypes, "RTLD_GLOBAL", 0))
                candidates.append(libdir / "libOrbbecSDK.so")
                candidates.extend(sorted(libdir.glob("libOrbbecSDK.so.*"),
                                         reverse=True))
            found = ctypes.util.find_library("OrbbecSDK")
            if found:
                candidates.append(found)
            for path in candidates:
                try:
                    lib = ctypes.CDLL(str(path))
                except OSError:
                    continue
                cls._bind(lib)
                cls._lib = lib
                return lib
        raise OSError("libOrbbecSDK не найдена "
                      "(положите SDK в robov_core/vendor/orbbec или задайте "
                      "R2_ORBBEC_LIB)")

    @staticmethod
    def _bind(lib) -> None:
        def fn(name, restype, *argtypes):
            func = getattr(lib, name)
            func.restype = restype
            func.argtypes = list(argtypes)
            return func

        p = ctypes.POINTER(ctypes.c_void_p)
        lib.ob_set_logger_severity = fn(
            "ob_set_logger_severity", None, ctypes.c_int, p)
        lib.ob_create_context = fn("ob_create_context", ctypes.c_void_p, p)
        lib.ob_delete_context = fn("ob_delete_context", None, ctypes.c_void_p, p)
        lib.ob_query_device_list = fn(
            "ob_query_device_list", ctypes.c_void_p, ctypes.c_void_p, p)
        lib.ob_delete_device_list = fn(
            "ob_delete_device_list", None, ctypes.c_void_p, p)
        lib.ob_device_list_device_count = fn(
            "ob_device_list_device_count", ctypes.c_uint32, ctypes.c_void_p, p)
        lib.ob_device_list_get_device = fn(
            "ob_device_list_get_device", ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint32, p)
        lib.ob_delete_device = fn("ob_delete_device", None, ctypes.c_void_p, p)
        lib.ob_device_set_bool_property = fn(
            "ob_device_set_bool_property", None, ctypes.c_void_p, ctypes.c_int,
            ctypes.c_bool, p)
        lib.ob_device_get_bool_property = fn(
            "ob_device_get_bool_property", ctypes.c_bool, ctypes.c_void_p,
            ctypes.c_int, p)
        lib.ob_device_get_device_info = fn(
            "ob_device_get_device_info", ctypes.c_void_p, ctypes.c_void_p, p)
        lib.ob_delete_device_info = fn(
            "ob_delete_device_info", None, ctypes.c_void_p, p)
        lib.ob_device_info_name = fn(
            "ob_device_info_name", ctypes.c_char_p, ctypes.c_void_p, p)
        lib.ob_create_pipeline_with_device = fn(
            "ob_create_pipeline_with_device", ctypes.c_void_p, ctypes.c_void_p, p)
        lib.ob_delete_pipeline = fn(
            "ob_delete_pipeline", None, ctypes.c_void_p, p)
        lib.ob_pipeline_get_stream_profile_list = fn(
            "ob_pipeline_get_stream_profile_list", ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, p)
        lib.ob_delete_stream_profile_list = fn(
            "ob_delete_stream_profile_list", None, ctypes.c_void_p, p)
        lib.ob_stream_profile_list_get_video_stream_profile = fn(
            "ob_stream_profile_list_get_video_stream_profile", ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, p)
        lib.ob_delete_stream_profile = fn(
            "ob_delete_stream_profile", None, ctypes.c_void_p, p)
        lib.ob_create_config = fn("ob_create_config", ctypes.c_void_p, p)
        lib.ob_delete_config = fn("ob_delete_config", None, ctypes.c_void_p, p)
        lib.ob_config_enable_stream = fn(
            "ob_config_enable_stream", None, ctypes.c_void_p, ctypes.c_void_p, p)
        lib.ob_config_set_align_mode = fn(
            "ob_config_set_align_mode", None, ctypes.c_void_p, ctypes.c_int, p)
        lib.ob_pipeline_start_with_config = fn(
            "ob_pipeline_start_with_config", None,
            ctypes.c_void_p, ctypes.c_void_p, p)
        lib.ob_pipeline_stop = fn(
            "ob_pipeline_stop", None, ctypes.c_void_p, p)
        lib.ob_pipeline_get_camera_param = fn(
            "ob_pipeline_get_camera_param", _CameraParam, ctypes.c_void_p, p)
        lib.ob_pipeline_wait_for_frameset = fn(
            "ob_pipeline_wait_for_frameset", ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_uint32, p)
        lib.ob_frameset_color_frame = fn(
            "ob_frameset_color_frame", ctypes.c_void_p, ctypes.c_void_p, p)
        lib.ob_frameset_depth_frame = fn(
            "ob_frameset_depth_frame", ctypes.c_void_p, ctypes.c_void_p, p)
        lib.ob_delete_frame = fn("ob_delete_frame", None, ctypes.c_void_p, p)
        lib.ob_frame_data = fn("ob_frame_data", ctypes.c_void_p, ctypes.c_void_p, p)
        lib.ob_frame_data_size = fn(
            "ob_frame_data_size", ctypes.c_uint32, ctypes.c_void_p, p)
        lib.ob_frame_format = fn(
            "ob_frame_format", ctypes.c_int, ctypes.c_void_p, p)
        lib.ob_video_frame_width = fn(
            "ob_video_frame_width", ctypes.c_uint32, ctypes.c_void_p, p)
        lib.ob_video_frame_height = fn(
            "ob_video_frame_height", ctypes.c_uint32, ctypes.c_void_p, p)
        lib.ob_depth_frame_get_value_scale = fn(
            "ob_depth_frame_get_value_scale", ctypes.c_float, ctypes.c_void_p, p)
        lib.ob_error_message = fn(
            "ob_error_message", ctypes.c_char_p, ctypes.c_void_p)
        lib.ob_delete_error = fn("ob_delete_error", None, ctypes.c_void_p)

    @staticmethod
    def _clear_error(lib, err: ctypes.c_void_p) -> None:
        OrbbecCamera._error_text(lib, err)

    @staticmethod
    def _error_text(lib, err: ctypes.c_void_p) -> str:
        if not err.value:
            return ""
        try:
            msg = lib.ob_error_message(err.value)
            return msg.decode(errors="replace") if msg else ""
        finally:
            lib.ob_delete_error(err.value)
            err.value = None

    # -- открытие/закрытие -------------------------------------------------
    def open(self) -> bool:
        try:
            lib = self._load_library()
        except OSError as exc:
            print(f"[Orbbec] {exc}", flush=True)
            return False
        _tune_usbfs()
        with self._io_lock:
            if self._pipe is not None:
                return True
            try:
                lib.ob_set_logger_severity(OB_LOG_SEVERITY_ERROR, None)
                self._ctx = lib.ob_create_context(None)
                if not self._ctx:
                    return False
                err = ctypes.c_void_p()
                devices = lib.ob_query_device_list(self._ctx, ctypes.byref(err))
                count = (lib.ob_device_list_device_count(devices, ctypes.byref(err))
                         if devices else 0)
                if count <= 0:
                    self._release_locked(lib)
                    return False
                self._dev = lib.ob_device_list_get_device(
                    devices, 0, ctypes.byref(err))
                lib.ob_delete_device_list(devices, None)
                if not self._dev:
                    self._release_locked(lib)
                    return False
                info = lib.ob_device_get_device_info(self._dev, ctypes.byref(err))
                if info:
                    name = lib.ob_device_info_name(info, ctypes.byref(err))
                    if name:
                        self.name = name.decode(errors="replace")
                    lib.ob_delete_device_info(info, None)
                # ИК-подсветка: без неё Astra отдаёт нулевую глубину.
                try:
                    lib.ob_device_set_bool_property(
                        self._dev, OB_PROP_LASER_BOOL, True, ctypes.byref(err))
                    self._clear_error(lib, err)
                except Exception:
                    pass
                self._pipe = lib.ob_create_pipeline_with_device(
                    self._dev, ctypes.byref(err))
                if not self._pipe:
                    self._release_locked(lib)
                    return False
                color_list = lib.ob_pipeline_get_stream_profile_list(
                    self._pipe, OB_SENSOR_COLOR, ctypes.byref(err))
                depth_list = lib.ob_pipeline_get_stream_profile_list(
                    self._pipe, OB_SENSOR_DEPTH, ctypes.byref(err))
                if not color_list or not depth_list:
                    self._release_locked(lib)
                    return False
                color_profile = lib.ob_stream_profile_list_get_video_stream_profile(
                    color_list, self.width, self.height, OB_FORMAT_RGB,
                    self.fps, ctypes.byref(err))
                depth_profile = lib.ob_stream_profile_list_get_video_stream_profile(
                    depth_list, self.width, self.height, OB_FORMAT_Y11,
                    self.fps, ctypes.byref(err))
                lib.ob_delete_stream_profile_list(color_list, None)
                lib.ob_delete_stream_profile_list(depth_list, None)
                if not color_profile or not depth_profile:
                    self._release_locked(lib)
                    return False
                self._config = lib.ob_create_config(ctypes.byref(err))
                lib.ob_config_enable_stream(
                    self._config, color_profile, ctypes.byref(err))
                lib.ob_config_enable_stream(
                    self._config, depth_profile, ctypes.byref(err))
                lib.ob_config_set_align_mode(
                    self._config, ALIGN_D2C_SW, ctypes.byref(err))
                lib.ob_delete_stream_profile(color_profile, None)
                lib.ob_delete_stream_profile(depth_profile, None)
                self._clear_error(lib, err)
                lib.ob_pipeline_start_with_config(
                    self._pipe, self._config, ctypes.byref(err))
                if err.value:
                    print(f"[Orbbec] start: {self._error_text(lib, err)}",
                          flush=True)
                    self._release_locked(lib)
                    return False
                param = lib.ob_pipeline_get_camera_param(
                    self._pipe, ctypes.byref(err))
                ci = param.rgbIntrinsic
                if ci.fx > 0 and ci.fy > 0:
                    self.intrinsics = Intrinsics(ci.fx, ci.fy, ci.cx, ci.cy,
                                                 ci.width, ci.height)
                suffix = (f"FOV {self.fov_h:.0f}°" if self.intrinsics
                          else "нет интринсиков")
                print(f"[Orbbec] {self.name}: {self.width}x{self.height} "
                      f"@ {self.fps} fps, {suffix}", flush=True)
                return True
            except Exception as exc:
                print(f"[Orbbec] open error: {exc}", flush=True)
                self._release_locked(lib)
                return False

    def _release_locked(self, lib) -> None:
        if self._pipe:
            try:
                lib.ob_pipeline_stop(self._pipe, None)
            except Exception:
                pass
        for attr in ("_config", "_pipe", "_dev", "_ctx"):
            pointer = getattr(self, attr)
            setattr(self, attr, None)
            if not pointer:
                continue
            delete = {
                "_config": lib.ob_delete_config,
                "_pipe": lib.ob_delete_pipeline,
                "_dev": lib.ob_delete_device,
                "_ctx": lib.ob_delete_context,
            }[attr]
            try:
                delete(pointer, None)
            except Exception:
                pass

    def close(self) -> None:
        with self._io_lock:
            lib = self._lib
            if lib is not None:
                try:
                    self._release_locked(lib)
                except Exception:
                    pass
            self.intrinsics = None

    # -- кадры --------------------------------------------------------------
    @staticmethod
    def _frame_array(lib, frame, dtype) -> Optional[np.ndarray]:
        if not frame:
            return None
        size = lib.ob_frame_data_size(frame, None)
        ptr = lib.ob_frame_data(frame, None)
        if not size or not ptr:
            return None
        buf = (ctypes.c_uint8 * size).from_address(ptr)
        return np.ctypeslib.as_array(buf).copy().view(dtype)

    @staticmethod
    def _color_bgr(lib, frame) -> Optional[np.ndarray]:
        fmt = lib.ob_frame_format(frame, None)
        w = lib.ob_video_frame_width(frame, None)
        h = lib.ob_video_frame_height(frame, None)
        data = OrbbecCamera._frame_array(lib, frame, np.uint8)
        if data is None or w <= 0 or h <= 0:
            return None
        try:
            if fmt == OB_FORMAT_RGB:
                return cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
            if fmt == OB_FORMAT_UYVY:
                return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
            if fmt == OB_FORMAT_MJPG:
                return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None
        return None

    def read(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """(BGR-кадр, глубина uint16 в мм, выровненная по цвету)."""
        with self._io_lock:
            lib = self._lib
            if lib is None or self._pipe is None:
                return None, None
            try:
                frames = lib.ob_pipeline_wait_for_frameset(self._pipe, 1000, None)
                if not frames:
                    return None, None
                color_frame = lib.ob_frameset_color_frame(frames, None)
                depth_frame = lib.ob_frameset_depth_frame(frames, None)
                color = self._color_bgr(lib, color_frame) if color_frame else None
                depth = None
                if depth_frame:
                    raw = self._frame_array(lib, depth_frame, np.uint16)
                    if raw is not None:
                        h = lib.ob_video_frame_height(depth_frame, None)
                        w = lib.ob_video_frame_width(depth_frame, None)
                        if h > 0 and w > 0 and raw.size == h * w:
                            raw = raw.reshape(h, w)
                        scale = lib.ob_depth_frame_get_value_scale(depth_frame, None)
                        scale = float(scale) if scale and scale > 0 else 1.0
                        depth = np.clip(raw.astype(np.float32) * scale,
                                        0, 65535).astype(np.uint16)
                if color_frame:
                    lib.ob_delete_frame(color_frame, None)
                if depth_frame:
                    lib.ob_delete_frame(depth_frame, None)
                lib.ob_delete_frame(frames, None)
                return color, depth
            except Exception:
                return None, None

    @property
    def fov_h(self) -> float:
        if self.intrinsics and self.intrinsics.fx > 0:
            return float(np.degrees(2.0 * np.arctan(
                self.width / (2.0 * self.intrinsics.fx))))
        return 60.0
