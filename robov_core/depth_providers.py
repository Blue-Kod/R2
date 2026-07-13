from __future__ import annotations

import os
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class DepthResult:
    disparity: np.ndarray
    confidence: Optional[np.ndarray] = None


class DepthProvider(ABC):

    @abstractmethod
    def setup(self, **kwargs) -> None:
        pass

    @abstractmethod
    def compute(self, left: np.ndarray, right: np.ndarray) -> DepthResult:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    def release(self) -> None:
        pass


class StereoSGBMDepthProvider(DepthProvider):

    def __init__(self) -> None:
        self.left_matcher: Optional[cv2.StereoSGBM] = None
        self.right_matcher = None
        self.wls_filter = None
        self._wls_enabled: bool = True

    def setup(self, **kwargs) -> None:
        num_disp: int = kwargs.get("num_disp", 160)
        window_size: int = kwargs.get("window_size", 11)
        min_disp: int = kwargs.get("min_disp", 0)
        self._wls_enabled = kwargs.get("wls_enabled", True)
        wls_lambda: float = kwargs.get("wls_lambda", 8000.0)
        wls_sigma: float = kwargs.get("wls_sigma", 1.5)

        self.left_matcher = cv2.StereoSGBM_create(
            minDisparity=min_disp,
            numDisparities=num_disp,
            blockSize=window_size,
            P1=8 * 3 * window_size ** 2,
            P2=32 * 3 * window_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=200,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )

        if self._wls_enabled:
            self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)
            self.wls_filter = cv2.ximgproc.createDisparityWLSFilter(
                matcher_left=self.left_matcher,
            )
            self.wls_filter.setLambda(wls_lambda)
            self.wls_filter.setSigmaColor(wls_sigma)
        else:
            self.right_matcher = None
            self.wls_filter = None

    def compute(self, left: np.ndarray, right: np.ndarray) -> DepthResult:
        grayL = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        disp = self.left_matcher.compute(grayL, grayR)
        disp[disp < 0] = 0

        if self._wls_enabled and self.right_matcher and self.wls_filter:
            disp_r = self.right_matcher.compute(grayR, grayL)
            disp = self.wls_filter.filter(disp, grayL, disparity_map_right=disp_r)
            disp[disp < 0] = 0

        disp = cv2.medianBlur(disp, 3)
        return DepthResult(disparity=disp)

    @property
    def name(self) -> str:
        return "SGBM"

    def release(self) -> None:
        self.left_matcher = None
        self.right_matcher = None
        self.wls_filter = None


class LAS2DepthProvider(DepthProvider):
    """LAS2-S stereo depth via ONNX Runtime. No PyTorch needed at runtime."""

    def __init__(self) -> None:
        self._session = None
        self._max_disp: int = 192
        self._input_h: int = 384
        self._input_w: int = 640
        self._crop_h: int = 360

    def setup(self, **kwargs) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise RuntimeError("onnxruntime is required for LAS2 provider")

        onnx_path: str = kwargs.get("onnx_path", "")
        self._max_disp: int = kwargs.get("max_disp", 192)

        if not onnx_path:
            if self._session is not None:
                return
            raise FileNotFoundError("LAS2 ONNX model path not provided")

        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"LAS2 ONNX model not found: {onnx_path}")

        self._session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )

        inp = self._session.get_inputs()[0]
        parts = inp.shape
        if len(parts) == 4:
            self._input_h = int(parts[2])
            self._input_w = int(parts[3])

        self._crop_h = kwargs.get("crop_h", 360)

    def compute(self, left: np.ndarray, right: np.ndarray) -> DepthResult:
        if self._session is None:
            raise RuntimeError("LAS2 model not loaded")

        h, w = left.shape[:2]

        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        need_pad = (h != self._input_h) or (w != self._input_w)
        if need_pad:
            left_rgb = cv2.resize(left_rgb, (self._input_w, self._input_h))
            right_rgb = cv2.resize(right_rgb, (self._input_w, self._input_h))

        left_t = left_rgb.astype(np.float32).transpose(2, 0, 1)[None]
        right_t = right_rgb.astype(np.float32).transpose(2, 0, 1)[None]

        disp_float = self._session.run(
            ["disparity"], {"left": left_t, "right": right_t}
        )[0][0, 0]

        if need_pad and self._crop_h < self._input_h:
            disp_float = disp_float[:h, :w]

        disp_int16 = np.clip(disp_float * 16.0, 0, 32767).astype(np.int16)
        return DepthResult(disparity=disp_int16)

    @property
    def name(self) -> str:
        return "LAS2"

    def release(self) -> None:
        self._session = None


class RKNNDepthProvider(DepthProvider):
    """LAS2-S stereo depth via Rockchip NPU (rknn-toolkit-lite2).

    Falls back to ONNX Runtime if rknn-lite2 is unavailable or .rknn file missing.
    """

    def __init__(self) -> None:
        self._rknn = None
        self._fallback: Optional[LAS2DepthProvider] = None
        self._input_h: int = 384
        self._input_w: int = 640
        self._crop_h: int = 360
        self._core_mask: int = 0  # 0 = auto

    def setup(self, **kwargs) -> None:
        rknn_path: str = kwargs.get("rknn_path", "")
        onnx_path: str = kwargs.get("onnx_path", "")
        self._crop_h: int = kwargs.get("crop_h", 360)
        self._core_mask: int = kwargs.get("core_mask", 0)

        if not rknn_path or not os.path.isfile(rknn_path):
            raise FileNotFoundError(f"RKNN model not found: {rknn_path}")

        try:
            from rknnlite.api import RKNNLite
        except ImportError:
            raise ImportError(
                "rknn-toolkit-lite2 not installed. "
                "Install on the NPU device: pip install rknn-toolkit-lite2"
            )

        self._rknn = RKNNLite()
        if self._rknn.load_rknn(rknn_path) != 0:
            raise RuntimeError(f"Failed to load RKNN model: {rknn_path}")

        core_mask = RKNNLite.NPU_CORE_0_1_2
        if self._core_mask == 1:
            core_mask = RKNNLite.NPU_CORE_0
        elif self._core_mask == 2:
            core_mask = RKNNLite.NPU_CORE_1
        elif self._core_mask == 3:
            core_mask = RKNNLite.NPU_CORE_2

        if self._rknn.init_runtime(target='a733', core_mask=core_mask) != 0:
            raise RuntimeError("Failed to init RKNN runtime")

        self._rknn_sdk_version = self._rknn.get_sdk_version()
        self._input_h = 384
        self._input_w = 640
        return

    def _setup_fallback(self, onnx_path: str) -> None:
        if self._fallback is not None:
            return
        if not onnx_path or not os.path.isfile(onnx_path):
            return
        self._fallback = LAS2DepthProvider()
        self._fallback.setup(onnx_path=onnx_path, crop_h=self._crop_h)

    def compute(self, left: np.ndarray, right: np.ndarray) -> DepthResult:
        if self._rknn is None:
            if self._fallback is not None:
                return self._fallback.compute(left, right)
            raise RuntimeError("Neither RKNN model nor fallback loaded")

        h, w = left.shape[:2]

        left_rgb = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right, cv2.COLOR_BGR2RGB)

        need_resize = (h != self._input_h) or (w != self._input_w)
        if need_resize:
            left_rgb = cv2.resize(left_rgb, (self._input_w, self._input_h))
            right_rgb = cv2.resize(right_rgb, (self._input_w, self._input_h))

        left_input = left_rgb.astype(np.float32).transpose(2, 0, 1)[None]
        right_input = right_rgb.astype(np.float32).transpose(2, 0, 1)[None]

        try:
            outputs = self._rknn.inference(inputs=[left_input, right_input])
        except Exception as e:
            if self._fallback is not None:
                return self._fallback.compute(left, right)
            raise RuntimeError(f"RKNN inference failed: {e}")

        disp_float = outputs[0][0, 0]

        if need_resize and self._crop_h < self._input_h:
            disp_float = disp_float[:h, :w]

        disp_int16 = np.clip(disp_float * 16.0, 0, 32767).astype(np.int16)
        return DepthResult(disparity=disp_int16)

    @property
    def name(self) -> str:
        return "RKNN"

    def release(self) -> None:
        if self._rknn is not None:
            try:
                self._rknn.release()
            except Exception:
                pass
            self._rknn = None
        if self._fallback is not None:
            self._fallback.release()
            self._fallback = None
