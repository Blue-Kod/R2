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

        if not onnx_path or not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"LAS2 ONNX model not found: {onnx_path}")

        self._session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )

        inp = self._session.get_inputs()[0]
        parts = inp.shape  # e.g. [1, 3, 384, 640]
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
