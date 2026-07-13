from __future__ import annotations

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
    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> DepthResult:
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

    def compute(self, left_gray: np.ndarray, right_gray: np.ndarray) -> DepthResult:
        disp = self.left_matcher.compute(left_gray, right_gray)
        disp[disp < 0] = 0

        if self._wls_enabled and self.right_matcher and self.wls_filter:
            disp_r = self.right_matcher.compute(right_gray, left_gray)
            disp = self.wls_filter.filter(disp, left_gray, disparity_map_right=disp_r)
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
