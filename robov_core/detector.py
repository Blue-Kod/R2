from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _HAS_ORT = True
except ImportError:
    _HAS_ORT = False

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


@dataclass
class Detection:
    name: str
    class_id: int
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    center_x: int
    center_y: int


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)


class ObjectDetector:
    INPUT_SIZE = 640

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.35) -> None:
        self._session: Optional[ort.InferenceSession] = None
        self._conf_threshold = conf_threshold
        self._input_name: str = ""
        self._names: Dict[int, str] = {}
        self._num_classes: int = 0
        self._last_infer_ms: float = 0.0
        self._format: str = "raw"
        self._num_dets: int = 300

        if model_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for candidate in ["yoloe-11s-seg.onnx", "yolov8s.onnx"]:
                p = os.path.join(base, "models", candidate)
                if os.path.isfile(p):
                    model_path = p
                    break
        self._load_names(model_path)
        if os.path.isfile(model_path):
            try:
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 2
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(
                    model_path, opts, providers=["CPUExecutionProvider"]
                )
                self._input_name = self._session.get_inputs()[0].name
                out_shape = self._session.get_outputs()[0].shape
                if len(out_shape) == 3 and out_shape[2] <= 64:
                    self._format = "nms"
                    self._num_dets = int(out_shape[1])
                    self._num_classes = int(out_shape[2]) - 6
                else:
                    self._format = "raw"
                    if len(out_shape) >= 2:
                        self._num_classes = int(out_shape[1]) - 4
                if self._num_classes == len(COCO_NAMES):
                    self._names = {i: n for i, n in enumerate(COCO_NAMES)}
            except Exception:
                self._session = None

    def _load_names(self, onnx_path: str) -> None:
        base_dir = os.path.dirname(onnx_path)
        model_stem = os.path.splitext(os.path.basename(onnx_path))[0]
        candidates = [
            os.path.join(base_dir, model_stem + "_names.json"),
            os.path.join(base_dir, "yoloe_names.json"),
            os.path.join(base_dir, "lvis_classes.json"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                with open(path) as f:
                    raw = json.load(f)
                self._names = {int(k): v for k, v in raw.items()}
                self._num_classes = len(self._names)
                return

    @property
    def available(self) -> bool:
        return self._session is not None

    @property
    def last_infer_ms(self) -> float:
        return self._last_infer_ms

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.available:
            return []
        t0 = time.monotonic()
        img, ratio, (pad_w, pad_h) = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: img})
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]
        if self._format == "nms":
            dets = self._postprocess_nms(raw, frame.shape, ratio, pad_w, pad_h)
        else:
            dets = self._postprocess_raw(raw, frame.shape, ratio, pad_w, pad_h)
        self._last_infer_ms = (time.monotonic() - t0) * 1000
        return dets

    def find(self, name: str, frame: np.ndarray) -> List[Detection]:
        name_lower = name.lower().strip()
        all_dets = self.detect(frame)
        matched = [d for d in all_dets if name_lower in d.name.lower()]
        matched.sort(key=lambda d: d.confidence, reverse=True)
        return matched

    def annotate(self, frame: np.ndarray, detections: List[Detection],
                 labels: Optional[List[str]] = None) -> np.ndarray:
        vis = frame.copy()
        colors = [
            (46, 204, 113), (52, 152, 219), (231, 76, 60),
            (241, 196, 15), (155, 89, 182), (26, 188, 156),
        ]
        for i, det in enumerate(detections):
            color = colors[i % len(colors)]
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 2)
            label = labels[i] if labels else f"{det.name} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (det.x1, det.y1 - th - 8), (det.x1 + tw + 4, det.y1), color, -1)
            cv2.putText(vis, label, (det.x1 + 2, det.y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        return vis

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = frame.shape[:2]
        scale = self.INPUT_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        pad_w = (self.INPUT_SIZE - new_w) // 2
        pad_h = (self.INPUT_SIZE - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, (pad_w, pad_h)

    def _postprocess_nms(self, raw: np.ndarray, orig_shape: Tuple[int, int],
                         ratio: float, pad_w: int, pad_h: int) -> List[Detection]:
        oh, ow = orig_shape[:2]
        mask = raw[:, 4] > self._conf_threshold
        rows = raw[mask]
        if len(rows) == 0:
            return []
        results = []
        for row in rows:
            x1, y1, x2, y2, conf, cls_id = row[:6]
            x1 = int((float(x1) - pad_w) / ratio)
            y1 = int((float(y1) - pad_h) / ratio)
            x2 = int((float(x2) - pad_w) / ratio)
            y2 = int((float(y2) - pad_h) / ratio)
            x1 = max(0, min(x1, ow - 1))
            y1 = max(0, min(y1, oh - 1))
            x2 = max(0, min(x2, ow - 1))
            y2 = max(0, min(y2, oh - 1))
            cid = int(cls_id)
            name = self._names.get(cid, str(cid))
            results.append(Detection(
                name=name, class_id=cid, confidence=float(conf),
                x1=x1, y1=y1, x2=x2, y2=y2,
                center_x=(x1 + x2) // 2, center_y=(y1 + y2) // 2,
            ))
        return results

    def _postprocess_raw(self, raw: np.ndarray, orig_shape: Tuple[int, int],
                         ratio: float, pad_w: int, pad_h: int) -> List[Detection]:
        num_classes = self._num_classes

        boxes_xywh = raw[:4, :].T
        scores_raw = raw[4:4 + num_classes, :].T

        if scores_raw.min() < 0.0 or scores_raw.max() > 1.0:
            scores = 1.0 / (1.0 + np.exp(-scores_raw))
        else:
            scores = scores_raw

        oh, ow = orig_shape[:2]
        max_scores = scores.max(axis=1)
        class_ids = scores.argmax(axis=1)
        mask = max_scores > self._conf_threshold
        boxes_xywh = boxes_xywh[mask]
        max_scores = max_scores[mask]
        class_ids = class_ids[mask]

        if len(max_scores) == 0:
            return []

        boxes_x1y1x2y2 = np.zeros_like(boxes_xywh)
        boxes_x1y1x2y2[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        boxes_x1y1x2y2[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        boxes_x1y1x2y2[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        boxes_x1y1x2y2[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2

        keep = _nms(boxes_x1y1x2y2, max_scores, 0.45)

        results = []
        for i in keep:
            cx, cy, bw, bh = boxes_xywh[i]
            x1 = int((cx - bw / 2 - pad_w) / ratio)
            y1 = int((cy - bh / 2 - pad_h) / ratio)
            x2 = int((cx + bw / 2 - pad_w) / ratio)
            y2 = int((cy + bh / 2 - pad_h) / ratio)
            x1 = max(0, min(x1, ow - 1))
            y1 = max(0, min(y1, oh - 1))
            x2 = max(0, min(x2, ow - 1))
            y2 = max(0, min(y2, oh - 1))
            cid = int(class_ids[i])
            name = self._names.get(cid, str(cid))
            results.append(Detection(
                name=name, class_id=cid, confidence=float(max_scores[i]),
                x1=x1, y1=y1, x2=x2, y2=y2,
                center_x=(x1 + x2) // 2, center_y=(y1 + y2) // 2,
            ))
        return results
