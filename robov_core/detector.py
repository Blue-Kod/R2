from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
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
    mask: Optional[np.ndarray] = field(default=None, repr=False)


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

    def __init__(self, model_path: Optional[str] = None, conf_threshold: float = 0.35,
                 threads_inter: int = 2, threads_intra: int = 4) -> None:
        self._session: Optional[ort.InferenceSession] = None
        self._conf_threshold = conf_threshold
        self._input_name: str = ""
        self._input_size: int = 640
        self._names: Dict[int, str] = {}
        self._num_classes: int = 0
        self._last_infer_ms: float = 0.0
        self._format: str = "raw"
        self._num_dets: int = 300
        self._has_mask_output: bool = False
        self._mask_proto_name: str = ""
        self._model_path: str = ""

        if model_path is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            for candidate in ["yoloe-11s-seg-640.onnx", "yoloe-11s-seg-320.onnx", "yoloe-11s-seg.onnx", "yolov8s.onnx"]:
                p = os.path.join(base, "models", candidate)
                if os.path.isfile(p):
                    model_path = p
                    break
        self._model_path = model_path
        self._load_names(model_path)
        if os.path.isfile(model_path):
            try:
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = threads_inter
                opts.intra_op_num_threads = threads_intra
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._session = ort.InferenceSession(
                    model_path, opts, providers=["CPUExecutionProvider"]
                )
                self._input_name = self._session.get_inputs()[0].name
                in_shape = self._session.get_inputs()[0].shape
                if len(in_shape) >= 4 and isinstance(in_shape[2], int):
                    self._input_size = in_shape[2]
                outputs = self._session.get_outputs()
                out_shape = outputs[0].shape
                if len(out_shape) == 3 and out_shape[2] <= 64:
                    self._format = "nms"
                    self._num_dets = int(out_shape[1])
                    self._num_classes = int(out_shape[2]) - 6
                else:
                    self._format = "raw"
                    if len(out_shape) >= 2:
                        self._num_classes = int(out_shape[1]) - 4
                if len(outputs) >= 2:
                    self._has_mask_output = True
                    self._mask_proto_name = outputs[1].name
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

    @property
    def model_name(self) -> str:
        return os.path.basename(self._model_path) if self._model_path else ""

    def reinit_object_detection(self, model_name: str) -> bool:
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(base, "models", model_name)
        if not os.path.isfile(model_path):
            return False
        try:
            self._session.close() if self._session else None
        except Exception:
            pass
        self._session = None
        self._model_path = model_path
        self._names = {}
        self._num_classes = 0
        self._format = "raw"
        self._num_dets = 300
        self._has_mask_output = False
        self._mask_proto_name = ""
        self._input_name = ""
        self._load_names(model_path)
        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(
                model_path, opts, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
            in_shape = self._session.get_inputs()[0].shape
            if len(in_shape) >= 4 and isinstance(in_shape[2], int):
                self._input_size = in_shape[2]
            outputs = self._session.get_outputs()
            out_shape = outputs[0].shape
            if len(out_shape) == 3 and out_shape[2] <= 64:
                self._format = "nms"
                self._num_dets = int(out_shape[1])
                self._num_classes = int(out_shape[2]) - 6
            else:
                self._format = "raw"
                if len(out_shape) >= 2:
                    self._num_classes = int(out_shape[1]) - 4
            if len(outputs) >= 2:
                self._has_mask_output = True
                self._mask_proto_name = outputs[1].name
            if self._num_classes == len(COCO_NAMES):
                self._names = {i: n for i, n in enumerate(COCO_NAMES)}
            return True
        except Exception:
            self._session = None
            return False

    def detect(self, frame: np.ndarray) -> List[Detection]:
        if not self.available:
            return []
        t0 = time.monotonic()
        img, ratio, (pad_w, pad_h) = self._preprocess(frame)
        outputs = self._session.run(None, {self._input_name: img})
        raw = outputs[0]
        if raw.ndim == 3:
            raw = raw[0]
        proto = outputs[1][0] if len(outputs) >= 2 else None
        if self._format == "nms":
            dets = self._postprocess_nms(raw, proto, frame.shape, ratio, pad_w, pad_h)
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
        font_scale = 0.35
        font_thick = 1
        for i, det in enumerate(detections):
            color = colors[i % len(colors)]
            if det.mask is not None:
                overlay = vis.copy()
                overlay[det.mask > 0] = color
                cv2.addWeighted(vis, 0.6, overlay, 0.4, 0, dst=vis)
                contours, _ = cv2.findContours(
                    det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(vis, contours, -1, color, 2)
            cv2.rectangle(vis, (det.x1, det.y1), (det.x2, det.y2), color, 1)
            label = labels[i] if labels else f"{det.name} {det.confidence:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            cv2.rectangle(vis, (det.x1, det.y1 - th - 6), (det.x1 + tw + 3, det.y1), color, -1)
            cv2.putText(vis, label, (det.x1 + 1, det.y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)
        return vis

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        h, w = frame.shape[:2]
        sz = self._input_size
        scale = sz / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((sz, sz, 3), 114, dtype=np.uint8)
        pad_w = (sz - new_w) // 2
        pad_h = (sz - new_h) // 2
        canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, axis=0)
        return blob, scale, (pad_w, pad_h)

    def _decode_mask(self, mask_coeffs: np.ndarray, proto: np.ndarray,
                     orig_h: int, orig_w: int,
                     ratio: float, pad_w: int, pad_h: int,
                     bbox: Optional[Tuple[int, int, int, int]] = None) -> Optional[np.ndarray]:
        if proto is None or mask_coeffs is None:
            return None
        try:
            mask_coeffs = mask_coeffs.astype(np.float32)
            raw_mask = mask_coeffs @ proto.reshape(proto.shape[0], -1)
            raw_mask = raw_mask.reshape(proto.shape[1], proto.shape[2])
            raw_mask = 1.0 / (1.0 + np.exp(-raw_mask))
            mask_bin = (raw_mask > 0.5).astype(np.uint8)
            new_h = int(orig_h * ratio)
            new_w = int(orig_w * ratio)
            valid_mask = mask_bin[pad_h:pad_h + new_h, pad_w:pad_w + new_w]
            full_mask = cv2.resize(valid_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                clip = np.zeros_like(full_mask)
                clip[max(0, y1):min(orig_h, y2), max(0, x1):min(orig_w, x2)] = 1
                full_mask = full_mask & clip
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            full_mask = cv2.morphologyEx(full_mask, cv2.MORPH_OPEN, kernel)
            n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(full_mask, connectivity=8)
            if n_labels > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                best = int(np.argmax(areas)) + 1
                full_mask = (labels == best).astype(np.uint8)
            return full_mask
        except Exception:
            return None

    def _postprocess_nms(self, raw: np.ndarray, proto: Optional[np.ndarray],
                         orig_shape: Tuple[int, int],
                         ratio: float, pad_w: int, pad_h: int) -> List[Detection]:
        oh, ow = orig_shape[:2]
        mask = raw[:, 4] > self._conf_threshold
        rows = raw[mask]
        if len(rows) == 0:
            return []
        results = []
        for row in rows:
            x1, y1, x2, y2, conf, cls_id = row[:6]
            ix1 = int((float(x1) - pad_w) / ratio)
            iy1 = int((float(y1) - pad_h) / ratio)
            ix2 = int((float(x2) - pad_w) / ratio)
            iy2 = int((float(y2) - pad_h) / ratio)
            ix1 = max(0, min(ix1, ow - 1))
            iy1 = max(0, min(iy1, oh - 1))
            ix2 = max(0, min(ix2, ow - 1))
            iy2 = max(0, min(iy2, oh - 1))
            cid = int(cls_id)
            name = self._names.get(cid, str(cid))
            det_mask = None
            if proto is not None and len(row) > 6:
                det_mask = self._decode_mask(
                    row[6:6 + proto.shape[0]], proto,
                    oh, ow, ratio, pad_w, pad_h,
                    bbox=(ix1, iy1, ix2, iy2)
                )
            results.append(Detection(
                name=name, class_id=cid, confidence=float(conf),
                x1=ix1, y1=iy1, x2=ix2, y2=iy2,
                center_x=(ix1 + ix2) // 2, center_y=(iy1 + iy2) // 2,
                mask=det_mask,
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
