"""Shared preprocessing, postprocessing, and annotation utilities for YOLOv8-nano inference.

This module is an internal dependency of ``inference_pytorch.py`` and
``inference_onnx.py``.  All functions are pure NumPy / OpenCV — no framework
(PyTorch, ONNX Runtime, Ultralytics) is imported here.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yaml

# (class_id, conf, x1, y1, x2, y2) — all coordinates in original image pixels
Detection = tuple[int, float, float, float, float, float]

IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

CLASS_NAMES: dict[int, str] = {0: "truck", 1: "car", 2: "van", 3: "bus"}

# BGR colours matching visualize_annotations.py
_COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 255),    # truck  — red
    1: (0, 255, 0),    # car    — green
    2: (255, 0, 0),    # van    — blue
    3: (0, 255, 255),  # bus    — yellow
}

_FILL_VALUE: int = 114   # standard YOLOv8 letterbox grey
DEFAULT_IMGSZ: int = 640


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def letterbox(
    image: np.ndarray,
    target_size: int = DEFAULT_IMGSZ,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize *image* to *target_size* × *target_size* with letterbox padding.

    The image is scaled uniformly so that its longer side equals *target_size*.
    Grey padding (pixel value 114) is symmetrically added on the shorter sides.

    Args:
        image: HWC uint8 RGB image.
        target_size: Square output side length in pixels.

    Returns:
        padded: Padded image of shape ``(target_size, target_size, 3)``.
        scale: Uniform scale factor applied to the original image.
        pad: ``(left, top)`` padding in pixels added after scaling.
    """
    orig_h, orig_w = image.shape[:2]
    scale = target_size / max(orig_h, orig_w)
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw = target_size - new_w
    dh = target_size - new_h
    pad_left = dw // 2
    pad_top = dh // 2

    padded = cv2.copyMakeBorder(
        image,
        pad_top,
        dh - pad_top,
        pad_left,
        dw - pad_left,
        cv2.BORDER_CONSTANT,
        value=(_FILL_VALUE, _FILL_VALUE, _FILL_VALUE),
    )
    return padded, scale, (pad_left, pad_top)


def preprocess(
    image_bgr: np.ndarray,
    target_size: int = DEFAULT_IMGSZ,
) -> tuple[np.ndarray, float, tuple[int, int], tuple[int, int]]:
    """Convert a BGR image to a normalised float32 NCHW array for YOLOv8.

    Args:
        image_bgr: HWC uint8 BGR image as returned by ``cv2.imread``.
        target_size: Inference image side length (default 640).

    Returns:
        nchw: C-contiguous float32 array of shape
            ``(1, 3, target_size, target_size)`` with values in ``[0, 1]``.
        scale: Uniform scale factor used during letterboxing.
        pad: ``(left, top)`` padding in pixels applied after scaling.
        orig_shape: ``(height, width)`` of the original image.
    """
    orig_shape: tuple[int, int] = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    padded, scale, pad = letterbox(image_rgb, target_size)
    arr = padded.astype(np.float32) / 255.0
    nchw = np.ascontiguousarray(np.expand_dims(arr.transpose(2, 0, 1), 0))
    return nchw, scale, pad, orig_shape


# ---------------------------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------------------------


def nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> np.ndarray:
    """Greedy non-maximum suppression on ``x1 y1 x2 y2`` boxes.

    Args:
        boxes: ``(N, 4)`` float array of ``x1 y1 x2 y2`` coordinates.
        scores: ``(N,)`` float array of confidence scores.
        iou_threshold: Boxes whose IoU with the highest-scoring kept box
            exceeds this value are suppressed.
        max_det: Maximum number of candidates considered before NMS.

    Returns:
        Integer array of kept box indices sorted by descending score.
    """
    order = scores.argsort()[::-1][:max_det]
    keep: list[int] = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])

        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        area_rest = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        union = area_i + area_rest - inter
        iou = np.where(union > 0.0, inter / union, 0.0)
        order = rest[iou <= iou_threshold]

    return np.array(keep, dtype=np.int64)


# ---------------------------------------------------------------------------
# Postprocessing
# ---------------------------------------------------------------------------


def postprocess(
    raw_output: np.ndarray,
    scale: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> list[Detection]:
    """Decode raw YOLOv8 output into per-detection tuples in original image space.

    Expected *raw_output* layout (shape ``(1, 4 + nc, n_proposals)``):

    * ``[:, :4, :]`` — decoded ``cx cy w h`` in model-input pixel space.
    * ``[:, 4:, :]`` — per-class sigmoid scores.

    Args:
        raw_output: Float32 array of shape ``(1, 4 + nc, n_proposals)``.
        scale: Uniform scale applied during letterboxing.
        pad: ``(left, top)`` padding added during letterboxing.
        orig_shape: ``(height, width)`` of the original image.
        conf_threshold: Minimum per-class score to retain a detection.
        iou_threshold: IoU threshold for NMS.
        max_det: Maximum detections returned per image.

    Returns:
        List of :data:`Detection` tuples ``(class_id, conf, x1, y1, x2, y2)``
        with coordinates in original image pixels, sorted by descending confidence.
    """
    pred = raw_output[0].T  # (n_proposals, 4+nc)

    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]

    conf = class_scores.max(axis=1)
    cls = class_scores.argmax(axis=1)

    mask = conf >= conf_threshold
    boxes_cxcywh = boxes_cxcywh[mask]
    conf = conf[mask]
    cls = cls[mask]

    if len(conf) == 0:
        return []

    # cx, cy, w, h → x1, y1, x2, y2 (model pixel space)
    half_w = boxes_cxcywh[:, 2] / 2
    half_h = boxes_cxcywh[:, 3] / 2
    x1 = boxes_cxcywh[:, 0] - half_w
    y1 = boxes_cxcywh[:, 1] - half_h
    x2 = boxes_cxcywh[:, 0] + half_w
    y2 = boxes_cxcywh[:, 1] + half_h

    # Remove letterbox padding and rescale to original image pixel space
    pad_left, pad_top = pad
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top) / scale

    orig_h, orig_w = orig_shape
    x1 = np.clip(x1, 0.0, orig_w)
    y1 = np.clip(y1, 0.0, orig_h)
    x2 = np.clip(x2, 0.0, orig_w)
    y2 = np.clip(y2, 0.0, orig_h)

    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    keep = nms(boxes_xyxy, conf, iou_threshold=iou_threshold, max_det=max_det)

    boxes_xyxy = boxes_xyxy[keep]
    conf = conf[keep]
    cls = cls[keep]

    order = np.argsort(-conf)
    boxes_xyxy = boxes_xyxy[order]
    conf = conf[order]
    cls = cls[order]

    return [
        (int(cls[i]), float(conf[i]), float(x1), float(y1), float(x2), float(y2))
        for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy)
    ]


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------


def draw_detections(
    image_bgr: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """Draw bounding boxes and class labels onto *image_bgr*.

    Label format is ``<class_name> <conf>``.  The font scale and line
    thickness are derived from the shorter image dimension so annotations
    look consistent across different resolutions.

    Args:
        image_bgr: HWC uint8 BGR image to annotate (not modified in place).
        detections: Output of :func:`postprocess`.

    Returns:
        Annotated copy of *image_bgr*.
    """
    canvas = image_bgr.copy()
    short_side = min(canvas.shape[:2])
    font_scale = 0.5
    thickness = 2

    for cls_id, conf, x1, y1, x2, y2 in detections:
        color = _COLORS.get(cls_id, (128, 128, 128))
        label = f"{CLASS_NAMES.get(cls_id, str(cls_id))} {conf:.2f}"
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2), int(y2))

        cv2.rectangle(canvas, pt1, pt2, color, thickness=thickness)

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        # Place label above box; clamp to image top if needed
        label_y0 = max(pt1[1] - th - baseline - 4, 0)
        cv2.rectangle(
            canvas,
            (pt1[0], label_y0),
            (pt1[0] + tw + 2, label_y0 + th + baseline + 4),
            color,
            -1,
        )
        cv2.putText(
            canvas,
            label,
            (pt1[0] + 1, label_y0 + th + 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------------
# Config / filesystem helpers
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    """Load and return the YAML configuration file.

    Args:
        config_path: Path to ``config.yaml``.

    Returns:
        Parsed configuration dictionary (empty dict if file is blank).

    Raises:
        FileNotFoundError: If *config_path* does not exist.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open() as fh:
        return yaml.safe_load(fh) or {}


def collect_image_paths(directory: Path) -> list[Path]:
    """Return sorted image paths in *directory* matching supported extensions.

    Args:
        directory: Directory to scan for images.

    Returns:
        Sorted list of image :class:`~pathlib.Path` objects.
    """
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)
