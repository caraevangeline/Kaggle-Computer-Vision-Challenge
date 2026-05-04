"""ONNX Runtime inference for YOLOv8-nano, fully independent of Ultralytics.

All letterbox resizing, NMS, and coordinate conversion delegate to
:mod:`_inference_utils` and are implemented with NumPy / OpenCV only.
The Ultralytics package is not imported at any point.

Export a compatible ONNX model from a trained ``.pt`` checkpoint with::

    from ultralytics import YOLO
    YOLO("best.pt").export(format="onnx", imgsz=640, simplify=True)

This produces ``best.onnx`` with a single output of shape
``(1, 4 + nc, 8400)`` containing decoded ``cx cy w h`` coordinates (in
model-input pixel space) followed by per-class sigmoid scores.

Typical usage::

    python inference_onnx.py --weights best.onnx
    python inference_onnx.py --weights best.onnx --device cpu
    python inference_onnx.py --test-dir data/val/images --output-dir out/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import onnxruntime as ort

from utils.ops import (
    DEFAULT_IMGSZ,
    Detection,
    collect_image_paths,
    draw_detections,
    load_config,
    postprocess,
    preprocess,
)


# ---------------------------------------------------------------------------
# Weights resolution
# ---------------------------------------------------------------------------


def _resolve_weights(cfg: dict, override: Optional[str]) -> Path:
    """Determine the ``.onnx`` model path from CLI override or config.

    Search order:

    1. CLI ``--weights`` override.
    2. ``predict.onnx_weights`` key in config.
    3. ``runs/detect/<run_name>/weights/best.onnx``.
    4. Most recently modified ``best.onnx`` under ``runs/detect/``.

    Args:
        cfg: Parsed ``config.yaml`` dictionary.
        override: Optional path string from ``--weights`` CLI argument.

    Returns:
        Resolved :class:`~pathlib.Path` to the ONNX model file.

    Raises:
        FileNotFoundError: If no ``.onnx`` file can be located.
    """
    if override:
        return Path(override)

    predict_cfg = cfg.get("predict", {})
    if predict_cfg.get("onnx_weights"):
        return Path(predict_cfg["onnx_weights"])

    paths_cfg = cfg.get("paths", {})
    runs_root = Path(paths_cfg.get("runs_detect_root", "runs/detect"))
    run_name = cfg.get("training", {}).get("run_name", "")
    candidate = runs_root / run_name / "weights" / "best.onnx"
    if candidate.exists():
        return candidate

    onnx_files = sorted(runs_root.rglob("best.onnx"), key=lambda p: p.stat().st_mtime)
    if not onnx_files:
        raise FileNotFoundError(
            f"No best.onnx found under {runs_root}. "
            "Export one with: YOLO('best.pt').export(format='onnx', imgsz=640)"
        )
    return onnx_files[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_session(
    onnx_path: Path,
    providers: Optional[list[str]] = None,
) -> ort.InferenceSession:
    """Load a YOLOv8-nano ONNX model and return an inference session.

    Args:
        onnx_path: Path to the ``.onnx`` model file.
        providers: ONNX Runtime execution providers in priority order.
            Defaults to ``["CUDAExecutionProvider", "CPUExecutionProvider"]``.

    Returns:
        Configured :class:`onnxruntime.InferenceSession`.

    Raises:
        FileNotFoundError: If *onnx_path* does not exist.
    """
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    if providers is None:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    print(f"ONNX Runtime active providers: {session.get_providers()}")
    return session


def predict_image(
    session: ort.InferenceSession,
    image_path: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    imgsz: int = DEFAULT_IMGSZ,
    max_det: int = 300,
) -> list[Detection]:
    """Run ONNX inference on a single image and return detections in pixel space.

    Performs letterbox preprocessing, a single ONNX Runtime forward pass, and
    custom NMS + coordinate postprocessing without any dependency on Ultralytics
    or PyTorch.

    Args:
        session: Loaded ONNX Runtime session (from :func:`load_session`).
        image_path: Path to the input image.
        conf_threshold: Minimum per-class confidence to keep a detection.
        iou_threshold: IoU threshold for NMS.
        imgsz: Square inference image side length.
        max_det: Maximum detections per image.

    Returns:
        List of :data:`~_inference_utils.Detection` tuples
        ``(class_id, conf, x1, y1, x2, y2)`` in original image pixel
        coordinates, sorted by descending confidence.

    Raises:
        ValueError: If the image cannot be read from *image_path*.
    """
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"Cannot read image: {image_path}")

    nchw, scale, pad, orig_shape = preprocess(image_bgr, target_size=imgsz)

    input_name: str = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: nchw})[0]  # (1, 4+nc, n_proposals)

    return postprocess(
        raw_output,
        scale=scale,
        pad=pad,
        orig_shape=orig_shape,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
        max_det=max_det,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run ONNX inference on test images and save annotated results to a directory."""
    parser = argparse.ArgumentParser(
        description="YOLOv8-nano ONNX inference — saves annotated images.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--weights", default=None, help="ONNX model path (.onnx)")
    parser.add_argument("--test-dir", default=None, help="Input image directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for annotated images")
    parser.add_argument(
        "--device",
        default=None,
        help="Execution provider: 'cuda' (default) or 'cpu'",
    )
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=None, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=None, help="Inference image size")
    args = parser.parse_args()

    cfg = load_config(Path(args.config))
    predict_cfg = cfg.get("predict", {})
    paths_cfg = cfg.get("paths", {})

    conf_threshold: float = args.conf or predict_cfg.get("conf", 0.25)
    iou_threshold: float = args.iou or predict_cfg.get("iou", 0.7)
    imgsz: int = args.imgsz or predict_cfg.get("imgsz", DEFAULT_IMGSZ)
    max_det: int = predict_cfg.get("max_det", 300)

    raw_device: str = str(args.device or predict_cfg.get("device", "cuda")).lower()
    providers = (
        ["CPUExecutionProvider"]
        if raw_device == "cpu"
        else ["CUDAExecutionProvider", "CPUExecutionProvider"]
    )

    test_dir = Path(args.test_dir or paths_cfg.get("test_images", "data/test/images"))
    output_dir = Path(args.output_dir or paths_cfg.get("output_dir", "output/annotated"))
    onnx_path = _resolve_weights(cfg, args.weights)

    print(f"ONNX model: {onnx_path}")
    print(f"Test dir  : {test_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Providers : {providers}  conf={conf_threshold}  iou={iou_threshold}  imgsz={imgsz}")

    session = load_session(onnx_path, providers=providers)

    image_paths = collect_image_paths(test_dir)
    if not image_paths:
        raise RuntimeError(f"No images found in {test_dir}")
    print(f"Found {len(image_paths)} image(s).")

    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_paths:
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"  [skip] cannot read {img_path.name}")
            continue

        detections = predict_image(
            session,
            img_path,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
            max_det=max_det,
        )
        annotated = draw_detections(image_bgr, detections)
        out_path = output_dir / img_path.name
        cv2.imwrite(str(out_path), annotated)
        print(f"  {img_path.name}: {len(detections)} detection(s) → {out_path}")

    print(f"Done. Annotated images saved to {output_dir}")


if __name__ == "__main__":
    main()
