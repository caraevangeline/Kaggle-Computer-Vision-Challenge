"""PyTorch inference for YOLOv8-nano with custom preprocessing and postprocessing.

Loads the model via the Ultralytics YOLO API for checkpoint handling only.
All letterbox resizing, tensor construction, NMS, and coordinate conversion
delegate to :mod:`_inference_utils` and are independent of the Ultralytics
inference pipeline.

Typical usage::

    python inference_pytorch.py                          # uses config.yaml defaults
    python inference_pytorch.py --weights best.pt --device cpu
    python inference_pytorch.py --test-dir data/val/images --output-dir out/
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import cv2
import torch
from ultralytics import YOLO

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
    """Determine the ``.pt`` weights path from CLI override or config.

    Search order:

    1. CLI ``--weights`` override.
    2. ``predict.weights`` key in config.
    3. ``runs/detect/<run_name>/weights/best.pt``.
    4. Most recently modified ``best.pt`` under ``runs/detect/``.

    Args:
        cfg: Parsed ``config.yaml`` dictionary.
        override: Optional path string from ``--weights`` CLI argument.

    Returns:
        Resolved :class:`~pathlib.Path` to the weights file.

    Raises:
        FileNotFoundError: If no weights file can be located.
    """
    if override:
        return Path(override)

    predict_cfg = cfg.get("predict", {})
    if predict_cfg.get("weights"):
        return Path(predict_cfg["weights"])

    paths_cfg = cfg.get("paths", {})
    runs_root = Path(paths_cfg.get("runs_detect_root", "runs/detect"))
    run_name = cfg.get("training", {}).get("run_name", "")
    candidate = runs_root / run_name / "weights" / "best.pt"
    if candidate.exists():
        return candidate

    best_files = sorted(runs_root.rglob("best.pt"), key=lambda p: p.stat().st_mtime)
    if not best_files:
        raise FileNotFoundError(f"No best.pt found under {runs_root}")
    return best_files[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_model(
    weights_path: Path,
    device: int | str = 0,
) -> tuple[torch.nn.Module, torch.device]:
    """Load a YOLOv8-nano checkpoint and return the underlying PyTorch module.

    The Ultralytics :class:`~ultralytics.YOLO` wrapper is used only for
    checkpoint loading.  The returned module is placed in evaluation mode so
    that its :meth:`forward` produces decoded predictions rather than training
    targets.

    Args:
        weights_path: Path to the ``.pt`` weights file.
        device: CUDA device index (``int``) or ``"cpu"``.

    Returns:
        A ``(torch_module, device)`` 2-tuple ready for manual forward passes.

    Raises:
        FileNotFoundError: If *weights_path* does not exist.
    """
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    yolo = YOLO(str(weights_path))
    torch_model: torch.nn.Module = yolo.model

    if isinstance(device, int):
        resolved = torch.device(f"cuda:{device}" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)

    torch_model = torch_model.to(resolved).eval()
    return torch_model, resolved


def predict_image(
    torch_model: torch.nn.Module,
    device: torch.device,
    image_path: Path,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.7,
    imgsz: int = DEFAULT_IMGSZ,
    max_det: int = 300,
) -> list[Detection]:
    """Run inference on a single image and return detections in pixel space.

    Performs letterbox preprocessing, a direct forward pass through the raw
    PyTorch module, and custom NMS + coordinate postprocessing without invoking
    any Ultralytics inference utilities.

    In eval mode the Ultralytics ``DetectionModel`` returns a
    ``(decoded, features)`` tuple; only the decoded tensor of shape
    ``(1, 4 + nc, n_proposals)`` is consumed.

    Args:
        torch_model: PyTorch module in eval mode (from :func:`load_model`).
        device: Device on which to run inference.
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
    tensor = torch.from_numpy(nchw.copy()).to(device)

    with torch.inference_mode():
        output = torch_model(tensor)

    # Eval mode returns (decoded_tensor, raw_features); export mode returns
    # the tensor directly.  Normalise to always obtain the decoded tensor.
    decoded: torch.Tensor = output[0] if isinstance(output, (tuple, list)) else output

    return postprocess(
        decoded.cpu().numpy(),
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
    """Run inference on test images and save annotated results to a directory."""
    parser = argparse.ArgumentParser(
        description="YOLOv8-nano PyTorch inference — saves annotated images.",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--weights", default=None, help="Model weights path (.pt)")
    parser.add_argument("--test-dir", default=None, help="Input image directory")
    parser.add_argument("--output-dir", default=None, help="Output directory for annotated images")
    parser.add_argument("--device", default=None, help="Device: GPU index (int) or 'cpu'")
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

    raw_device = args.device or predict_cfg.get("device", 0)
    device_arg: int | str = int(raw_device) if str(raw_device).isdigit() else str(raw_device)

    test_dir = Path(args.test_dir or paths_cfg.get("test_images", "data/test/images"))
    output_dir = Path(args.output_dir or paths_cfg.get("output_dir", "output/annotated"))
    weights_path = _resolve_weights(cfg, args.weights)

    print(f"Weights   : {weights_path}")
    print(f"Test dir  : {test_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Device    : {device_arg}  conf={conf_threshold}  iou={iou_threshold}  imgsz={imgsz}")

    torch_model, device = load_model(weights_path, device=device_arg)

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
            torch_model,
            device,
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
