#!/usr/bin/env python3
"""Run inference on test images and write a Kaggle submission CSV.

Two pipelines are available, controlled by ``predict.pipeline`` in config.yaml:

- ``memory`` (default): chunked in-GPU prediction — best for ~8 GB GPUs after
  training because it avoids ``save_txt`` stream spikes.
- ``txt``: Ultralytics ``save_txt`` export followed by label-file parsing.
  Slightly slower but produces a persistent label directory on disk.

All paths and hyperparameters are read from ``config.yaml``.
"""

from __future__ import annotations

import csv
import gc
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tlc_ultralytics import YOLO

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x: Any, **kwargs: Any) -> Any:  # type: ignore[misc]
        return x

WORK_DIR: Path = Path(__file__).resolve().parent
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# ---------------------------------------------------------------------------
# Ultralytics 8.3+ compatibility shim (identical to train.py)
# ---------------------------------------------------------------------------

def _apply_ultralytics_83_compat() -> None:
    """Patch the 3LC validator for dict-style predictions (Ultralytics 8.3+).

    Must be called before constructing a YOLO model so that validation and
    collection paths can handle dict-style predictions. Idempotent.
    """
    from tlc_ultralytics.detect import validator as det_val

    if getattr(det_val.TLCDetectionValidator, "_ua_detrac_compat_patched", False):
        return

    original = det_val.TLCDetectionValidator._process_detection_predictions

    def _pred_dict_to_tensor(predictions: Any) -> torch.Tensor:
        if isinstance(predictions, torch.Tensor):
            return predictions
        bboxes: torch.Tensor = predictions["bboxes"]
        conf: torch.Tensor = predictions["conf"]
        cls: torch.Tensor = predictions["cls"]
        if bboxes.shape[0] == 0:
            return bboxes.new_empty((0, 6))
        return torch.cat([bboxes, conf.unsqueeze(1), cls.unsqueeze(1).float()], dim=1)

    def patched(self: Any, preds: list[Any], batch: dict[str, Any]) -> list[Any]:
        from tlc_ultralytics.detect.utils import construct_bbox_struct
        from ultralytics.utils import metrics, ops

        predicted_boxes: list[Any] = []
        for i, predictions in enumerate(preds):
            predictions = _pred_dict_to_tensor(predictions)
            ori_shape = batch["ori_shape"][i]
            resized_shape = batch["resized_shape"][i]
            ratio_pad = batch["ratio_pad"][i]
            height, width = ori_shape

            if len(predictions) == 0:
                predicted_boxes.append(
                    construct_bbox_struct([], image_width=width, image_height=height)
                )
                continue

            predictions = predictions.clone()
            predictions = predictions[predictions[:, 4] > self._settings.conf_thres]
            predictions = predictions[
                predictions[:, 4].argsort(descending=True)[: self._settings.max_det]
            ]

            pred_box = predictions[:, :4].clone()
            pred_scaled = ops.scale_boxes(resized_shape, pred_box, ori_shape, ratio_pad)

            pbatch = self._prepare_batch(i, batch)
            gt_bbox = pbatch.get("bbox", pbatch.get("bboxes"))
            if gt_bbox is not None and gt_bbox.shape[0]:
                ious = metrics.box_iou(gt_bbox, pred_scaled)
                box_ious: list[float] = ious.max(dim=0)[0].cpu().tolist()
            else:
                box_ious = [0.0] * pred_scaled.shape[0]

            pred_xywh = ops.xyxy2xywhn(pred_scaled, w=width, h=height)
            conf_list: list[float] = predictions[:, 4].cpu().tolist()
            pred_cls_list: list[float] = predictions[:, 5].cpu().tolist()

            annotations = [
                {
                    "score": conf_list[pi],
                    "category_id": self.data["range_to_3lc_class"][int(pred_cls_list[pi])],
                    "bbox": pred_xywh[pi, :].cpu().tolist(),
                    "iou": box_ious[pi],
                }
                for pi in range(len(predictions))
            ]
            predicted_boxes.append(
                construct_bbox_struct(annotations, image_width=width, image_height=height)
            )

        return predicted_boxes

    det_val.TLCDetectionValidator._process_detection_predictions = patched
    det_val.TLCDetectionValidator._ua_detrac_compat_patched = True
    det_val.TLCDetectionValidator._ua_detrac_compat_original = original


# ---------------------------------------------------------------------------
# Config + weight resolution
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    """Load ``config.yaml`` from WORK_DIR and return it as a dict."""
    import yaml

    p = WORK_DIR / "config.yaml"
    if not p.is_file():
        print(f"ERROR: Missing {p}", file=sys.stderr)
        sys.exit(1)
    with p.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _resolve(path_str: str) -> Path:
    """Resolve *path_str* relative to WORK_DIR if it is not already absolute."""
    p = Path(path_str)
    return p if p.is_absolute() else (WORK_DIR / p).resolve()


def _resolve_weights(cfg: dict[str, Any]) -> Path:
    """Find the best available checkpoint, preferring config-specified paths.

    Search order:
    1. ``predict.weights`` in config (if set and exists).
    2. ``runs/detect/<run_name>/weights/best.pt``.
    3. Most recently modified ``best.pt`` anywhere under ``runs/detect/``.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    paths_cfg: dict[str, Any] = cfg.get("paths", {})
    training: dict[str, Any] = cfg.get("training", {})
    predict_cfg: dict[str, Any] = cfg.get("predict", {})

    runs_root = str(paths_cfg.get("runs_detect_root", "runs/detect"))
    run_name = str(training.get("run_name", "yolov8n_baseline"))

    explicit = predict_cfg.get("weights")
    if explicit:
        p = _resolve(str(explicit))
        if p.is_file():
            return p

    primary = WORK_DIR / runs_root / run_name / "weights" / "best.pt"
    if primary.is_file():
        return primary.resolve()

    candidates = sorted(
        (WORK_DIR / runs_root).glob("*/weights/best.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0].resolve()

    raise FileNotFoundError(
        "No weights found. Run train.py first or set predict.weights in config.yaml."
    )


# ---------------------------------------------------------------------------
# Prediction-string helpers
# ---------------------------------------------------------------------------

def _label_file_to_prediction_string(label_path: Path) -> str:
    """Convert a saved Ultralytics label file (cls cx cy w h conf) to a submission string.

    Returns ``"no box"`` for empty or missing files.
    """
    if not label_path.is_file() or label_path.stat().st_size == 0:
        return "no box"
    tokens: list[str] = []
    with label_path.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 6:
                cls_id, xc, yc, w, h, conf = parts[:6]
                tokens.append(f"{cls_id} {conf} {xc} {yc} {w} {h}")
    return " ".join(tokens) if tokens else "no box"


def _result_to_prediction_string(result: Any) -> str:
    """Convert an Ultralytics result object to a submission prediction string.

    Format: ``cls conf cx cy w h`` repeated and space-separated, sorted by
    descending confidence. Returns ``"no box"`` when there are no detections.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return "no box"

    xywhn: np.ndarray = boxes.xywhn.cpu().numpy()
    cls: np.ndarray = boxes.cls.cpu().numpy().astype(int)
    conf: np.ndarray = boxes.conf.cpu().numpy()
    order: np.ndarray = np.argsort(-conf)

    parts: list[str] = []
    for i in order:
        c = int(cls[i])
        cf = float(conf[i])
        x, y, w, h = (float(v) for v in xywhn[i])
        parts.extend([
            str(c),
            f"{cf:.6f}",
            f"{min(1.0, max(0.0, x)):.6f}",
            f"{min(1.0, max(0.0, y)):.6f}",
            f"{min(1.0, max(0.0, w)):.6f}",
            f"{min(1.0, max(0.0, h)):.6f}",
        ])
    return " ".join(parts)


def _find_image(test_dir: Path, stem: str) -> Path | None:
    """Return the first image file matching *stem* in *test_dir*, or ``None``."""
    return next(
        (test_dir / f"{stem}{ext}" for ext in IMAGE_EXTS if (test_dir / f"{stem}{ext}").is_file()),
        None,
    )


# ---------------------------------------------------------------------------
# Inference pipelines
# ---------------------------------------------------------------------------

def _pipeline_txt(
    model: YOLO,
    test_dir: Path,
    stems: list[str],
    pred_dir_name: str,
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device: Any,
    max_det: int,
    batch: int,
) -> dict[str, str]:
    """Run inference using ``save_txt`` and convert the label files to prediction strings.

    Args:
        model:        Loaded YOLO model.
        test_dir:     Directory containing test images.
        stems:        Image filename stems (no extension) from sample_submission.csv.
        pred_dir_name: Temporary Ultralytics output directory name.
        imgsz:        Inference image size.
        conf:         Confidence threshold.
        iou:          NMS IoU threshold.
        device:       PyTorch device identifier.
        max_det:      Maximum detections per image.
        batch:        Inference batch size.

    Returns:
        Mapping of stem → prediction string.
    """
    pred_root = WORK_DIR / pred_dir_name
    if pred_root.exists():
        shutil.rmtree(pred_root)

    paths = [p for stem in stems if (p := _find_image(test_dir, stem)) is not None]
    if not paths:
        raise RuntimeError(f"No test images found under {test_dir}")

    results = model.predict(
        source=[str(p) for p in paths],
        save=False,
        save_txt=True,
        save_conf=True,
        conf=float(conf),
        iou=float(iou),
        imgsz=int(imgsz),
        device=device,
        max_det=int(max_det),
        batch=max(1, int(batch)),
        project=str(WORK_DIR),
        name=pred_dir_name,
        exist_ok=False,
        verbose=False,
        stream=True,
    )
    for _ in tqdm(results, total=len(paths), desc="Predicting", unit="img"):
        pass

    labels_dir = pred_root / "labels"
    pred_by_stem: dict[str, str] = {
        stem: _label_file_to_prediction_string(labels_dir / f"{stem}.txt")
        for stem in stems
    }

    if pred_root.exists():
        shutil.rmtree(pred_root)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return pred_by_stem


def _pipeline_memory(
    model: YOLO,
    test_dir: Path,
    stems: list[str],
    *,
    imgsz: int,
    conf: float,
    iou: float,
    device: Any,
    max_det: int,
    batch: int,
) -> dict[str, str]:
    """Run inference in GPU-memory chunks and convert results to prediction strings.

    Args:
        model:    Loaded YOLO model.
        test_dir: Directory containing test images.
        stems:    Image filename stems from sample_submission.csv.
        imgsz:    Inference image size.
        conf:     Confidence threshold.
        iou:      NMS IoU threshold.
        device:   PyTorch device identifier.
        max_det:  Maximum detections per image.
        batch:    Chunk size for batched prediction.

    Returns:
        Mapping of stem → prediction string.
    """
    paths = [p for stem in stems if (p := _find_image(test_dir, stem)) is not None]
    bs = max(1, int(batch))
    pred_by_stem: dict[str, str] = {}

    for start in tqdm(range(0, len(paths), bs), desc="Predicting", unit="batch"):
        chunk = paths[start : start + bs]
        results = model.predict(
            source=[str(p) for p in chunk],
            imgsz=int(imgsz),
            conf=float(conf),
            iou=float(iou),
            device=device,
            max_det=int(max_det),
            batch=len(chunk),
            verbose=False,
        )
        for res, p in zip(results, chunk):
            pred_by_stem[p.stem] = _result_to_prediction_string(res)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return pred_by_stem


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Load config, run inference, and write submission.csv. Returns exit code."""
    os.chdir(WORK_DIR)
    _apply_ultralytics_83_compat()
    cfg = _load_config()

    paths_cfg: dict[str, Any] = cfg.get("paths", {})
    training: dict[str, Any] = cfg.get("training", {})
    predict_cfg: dict[str, Any] = cfg.get("predict", {})

    sample_path = _resolve(str(paths_cfg.get("sample_submission", "sample_submission.csv")))
    out_path    = _resolve(str(paths_cfg.get("submission_csv", "submission.csv")))
    test_dir    = _resolve(str(paths_cfg.get("test_images", "data/test/images")))

    conf     = float(predict_cfg.get("conf", 0.25))
    iou      = float(predict_cfg.get("iou", 0.7))
    imgsz    = int(predict_cfg.get("imgsz") or training.get("image_size", 640))
    device   = predict_cfg.get("device") if predict_cfg.get("device") is not None else training.get("device", 0)
    max_det  = int(predict_cfg.get("max_det", 300))
    batch    = int(predict_cfg.get("batch", 1))
    pipeline = str(predict_cfg.get("pipeline", "memory")).lower()
    pred_dir = str(predict_cfg.get("pred_dir_name", "predictions"))

    print("=" * 70)
    print("PREDICTIONS → SUBMISSION CSV")
    print("=" * 70)

    if not sample_path.is_file():
        print(f"ERROR: sample submission not found: {sample_path}", file=sys.stderr)
        return 1
    if not test_dir.is_dir():
        print(f"ERROR: test images directory not found: {test_dir}", file=sys.stderr)
        return 1

    try:
        weights = _resolve_weights(cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    with sample_path.open(newline="", encoding="utf-8") as fh:
        rows_in: list[dict[str, str]] = list(csv.DictReader(fh))

    stems = [r["image_id"].strip() for r in rows_in]

    print(f"\n  Rows (sample) : {len(rows_in)}")
    print(f"  Test dir      : {test_dir}")
    print(f"  Weights       : {weights}")
    print(f"  Pipeline      : {pipeline}")
    print(f"  conf={conf}, iou={iou}, imgsz={imgsz}, batch={batch}")

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = YOLO(str(weights))
    print("  Model loaded.")

    pipeline_kwargs: dict[str, Any] = dict(
        imgsz=imgsz, conf=conf, iou=iou, device=device, max_det=max_det, batch=batch
    )
    if pipeline == "txt":
        pred_by_stem = _pipeline_txt(model, test_dir, stems, pred_dir, **pipeline_kwargs)
    else:
        pred_by_stem = _pipeline_memory(model, test_dir, stems, **pipeline_kwargs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "image_id", "prediction_string"])
        writer.writeheader()
        writer.writerows(
            tqdm(
                (
                    {
                        "id": row["id"],
                        "image_id": stem,
                        "prediction_string": pred_by_stem.get(stem, "no box"),
                    }
                    for row, stem in zip(rows_in, stems)
                ),
                total=len(rows_in),
                desc="Writing CSV",
                unit="row",
            )
        )

    nonempty = sum(1 for s in stems if pred_by_stem.get(s, "no box") != "no box")
    n_boxes = sum(
        len(pred_by_stem[s].split()) // 6
        for s in stems
        if pred_by_stem.get(s, "no box") != "no box"
    )

    print("\n" + "=" * 70)
    print("OK — SUBMISSION READY")
    print("=" * 70)
    print(f"  File              : {out_path}")
    print(f"  Images with ≥1 box: {nonempty} / {len(rows_in)}")
    print(f"  Total boxes       : {n_boxes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
