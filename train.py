#!/usr/bin/env python3
"""Train YOLOv8n on 3LC train/val tables. All settings come from config.yaml.

Competition rules enforced here:
- YOLOv8n only (``config training.model``).
- Train from scratch only (``yolov8n.yaml``); COCO or other pretrained
  checkpoints are not permitted.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tlc
from tlc_ultralytics import YOLO, Settings

WORK_DIR: Path = Path(__file__).resolve().parent
_LOCKED_ARCH: str = "yolov8n"
_YOLOV8N_YAML: str = "yolov8n.yaml"


# ---------------------------------------------------------------------------
# Ultralytics 8.3+ compatibility shim
# ---------------------------------------------------------------------------

def _apply_ultralytics_83_compat() -> None:
    """Patch the 3LC detection validator for Ultralytics 8.3+ dict-style predictions.

    In some Ultralytics versions, detection predictions arrive as a dict with
    keys ``bboxes``, ``conf``, and ``cls`` instead of a single Nx6 tensor. The
    3LC validator expects a tensor; this shim normalises the input before
    passing it on. It is idempotent — applying it twice is safe.
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
# Helpers
# ---------------------------------------------------------------------------

def _load_config() -> dict[str, Any]:
    """Load and return the ``config.yaml`` from WORK_DIR. Exits on missing file."""
    import yaml

    cfg_path = WORK_DIR / "config.yaml"
    if not cfg_path.is_file():
        print(f"ERROR: Missing {cfg_path}", file=sys.stderr)
        sys.exit(1)
    with cfg_path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _normalize_model_stem(name: str) -> str:
    """Strip known extensions from a model name and return a lowercase stem."""
    s = str(name).lower().strip()
    for suffix in (".pt", ".yaml", ".yml"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def _assert_yolov8n_only(training: dict[str, Any]) -> None:
    """Raise SystemExit if ``training.model`` is set to anything other than YOLOv8n."""
    raw = training.get("model")
    if raw is None:
        return
    stem = _normalize_model_stem(str(raw))
    if stem != _LOCKED_ARCH:
        raise SystemExit(
            f"train.py is locked to YOLOv8n only. config training.model={raw!r} "
            f"resolves to {stem!r}. Remove training.model or set it to yolov8n."
        )


def _reject_pretrained_config(training: dict[str, Any]) -> None:
    """Raise SystemExit if ``training.pretrained`` is truthy."""
    if training.get("pretrained"):
        raise SystemExit(
            "Competition starter does not allow pretrained weights. Remove "
            "`training.pretrained` from config.yaml (training is always from scratch "
            f"via {_YOLOV8N_YAML})."
        )


def _apply_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility."""
    if seed is None:
        return
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _check_umap(emb_dim: int, reducer: str) -> None:
    """Fail fast if ``umap-learn`` is required but not installed."""
    if emb_dim <= 0 or reducer != "umap":
        return
    try:
        import umap  # noqa: F401
    except ImportError:
        raise SystemExit(
            "\n  umap-learn is required for image embeddings but is not installed.\n"
            "  Fix:  pip install umap-learn\n"
            "  Or:   set image_embeddings_dim: 0 in config.yaml to skip embeddings.\n"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse config, set up 3LC, and launch YOLOv8n training. Returns exit code."""
    os.chdir(WORK_DIR)
    _apply_ultralytics_83_compat()
    cfg = _load_config()

    tlc_cfg: dict[str, Any] = cfg.get("tlc", {})
    training: dict[str, Any] = cfg.get("training", {})
    repro: dict[str, Any] = cfg.get("reproducibility", {})

    _assert_yolov8n_only(training)
    _reject_pretrained_config(training)
    _apply_seed(repro.get("seed"))

    project_name = str(tlc_cfg.get("project_name", "ua_detrac_vehicle_detection"))
    dataset_name = str(tlc_cfg.get("dataset_name", "ua_detrac_10k"))
    train_name = str(tlc_cfg.get("train_table_name", f"{dataset_name}-train"))
    val_name = str(tlc_cfg.get("val_table_name", f"{dataset_name}-val"))
    emb_dim = int(tlc_cfg.get("image_embeddings_dim", 3))
    emb_reducer = str(tlc_cfg.get("image_embeddings_reducer", "umap"))
    _check_umap(emb_dim, emb_reducer)

    run_name = str(training.get("run_name", "yolov8n_baseline"))
    run_desc = str(training.get("run_description", ""))
    epochs = int(training.get("epochs", 10))
    batch_size = int(training.get("batch_size", 16))
    image_size = int(training.get("image_size", 640))
    device = training.get("device", 0)
    workers = int(training.get("workers", 4))
    lr0 = float(training.get("lr0", 0.01))
    use_aug = bool(training.get("use_augmentation", True))

    print("=" * 70)
    print("TRAINING (YOLOv8n + 3LC)")
    print("=" * 70)
    print(f"\n  PyTorch: {torch.__version__}, 3LC: {tlc.__version__}")
    print(f"  CUDA   : {torch.cuda.is_available()}")

    print("\n  Loading tables (.latest() — includes Dashboard edits)...")
    train_table = tlc.Table.from_names(
        project_name=project_name, dataset_name=dataset_name, table_name=train_name
    ).latest()
    val_table = tlc.Table.from_names(
        project_name=project_name, dataset_name=dataset_name, table_name=val_name
    ).latest()
    print(f"  Train: {len(train_table)} | Val: {len(val_table)}")

    tables_used = WORK_DIR / "tables_used.txt"
    tables_used.write_text(
        f"train_url={train_table.url}\nval_url={val_table.url}\n",
        encoding="utf-8",
    )
    print(f"  Wrote {tables_used}")

    print(
        f"\n  Model  : YOLOv8n from scratch ({_YOLOV8N_YAML})"
        f" | run: {run_name} | epochs: {epochs}"
        f" | batch: {batch_size} | imgsz: {image_size} | aug: {use_aug}"
    )

    settings = Settings(
        project_name=project_name,
        run_name=run_name,
        run_description=run_desc,
        image_embeddings_dim=emb_dim,
        image_embeddings_reducer=emb_reducer,
    )

    model = YOLO(_YOLOV8N_YAML)
    train_args: dict[str, Any] = {
        "tables": {"train": train_table, "val": val_table},
        "name": run_name,
        "epochs": epochs,
        "imgsz": image_size,
        "batch": batch_size,
        "device": device,
        "workers": workers,
        "lr0": lr0,
        "settings": settings,
        "val": True,
    }
    if use_aug:
        train_args.update({"mosaic": 1.0, "mixup": 0.05, "copy_paste": 0.1})

    model.train(**train_args)

    runs_root = cfg.get("paths", {}).get("runs_detect_root", "runs/detect")
    best_weights = WORK_DIR / runs_root / run_name / "weights" / "best.pt"
    print("\n" + "=" * 70)
    print("OK — TRAINING COMPLETE")
    print("=" * 70)
    print(f"\n  Weights: {best_weights}")
    print("  Next   : python predict.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
