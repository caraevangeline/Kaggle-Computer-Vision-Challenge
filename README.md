# UA-DETRAC Vehicle Detection — Starter Kit

Real-time vehicle detection across four classes — **truck, car, van, bus** — on overhead traffic camera footage from the [UA-DETRAC](http://detrac-db.rit.albany.edu/) benchmark, evaluated at COCO mAP (IoU 0.50:0.95).

**YOLOv8n only · 3LC tables · CSV submission format**

---

## Quick start

```bash
python verify_setup.py       # check environment before anything else
3lc login YOUR_API_KEY        # one-time per machine
python register_tables.py     # create 3LC tables (train + val)
python train.py               # train YOLOv8n from scratch
python predict.py             # generate submission.csv
# Upload submission.csv to Kaggle
```

### Expected timings

| Step | GPU (~8 GB VRAM) | CPU only |
|------|-----------------|----------|
| `verify_setup.py` | ~2 sec | ~2 sec |
| `register_tables.py` | ~15 sec | ~15 sec |
| `train.py` (10 epochs) | **~25 min** | **~2–4 hours** |
| `predict.py` (982 images) | ~25 sec | ~5 min |

If training seems stuck at the start, **wait 60 seconds** — PyTorch, 3LC, and data-loader workers need time to initialize. This is normal.

**CPU-only?** See the commented-out overrides at the bottom of `config.yaml` for recommended settings.

---

## Installation

Install the packages listed on the competition **Environment Setup** page:

```bash
pip install 3lc-ultralytics umap-learn torch PyYAML tqdm opencv-python
```

Git is not required for participants running the kit locally.

---

## The core insight: label quality beats model capacity

Most teams iterate on model architecture. This pipeline iterates on **data quality** — the more leveraged variable at this scale.

The dataset labels originate from an automated pipeline and carry systematic noise: missed detections, loose boundaries, and class confusion between visually similar vehicles (van vs. truck). Training directly on this noise teaches the model to reproduce it. The solution is to treat annotation correctness as a first-class engineering problem, not a one-time preprocessing step.

---

## Pipeline

```
Raw YOLO labels
      │
      ▼
Register with 3LC ──► Dashboard: visualise, filter, edit annotations
      │
      ▼
Train YOLOv8n (round N)
      │
      ▼  per-sample metrics + embeddings written back to 3LC
Co-DETR pseudo-labels ──► diff against current labels ──► promote high-conf corrections
      │
      ▼
Corrected table revision (picked up automatically via .latest())
      │
      └──► Train round N+1  →  repeat until mAP plateaus
```

Each round produces a better model, which produces better pseudo-labels for the next round — a self-reinforcing loop that converges toward clean annotations without manually inspecting every image.

---

## Folder layout

```
├── README.md
├── config.yaml              # Paths + hyperparameters (edit here only)
├── dataset.yaml             # YOLO paths → data/train, data/val, data/test
├── sample_submission.csv    # Official row ids + image_ids (from competition Data tab)
├── verify_setup.py          # Pre-flight environment check (run first!)
├── register_tables.py       # Create 3LC train/val tables
├── train.py                 # Train YOLOv8n from scratch via 3LC
├── predict.py               # Run inference and write submission.csv
├── visualize_annotations.py # Spot-check ground-truth labels before training
└── data/
    ├── train/
    │   ├── images/          # *.jpg (or .png)
    │   └── labels/          # one .txt per image (YOLO: class xc yc w h)
    ├── val/
    │   ├── images/
    │   └── labels/
    └── test/
        └── images/          # test only — no labels folder for participants
```

All scripts resolve paths relative to the folder that contains this README (`Path(__file__).parent`). Run from that directory:

```bash
cd /path/to/this/folder
python train.py
```

On **Kaggle Notebooks**, after copying or unzipping the kit:

```python
import os
os.chdir("/kaggle/working/starter")  # path where these files live
```

---

## Scripts

### `verify_setup.py`
Pre-flight check — validates Python version, package imports, GPU availability, and that all required data files are present. Fix every `[FAIL]` before running anything else.

### `register_tables.py`
Registers `data/train` and `data/val` into 3LC tables. Idempotent — safe to re-run without overwriting. Test images are **not** registered; they stay on disk.

### `train.py`
Trains YOLOv8n from scratch using the 3LC train/val tables. Always calls `.latest()` on both tables, so any corrections made via the 3LC Dashboard are automatically included in the next training run without file management. Weights land at `runs/detect/<run_name>/weights/best.pt`.

Competition rules enforced in code: YOLOv8n architecture only, no pretrained weights.

### `predict.py`
Reads `sample_submission.csv`, runs inference over `data/test/images`, and writes `submission.csv`. Default pipeline is `memory` (suitable for ~8 GB GPUs). Switch to `pipeline: txt` in `config.yaml` if you hit memory issues.

### `visualize_annotations.py`
Draws ground-truth YOLO boxes on a random sample of training images — useful for spot-checking label quality before investing in a training run.

```bash
python visualize_annotations.py --n 20              # show 20 images interactively
python visualize_annotations.py --n 50 --no-show --save-dir out/  # save to disk instead
```

Classes and colors: truck (red), car (green), van (blue), bus (yellow).

---

## Configuration (`config.yaml`)

All scripts read from `config.yaml` — do **not** hardcode paths or parameters inside the `.py` files.

| Section | Purpose |
|---------|---------|
| `paths` | `dataset_yaml`, `sample_submission`, `submission_csv`, `test_images`, `runs_detect_root` — all relative to the kit root unless you use an absolute path |
| `tlc` | `project_name`, `dataset_name`, `train_table_name`, `val_table_name`, `image_embeddings_dim` |
| `training` | `model` must stay **`yolov8n`**; epochs, batch, device, etc. No pretrained-weights switch — `train.py` always uses `yolov8n.yaml` |
| `predict` | `pipeline`: **`memory`** or **`txt`**; `batch`, `conf`, `iou` |
| `reproducibility` | `seed` — applied to Python, NumPy, and PyTorch RNGs |

---

## Submission format

`submission.csv` must match **`sample_submission.csv`** exactly: columns **`id`**, **`image_id`**, **`prediction_string`**.

Per image, `prediction_string` is space-separated **`class conf xc yc w h`** (normalized YOLO coordinates), repeated for each box. Images with no detections use the literal string **`no box`**.

---

## Why Co-DETR for label correction

[Co-DETR](https://arxiv.org/pdf/2211.12860) (Collaborative Hybrid Assignments Training) achieves **66.0 mAP on COCO** — the current state of the art in object detection. Running it over the training set and diffing its predictions against existing labels gives a principled signal for where annotation quality is weakest:

- **Missed detections** — high-confidence Co-DETR boxes with no matching ground-truth are promoted directly as new annotations.
- **Class corrections** — disagreements on class assignment (e.g. Co-DETR predicts `truck`, label says `van`) are flagged for human review in the 3LC Dashboard.
- **Boundary tightening** — significant IoU gap between Co-DETR and the current box indicates a loose or misaligned label.

The effect is asymmetric: Co-DETR at 66 mAP handles the easy 80% of corrections automatically. Human review time is reserved for genuinely ambiguous cases — occluded vehicles, unusual viewpoints, class boundaries — where expert judgement is irreplaceable.

---

## 3LC — annotation tooling that closes the loop

[3LC](https://3lc.ai) is the data-management layer. Every training run writes metrics and per-image embeddings back into 3LC tables, making annotation review data-driven rather than random:

- **Embedding clusters** surface systematic label errors across similar images at once rather than one by one.
- **Per-sample metrics** (loss, IoU at eval) pinpoint which images the model struggles with most, directing review effort where it has the highest return.
- **Versioned table revisions** mean every correction is auditable and reversible. `train.py` always calls `.latest()`, so corrections are live in the next run with no file management.

A reviewer working in correction mode can validate roughly **10× more annotations per hour** than working from scratch.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Detector (training) | YOLOv8n | Fast iteration on constrained compute |
| Detector (label correction) | Co-DETR (66 mAP) | Strongest available signal for annotation noise |
| Data management | 3LC | Versioned tables, Dashboard review, embedding-based filtering |
| Language | Python 3.11, fully typed | Maintainability at pipeline scale |

---

## Troubleshooting

| Issue | What to try |
|-------|-------------|
| `dataset.yaml` not found | Run scripts from the kit root; check `paths.dataset_yaml` in `config.yaml` |
| CUDA OOM on `predict.py` | New terminal after training; set `predict.batch: 1`; or `predict.device: cpu` |
| Wrong submission shape | Re-download `sample_submission.csv` from the competition **Data** tab |
| Crash at "Reducing image embeddings" | `pip install umap-learn`, or set `image_embeddings_dim: 0` in `config.yaml` to skip |
| Training seems frozen at start | Normal — PyTorch and data loaders take up to 60 sec to initialize |
| `CUDA: False` despite having GPU | Install the CUDA build of PyTorch **before** `3lc-ultralytics` |
| Not sure if setup is correct | Run `python verify_setup.py` to check everything at once |

---

## Before you zip this kit for upload

**Include:** everything in the folder layout above (with real images/labels as distributed by the competition host).

**Exclude** (optional — smaller zip, no leakage):
- `runs/` — local training outputs
- `predictions/` — temporary predict output
- `submission.csv`, `tables_used.txt` — local artifacts
- any `labels_backup_*` folders if you created exports locally

---

## License / data use

Follow the official competition **Rules** and data license.
