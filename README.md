# Vehicle Detection in Traffic Surveillance

Real-time vehicle detection across four classes - **truck, car, van, bus** - on overhead traffic camera footage from the UA-DETRAC benchmark, evaluated at COCO mAP (IoU 0.50:0.95).

---

## The core insight: label quality beats model capacity

Most teams iterate on model architecture. This pipeline iterates on **data quality** - the more leveraged variable at this scale.

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

Each round produces a better model, which produces better pseudo-labels for the next round - a self-reinforcing loop that converges toward clean annotations without manually inspecting every image.

---

## Why Co-DETR for label correction

[Co-DETR](https://arxiv.org/pdf/2211.12860) (Collaborative Hybrid Assignments Training) achieves **66.0 mAP on COCO** - the current state of the art in object detection. Running it over the training set and diffing its predictions against existing labels gives a principled signal for where annotation quality is weakest:

- **Missed detections**: high-confidence Co-DETR boxes with no matching ground-truth are promoted directly as new annotations.
- **Class corrections**: disagreements on class assignment (e.g. Co-DETR predicts `truck`, label says `van`) are flagged for human review in the 3LC Dashboard.
- **Boundary tightening**: significant IoU gap between Co-DETR and the current box indicates a loose or misaligned label.

The effect is asymmetric: Co-DETR at 66 mAP handles the easy 80% of corrections automatically. Human review time is reserved for genuinely ambiguous cases - occluded vehicles, unusual viewpoints, class boundaries - where expert judgement is irreplaceable.

---

## 3LC - annotation tooling that closes the loop

[3LC](https://3lc.ai) is the data-management layer. Every training run writes metrics and per-image embeddings back into 3LC tables, making annotation review data-driven rather than random:

- **Embedding clusters** surface systematic label errors across similar images at once rather than one by one.
- **Per-sample metrics** (loss, IoU at eval) pinpoint which images the model struggles with most, directing review effort where it has the highest return.
- **Versioned table revisions** mean every correction is auditable and reversible. `train.py` always calls `.latest()`, so corrections are live in the next run with no file management.

The result: a reviewer working in correction mode can validate roughly **10× more annotations per hour** than working from scratch - the difference between a one-day and a two-week data quality sprint.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| Detector (training) | YOLOv8n | Fast iteration on constrained compute |
| Detector (label correction) | Co-DETR (66 mAP) | Strongest available signal for annotation noise |
| Data management | 3LC | Versioned tables, Dashboard review, embedding-based filtering |
| Language | Python 3.11, fully typed | Maintainability at pipeline scale |
