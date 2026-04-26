#!/usr/bin/env python3
"""Visualize ground-truth YOLO annotations on training images using OpenCV."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

WORK_DIR: Path = Path(__file__).resolve().parent
IMAGES_DIR: Path = WORK_DIR / "data" / "train" / "images"
LABELS_DIR: Path = WORK_DIR / "data" / "train" / "labels"

CLASS_NAMES: dict[int, str] = {0: "truck", 1: "car", 2: "van", 3: "bus"}
COLORS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 255),    # truck  — red
    1: (0, 255, 0),    # car    — green
    2: (255, 0, 0),    # van    — blue
    3: (0, 255, 255),  # bus    — yellow
}
IMAGE_EXTS: tuple[str, ...] = (".jpg", ".jpeg", ".png")


def draw_yolo_boxes(image_path: Path, label_path: Path) -> np.ndarray | None:
    """Draw YOLO ground-truth boxes onto an image and return the annotated array.

    Args:
        image_path: Path to the source image.
        label_path: Path to the YOLO label file (cls cx cy w h, normalised).

    Returns:
        Annotated BGR array, or ``None`` if the image could not be read.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read image: {image_path}")
        return None

    ih, iw = img.shape[:2]

    if not label_path.is_file():
        print(f"No label file: {label_path}")
        return img

    with label_path.open() as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            x1 = int((cx - bw / 2) * iw)
            y1 = int((cy - bh / 2) * ih)
            x2 = int((cx + bw / 2) * iw)
            y2 = int((cy + bh / 2) * ih)

            color = COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                img,
                CLASS_NAMES.get(cls, str(cls)),
                (x1, max(y1 - 6, 14)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    return img


def main(n: int = 10, show: bool = True, save_dir: Path | None = None) -> None:
    """Sample *n* training images, annotate them, and optionally display or save.

    Args:
        n:        Number of images to process.
        show:     Display each image in an OpenCV window when ``True``.
        save_dir: Write annotated images here; skipped when ``None``.
    """
    image_files = [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in IMAGE_EXTS]
    sample = random.sample(image_files, min(n, len(image_files)))

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for img_path in sample:
        label_path = LABELS_DIR / f"{img_path.stem}.txt"
        annotated = draw_yolo_boxes(img_path, label_path)
        if annotated is None:
            continue

        if save_dir is not None:
            out = save_dir / img_path.name
            cv2.imwrite(str(out), annotated)
            print(f"Saved: {out}")

        if show:
            cv2.imshow(img_path.name, annotated)
            if cv2.waitKey(0) == ord("q"):
                cv2.destroyAllWindows()
                break
            cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO annotations on training images.")
    parser.add_argument("--n", type=int, default=10, help="Number of images to display (default: 10)")
    parser.add_argument("--no-show", action="store_true", help="Skip interactive display")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save annotated images")
    args = parser.parse_args()

    main(
        n=args.n,
        show=not args.no_show,
        save_dir=Path(args.save_dir) if args.save_dir else None,
    )
