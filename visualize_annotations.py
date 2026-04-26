import cv2
import os
import random
import argparse

IMAGES_DIR = "data/train/images"
LABELS_DIR = "data/train/labels"

CLASS_NAMES = {0: "truck", 1: "car", 2: "van", 3: "bus"}
COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 255, 255)}


def draw_yolo_boxes(image_path, label_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return None

    h, w = img.shape[:2]

    if not os.path.exists(label_path):
        print(f"No label file: {label_path}")
        return img

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls, cx, cy, bw, bh = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            color = COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = CLASS_NAMES.get(cls, str(cls))
            cv2.putText(img, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    return img


def main(n=10, show=True, save_dir=None):
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    sample = random.sample(image_files, min(n, len(image_files)))

    for fname in sample:
        stem = os.path.splitext(fname)[0]
        image_path = os.path.join(IMAGES_DIR, fname)
        label_path = os.path.join(LABELS_DIR, stem + ".txt")

        annotated = draw_yolo_boxes(image_path, label_path)
        if annotated is None:
            continue

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            cv2.imwrite(os.path.join(save_dir, fname), annotated)
            print(f"Saved: {os.path.join(save_dir, fname)}")

        if show:
            cv2.imshow(fname, annotated)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == ord("q"):
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize YOLO annotations on training images.")
    parser.add_argument("--n", type=int, default=10, help="Number of images to display (default: 10)")
    parser.add_argument("--no-show", action="store_true", help="Do not display images interactively")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save annotated images")
    args = parser.parse_args()

    main(n=args.n, show=not args.no_show, save_dir=args.save_dir)
