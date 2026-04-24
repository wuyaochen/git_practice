#!/usr/bin/env python3
"""
YOLO 資料擴增（flip / rotate / translate）並同步更新 bbox。

用途：
- 讀取 images_dir + labels_dir（YOLO txt）
- 每張圖固定產生三張擴增圖：flip、rotate、translate
- 自動輸出對應 txt 到 out_labels_dir

範例：
  uv run --python .venv\Scripts\python.exe scripts/augment_yolo.py \
    --images-dir output_train_cut/datasets/images/train \
    --labels-dir output_train_cut/datasets/labels/train \
    --out-images-dir output_train_cut/datasets/images/aug_train \
    --out-labels-dir output_train_cut/datasets/labels/aug_train
"""

from __future__ import annotations

import argparse
import logging
import random
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger("augment_yolo")


BBox = Tuple[int, float, float, float, float]  # class_id, cx, cy, w, h (normalized)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")


def list_images(images_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted([p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])


def read_yolo_labels(label_path: Path) -> List[BBox]:
    if not label_path.exists():
        return []

    boxes: List[BBox] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls_id = int(float(parts[0]))
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
        except ValueError:
            continue
        boxes.append((cls_id, cx, cy, w, h))
    return boxes


def write_yolo_labels(label_path: Path, boxes: Sequence[BBox]) -> None:
    lines = [f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for cls_id, cx, cy, w, h in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def yolo_to_corners(box: BBox, img_w: int, img_h: int) -> Tuple[int, np.ndarray]:
    cls_id, cx, cy, bw, bh = box
    x_c = cx * img_w
    y_c = cy * img_h
    w = bw * img_w
    h = bh * img_h

    x1 = x_c - w / 2.0
    y1 = y_c - h / 2.0
    x2 = x_c + w / 2.0
    y2 = y_c + h / 2.0

    corners = np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )
    return cls_id, corners


def corners_to_yolo(cls_id: int, corners: np.ndarray, img_w: int, img_h: int) -> BBox | None:
    xs = corners[:, 0]
    ys = corners[:, 1]

    x1 = float(np.clip(np.min(xs), 0, img_w - 1))
    y1 = float(np.clip(np.min(ys), 0, img_h - 1))
    x2 = float(np.clip(np.max(xs), 0, img_w - 1))
    y2 = float(np.clip(np.max(ys), 0, img_h - 1))

    bw = x2 - x1
    bh = y2 - y1
    if bw < 1.0 or bh < 1.0:
        return None

    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    nw = bw / img_w
    nh = bh / img_h

    if nw <= 0 or nh <= 0:
        return None

    return cls_id, cx, cy, nw, nh


def apply_affine_to_corners(corners: np.ndarray, M: np.ndarray) -> np.ndarray:
    ones = np.ones((corners.shape[0], 1), dtype=np.float32)
    points = np.hstack([corners, ones])
    transformed = points @ M.T
    return transformed[:, :2]


def augment_flip(image: np.ndarray, boxes: Sequence[BBox], mode: str = "h") -> Tuple[np.ndarray, List[BBox]]:
    h, w = image.shape[:2]

    if mode == "random":
        # 依需求隨機選擇左右或上下翻轉
        mode = random.choice(["h", "v"])

    if mode == "h":
        out_img = cv2.flip(image, 1)
        out_boxes = [(cls, 1.0 - cx, cy, bw, bh) for cls, cx, cy, bw, bh in boxes]
    elif mode == "v":
        out_img = cv2.flip(image, 0)
        out_boxes = [(cls, cx, 1.0 - cy, bw, bh) for cls, cx, cy, bw, bh in boxes]
    else:
        # hv
        out_img = cv2.flip(image, -1)
        out_boxes = [(cls, 1.0 - cx, 1.0 - cy, bw, bh) for cls, cx, cy, bw, bh in boxes]

    # 保底 clamp
    clamped: List[BBox] = []
    for cls, cx, cy, bw, bh in out_boxes:
        clamped.append((cls, float(np.clip(cx, 0, 1)), float(np.clip(cy, 0, 1)), bw, bh))
    return out_img, clamped


def augment_rotate(
    image: np.ndarray,
    boxes: Sequence[BBox],
    max_abs_degree: float,
) -> Tuple[np.ndarray, List[BBox], float]:
    h, w = image.shape[:2]
    angle = random.uniform(-max_abs_degree, max_abs_degree)

    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, 1.0).astype(np.float32)
    out_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    out_boxes: List[BBox] = []
    for b in boxes:
        cls_id, corners = yolo_to_corners(b, w, h)
        trans_corners = apply_affine_to_corners(corners, M)
        yolo_box = corners_to_yolo(cls_id, trans_corners, w, h)
        if yolo_box is not None:
            out_boxes.append(yolo_box)

    return out_img, out_boxes, angle


def augment_translate(
    image: np.ndarray,
    boxes: Sequence[BBox],
    max_shift_ratio: float,
) -> Tuple[np.ndarray, List[BBox], float, float]:
    h, w = image.shape[:2]
    tx = random.uniform(-max_shift_ratio, max_shift_ratio) * w
    ty = random.uniform(-max_shift_ratio, max_shift_ratio) * h

    M = np.array([[1.0, 0.0, tx], [0.0, 1.0, ty]], dtype=np.float32)
    out_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    out_boxes: List[BBox] = []
    for b in boxes:
        cls_id, corners = yolo_to_corners(b, w, h)
        trans_corners = apply_affine_to_corners(corners, M)
        yolo_box = corners_to_yolo(cls_id, trans_corners, w, h)
        if yolo_box is not None:
            out_boxes.append(yolo_box)

    return out_img, out_boxes, tx, ty


def run_augmentation(
    images_dir: Path,
    labels_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    flip_mode: str,
    rotate_deg: float,
    translate_ratio: float,
    seed: int,
) -> None:
    random.seed(seed)
    np.random.seed(seed)

    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(images_dir)
    if not image_paths:
        raise FileNotFoundError(f"No image files found under: {images_dir}")

    LOGGER.info("Found %d images", len(image_paths))

    total_outputs = 0
    for idx, img_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(img_path))
        if image is None:
            LOGGER.warning("Skip unreadable image: %s", img_path)
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = read_yolo_labels(label_path)

        # 0) Copy original image and label to output directory
        shutil.copy2(img_path, out_images_dir / img_path.name)
        if label_path.exists():
            shutil.copy2(label_path, out_labels_dir / label_path.name)
        total_outputs += 1

        # 1) flip
        flip_img, flip_boxes = augment_flip(image, boxes, mode=flip_mode)
        flip_img_name = f"{img_path.stem}_flip{img_path.suffix.lower()}"
        flip_lbl_name = f"{img_path.stem}_flip.txt"
        cv2.imwrite(str(out_images_dir / flip_img_name), flip_img)
        write_yolo_labels(out_labels_dir / flip_lbl_name, flip_boxes)
        total_outputs += 1

        # 2) rotate
        rot_img, rot_boxes, angle = augment_rotate(image, boxes, max_abs_degree=rotate_deg)
        rot_img_name = f"{img_path.stem}_rot{img_path.suffix.lower()}"
        rot_lbl_name = f"{img_path.stem}_rot.txt"
        cv2.imwrite(str(out_images_dir / rot_img_name), rot_img)
        write_yolo_labels(out_labels_dir / rot_lbl_name, rot_boxes)
        total_outputs += 1

        # 3) translate
        trans_img, trans_boxes, tx, ty = augment_translate(image, boxes, max_shift_ratio=translate_ratio)
        trans_img_name = f"{img_path.stem}_trans{img_path.suffix.lower()}"
        trans_lbl_name = f"{img_path.stem}_trans.txt"
        cv2.imwrite(str(out_images_dir / trans_img_name), trans_img)
        write_yolo_labels(out_labels_dir / trans_lbl_name, trans_boxes)
        total_outputs += 1

        if idx % 100 == 0 or idx == len(image_paths):
            LOGGER.info(
                "Progress %d/%d | latest: %s | rot=%.2f deg, tx=%.1f, ty=%.1f",
                idx,
                len(image_paths),
                img_path.name,
                angle,
                tx,
                ty,
            )

    LOGGER.info("Done. Generated %d total images/labels (original + augmented) under aug_train.", total_outputs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO bbox-aware augmentation: flip/rotate/translate")
    parser.add_argument("--images-dir", type=Path, required=True, help="Input images directory")
    parser.add_argument("--labels-dir", type=Path, required=True, help="Input YOLO labels directory")
    parser.add_argument("--out-images-dir", type=Path, required=True, help="Output augmented images directory")
    parser.add_argument("--out-labels-dir", type=Path, required=True, help="Output augmented labels directory")
    parser.add_argument(
        "--flip-mode",
        type=str,
        default="random",
        choices=["h", "v", "hv", "random"],
        help="Flip mode: h(horizontal), v(vertical), hv(both), random(隨機左右/上下)",
    )
    parser.add_argument("--rotate-deg", type=float, default=8.0, help="Max absolute rotation degree")
    parser.add_argument(
        "--translate-ratio",
        type=float,
        default=0.06,
        help="Max translate ratio relative to width/height",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)

    run_augmentation(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        out_images_dir=args.out_images_dir,
        out_labels_dir=args.out_labels_dir,
        flip_mode=args.flip_mode,
        rotate_deg=args.rotate_deg,
        translate_ratio=args.translate_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
