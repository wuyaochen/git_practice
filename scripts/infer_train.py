#!/usr/bin/env python3
"""
YOLOv11 推論與裁切脚本 — 使用預訓練權重自動框選，替代 labelimg

用法：
  python scripts/infer_train.py \
    --weights runs/detect/yolov11m_aug/weights/best.pt \
    --source data260420/train \
    --out output_train \
    --device cuda \
    --imgsz 640 --conf 0.25 --iou 0.45 \
    --save-crops --save-json

輸出：
  output_train/crops/{class}/                 ← 按類別裁切圖像（jpg）
  output_train/crops/{class}/{image}_*.txt    ← 對應裁切的 YOLO 標註
  output_train/infer_results.json             ← 推論結果彙整
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def setup_output_dirs(output_dir: Path, classes: List[str] = None) -> Path:
    """建立輸出目錄結構"""
    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    # 建立按類別的子目錄
    if classes:
        for cls in classes:
            (crops_dir / cls).mkdir(parents=True, exist_ok=True)

    (output_dir / "annotated").mkdir(parents=True, exist_ok=True)
    return crops_dir


def to_yolo_xywh(x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """將 xyxy 座標轉為 YOLO 正規化 xywh。"""
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0

    # clamp 到影像範圍，避免數值越界
    cx = min(max(cx, 0.0), float(img_w))
    cy = min(max(cy, 0.0), float(img_h))
    bw = min(max(bw, 0.0), float(img_w))
    bh = min(max(bh, 0.0), float(img_h))

    if img_w <= 0 or img_h <= 0:
        return 0.5, 0.5, 0.0, 0.0

    return cx / img_w, cy / img_h, bw / img_w, bh / img_h


def save_crop_with_annotation(
    image: np.ndarray,
    crop_box: Tuple[int, int, int, int],
    class_id: int,
    class_name: str,
    orig_image_path: Path,
    image_id,
    crops_dir: Path,
) -> Tuple[Path, Path]:
    """
    保存裁切圖像與對應的 YOLO 標註檔（txt）

    Args:
        image: 原始圖像（numpy array）
        crop_box: (x1, y1, x2, y2) 邊界框座標
        class_id: 類別ID
        class_name: 類別名稱
        orig_image_path: 原始圖像路径
        image_id: image index 用於命名
        crops_dir: 輸出目錄

    Returns:
        (crop_image_path, annotation_path)
    """
    x1, y1, x2, y2 = crop_box
    x1, y1, x2, y2 = max(0, int(x1)), max(0, int(y1)), int(x2), int(y2)

    # 確保座標有效
    x2 = min(x2, image.shape[1])
    y2 = min(y2, image.shape[0])

    if x2 <= x1 or y2 <= y1:
        return None, None

    # 裁切圖像
    crop = image[y1:y2, x1:x2].copy()
    if crop.size == 0:
        return None, None

    # 存檔名稱
    stem = orig_image_path.stem
    crop_filename = f"{stem}_crop_{image_id}.jpg"
    txt_filename = f"{stem}_crop_{image_id}.txt"

    class_dir = crops_dir / class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    crop_path = class_dir / crop_filename
    txt_path = class_dir / txt_filename

    # 保存裁切圖像
    cv2.imwrite(str(crop_path), crop)

    # 保存 YOLO 標註（單物件，格式：class_id cx cy w h，normalized）
    # 對於裁切後的圖像，物件幾乎佔滿整個圖，所以中心接近 (0.5, 0.5)，寬高接近 1.0
    h, w = crop.shape[:2]
    cx, cy = 0.5, 0.5
    norm_w, norm_h = 1.0, 1.0  # 裁切圖像的邊界框基本是整個圖像

    with open(txt_path, "w") as f:
        f.write(f"{class_id} {cx:.6f} {cy:.6f} {norm_w:.6f} {norm_h:.6f}\n")

    return crop_path, txt_path


def infer_and_crop(
    weights: str,
    source: str,
    output_dir: str,
    imgsz: int = 640,
    conf: float = 0.25,
    iou: float = 0.45,
    device: str = "cuda",
    save_crops: bool = False,
    save_json: bool = True,
    save_annotated: bool = False,
    single_ng_class: bool = True,
) -> None:
    """
    使用 YOLO 模型推論並裁切

    Args:
        weights: 模型權重檔案路徑
        source: 輸入圖像資料夾
        output_dir: 輸出目錄
        imgsz: 推論圖像大小
        conf: 信心度閾值
        iou: NMS IoU 閾值
        device: 裝置（cuda / cpu）
        save_crops: 是否保存裁切圖像
        save_json: 是否保存推論結果為 JSON
        save_annotated: 是否保存標註圖像
    """
    output_path = Path(output_dir)
    source_path = Path(source)

    # 加載模型
    logger.info(f"Loading model from {weights}")
    model = YOLO(weights)
    
    # 獲取類別名稱
    class_names = model.names  # {0: 'class_a', 1: 'class_b', ...}
    logger.info(f"Classes: {class_names}")

    # 建立輸出目錄
    output_classes = ["NG"] if single_ng_class else [name for name in class_names.values()]
    crops_dir = setup_output_dirs(output_path, output_classes)
    labels_dir = output_path / "labels"

    # 推論
    logger.info(f"Running inference on {source_path}")
    results = model(
        source=str(source_path),
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )

    infer_results = []
    crop_count = 0

    video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}

    for frame_idx, result in enumerate(tqdm(results, desc="Processing detections")):
        source_item_path = Path(result.path)

        # 支援影片來源：優先使用 result.orig_img（每個 frame 的影像）
        if getattr(result, "orig_img", None) is not None:
            image = result.orig_img.copy()
        else:
            image = cv2.imread(str(source_item_path))

        if image is None:
            logger.warning(f"Could not read {source_item_path}")
            continue

        # 影片來源下每個 frame 需唯一命名，避免覆蓋
        if source_item_path.suffix.lower() in video_suffixes:
            logical_image_path = Path(f"{source_item_path.stem}_frame_{frame_idx:06d}.jpg")
        else:
            logical_image_path = source_item_path

        h, w = image.shape[:2]

        label_lines = []

        # 如果有檢測結果
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for idx, box in enumerate(boxes):
                # 提取邊界框與信心度
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf_score = box.conf[0].cpu().numpy()
                original_class_id = int(box.cls[0].cpu().numpy())
                original_class_name = class_names[original_class_id]

                # 輸出標註可選擇合併為單一 NG 類別
                if single_ng_class:
                    export_class_id = 0
                    export_class_name = "NG"
                else:
                    export_class_id = original_class_id
                    export_class_name = original_class_name

                # 針對原圖輸出 YOLO 標註（不是裁切圖）
                y_cx, y_cy, y_w, y_h = to_yolo_xywh(float(x1), float(y1), float(x2), float(y2), w, h)
                label_lines.append(f"{export_class_id} {y_cx:.6f} {y_cy:.6f} {y_w:.6f} {y_h:.6f}")

                # 保存裁切與標註
                if save_crops:
                    crop_path, txt_path = save_crop_with_annotation(
                        image=image,
                        crop_box=(x1, y1, x2, y2),
                        class_id=export_class_id,
                        class_name=export_class_name,
                        orig_image_path=logical_image_path,
                        image_id=f"{frame_idx}_{idx}",
                        crops_dir=crops_dir,
                    )

                    if crop_path:
                        crop_count += 1
                        infer_results.append(
                            {
                                "image": logical_image_path.name,
                                "crop_id": idx,
                                "class_id": export_class_id,
                                "class_name": export_class_name,
                                "original_class_id": original_class_id,
                                "original_class_name": original_class_name,
                                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                                "yolo_bbox": [float(y_cx), float(y_cy), float(y_w), float(y_h)],
                                "confidence": float(conf_score),
                                "crop_path": str(crop_path.relative_to(output_path)),
                                "annotation_path": str(txt_path.relative_to(output_path)),
                            }
                        )
                else:
                    infer_results.append(
                        {
                            "image": logical_image_path.name,
                            "crop_id": idx,
                            "class_id": export_class_id,
                            "class_name": export_class_name,
                            "original_class_id": original_class_id,
                            "original_class_name": original_class_name,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "yolo_bbox": [float(y_cx), float(y_cy), float(y_w), float(y_h)],
                            "confidence": float(conf_score),
                        }
                    )

                # 保存標註圖像（可選）
                if save_annotated:
                    result_plot = result.plot()
                    annotated_path = output_path / "annotated" / logical_image_path.name
                    cv2.imwrite(str(annotated_path), result_plot)
        else:
            # 沒有檢測結果的圖像
            if save_annotated:
                annotated_path = output_path / "annotated" / logical_image_path.name
                cv2.imwrite(str(annotated_path), image)

        # 每張原圖/每個 frame 對應一個 labels txt（可為空檔）
        label_path = labels_dir / f"{logical_image_path.stem}.txt"
        with open(label_path, "w", encoding="utf-8") as f:
            if label_lines:
                f.write("\n".join(label_lines) + "\n")

    # 保存推論結果摘要
    logger.info(f"Total crops saved: {crop_count}")

    if save_json:
        json_path = output_path / "infer_results.json"
        with open(json_path, "w") as f:
            json.dump(
                {
                    "model": weights,
                    "source": str(source_path),
                    "total_images": len(results),
                    "total_detections": crop_count,
                    "imgsz": imgsz,
                    "conf": conf,
                    "iou": iou,
                    "single_ng_class": single_ng_class,
                    "results": infer_results,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"Inference results saved to {json_path}")

    logger.info("Inference and cropping completed!")


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv11 推論與裁切 — 自動框選替代 labelimg"
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="模型權重檔案路徑",
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="輸入圖像資料夾（data260420/train）",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output_train",
        help="輸出目錄（default: output_train）",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="推論圖像大小（default: 640）",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="信心度閾值（default: 0.25）",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="NMS IoU 閾值（default: 0.45）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="裝置：cuda / cpu（default: cuda）",
    )
    parser.add_argument(
        "--save-crops",
        dest="save_crops",
        action="store_true",
        help="保存裁切圖像與裁切標註檔（預設關閉）",
    )
    parser.add_argument(
        "--no-save-crops",
        dest="save_crops",
        action="store_false",
        help="不保存裁切圖像，僅輸出原圖 labels（預設）",
    )
    parser.set_defaults(save_crops=False)
    parser.add_argument(
        "--save-json",
        action="store_true",
        default=True,
        help="保存推論結果為 JSON",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        default=False,
        help="保存標註圖像（可選）",
    )
    parser.add_argument(
        "--single-ng-class",
        dest="single_ng_class",
        action="store_true",
        default=True,
        help="將所有檢測輸出為單一類別 NG（class id = 0，default）",
    )
    parser.add_argument(
        "--keep-original-classes",
        dest="single_ng_class",
        action="store_false",
        help="保留模型原始類別（例如 line/sq）輸出",
    )

    args = parser.parse_args()

    infer_and_crop(
        weights=args.weights,
        source=args.source,
        output_dir=args.out,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_crops=args.save_crops,
        save_json=args.save_json,
        save_annotated=args.save_annotated,
        single_ng_class=args.single_ng_class,
    )


if __name__ == "__main__":
    main()
