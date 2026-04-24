#!/usr/bin/env python3
"""
裁切圖像前處理腳本 — 支援多種濾波方式

用法：
  # 高斯濾波
  python scripts/preprocess.py \
    --input output_train/crops \
    --output output_train/crops_preprocessed \
    --method gaussian --kernel 5

  # 中值濾波
  python scripts/preprocess.py \
    --input output_train/crops \
    --output output_train/crops_preprocessed \
    --method median --kernel 5

  # 雙邊濾波
  python scripts/preprocess.py \
    --input output_train/crops \
    --output output_train/crops_preprocessed \
    --method bilateral --d 9 --sigma-color 75 --sigma-space 75

  # 無濾波（直接複製）
  python scripts/preprocess.py \
    --input output_train/crops \
    --output output_train/crops_preprocessed \
    --method none

前處理方式：
  - none       : 不處理（直接複製）
  - gaussian   : 高斯濾波（參數: kernel）
  - median     : 中值濾波（參數: kernel）
  - bilateral  : 雙邊濾波（參數: d, sigma_color, sigma_space）
  - morphopen  : 形態開運算（參數: kernel）
  - morphclose : 形態閉運算（參數: kernel）
  - clahe      : 對比度拉伸（CLAHE）（參數: clip_limit, tile_size）
"""

import argparse
import logging
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from tqdm import tqdm

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """圖像前處理類"""

    def __init__(self):
        self.methods = {
            "none": self.preprocess_none,
            "gaussian": self.preprocess_gaussian,
            "median": self.preprocess_median,
            "bilateral": self.preprocess_bilateral,
            "morphopen": self.preprocess_morphopen,
            "morphclose": self.preprocess_morphclose,
            "clahe": self.preprocess_clahe,
        }

    def preprocess_none(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """不處理，直接返回"""
        return image

    def preprocess_gaussian(self, image: np.ndarray, kernel: int = 5, **kwargs) -> np.ndarray:
        """高斯濾波"""
        if kernel % 2 == 0:
            kernel += 1
        return cv2.GaussianBlur(image, (kernel, kernel), 0)

    def preprocess_median(self, image: np.ndarray, kernel: int = 5, **kwargs) -> np.ndarray:
        """中值濾波"""
        if kernel % 2 == 0:
            kernel += 1
        return cv2.medianBlur(image, kernel)

    def preprocess_bilateral(
        self,
        image: np.ndarray,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
        **kwargs,
    ) -> np.ndarray:
        """雙邊濾波（保留邊緣同時平滑）"""
        return cv2.bilateralFilter(
            image,
            d=d,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
        )

    def preprocess_morphopen(self, image: np.ndarray, kernel: int = 5, **kwargs) -> np.ndarray:
        """形態開運算（開運算 = 腐蝕 → 膨脹）"""
        kernel_mat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_mat)

    def preprocess_morphclose(self, image: np.ndarray, kernel: int = 5, **kwargs) -> np.ndarray:
        """形態閉運算（閉運算 = 膨脹 → 腐蝕）"""
        kernel_mat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_mat)

    def preprocess_clahe(
        self,
        image: np.ndarray,
        clip_limit: float = 2.0,
        tile_size: int = 8,
        **kwargs,
    ) -> np.ndarray:
        """對比度拉伸（CLAHE - Contrast Limited Adaptive Histogram Equalization）"""
        if len(image.shape) == 3 and image.shape[2] == 3:
            # 轉為 LAB 色彩空間
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]

            # 進行 CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            l_channel_clahe = clahe.apply(l_channel)

            # 融合回 LAB
            lab[:, :, 0] = l_channel_clahe
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            # 灰度圖像
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            return clahe.apply(image)

    def process(
        self,
        image: np.ndarray,
        method: str,
        **kwargs,
    ) -> np.ndarray:
        """調用指定的前處理方法"""
        if method not in self.methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.methods.keys())}")

        return self.methods[method](image, **kwargs)


def preprocess_crops(
    input_dir: str,
    output_dir: str,
    method: str = "gaussian",
    **kwargs,
) -> None:
    """
    對裁切資料夾進行前處理

    Args:
        input_dir: 輸入資料夾（output_train/crops）
        output_dir: 輸出資料夾
        method: 前處理方式
        **kwargs: 傳給前處理方法的參數
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_path}")
        return

    # 建立輸出目錄結構
    output_path.mkdir(parents=True, exist_ok=True)

    # 獲取所有類別資料夾
    class_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    if not class_dirs:
        logger.warning(f"No class directories found in {input_path}")
        return

    preprocessor = ImagePreprocessor()
    total_processed = 0

    # 遍歷每個類別
    for class_dir in class_dirs:
        class_name = class_dir.name
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        # 獲取所有圖像檔案
        image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
        if not image_files:
            logger.warning(f"No images found in {class_dir}")
            continue

        logger.info(f"Processing class '{class_name}' with {len(image_files)} images")

        for img_path in tqdm(image_files, desc=f"Preprocessing {class_name}"):
            # 讀取圖像
            image = cv2.imread(str(img_path))
            if image is None:
                logger.warning(f"Could not read {img_path}")
                continue

            # 應用前處理
            try:
                processed_image = preprocessor.process(image, method, **kwargs)
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

            # 保存前處理後的圖像
            output_img_path = output_class_dir / img_path.name
            cv2.imwrite(str(output_img_path), processed_image)
            total_processed += 1

            # 複製對應的標註檔（txt）
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                output_txt_path = output_class_dir / txt_path.name
                with open(txt_path, "r") as src, open(output_txt_path, "w") as dst:
                    dst.write(src.read())

    logger.info(f"Preprocessing completed! Total images processed: {total_processed}")
    logger.info(f"Output saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="裁切圖像前處理 — 支援多種濾波方式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例：
  # 高斯濾波
  python scripts/preprocess.py --input output_train/crops --output output_train/crops_preprocessed --method gaussian --kernel 5

  # 中值濾波
  python scripts/preprocess.py --input output_train/crops --output output_train/crops_preprocessed --method median --kernel 5

  # 雙邊濾波
  python scripts/preprocess.py --input output_train/crops --output output_train/crops_preprocessed --method bilateral --d 9 --sigma-color 75 --sigma-space 75

  # CLAHE
  python scripts/preprocess.py --input output_train/crops --output output_train/crops_preprocessed --method clahe --clip-limit 2.0 --tile-size 8
        """,
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="輸入資料夾（含按類別組織的裁切圖像）",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="輸出資料夾",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gaussian",
        choices=["none", "gaussian", "median", "bilateral", "morphopen", "morphclose", "clahe"],
        help="前處理方式（default: gaussian）",
    )
    parser.add_argument(
        "--kernel",
        type=int,
        default=5,
        help="核大小（用於 gaussian/median/morphopen/morphclose）（default: 5）",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=9,
        help="雙邊濾波直徑（default: 9）",
    )
    parser.add_argument(
        "--sigma-color",
        type=float,
        default=75.0,
        help="雙邊濾波顏色標準差（default: 75.0）",
    )
    parser.add_argument(
        "--sigma-space",
        type=float,
        default=75.0,
        help="雙邊濾波空間標準差（default: 75.0）",
    )
    parser.add_argument(
        "--clip-limit",
        type=float,
        default=2.0,
        help="CLAHE 剪裁限制（default: 2.0）",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=8,
        help="CLAHE 瓦片大小（default: 8）",
    )

    args = parser.parse_args()

    # 準備參數字典
    kwargs = {}
    if args.method in ["gaussian", "median", "morphopen", "morphclose"]:
        kwargs["kernel"] = args.kernel
    elif args.method == "bilateral":
        kwargs.update({
            "d": args.d,
            "sigma_color": args.sigma_color,
            "sigma_space": args.sigma_space,
        })
    elif args.method == "clahe":
        kwargs.update({
            "clip_limit": args.clip_limit,
            "tile_size": args.tile_size,
        })

    logger.info(f"Preprocessing method: {args.method}")
    logger.info(f"Parameters: {kwargs}")

    preprocess_crops(
        input_dir=args.input,
        output_dir=args.output,
        method=args.method,
        **kwargs,
    )


if __name__ == "__main__":
    main()
