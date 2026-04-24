import argparse
import os
from pathlib import Path

from PIL import Image

# 定義裁剪中心圖片的函數，新增水平偏移量參數
def crop_center(image, crop_width, crop_height, offset_x=0):
    img_width, img_height = image.size
    left = (img_width - crop_width) / 2 + offset_x
    top = (img_height - crop_height) / 2
    right = (img_width + crop_width) / 2 + offset_x
    bottom = (img_height + crop_height) / 2
    return image.crop((left, top, right, bottom))

# 定義處理圖片的函數
def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path)

            img_width, img_height = img.size

            # 判斷圖片尺寸並裁剪
            if img_width == 1280 and img_height == 1024:
                # 裁剪出1280x1024圖片的中央區域
                cropped_img = crop_center(img, 550, 1024)
                #cropped_img.show()  # 在裁剪完後顯示圖片
                print(f"已裁剪 {filename}: 1280x1024 -> 550x1024")
            elif img_width == 1024 and img_height == 768:
                # 裁剪出1024x768圖片的中央區域，偏左
                cropped_img = crop_center(img, 400, 768, offset_x=-40)  # 偏移量 -50 像素向左
                #cropped_img.show()  # 在裁剪完後顯示圖片
                print(f"已裁剪 {filename}: 1024x768 -> 400x768，偏左")
            elif img_width == 1280 and img_height == 720:
                # 裁剪出1280x720圖片的中央區域，偏右
                cropped_img = crop_center(img, 500, 720, offset_x=40)  # 偏移量 -50 像素向左
                #cropped_img.show()  # 在裁剪完後顯示圖片
                print(f"已裁剪 {filename}: 1280x720 -> 500x720，偏右")
            else:
                print(f"跳過 {filename}: 不符合1280x1024或1024x768")
                continue

            # 保存處理過的圖片
            output_path = os.path.join(output_folder, filename)
            cropped_img.save(output_path)
            print(f"已處理並保存: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="批次中心裁切圖片")
    parser.add_argument("--input", type=str, required=True, help="輸入圖片資料夾")
    parser.add_argument("--output", type=str, required=True, help="輸出圖片資料夾")
    args = parser.parse_args()

    input_folder = Path(args.input)
    output_folder = Path(args.output)
    process_images(str(input_folder), str(output_folder))


if __name__ == "__main__":
    main()