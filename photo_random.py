import os
import random
import shutil

def split_photos(images, train, test, val, ratio=(8, 1, 1)):
    # 確保目標資料夾存在
    os.makedirs(train, exist_ok=True)
    os.makedirs(test, exist_ok=True)
    os.makedirs(val, exist_ok=True)

    # 獲取所有照片的路徑
    photos = [f for f in os.listdir(images) if os.path.isfile(os.path.join(images, f))]

    # 隨機打亂照片順序
    random.shuffle(photos)	

    # 計算每個資料夾應該分配的照片數量
    total_photos = len(photos)
    num_b = (total_photos * ratio[0]) // sum(ratio)
    num_c = (total_photos * ratio[1]) // sum(ratio)
    num_d = total_photos - num_b - num_c

    # 分配照片
    photos_for_b = photos[:num_b]
    photos_for_c = photos[num_b:num_b + num_c]
    photos_for_d = photos[num_b + num_c:]

    # 移動照片到目標資料夾
    for photo in photos_for_b:
        shutil.move(os.path.join(images, photo), os.path.join(train, photo))

    for photo in photos_for_c:
        shutil.move(os.path.join(images, photo), os.path.join(test, photo))

    for photo in photos_for_d:
        shutil.move(os.path.join(images, photo), os.path.join(val, photo))
    print(f"Total photos: {total_photos}, Train: {num_b}, Test: {num_c}, Val: {num_d}")

# 使用範例
images = r'C:\Users\User\wyc\cut_datasets0411_aug\augmented_images'  # A 資料夾的路徑
train = r'C:\Users\User\wyc\cut_datasets0411_aug\Images\train'  # B 資料夾的路徑
test = r'C:\Users\User\wyc\cut_datasets0411_aug\Images\test'  # C 資料夾的路徑
val = r'C:\Users\User\wyc\cut_datasets0411_aug\Images\val'  # D 資料夾的路徑

split_photos(images, train, test, val)
