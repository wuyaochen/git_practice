import os
import shutil

def match_and_copy(source_E, target_B, target_C, target_D, dest_F, dest_G, dest_H, ext=".xml"):
    # 確保目標資料夾存在
    os.makedirs(dest_F, exist_ok=True)
    os.makedirs(dest_G, exist_ok=True)
    os.makedirs(dest_H, exist_ok=True)

    # 取得 B/C/D 的檔名（不含副檔名）
    names_B = {os.path.splitext(f)[0] for f in os.listdir(target_B)}
    names_C = {os.path.splitext(f)[0] for f in os.listdir(target_C)}
    names_D = {os.path.splitext(f)[0] for f in os.listdir(target_D)}

    # 開始尋找 E 中的檔案，依據檔名對應搬移
    for f in os.listdir(source_E):
        name, file_ext = os.path.splitext(f)
        if file_ext.lower() != ext.lower():
            continue  # 忽略非目標副檔名

        src_file = os.path.join(source_E, f)

        if name in names_B:
            shutil.copy(src_file, os.path.join(dest_F, f))
        elif name in names_C:
            shutil.copy(src_file, os.path.join(dest_G, f))
        elif name in names_D:
            shutil.copy(src_file, os.path.join(dest_H, f))

# 使用範例
source_E = r'C:\Users\User\wyc\cut_datasets0411_aug\augmented_labels'       # E 的路徑（如 XML 標註檔）
target_B = r'C:\Users\User\wyc\cut_datasets0411_aug\Images\train'      # B
target_C = r'C:\Users\User\wyc\cut_datasets0411_aug\Images\test'       # C
target_D = r'C:\Users\User\wyc\cut_datasets0411_aug\Images\val'        # D

dest_F = r'C:\Users\User\wyc\cut_datasets0411_aug\labels\train' # F
dest_G = r'C:\Users\User\wyc\cut_datasets0411_aug\labels\test'  # G
dest_H = r'C:\Users\User\wyc\cut_datasets0411_aug\labels\val'   # H

match_and_copy(source_E, target_B, target_C, target_D, dest_F, dest_G, dest_H, ext=".xml")
