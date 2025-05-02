from moviepy.editor import *
import os

# 圖片資料夾名稱
folder_name = "dataset"

# 檢查圖片資料夾是否存在，若不存在則建立
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f"資料夾 '{folder_name}' 已建立")
else:
    print(f"資料夾 '{folder_name}' 已存在")

# 獲取資料夾中已存在的圖片數量，確保圖片編號延續
existing_frames = len([name for name in os.listdir(folder_name) if name.endswith(".jpg")])
frame_count = existing_frames

# 設定要處理的影片列表
video_files = ["monkey.mp4", "monkey2.mp4"]  # 在這裡加上其他影片的檔案名稱

for video_file in video_files:
    # 讀取影片
    video = VideoFileClip(video_file)
    print(f"處理影片: {video_file}")

    # 設定截圖間隔時間 (秒)
    interval = 1
    duration = video.duration  # 影片總長度（秒）

    # 迴圈每隔 0.1 秒截圖
    t = 0.0

    while t < duration:
        # 儲存每一幀為圖片到指定資料夾
        frame_path = os.path.join(folder_name, f"frame{frame_count + 1}.jpg")
        frame = video.save_frame(frame_path, t=t)
        print(f"Saved frame at {t} seconds as {frame_path}")
        
        # 更新時間和幀計數器
        t = t + interval
        frame_count += 1

    print(f"影片 {video_file} 的所有幀已保存")

print("所有影片的幀已保存")
