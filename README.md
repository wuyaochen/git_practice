# YOLOv11 數據準備工具包

這是一個用於 YOLOv11 和 Faster R-CNN 物體檢測的數據準備工具包，提供了從影片提取幀、圖像裁剪、數據增強、格式轉換到數據集分割的完整工作流程。

## 📋 目錄

- [功能特點](#功能特點)
- [環境要求](#環境要求)
- [安裝](#安裝)
- [完整工作流程](#完整工作流程)
- [腳本詳細說明](#腳本詳細說明)
- [目錄結構](#目錄結構)
- [使用注意事項](#使用注意事項)

## ✨ 功能特點

- **影片幀提取**：從影片中按指定間隔提取幀
- **智能圖像裁剪**：自動識別圖像尺寸並裁剪中心區域
- **數據增強**：支持 8 種增強方式（翻轉、旋轉、亮度調整、模糊、噪聲等）
- **格式轉換**：XML (Pascal VOC) 到 YOLO 格式的轉換
- **數據集分割**：自動將數據集分為訓練集、測試集和驗證集
- **標籤同步**：自動同步圖像和標籤文件到對應目錄

## 🔧 環境要求

### Python 版本
- Python 3.7+

### 依賴庫
```bash
opencv-python (cv2)
albumentations
Pillow (PIL)
moviepy
xml.etree.ElementTree
```

## 📦 安裝

1. 克隆此倉庫：
```bash
git clone https://github.com/wuyaochen/yolov11_wyc.git
cd yolov11_wyc
```

2. 安裝依賴：
```bash
pip install opencv-python albumentations pillow moviepy
```

## 🚀 完整工作流程

按照以下順序執行腳本，完成從影片到 YOLO/Faster R-CNN 數據集的完整準備：

### 步驟 1：提取影片幀
使用 `dataset.py` 從影片中提取圖像幀：
```bash
python dataset.py
```
- **輸入**：影片文件（如 `monkey.mp4`）
- **輸出**：JPEGImage 資料夾中的圖像幀

### 步驟 2：裁剪圖像
使用 `cut3.py` 裁剪圖像，保留中心管子部分：
```bash
python cut3.py
```
- **支持的尺寸**：
  - 1280×1024 → 550×1024
  - 1024×768 → 400×768（偏左 40px）
  - 1280×720 → 500×720（偏右 40px）

### 步驟 3：標注圖像
使用 **LabelImg** 工具標注圖像：
- 標注後的圖像存放在：`JPEGImage/`
- 標注的 XML 文件存放在：`Annotations/`

### 步驟 4：數據增強
使用 `augment2.py` 對圖像和標注進行數據增強：
```bash
python augment2.py
```
- **輸入**：`JPEGImages/` 和 `Annotations/`
- **輸出**：`augmented_images/` 和 `augmented_labels/`
- **增強方式**：
  - 水平翻轉
  - 垂直翻轉
  - 旋轉、縮放、位移
  - 亮度對比度調整
  - 模糊
  - 高斯噪聲
  - RGB 偏移
  - 色調飽和度調整

**注意**：`augment.py` 僅用於單張圖像測試，實際批量處理請使用 `augment2.py`

### 步驟 5：XML 轉 YOLO 格式
使用 `xml2yolo.py` 將 XML 標注轉換為 YOLO 格式：
```bash
python xml2yolo.py
```
- **輸入**：XML 標注文件
- **輸出**：YOLO 格式的 `.txt` 文件
- **格式**：`<class_id> <x_center> <y_center> <width> <height>`（歸一化坐標）

### 步驟 6：分割數據集
使用 `photo_random.py` 將圖像隨機分配到訓練集、測試集和驗證集：
```bash
python photo_random.py
```
- **輸入**：`augmented_images/`
- **輸出**：
  - `Images/train/`
  - `Images/test/`
  - `Images/val/`
- **默認比例**：8:1:1（訓練:測試:驗證）

### 步驟 7：同步標籤文件
使用 `label_move.py` 將標籤文件同步到對應的目錄：
```bash
python label_move.py
```
- **功能**：根據 `Images/` 中各子目錄的文件名，將對應的標籤從 `augmented_labels/` 複製到 `labels/` 的對應子目錄
- **輸出**：
  - `labels/train/`
  - `labels/test/`
  - `labels/val/`

### 步驟 8：完成
數據集準備完成，可以分別用於訓練：
- **Faster R-CNN**：使用 XML 格式的 `Annotations/`
- **YOLO**：使用 YOLO 格式的 `labels/`

## 📚 腳本詳細說明

### dataset.py
從影片中按固定間隔提取幀並保存為圖像。

**配置參數**：
- `video_files`：要處理的影片文件列表
- `interval`：截圖間隔（秒）
- `folder_name`：輸出資料夾名稱

**特點**：
- 自動延續已有的圖像編號
- 支持批量處理多個影片

### cut3.py
智能裁剪圖像，保留中心區域。

**支持的圖像尺寸**：
- 1280×1024：裁剪為 550×1024
- 1024×768：裁剪為 400×768（偏左 40 像素）
- 1280×720：裁剪為 500×720（偏右 40 像素）

**配置參數**：
- `input_folder`：輸入圖像路徑
- `output_folder`：輸出圖像路徑

### augment.py
單張圖像的數據增強測試腳本。

**用途**：
- 測試增強效果
- 驗證標注框是否正確

**不建議用於批量處理**，請使用 `augment2.py`。

### augment2.py
批量數據增強腳本，同時處理圖像和對應的 XML 標注。

**增強方式**（8種）：
1. 水平翻轉
2. 垂直翻轉
3. 旋轉、縮放、位移
4. 亮度對比度調整
5. 模糊
6. 高斯噪聲
7. RGB 偏移
8. 色調飽和度調整

**特點**：
- 自動延續圖像編號
- 保證增強後的標注框不超出圖像邊界（`min_visibility=0.3`）
- 包含原圖（共輸出 9 張圖）

### xml2yolo.py
將 Pascal VOC 格式的 XML 標注轉換為 YOLO 格式。

**配置**：
- `lut`：類別名稱到類別 ID 的映射字典
  ```python
  lut = {
      "NG": 0,
      # 添加更多類別...
  }
  ```

**轉換公式**：
```
x_center = (xmin + xmax) / 2 / width
y_center = (ymin + ymax) / 2 / height
box_width = (xmax - xmin) / width
box_height = (ymax - ymin) / height
```

### photo_random.py
隨機分割數據集到訓練集、測試集和驗證集。

**配置參數**：
- `images`：輸入圖像路徑
- `train`, `test`, `val`：輸出路徑
- `ratio`：分割比例（默認 8:1:1）

**特點**：
- 隨機打亂順序
- 自動創建目錄
- 移動文件（非複製）

### label_move.py
根據圖像文件名同步標籤文件。

**工作原理**：
1. 掃描 `train/`, `test/`, `val/` 中的圖像文件名
2. 從源標籤目錄中查找同名標籤文件
3. 複製到對應的標籤子目錄

**配置參數**：
- `source_E`：源標籤目錄
- `target_B/C/D`：圖像目錄（train/test/val）
- `dest_F/G/H`：目標標籤目錄（train/test/val）
- `ext`：標籤文件擴展名（`.xml` 或 `.txt`）

## 📁 目錄結構

推薦的數據集目錄結構：

```
project/
├── dataset/                    # 從影片提取的原始幀
├── JPEGImages/                 # 裁剪後的圖像
├── Annotations/                # LabelImg 標注的 XML 文件
├── augmented_images/           # 增強後的圖像
├── augmented_labels/           # 增強後的 XML 標注
├── Images/                     # 分割後的圖像
│   ├── train/
│   ├── test/
│   └── val/
└── labels/                     # YOLO 格式的標籤
    ├── train/
    ├── test/
    └── val/
```

## 📝 使用注意事項

### 重要提示

1. **路徑配置**：所有腳本中的路徑都需要根據實際情況修改
2. **類別映射**：在 `xml2yolo.py` 中需要正確配置 `lut` 字典
3. **備份數據**：在執行數據處理前建議備份原始數據
4. **文件命名**：確保圖像和標注文件使用相同的文件名（不含擴展名）

### 常見問題

**Q: augment.py 和 augment2.py 有什麼區別？**  
A: `augment.py` 用於單張圖像測試，`augment2.py` 用於批量處理整個數據集。

**Q: 為什麼需要 label_move.py？**  
A: 因為 `photo_random.py` 只分割圖像，標籤需要單獨同步到對應目錄。

**Q: 支持哪些圖像格式？**  
A: 支持常見格式如 `.jpg`, `.jpeg`, `.png`。

**Q: 數據增強會增加多少數據？**  
A: 使用 `augment2.py` 會將每張原圖增強為 9 張（包含原圖）。

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權

本項目採用 MIT 授權協議。

## 👤 作者

- GitHub: [@wuyaochen](https://github.com/wuyaochen)

---

**注意**：此工具包主要用於學術研究和實驗，請確保您擁有所使用數據的適當權限。

