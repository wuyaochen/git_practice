# YOLOv11 缺陷檢測 — 完整專案包

此資料夾包含訓練 YOLOv11m 模型所需的所有檔案，可直接用於 GitHub 上傳或本地開發。

## 📁 資料夾結構

```
yolov11_new/
├── .github/
│   └── copilot-instructions.md       ← 專案開發指南
├── scripts/
│   ├── augment_yolo.py               ← 資料擴增腳本
│   ├── preprocess.py                 ← 前處理腳本
│   ├── infer_train.py                ← 推論與裁切
│   └── eval.py                       ← 驗證腳本
├── output_train_cut/
│   └── datasets/
│       ├── images/
│       │   ├── aug_train/   (4064張) ← 擴增後訓練集
│       │   ├── train/       (1016張) ← 原始訓練集
│       │   ├── val/         (266張)  ← 驗證集
│       │   └── test/        (265張)  ← 測試集
│       └── labels/          (YOLO 標註檔)
├── train.py                          ← 主訓練腳本 ⭐
├── data_augmented.yaml               ← 資料集配置 ⭐
├── yolo11m.pt                        ← 預訓練權重 ⭐
├── cut.py                            ← ROI 切割
├── mp4_to_jpg.py                     ← 視頻幀提取
└── README.md                         ← 完整說明文件
```

## 🚀 快速開始

### 前置條件
```powershell
# 1. 安裝 uv (Python 環境管理器)
pipx install uv

# 2. 建立虛擬環境
uv venv

# 3. 激活虛擬環境
.\.venv\Scripts\Activate.ps1

# 4. 安裝依賴
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install ultralytics opencv-python pandas numpy tqdm pillow
```

### 執行訓練
```powershell
# 依序運行兩個實驗 (freeze=0 和 freeze=10)
python train.py

# 或使用 uv
uv run --python .venv\Scripts\python.exe train.py
```

## 📊 資料集統計

| Split | 數量 | 說明 |
|-------|------|------|
| aug_train | 4064 | 擴增後訓練集 (1016 原始 + 3048 擴增) |
| train | 1016 | 原始訓練集 |
| val | 266 | 驗證集 |
| test | 265 | 測試集 |
| **總計** | **5611** | 不含重複 |

## 🔬 實驗設計

### Exp A: 完整微調 (freeze=0)
- 所有網路層參與訓練
- 充分利用 COCO 預訓練特徵
- 結果: `runs/pretrained_study/Exp_A_freeze0_FullFT/`

### Exp B: 凍結 Backbone (freeze=10)
- 凍結前 10 層（Backbone）
- 只訓練頭部層
- 結果: `runs/pretrained_study/Exp_B_freeze10_BackboneFrozen/`

## 📈 訓練輸出

完成後的結果結構：
```
runs/pretrained_study/
├── Exp_A_freeze0_FullFT/
│   ├── weights/best.pt              ← 最佳模型
│   ├── results.csv                  ← 訓練曲線數據
│   └── events.out.*                 ← TensorBoard 日誌
└── Exp_B_freeze10_BackboneFrozen/
    ├── weights/best.pt
    ├── results.csv
    └── events.out.*
```

## 🔧 配置說明

**data_augmented.yaml** （資料集配置）：
```yaml
path: C:/Users/User/wyc/yolov11/output_train_cut/datasets
train: images/aug_train    # 使用擴增後的訓練集
val: images/val
test: images/test
nc: 1                      # 單一類別 (NG 缺陷)
names: ['NG']
```

**train.py** 中的超參數：
- `epochs=300`：最多 300 輪
- `patience=50`：早停機制（50 輪無進步停止）
- `batch=16`：批量大小
- `imgsz=640`：輸入尺寸
- `freeze=0/10`：凍結層數（實驗變量）

## ⚠️ 常見問題

### 1. GPU 記憶體不足
```powershell
# 修改 train.py 中的 batch 大小
batch=8  # 改小一點
```

### 2. 路徑錯誤
- 檢查 `data_augmented.yaml` 中 path 是否指向正確的資料夾
- 在不同機器上可能需要調整絕對路徑

### 3. 模型權重缺失
- 確認 `yolo11m.pt` 在根目錄
- 若無法自動下載，手動下載放置

## 📝 關鍵檔案說明

| 檔案 | 用途 |
|------|------|
| **train.py** | 主訓練腳本，依序運行兩個實驗 |
| **data_augmented.yaml** | 資料集配置，指向訓練/驗證/測試檔案 |
| **scripts/augment_yolo.py** | 自動擴增（翻轉、旋轉、平移） |
| **cut.py** | ROI 切割腳本 |
| **output_train_cut/datasets/** | 完整的訓練資料集 |
| **.github/copilot-instructions.md** | 專案開發指南 |

## 🎯 後續分析

訓練完成後，建議：
1. 比較 `results.csv` 中的 mAP、precision、recall
2. 分析 freeze 層數對收斂速度的影響
3. 生成訓練曲線對比圖
4. 記錄最優超參數用於後續改進

## 📞 支援

- 詳細說明見 `.github/copilot-instructions.md`
- 原始 README 見 `README.md`
- 有問題請檢查訓練日誌輸出

---

**建立日期**: 2026年4月24日  
**資料集來源**: output_train_cut (4064 張擴增訓練集)  
**模型**: YOLOv11m 預訓練權重  
**類別**: NG (單一缺陷類別)
