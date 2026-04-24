# YOLOv11 缺陷框選與裁切 — 專案指南

## 概述

本專案使用 **YOLOv11m** 進行物件偵測與裁切：
- **資料流程**：重新拍攝 20 根管子，切分為 train/val/test
- **標註與擴增階段**：ROI 切割後先完成標註確認，再做資料擴增（翻轉、旋轉、平移）
- **訓練階段**：使用擴增後 train 資料訓練 YOLO
- **驗證階段**：使用 val/test 資料驗證模型性能

詳見 [README.md](../README.md)

## 環境與安裝

**Python 版本**：3.10–3.11

**套件管理**：使用 `uv` 管理虛擬環境，支援快速跨裝置遷移

### Windows PowerShell 安裝步驟

```powershell
# 1. 全域安裝 uv（若未安裝）
pipx install uv

# 2. 建立虛擬環境
uv venv

# 3. 激活虛擬環境
.\.venv\Scripts\Activate.ps1

# 4. 安裝依賴
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install ultralytics opencv-python pandas numpy tqdm pillow
```

### 環境遷移
```powershell
# 匯出依賴
uv pip freeze > requirements.txt

# 在新裝置快速安裝
uv venv && .\.venv\Scripts\Activate.ps1 && uv pip install -r requirements.txt
```

**GPU/CPU 選擇**：所有命令支援 `--device cuda`（需 CUDA 11.8+）；無 GPU 改用 `--device cpu`。

## 目錄結構與路徑約定

```
data260420/                    # 單一數據集根目錄（彼此獨立）
  train/                       ← 訓練來源（16 根管，含 4 無瑕疵）
  test/                        ← 測試來源（4 根管：2 瑕疵 + 2 ok）
  val/                         ← 驗證來源（4 根管：2 瑕疵 + 2 ok）

runs/detect/yolov11m_aug/
  weights/best.pt              ← 預訓練 YOLOv11m 權重

output_train/                  ← train 推論與裁切輸出
  crops/{class}/               ← 按類別組織的裁切 ROI（中間格式）
  aug_train/                   ← 擴增後 train 圖像（翻轉/旋轉/平移）
  infer_results.json           ← 推論結果（邊界框、信心度）
  annotated/                   ← 標註圖像（可選）
  datasets/                    ← 訓練用資料集（裁切 + 前處理後）
    images/
      train/                   ← 原始訓練裁切圖像
      aug_train/               ← 擴增後訓練裁切圖像
      val/                     ← 驗證裁切圖像
      test/                    ← 測試裁切圖像
    labels/
      train/                   ← 訓練標註檔（txt）
      aug_train/               ← 擴增後訓練標註檔（txt）
      val/                     ← 驗證標註檔（txt）
      test/                    ← 測試標註檔（txt）

runs/custom_train/             ← 自定義模型訓練輸出
  weights/best.pt              ← 最佳模型
  metrics.csv                  ← 訓練曲線

runs/custom_train/eval_*/      ← Test/Val 驗證輸出
  metrics.json                 ← 評估指標
```

## 關鍵腳本與工作流

### 1. 推論與裁切（Train 資料）— 替代 labelimg
[**scripts/infer_train.py**](../scripts/infer_train.py)

使用預訓練 YOLOv11m 自動框選，產出裁切圖像 + YOLO 標註檔：

```bash
python scripts/infer_train.py \
  --weights runs/detect/yolov11m_aug/weights/best.pt \
  --source data260420/train \
  --out output_train \
  --device cuda \
  --imgsz 640 --conf 0.25 --iou 0.45 \
  --save-crops --save-json
```

**輸出結構**：
```
output_train/crops/{class}/
  image1_crop_0.jpg + image1_crop_0.txt (YOLO format)
  image2_crop_0.jpg + image2_crop_0.txt
output_train/infer_results.json           (推論統計)
```

**參數說明**：
- `--conf`：信心度門檻（調低 → 更多裁切）
- `--iou`：NMS IoU（控制邊界框重疊）
- `--save-crops`：輸出按類別組織的裁切 + 對應 txt 標註
- `--save-json`：導出推論結果為 JSON

### 1-1. ROI 切割
[**cut.py**](../cut.py)

對抽幀後影像先做 ROI 切割，再進入擴增與標註。

### 1-2. 資料擴增（目前實驗設定）

Train 在「標註確認完成後」只做下列三種擴增：
- 翻轉（flip）
- 旋轉（rotate）
- 平移（translate）

擴增規則：
- 程式需根據原始標註 `txt` 自動計算擴增後框座標
- 影像與標註必須同步變換（同檔名對應）

數量定義：
- 抽幀後資料量：train/val/test = 1016 / 266 / 265
- 擴增後 train：3048（依目前實驗定義）

### 1-3. 前處理（可選）— 裁切後濾波
[**scripts/preprocess.py**](../scripts/preprocess.py)

對裁切圖像進行濾波增強，支援多種方式：

```bash
# 高斯濾波
python scripts/preprocess.py \
  --input output_train/crops \
  --output output_train/crops_gaussian \
  --method gaussian --kernel 5

# 中值濾波
python scripts/preprocess.py \
  --input output_train/crops \
  --output output_train/crops_median \
  --method median --kernel 5

# 雙邊濾波（保留邊緣）
python scripts/preprocess.py \
  --input output_train/crops \
  --output output_train/crops_bilateral \
  --method bilateral --d 9 --sigma-color 75 --sigma-space 75

# CLAHE（對比度增強）
python scripts/preprocess.py \
  --input output_train/crops \
  --output output_train/crops_clahe \
  --method clahe --clip-limit 2.0 --tile-size 8
```

**可用前處理方式**：
- `none` — 不處理（直接複製）
- `gaussian` — 高斯濾波（參數：kernel）
- `median` — 中值濾波（參數：kernel）
- `bilateral` — 雙邊濾波（參數：d, sigma_color, sigma_space）
- `morphopen` — 形態開運算（參數：kernel）
- `morphclose` — 形態閉運算（參數：kernel）
- `clahe` — 對比度拉伸（參數：clip_limit, tile_size）

**整理訓練資料**：
- train：保留原始 train 圖像；擴增圖像另存為 `aug_train/`
- val/test：使用 ROI 切割後資料（不做擴增）
- 原始 train 寫入 `output_train/datasets/images/train/` 與 `labels/train/`
- 擴增後 train 寫入 `output_train/datasets/images/aug_train/` 與 `labels/aug_train/`
- 影像移動/複製時，對應 `txt` 必須同步移動到相同 split（檔名需一致）

### 2. 訓練自定義模型 — 預訓練影響研究
[**train.py**](../train.py)

**研究目標**：比較 `freeze` 層數對模型訓練的影響
- **Exp A**：`freeze=0`（完整微調）— 所有層參與訓練
- **Exp B**：`freeze=10`（凍結 Backbone）— 只訓練頭部層

**執行方式**：

```bash
python train.py
```

或使用 `uv` 管理器：

```bash
uv run --python .venv\Scripts\python.exe train.py
```

**超參數共同設定**：
- `pretrained=True`：使用 YOLOv11m 預訓練權重
- `epochs=300, patience=50`：最多 300 輪，連續 50 輪無進步自動停止
- `batch=16, imgsz=640`：批量大小與輸入尺寸
- `seed=42, deterministic=True`：固定隨機種子確保結果可重複性
- `close_mosaic=10`：最後 10 輪關閉 Mosaic 增強

**變異參數** — 研究凍結層對訓練的影響：
- `freeze=0`：完整微調（Exp A）
- `freeze=10`：凍結 Backbone 層 0~9（Exp B）

**資料集配置** [**data_augmented.yaml**](../data_augmented.yaml)：

```yaml
path: C:/Users/User/wyc/yolov11/output_train_cut/datasets
train: images/aug_train          # 擴增後訓練集 (4064 張 = 1016 原始 + 3048 擴增)
val: images/val                  # 驗證集 (266 張)
test: images/test                # 測試集 (265 張)

nc: 1
names: ['NG']
```

**輸出結構**：
```
runs/pretrained_study/
  Exp_A_freeze0_FullFT/
    weights/best.pt
    results.csv                  ← 訓練曲線 (freeze=0)
    events.out.*
  Exp_B_freeze10_BackboneFrozen/
    weights/best.pt
    results.csv                  ← 訓練曲線 (freeze=10)
    events.out.*
```

**後續分析** — 比較兩個實驗：
- 檢視 `results.csv` 中的 mAP、precision、recall 等指標
- 分析 freeze 層數對收斂速度和最終性能的影響

### 3. 驗證 Test 與 Val 集
[**scripts/eval.py**](../scripts/eval.py)

```bash
# 評估 test 集
python scripts/eval.py \
  --weights runs/custom_train/weights/best.pt \
  --source data260420/test \
  --device cuda --out runs/custom_train/eval_test

# 評估 val 集
python scripts/eval.py \
  --weights runs/custom_train/weights/best.pt \
  --source data260420/val \
  --device cuda --out runs/custom_train/eval_val
```

## 程式碼風格與慣例

### Python
- **版本**：3.10+；使用 f-string、type hints（推薦）
- **導入順序**：torch/torchvision → ultralytics → cv2/pandas → 本地
- **命名**：
  - 函式/變數：`snake_case`（如 `infer_train`, `save_crops`）
  - 類別：`PascalCase`（如 `YOLOInference`, `CropDataset`）
  - 常數：`UPPER_SNAKE_CASE`（如 `IMG_SIZE`, `BATCH_SIZE`）
- **文件字符集**：UTF-8；支援繁體中文註解

### 資料流命名
- **裁切資料夾**：`crops/{class_name}/` — jpg 圖像 + txt 標註配對
- **推論輸出**：`infer_results.json` — 含邊界框、信心度、類別
- **前處理輸出**：`crops_{method_name}/` — 濾波後圖像 + 複製的 txt 標註
- **模型**：`best.pt`、`last.pt`（PyTorch 格式）

### 工作流約定
- **Train 資料**：抽幀 → ROI 切割 → 擴增（翻轉/旋轉/平移）→ 標註/訓練
- **Val/Test 資料**：抽幀 → ROI 切割 → 標註（不做擴增）
- **模型輸出**：`runs/custom_train/` 按實驗遞增編號（custom_train、custom_train_v2 等）
- **實驗追蹤**：所有超參在指令列明確化；每次訓練產生 `metrics.csv`

## 常見工作流與檢查表

### 換裝置快速使用
1. 複製整個專案資料夾（包含 data260420、runs 等）
2. 在新裝置激活虛擬環境並安裝依賴：
   ```powershell
   uv venv && .\.venv\Scripts\Activate.ps1 && uv pip install -r requirements.txt
   ```
3. 直接運行推論、前處理、訓練或驗證腳本

### 從頭開始工作流
1. **抽幀**：以 0.1 秒抽 1 張，得到 train/val/test = 1016/266/265
2. **ROI 切割**：執行 `cut.py` 對三個 split 做切割
3. **標註輸出**：執行 `infer_train.py` 產生 YOLO txt，並先確認標註品質
4. **Train 擴增**：只做翻轉、旋轉、平移，擴增到 4064(3048 + 1016)（依現有 txt 自動重算框座標）
5. **整理資料集**：
  - 保留原始 train（不覆蓋），擴增結果獨立存為 `aug_train/`
  - 將 train 原始 ROI 寫入 `output_train/datasets/images/train/`，標註寫入 `labels/train/`
  - 將 aug_train 寫入 `output_train/datasets/images/aug_train/`，標註寫入 `labels/aug_train/`
  - val/test 分別寫入各自 split 目錄
6. **訓練與驗證**：執行 `train_custom.py` 與 `eval.py`

### 疑難排除
- **GPU 記憶體不足**：減小 `--batch`、`--imgsz`；改用 `--device cpu`
- **推論結果不好**（檢測框太多/太少/位置不對）：
  - **框太多**：提高 `--conf` 或 `--iou` → 重新推論
  - **框太少**：降低 `--conf` → 重新推論
  - **圖像模糊**：使用 `preprocess.py` 濾波增強，無需重推論
- **前處理輸出目錄不存在**：確認 `--output` 路徑正確，會自動建立父目錄
- **跨裝置環境**：使用 `uv pip freeze > requirements.txt` 與快速安裝命令遷移
