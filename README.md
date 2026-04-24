# YOLOv11 缺陷框選與裁切 README

本文件說明 YOLOv11m 物件偵測模型的推論、裁切與訓練工作流。

## 1) 專案總覽

- **目標**：使用 YOLOv11m 框選 data260420 訓練集中的缺陷，產出裁切並訓練
- **資料集**：data260420（train / test / val 彼此獨立，不混用）
- **預訓練權重**：runs/detect/yolov11m_aug/weights/best.pt
- **工作流**：
  1. 對 train 資料進行推論與裁切（針對 txt 標註高速框選）
  2. 訓練自定義模型
  3. 驗證 test/val 集上的性能

## 2) 環境需求與安裝

- **推薦 Python 版本**：3.10–3.11
- **套件管理**：使用 `uv` 管理虛擬環境與依賴（支援跨裝置快速環境遷移）

### 安裝步驟（Windows PowerShell）

```powershell
# 1. 全域安裝 uv（若未安裝）
pipx install uv

# 2. 建立虛擬環境（自動在 .venv）
uv venv

# 3. 激活虛擬環境
.\.venv\Scripts\Activate.ps1

# 4. 安裝依賴
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install ultralytics opencv-python pandas numpy tqdm pillow

# 或者，使用 requirements.txt（見下文）
uv pip install -r requirements.txt
```

### 跨裝置環境遷移
```powershell
# 在舊裝置匯出依賴
uv pip freeze > requirements.txt

# 在新裝置上快速重建
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

- **GPU/CPU 選擇**：所有命令可用 `--device cuda` 啟用 GPU；若無 CUDA，改用 `--device cpu`。

## 3) 資料與路徑約定

```
data260420/
  train/              ← 推論 & 裁切對象
  test/               ← 驗證用，不進行任何前處理
  val/                ← 驗證用，不進行任何前處理

runs/detect/yolov11m_aug/weights/best.pt  ← 預訓練 YOLOv11m 權重

output_train/         ← train 推論與裁切輸出
  crops/{class}/      ← 按類別存儲的裁切 ROI（中間格式）
  infer_results.json  ← 推論結果（邊界框、信心度等）
  annotated/          ← 標註圖像（可選）
  datasets/           ← 訓練用資料集（裁切 + 前處理後）
    images/
      train/          ← 訓練裁切圖像
      val/            ← 驗證裁切圖像
      test/           ← 測試裁切圖像
    labels/
      train/          ← 訓練標註檔（txt）
      val/            ← 驗證標註檔（txt）
      test/           ← 測試標註檔（txt）
```

## 4) 工作流步驟

## 5) 步驟一：Train 資料推論與裁切

**目的**：使用預訓練模型自動框選 train 資料中的缺陷，產出 ROI 裁切及標註（替代 labelimg）

**關鍵腳本**：[scripts/infer_train.py](scripts/infer_train.py)

```bash
# 推論 data260420/train，輸出裁切 + YOLO 標註到 output_train/crops/{class}
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
output_train/
  crops/
    {class1}/
      image1_crop_0.jpg         ← 裁切圖像（中間格式）
      image1_crop_0.txt         ← 對應 YOLO 標註
      image2_crop_0.jpg
      image2_crop_0.txt
      ...
    {class2}/
      ...
  infer_results.json            ← 推論結果彙整（邊界框、信心度）
  annotated/                    ← 標註圖像（可選）
  datasets/                     ← 訓練用資料集（最終格式）
    images/
      train/                    ← 訓練集裁切圖像（來自 crops/ 與前處理）
      val/                      ← 驗證集裁切圖像
      test/                     ← 測試集裁切圖像
    labels/
      train/                    ← 訓練集標註檔
      val/                      ← 驗證集標註檔
      test/                     ← 測試集標註檔
```

**重點**：
- `crops/{class}/` — 推論後的中間格式（按類別存儲）
- `datasets/` — 最終訓練格式（裁切 + 前處理後的數據，按 train/val/test 分組）
  - 前處理後的圖像應該複製到此目錄的 `images/{train/val/test}/`
  - 對應標註檔複製到 `labels/{train/val/test}/`

**關鍵參數**：
- `--conf`：信心度門檻（調低 → 更多裁切）
- `--iou`：NMS IoU（控制邊界框重疊）
- `--save-crops`：輸出按類別組織的裁切 + 對應 txt 標註
- `--save-json`：導出推論結果為 JSON

## 5-1) 前處理（可選）

**目的**：對裁切圖像進行濾波或增強，支援多種方式

**關鍵腳本**：[scripts/preprocess.py](scripts/preprocess.py)

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

**組織訓練資料**：
前處理完成後，需將裁切圖像與標註整理到 `output_train/datasets/` 以供訓練：
```powershell
# 示例：使用高斯濾波後的結果
xcopy output_train\crops_gaussian\* output_train\datasets\images\ /S /Y
# （裁切圖像與標註會自動複製到 train/val/test/ 子資料夾）
```

## 6) 步驟二：訓練自定義模型

**目的**：在整理後的裁切數據上訓練模型

**關鍵腳本**：[scripts/train_custom.py](scripts/train_custom.py)（待編寫）

```bash
# 訓練示例（從 datasets/ 目錄讀取數據）
python scripts/train_custom.py \
  --train-root output_train/datasets \
  --epochs 50 --batch 64 --lr 1e-3 \
  --device cuda --out runs/custom_train
```

**輸出**：
- `runs/custom_train/weights/best.pt` — 最佳模型
- `runs/custom_train/metrics.csv` — 訓練曲線

## 7) 步驟三：驗證 Test & Val 集

**目的**：在 test 與 val 集上評估模型性能（不進行任何裁切或前處理）

**關鍵腳本**：[scripts/eval.py](scripts/eval.py)（待編寫）

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

**輸出**：指標報告（precision、recall、mAP 等）

## 8) 快速參考

| 步驟 | 目的 | 指令 |
|------|------|------|
| 環境 | 建立虛擬環境 | `uv venv && .\.venv\Scripts\Activate.ps1` |
| 安裝 | 裝套件 | `uv pip install -r requirements.txt` |
| 推論 | Train 自動框選裁切 | `python scripts/infer_train.py --weights ... --source data260420/train ...` |
| 前處理 | 濾波裁切圖像（可選） | `python scripts/preprocess.py --input output_train/crops --method gaussian ...` |
| 整理 | 複製到 datasets | 使用 PowerShell `xcopy` 或資源管理器複製裁切圖像到 `output_train/datasets/` |
| 訓練 | 自定義模型 | `python scripts/train_custom.py --train-root output_train/datasets ...` |
| 驗證 | 測試 Test/Val | `python scripts/eval.py --weights ... --source data260420/test ...` |

## 9) 常見問題

- **找不到 scripts 目錄**：確保已創建必要的腳本（見步驟一、二、三）
- **GPU 不可用**：改用 `--device cpu` 或檢查 CUDA 安裝
- **推論結果不好**（檢測框太多/太少/位置不對）：
  - **框太多**：提高 `--conf` 或 `--iou` → **重新推論** `infer_train.py`
  - **框太少**：降低 `--conf` → **重新推論** `infer_train.py`
  - **圖像模糊或品質差**：使用 `preprocess.py` 濾波增強，**無需重推論**
- **跨裝置環境**：使用 `uv pip freeze > requirements.txt` 與 `uv pip install -r requirements.txt` 快速遷移

## 10) 常見工作流與檢查表

### 換裝置快速使用
1. 複製整個專案資料夾（包含 data260420、runs 等）
2. 在新裝置激活虛擬環境並安裝依賴：
   ```powershell
   uv venv && .\.venv\Scripts\Activate.ps1 && uv pip install -r requirements.txt
   ```
3. 直接運行推論、前處理、訓練或驗證腳本

### 從頭開始工作流
1. **推論 Train**：執行 `infer_train.py` 產生裁切（調整 `--conf`/`--iou` 控制檢測框數量與質量）
   ```bash
   python scripts/infer_train.py --weights ... --source data260420/train --conf 0.25 --iou 0.45 ...
   ```
   - 輸出：`output_train/crops/{class}/` + jpg + txt
   - **如果檢測結果不好**（框太多/太少/不準）→ 調整參數**重新推論**

2. **前處理（可選）**：執行 `preprocess.py` 改善裁切圖像質量（不需要重推論）
   ```bash
   python scripts/preprocess.py --input output_train/crops --method gaussian --kernel 5 ...
   ```
   - 輸出：`output_train/crops_{method_name}/`
   - **用途**：濾波、去噪、對比度增強等（改善圖像品質）

3. **整理訓練資料**：複製裁切或前處理後的圖像 & 標註到 `output_train/datasets/images/{train|val|test}/` 與 `labels/{train|val|test}/`
   ```powershell
   # 示例：複製前處理後的圖像
   xcopy output_train\crops_gaussian\* output_train\datasets\ /S /Y
   ```

4. **訓練自定義模型**：執行 `train_custom.py`（從 `output_train/datasets` 讀取資料）
   ```bash
   python scripts/train_custom.py --train-root output_train/datasets ...
   ```

5. **驗證 Test/Val**：執行 `eval.py` 評估 test 與 val 集（不進行任何前處理）
   ```bash
   python scripts/eval.py --weights ... --source data260420/test ...
   ```