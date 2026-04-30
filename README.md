# YOLOv11 缺陷框選與裁切 — 專案指南

## 概述

本專案使用 **YOLOv11m** 進行物件偵測與裁切：
- **資料流程**：重新拍攝 20 根管子，切分為 train/val/test
- **標註與擴增階段**：ROI 切割後先完成標註確認，再做資料擴增（翻轉、旋轉、平移）
- **訓練階段**：使用擴增後 train 資料訓練 YOLO
- **驗證階段**：使用 val/test 資料驗證模型性能

## 最新成果（已完成）

### 預訓練權重凍結實驗（Freeze Study）

已完成 YOLOv11m 凍結層數對訓練效果的三組對照實驗（模型總層數 24；Backbone 0–9、Neck 10–22、Head 23）：

| Experiment | freeze | 說明 | epochs | mAP50 | mAP50-95 |
|---|---:|---|---:|---:|---:|
| Exp_A_freeze0_FullFT | 0 | 全部微調 | 109 | 0.8258 | (見 results.csv) |
| Exp_B_freeze10_BackboneFrozen | 10 | 凍結 Backbone（0–9） | 75 | **0.8512** | 0.4998 |
| Exp_C_freeze23_BackboneNeckFrozen | 23 | 凍結 Backbone+Neck（0–22） | 93 | 0.8171 | (見 results.csv) |

最佳模型（Exp_B）權重：
`runs/detect/runs/pretrained_study/Exp_B_freeze10_BackboneFrozen/weights/best.pt`

### Test 集最終評估腳本

已建立可在 test 集做最終評估與輸出報告的腳本：
- 主程式：`eval_test.py`
- 配置檔：`eval_configs/exp_b_test.yaml`
- 輸出位置（預設）：`runs/detect/runs/pretrained_study_eval/Exp_B_freeze10_test/`
  - `evaluation_report.txt`
  - `metrics.json`
  - `confusion_matrix.png`（若 ultralytics 版本在 val 回傳含 confusion_matrix；否則會提示）

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
\.venv\Scripts\Activate.ps1

# 4. 安裝依賴
uv pip install -r requirements.txt
```

## 目錄結構與路徑約定

```
data260420/                    # 單一數據集根目錄（彼此獨立）
  train/
  test/
  val/

# yolov11_new 內部（本資料夾）主要產物
output_train_cut/
  datasets/                    ← 目前訓練/驗證/測試主要資料集
    images/{aug_train|train|val|test}/
    labels/{aug_train|train|val|test}/

runs/detect/runs/pretrained_study/
  Exp_A_freeze0_FullFT/
  Exp_B_freeze10_BackboneFrozen/
  Exp_C_freeze23_BackboneNeckFrozen/

runs/detect/runs/pretrained_study_eval/
  Exp_B_freeze10_test/         ← eval_test.py 輸出
```

## 關鍵腳本與工作流

### 1) 推論與裁切（Train 資料）— 替代 labelimg

[scripts/infer_train.py](scripts/infer_train.py)

```bash
python scripts/infer_train.py \
  --weights yolo11m.pt \
  --source data260420/train \
  --out output_train \
  --device cuda \
  --imgsz 640 --conf 0.25 --iou 0.45 \
  --save-crops --save-json
```

### 2) 資料前處理（可選）

[scripts/preprocess.py](scripts/preprocess.py)

```bash
python scripts/preprocess.py \
  --input output_train/crops \
  --output output_train/crops_gaussian \
  --method gaussian --kernel 5
```

### 3) 訓練（Freeze Study：Exp A/B/C）

[train.py](train.py)

```bash
python train.py
```

### 4) 最終評估（Test 集）

[eval_test.py](eval_test.py)

```bash
python eval_test.py --config eval_configs/exp_b_test.yaml --device cuda
```

注意：`data_augmented.yaml` 的 `path:` 為絕對路徑，跨機器執行前請先更新成測試伺服器上的資料集位置。

## 快速參考

| 任務 | 指令 |
|---|---|
| 建立 + 啟用環境 | `uv venv && .\.venv\Scripts\Activate.ps1` |
| 安裝依賴 | `uv pip install -r requirements.txt` |
| Train 推論裁切（可選） | `python scripts/infer_train.py --weights yolo11m.pt --source data260420/train --out output_train --device cuda --imgsz 640 --conf 0.25 --iou 0.45 --save-crops --save-json` |
| 裁切前處理（可選） | `python scripts/preprocess.py --input output_train/crops --output output_train/crops_gaussian --method gaussian --kernel 5` |
| 執行三實驗訓練 | `python train.py` |
| Test 最終評估（Exp_B） | `python eval_test.py --config eval_configs/exp_b_test.yaml --device cuda` |

## 常見問題

- GPU 不可用：改用 `--device cpu`，或檢查 CUDA / 驅動與 torch CUDA 版本。
- 推論裁切框太多/太少：調整 `--conf` / `--iou` 後重新跑 `infer_train.py`。
- 跨機器執行：請先更新 `data_augmented.yaml` 的 `path:` 到測試伺服器的資料集實際位置。

## 建議工作流（論文/實驗一致）

1.（可選）用 `infer_train.py` 產生裁切與初始標註 → 人工快速 QC。
2.（可選）用 `preprocess.py` 做濾波/對比度增強（不影響標註座標）。
3. 用 `train.py` 跑 Exp_A/Exp_B/Exp_C（freeze=0/10/23）並比較 `results.csv`。
4. 用 `eval_test.py` 針對 Exp_B 最佳權重做 test 集最終評估並輸出報告。