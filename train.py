import os
import logging
from pathlib import Path

# 防止 Windows 環境下的重複庫錯誤
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
from multiprocessing import freeze_support

# 設定日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(exp_name: str, freeze: int, model_path: str = "yolo11m.pt", 
                   data_config: str = "data_augmented.yaml") -> dict:
    """
    執行單個訓練實驗。
    
    Args:
        exp_name: 實驗名稱（用於結果存檔）
        freeze: 凍結層數 (0=完整微調, 10=凍結 Backbone)
        model_path: 預訓練模型路徑
        data_config: 資料集配置檔
    
    Returns:
        訓練結果
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"開始實驗: {exp_name}")
    logger.info(f"凍結層數: freeze={freeze}")
    logger.info(f"{'='*60}\n")
    
    # 載入預訓練模型
    model = YOLO(model_path)
    
    # 共同的超參數 (確保公平比較)
    common_params = {
        "data": data_config,           # 使用新資料集配置
        "epochs": 300,                 # 設定 300 輪，配合早停機制
        "patience": 50,                # 連續 50 輪沒進步就自動停止
        "batch": 16,                   # RTX 3060 12G 可設 16
        "imgsz": 640,                  # 輸入尺寸
        "device": 0,                   # 指定 GPU
        "optimizer": "auto",           # 自動選擇優化器 (AdamW 或 SGD)
        "workers": 8,                  # 加速資料載入
        "close_mosaic": 10,            # 最後 10 輪關閉 Mosaic 增強
        "seed": 42,                    # 固定隨機種子確保可重複性
        "deterministic": True,         # 確定性模式
        
        # 實驗目標: 研究 pretrained 對訓練的影響
        "pretrained": True,            # 【核心】使用預訓練權重
        "freeze": freeze,              # 【變量】不同的凍結層數
        
        # 存檔管理
        "project": "runs/pretrained_study",
        "name": exp_name,
        "exist_ok": False,
    }
    
    results = model.train(**common_params)
    logger.info(f"✓ 實驗 {exp_name} 完成！結果已存檔至 runs/pretrained_study/{exp_name}\n")
    
    return results


def main():
    """主函數：順序執行兩個對比實驗"""
    
    logger.info("\n" + "="*60)
    logger.info("YOLOv11m 預訓練影響研究")
    logger.info("研究目標: 比較 freeze=0 vs freeze=10 的訓練效果")
    logger.info("資料集: output_train_cut/datasets (包含 4064 張擴增訓練集)")
    logger.info("="*60 + "\n")
    
    try:
        # 實驗 A: 完整微調 (freeze=0)
        logger.info("\n【實驗 A】完整微調 (freeze=0)")
        logger.info("說明: 所有層都參與訓練，充分利用預訓練特徵")
        results_A = run_experiment(
            exp_name="Exp_A_freeze0_FullFT",
            freeze=0
        )
        
        # 實驗 B: 凍結 Backbone (freeze=10)
        logger.info("\n【實驗 B】凍結 Backbone (freeze=10)")
        logger.info("說明: YOLO 定義 Layer 0~9 為 Backbone，凍結後只訓練頭部層")
        results_B = run_experiment(
            exp_name="Exp_B_freeze10_BackboneFrozen",
            freeze=10
        )
        
        logger.info("\n" + "="*60)
        logger.info("✓ 所有實驗完成！")
        logger.info("比較這兩個實驗的結果，分析 freeze 層數對訓練的影響:")
        logger.info("  - 檢視 runs/pretrained_study/Exp_A_freeze0_FullFT/results.csv")
        logger.info("  - 檢視 runs/pretrained_study/Exp_B_freeze10_BackboneFrozen/results.csv")
        logger.info("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"✗ 訓練過程出錯: {e}")
        raise


if __name__ == '__main__':
    freeze_support()
    main()