import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from ultralytics import YOLO
from multiprocessing import freeze_support

# 設定日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """載入 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def merge_configs(file_config, cli_args):
    """合併配置文件和命令行參數（命令行優先）"""
    config = file_config.copy()
    
    if cli_args.model:
        config['model']['path'] = cli_args.model
    if cli_args.device:
        config['inference']['device'] = cli_args.device
    if cli_args.batch_size:
        config['dataset']['batch_size'] = cli_args.batch_size
    if cli_args.output_dir:
        config['output']['base_dir'] = cli_args.output_dir
    
    return config


def evaluate_and_compare(model_path, data_config, config):
    """
    在 test 集上評估模型，並與訓練期間的 val 指標對比
    """
    logger.info("\n" + "="*100)
    logger.info("YOLOv11m Test Set Evaluation")
    logger.info("="*100 + "\n")
    
    # 1. 載入模型
    logger.info(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # 2. 運行驗證
    logger.info(f"Evaluating on test set with config: {data_config}")
    results = model.val(
        data=data_config,
        split=config['dataset'].get('split', 'val'),
        imgsz=config['dataset']['imgsz'],
        batch=config['dataset']['batch_size'],
        conf=config['inference']['conf_threshold'],
        iou=config['inference']['iou_threshold'],
        device=config['inference']['device'],
        verbose=True
    )
    
    # 3. 提取測試結果
    results_dict = getattr(results, 'results_dict', None) or {}
    box = getattr(results, 'box', None)

    def _first_present(dct, keys, default=None):
        for key in keys:
            if key in dct and dct[key] is not None:
                return dct[key]
        return default

    precision = _first_present(results_dict, ['metrics/precision(B)', 'metrics/precision'], default=None)
    recall = _first_present(results_dict, ['metrics/recall(B)', 'metrics/recall'], default=None)
    map50 = _first_present(results_dict, ['metrics/mAP50(B)', 'metrics/mAP50'], default=None)
    map5095 = _first_present(results_dict, ['metrics/mAP50-95(B)', 'metrics/mAP50-95'], default=None)

    if box is not None:
        if map50 is None and hasattr(box, 'map50'):
            map50 = box.map50
        if map5095 is None and hasattr(box, 'map'):
            map5095 = box.map

    test_metrics = {
        'precision': float(precision) if precision is not None else None,
        'recall': float(recall) if recall is not None else None,
        'mAP50': float(map50) if map50 is not None else None,
        'mAP50_95': float(map5095) if map5095 is not None else None,
        'total_images': config['dataset'].get('test_size'),
        'split': config['dataset'].get('split', 'val'),
    }
    
    logger.info("\n✓ Test Set Evaluation Results:")
    if test_metrics['mAP50'] is not None:
        logger.info(f"  mAP50: {test_metrics['mAP50']:.4f}")
    if test_metrics['mAP50_95'] is not None:
        logger.info(f"  mAP50-95: {test_metrics['mAP50_95']:.4f}")
    
    return results, test_metrics


def generate_comparison_report(test_metrics, config):
    """
    生成訓練 val vs 測試的對比報告
    """
    logger.info("\n" + "="*100)
    logger.info("Train Val vs Test Comparison")
    logger.info("="*100 + "\n")
    
    train_metrics = config['train_val_metrics']
    
    # 計算差異
    comparison = {
        'mAP50': {
            'train': train_metrics['mAP50'],
            'test': test_metrics['mAP50'],
            'diff': test_metrics['mAP50'] - train_metrics['mAP50'],
            'diff_pct': ((test_metrics['mAP50'] - train_metrics['mAP50']) / train_metrics['mAP50'] * 100) if train_metrics['mAP50'] > 0 else 0
        },
        'mAP50_95': {
            'train': train_metrics['mAP50_95'],
            'test': test_metrics['mAP50_95'],
            'diff': test_metrics['mAP50_95'] - train_metrics['mAP50_95'],
            'diff_pct': ((test_metrics['mAP50_95'] - train_metrics['mAP50_95']) / train_metrics['mAP50_95'] * 100) if train_metrics['mAP50_95'] > 0 else 0
        }
    }
    
    # 打印對比表
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│  Metric        │  Train Val  │  Test   │  Difference  │  Trend  │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    for metric_name, values in comparison.items():
        train_val = f"{values['train']:.4f}"
        test_val = f"{values['test']:.4f}"
        diff = f"{values['diff']:+.4f}"
        trend = "📈" if values['diff'] > 0 else "📉" if values['diff'] < 0 else "→"
        print(f"│ {metric_name:<14} │ {train_val:>11} │ {test_val:>7} │ {diff:>12} │ {trend:>7} │")
    
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # 過擬合分析
    logger.info("\nOverfitting Analysis:")
    if comparison['mAP50']['diff_pct'] < -5:
        logger.warning(f"  ⚠️  Possible overfitting detected!")
        logger.warning(f"     mAP50 dropped {abs(comparison['mAP50']['diff_pct']):.2f}% from train to test")
    elif comparison['mAP50']['diff_pct'] > 0:
        logger.info(f"  ✓ Test mAP50 is even better! ({comparison['mAP50']['diff_pct']:+.2f}%)")
    else:
        logger.info(f"  → Minor gap (acceptable)")
    
    return comparison


def generate_confusion_matrix(results, config, output_dir):
    """
    生成混淆矩陣圖表
    """
    logger.info("\nGenerating confusion matrix...")
    
    try:
        # YOLO 模型的混淆矩陣
        cm_obj = getattr(results, 'confusion_matrix', None)
        if cm_obj is None:
            logger.warning("  Confusion matrix not available in results (ultralytics may require plots=True).")
            return

        cm = getattr(cm_obj, 'matrix', None)
        if cm is None:
            logger.warning("  Confusion matrix matrix not found.")
            return

        cm = np.asarray(cm)
        if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
            logger.warning(f"  Unexpected confusion matrix shape: {cm.shape}")
            return

        # 嘗試從 ultralytics 取類別名稱（偵測任務通常會多一個 background）
        names = getattr(results, 'names', None)
        if isinstance(names, dict):
            labels = [names[i] for i in sorted(names.keys())]
        elif isinstance(names, (list, tuple)):
            labels = list(names)
        else:
            labels = [f"class_{i}" for i in range(cm.shape[0] - 1)]

        if len(labels) == cm.shape[0] - 1:
            labels.append('background')
        if len(labels) != cm.shape[0]:
            labels = [f"label_{i}" for i in range(cm.shape[0])]

        figure, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_title(f"Confusion Matrix - {config['model']['experiment_name']}", fontsize=14, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels, rotation=0)

        cm_path = output_dir / 'confusion_matrix.png'
        plt.tight_layout()
        plt.savefig(cm_path, dpi=config['visualization']['dpi'])
        logger.info(f"  ✓ Saved to {cm_path}")
        plt.close()
            
    except Exception as e:
        logger.warning(f"  Could not generate confusion matrix: {e}")


def generate_report_file(test_metrics, comparison, config, output_dir):
    """
    生成文字報告
    """
    report_path = output_dir / 'evaluation_report.txt'
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*100 + "\n")
        f.write("YOLOv11m Test Set Evaluation Report\n")
        f.write("="*100 + "\n\n")
        
        # 模型信息
        f.write("Model Information:\n")
        f.write(f"  Experiment: {config['model']['experiment_name']}\n")
        f.write(f"  Freeze: {config['model']['freeze_layers']} (Frozen Backbone)\n")
        f.write(f"  Description: {config['model']['description']}\n\n")
        
        # 測試集信息
        f.write("Test Set Information:\n")
        f.write(f"  Total images: {config['dataset']['test_size']}\n")
        f.write(f"  Batch size: {config['dataset']['batch_size']}\n")
        f.write(f"  Image size: {config['dataset']['imgsz']}\n")
        f.write(f"  Evaluation time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 測試結果
        f.write("Test Set Results:\n")
        f.write(f"  mAP50: {test_metrics['mAP50']:.4f}\n")
        f.write(f"  mAP50-95: {test_metrics['mAP50_95']:.4f}\n\n")
        
        # 對比分析
        f.write("Train Val vs Test Comparison:\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Metric':<20} {'Train Val':<20} {'Test':<20} {'Difference':<20} {'Trend':<20}\n")
        f.write("-"*100 + "\n")
        
        for metric_name, values in comparison.items():
            diff_str = f"{values['diff']:+.4f} ({values['diff_pct']:+.2f}%)"
            trend = "📈 Better" if values['diff'] > 0 else "📉 Worse" if values['diff'] < 0 else "→ Same"
            f.write(f"{metric_name:<20} {values['train']:<20.4f} {values['test']:<20.4f} {diff_str:<20} {trend:<20}\n")
        
        f.write("\n" + "="*100 + "\n")
        f.write("Recommendations:\n")
        f.write("-"*100 + "\n")
        
        # 根據結果生成建議
        if comparison['mAP50']['diff_pct'] < -5:
            f.write("⚠️  WARNING: Model shows overfitting signs\n")
            f.write("   - Test performance significantly lower than training\n")
            f.write("   - Consider: reduce model complexity, increase regularization, add more training data\n")
        elif comparison['mAP50']['diff_pct'] > 0:
            f.write("✓ Good: Model generalizes well!\n")
            f.write("   - Test performance even better than training\n")
            f.write("   - Model is ready for deployment\n")
        else:
            f.write("✓ Acceptable: Minor performance gap\n")
            f.write("   - Model generalizes reasonably well\n")
        
    logger.info(f"✓ Report saved to {report_path}")


def save_metrics_json(test_metrics, comparison, config, output_dir):
    """
    保存指標為 JSON
    """
    metrics_data = {
        'timestamp': datetime.now().isoformat(),
        'model': config['model'],
        'test_metrics': test_metrics,
        'comparison': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in comparison.items()},
        'dataset': config['dataset'],
    }
    
    json_path = output_dir / 'metrics.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✓ Metrics saved to {json_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv11 model on test set')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, help='Model path (overrides config)')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu, overrides config)')
    parser.add_argument('--batch-size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--output-dir', type=str, help='Output directory (overrides config)')
    
    args = parser.parse_args()
    
    # 載入配置
    if args.config:
        config = load_config(args.config)
        logger.info(f"Loaded config from: {args.config}")
    else:
        # 使用預設配置文件
        default_config = Path('eval_configs/exp_b_test.yaml')
        if default_config.exists():
            config = load_config(default_config)
            logger.info(f"Loaded default config from: {default_config}")
        else:
            raise FileNotFoundError("No config file found. Use --config to specify one.")
    
    # 合併命令行參數
    config = merge_configs(config, args)
    
    # 驗證模型檔案
    model_path = config['model']['path']
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # 驗證資料集配置
    data_config = config['dataset']['config']
    if not os.path.exists(data_config):
        raise FileNotFoundError(f"Dataset config not found: {data_config}")
    
    # 建立輸出目錄
    output_base = Path(config['output']['base_dir'])
    output_dir = output_base / config['output']['exp_subdir']
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # 1. 評估模型
    results, test_metrics = evaluate_and_compare(model_path, data_config, config)
    
    # 2. 生成對比分析
    comparison = generate_comparison_report(test_metrics, config)
    
    # 3. 生成混淆矩陣
    if config['visualization']['plot_confusion_matrix']:
        generate_confusion_matrix(results, config, output_dir)
    
    # 4. 保存報告
    generate_report_file(test_metrics, comparison, config, output_dir)
    
    # 5. 保存指標 JSON
    save_metrics_json(test_metrics, comparison, config, output_dir)
    
    logger.info("\n" + "="*100)
    logger.info("✓ Evaluation completed successfully!")
    logger.info(f"All results saved to: {output_dir}")
    logger.info("="*100 + "\n")


if __name__ == '__main__':
    freeze_support()
    main()
