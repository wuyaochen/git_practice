import os
import pandas as pd
from pathlib import Path
import json

# 結果路徑
base_path = r"C:\Users\User\wyc\yolov11\yolov11_new\runs\detect\runs\pretrained_study"

experiments = {
    "Exp_A_freeze0_FullFT": "完整微調 (所有層)",
    "Exp_B_freeze10_BackboneFrozen": "凍結 Backbone (層 0-9)",
    "Exp_C_freeze23_BackboneNeckFrozen": "凍結 Backbone + Neck (層 0-22)"
}

print("\n" + "="*120)
print("YOLOv11m 三個實驗結果對比分析")
print("="*120 + "\n")

results_dict = {}

for exp_name, description in experiments.items():
    csv_path = Path(base_path) / exp_name / "results.csv"
    
    if csv_path.exists():
        print(f"✓ 讀取: {exp_name}")
        print(f"  說明: {description}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # 取最後一行（最終結果）
            last_row = df.iloc[-1]
            
            # 存儲關鍵指標
            results_dict[exp_name] = {
                "total_epochs": len(df),
                "final_epoch": int(last_row.get("epoch", 0)) if "epoch" in df.columns else len(df),
                "box_loss": float(last_row.get("box_loss", 0)) if "box_loss" in df.columns else None,
                "cls_loss": float(last_row.get("cls_loss", 0)) if "cls_loss" in df.columns else None,
                "dfl_loss": float(last_row.get("dfl_loss", 0)) if "dfl_loss" in df.columns else None,
                "train_precision": float(last_row.get("metrics/precision", 0)) if "metrics/precision" in df.columns else None,
                "train_recall": float(last_row.get("metrics/recall", 0)) if "metrics/recall" in df.columns else None,
                "train_mAP50": float(last_row.get("metrics/mAP50", 0)) if "metrics/mAP50" in df.columns else None,
                "train_mAP50_95": float(last_row.get("metrics/mAP50-95", 0)) if "metrics/mAP50-95" in df.columns else None,
                "val_box_loss": float(last_row.get("val/box_loss", 0)) if "val/box_loss" in df.columns else None,
                "val_cls_loss": float(last_row.get("val/cls_loss", 0)) if "val/cls_loss" in df.columns else None,
                "val_precision": float(last_row.get("metrics/precision(B)", 0)) if "metrics/precision(B)" in df.columns else None,
                "val_recall": float(last_row.get("metrics/recall(B)", 0)) if "metrics/recall(B)" in df.columns else None,
                "val_mAP50": float(last_row.get("metrics/mAP50(B)", 0)) if "metrics/mAP50(B)" in df.columns else None,
                "val_mAP50_95": float(last_row.get("metrics/mAP50-95(B)", 0)) if "metrics/mAP50-95(B)" in df.columns else None,
            }
            
            print(f"  ✓ 成功讀取 ({len(df)} 行)")
            print(f"    - 最終 Epoch: {results_dict[exp_name]['final_epoch']}")
            
        except Exception as e:
            print(f"  ✗ 錯誤: {e}")
    else:
        print(f"✗ 找不到: {csv_path}")

# 構建對比表格
print("\n" + "="*120)
print("📊 關鍵指標對比表")
print("="*120 + "\n")

# 驗證指標
print("驗證集性能指標（越高越好）:")
print("-" * 120)
print(f"{'實驗名稱':<35} {'mAP50':<15} {'mAP50-95':<15} {'精度':<15} {'召回率':<15}")
print("-" * 120)

for exp_name in experiments.keys():
    if exp_name in results_dict:
        data = results_dict[exp_name]
        print(f"{exp_name:<35} {data['val_mAP50']:<15.4f} {data['val_mAP50_95']:<15.4f} {data['val_precision']:<15.4f} {data['val_recall']:<15.4f}")

print("\n訓練損失（越低越好）:")
print("-" * 120)
print(f"{'實驗名稱':<35} {'Box Loss':<15} {'Cls Loss':<15} {'DFL Loss':<15} {'Val Box Loss':<15}")
print("-" * 120)

for exp_name in experiments.keys():
    if exp_name in results_dict:
        data = results_dict[exp_name]
        print(f"{exp_name:<35} {data['box_loss']:<15.4f} {data['cls_loss']:<15.4f} {data['dfl_loss']:<15.4f} {data['val_box_loss']:<15.4f}")

# 分析和建議
print("\n" + "="*120)
print("📈 分析與建議")
print("="*120 + "\n")

if results_dict:
    # 找出最好的模型
    best_mAP50_exp = max(results_dict.items(), key=lambda x: x[1]['val_mAP50'] if x[1]['val_mAP50'] else 0)
    best_mAP50_95_exp = max(results_dict.items(), key=lambda x: x[1]['val_mAP50_95'] if x[1]['val_mAP50_95'] else 0)
    
    print(f"🏆 最高 mAP50: {best_mAP50_exp[0]} = {best_mAP50_exp[1]['val_mAP50']:.4f}")
    print(f"🏆 最高 mAP50-95: {best_mAP50_95_exp[0]} = {best_mAP50_95_exp[1]['val_mAP50_95']:.4f}")
    
    # 對比分析
    print("\n🔍 凍結層數影響分析:")
    A_data = results_dict.get("Exp_A_freeze0_FullFT", {})
    B_data = results_dict.get("Exp_B_freeze10_BackboneFrozen", {})
    C_data = results_dict.get("Exp_C_freeze23_BackboneNeckFrozen", {})
    
    if A_data and B_data:
        mAP50_diff = (B_data['val_mAP50'] - A_data['val_mAP50']) / A_data['val_mAP50'] * 100 if A_data['val_mAP50'] else 0
        print(f"  • Exp B vs A: mAP50 差異 {mAP50_diff:+.2f}%")
    
    if B_data and C_data:
        mAP50_diff = (C_data['val_mAP50'] - B_data['val_mAP50']) / B_data['val_mAP50'] * 100 if B_data['val_mAP50'] else 0
        print(f"  • Exp C vs B: mAP50 差異 {mAP50_diff:+.2f}%")
    
    print("\n💡 結論:")
    print("  • freeze=0 (完整微調): 最靈活，可充分適應新錯誤特徵")
    print("  • freeze=10 (凍結 Backbone): 平衡方案，保留預訓練特徵")
    print("  • freeze=23 (凍結 Neck): 保守方案，只訓練檢測頭")
    print("  → 選擇最高 mAP50 的實驗作為最佳模型")

else:
    print("✗ 無法讀取任何實驗結果")

print("\n" + "="*120)
