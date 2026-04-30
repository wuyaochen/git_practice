import os
from pathlib import Path

print("\n" + "="*100)
print("🔍 Exp_A 缺失診斷報告")
print("="*100 + "\n")

exp_a_path = Path(r"C:\Users\User\wyc\yolov11\yolov11_new\runs\detect\runs\pretrained_study\Exp_A_freeze0_FullFT")

# 1. 檢查資料夾是否存在
print(f"1️⃣ 資料夾存在性: {'✓' if exp_a_path.exists() else '✗'} {exp_a_path}")

# 2. 列出所有檔案
if exp_a_path.exists():
    print(f"\n2️⃣ 資料夾內容：")
    all_files = list(exp_a_path.rglob("*"))
    if all_files:
        for f in sorted(all_files):
            rel_path = f.relative_to(exp_a_path)
            if f.is_file():
                size_mb = f.stat().st_size / 1024 / 1024
                print(f"  📄 {rel_path} ({size_mb:.2f} MB)")
            else:
                print(f"  📁 {rel_path}/")
    else:
        print("  ⚠️ 資料夾為空")
    
    # 3. 尋找特定檔案
    print(f"\n3️⃣ 關鍵檔案檢查：")
    
    key_files = {
        "results.csv": exp_a_path / "results.csv",
        "weights/best.pt": exp_a_path / "weights" / "best.pt",
        "weights/last.pt": exp_a_path / "weights" / "last.pt",
        "events.out.*": exp_a_path,  # 在主目錄下找
        "runs.log": exp_a_path / "runs.log",
    }
    
    for name, path_obj in key_files.items():
        if "*" in name:
            # 查找通配符檔案
            import glob
            pattern = str(path_obj / "events.out.*")
            found = glob.glob(pattern)
            if found:
                print(f"  ✓ {name}: 找到 ({len(found)} 個)")
            else:
                print(f"  ✗ {name}: 未找到")
        else:
            if isinstance(path_obj, Path) and path_obj.exists():
                size = path_obj.stat().st_size
                print(f"  ✓ {name}: 存在 ({size / 1024 / 1024:.2f} MB)")
            else:
                print(f"  ✗ {name}: 未找到")
    
    # 4. 檢查是否有日誌或錯誤
    print(f"\n4️⃣ 查找日誌檔案：")
    
    for log_file in sorted(exp_a_path.rglob("*.log")):
        print(f"  📋 {log_file.name} ({log_file.stat().st_size} bytes)")
        # 顯示最後 5 行
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            if lines:
                print(f"    最後 5 行:")
                for line in lines[-5:]:
                    print(f"    > {line.strip()}")
    
    # 5. 檢查隱藏檔案或備份
    print(f"\n5️⃣ 檢查其他可能的檔案：")
    
    # 查找 Exp_A_freeze0_FullFT-2 (因為有看到這個資料夾)
    exp_a_2 = Path(r"C:\Users\User\wyc\yolov11\yolov11_new\runs\detect\runs\pretrained_study\Exp_A_freeze0_FullFT-2")
    if exp_a_2.exists():
        print(f"  ℹ️ 發現 Exp_A_freeze0_FullFT-2 資料夾（可能是重新訓練）")
        results_csv_2 = exp_a_2 / "results.csv"
        if results_csv_2.exists():
            print(f"    ✓ 此資料夾內有 results.csv!")
            print(f"    建議：可能需要改名或合併")
    
    # 6. 訓練狀態推論
    print(f"\n6️⃣ 訓練狀態推論：")
    
    results_csv = exp_a_path / "results.csv"
    weights_best = exp_a_path / "weights" / "best.pt"
    
    if results_csv.exists():
        print(f"  ✓ 訓練已完成（有 results.csv）")
        with open(results_csv, 'r') as f:
            lines = f.readlines()
            print(f"    訓練 epoch 數: {len(lines) - 1}")
    elif weights_best.exists():
        print(f"  ⚠️ 訓練可能進行中或被中斷（有權重但無結果）")
    else:
        print(f"  ✗ 訓練未開始或早期失敗")
        print(f"    可能原因：")
        print(f"    • 資料集路徑錯誤")
        print(f"    • GPU/CUDA 問題")
        print(f"    • 虛擬環境依賴缺失")
        print(f"    • 訓練被自動中斷（early stopping）")

else:
    print(f"\n✗ Exp_A 資料夾根本不存在！")
    print(f"可能原因：訓練從未執行過")

print("\n" + "="*100)
