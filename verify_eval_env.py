#!/usr/bin/env python3
"""快速驗證 eval_test.py 執行環境和配置"""

import os
import sys
from pathlib import Path

def verify_environment():
    """驗證環境和路徑"""
    print("🔍 Verifying eval_test.py environment...\n")
    
    # 1. 檢查關鍵文件
    checks = {
        'eval_test.py': Path('eval_test.py'),
        'eval_configs/exp_b_test.yaml': Path('eval_configs/exp_b_test.yaml'),
        'data_augmented.yaml': Path('data_augmented.yaml'),
        'runs/detect/runs/pretrained_study/Exp_B_freeze10_BackboneFrozen/weights/best.pt': 
            Path('runs/detect/runs/pretrained_study/Exp_B_freeze10_BackboneFrozen/weights/best.pt'),
    }
    
    print("📂 File Checks:")
    all_exist = True
    for name, path in checks.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        size = f"({path.stat().st_size / 1024 / 1024:.1f} MB)" if exists and path.is_file() else ""
        print(f"  {status} {name:<70} {size}")
        if not exists:
            all_exist = False
    
    print()
    
    # 2. 檢查必需的 Python 包
    print("📦 Required Packages:")
    packages = {
        'yaml': 'PyYAML',
        'numpy': 'NumPy',
        'matplotlib': 'Matplotlib',
        'ultralytics': 'Ultralytics YOLO',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
    }
    
    all_packages_ok = True
    for module_name, package_name in packages.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name:<30}")
        except ImportError:
            print(f"  ✗ {package_name:<30} (missing)")
            all_packages_ok = False
    
    print()
    
    # 3. 檢查配置內容
    print("⚙️  Configuration Check:")
    try:
        import yaml
        with open('eval_configs/exp_b_test.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print(f"  ✓ Model: {config['model']['experiment_name']}")
        print(f"  ✓ Freeze: {config['model']['freeze_layers']} layers")
        print(f"  ✓ Dataset: {config['dataset']['test_size']} test images")
        print(f"  ✓ Device: {config['inference']['device']}")
        print(f"  ✓ Train val mAP50: {config['train_val_metrics']['mAP50']}")
        
    except Exception as e:
        print(f"  ✗ Error reading config: {e}")
        all_packages_ok = False
    
    print()
    
    # 4. 結論
    if all_exist and all_packages_ok:
        print("✅ All checks passed! You can run eval_test.py\n")
        print("  Usage:")
        print("    python eval_test.py                               # Use default config")
        print("    python eval_test.py --device cuda                 # Override device")
        print("    python eval_test.py --device cpu --batch-size 8   # Multiple overrides")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        if not all_packages_ok:
            print("\n  Install missing packages with:")
            print("    pip install pyyaml numpy matplotlib ultralytics torch torchvision opencv-python")
        return False

if __name__ == '__main__':
    os.chdir(Path(__file__).parent / 'yolov11_new' if (Path(__file__).parent / 'yolov11_new').exists() else Path(__file__).parent)
    sys.exit(0 if verify_environment() else 1)
