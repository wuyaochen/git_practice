#!/usr/bin/env python3
"""快速驗證 eval_test.py 執行環境和配置"""

import argparse
import os
import sys
from pathlib import Path


def _load_yaml_config(config_path: Path) -> dict:
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def verify_environment(config_path: Path):
    """驗證環境和路徑"""
    print("🔍 Verifying eval_test.py environment...\n")

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return False

    try:
        cfg = _load_yaml_config(config_path)
    except Exception as e:
        print(f"❌ Failed to read config: {config_path}")
        print(f"   Error: {e}")
        return False
    
    # 1. 檢查關鍵文件
    dataset_cfg_path = Path(cfg['dataset']['config'])
    model_path = Path(cfg['model']['path'])
    checks = {
        'eval_test.py': Path('eval_test.py'),
        str(config_path.as_posix()): config_path,
        str(dataset_cfg_path.as_posix()): dataset_cfg_path,
        str(model_path.as_posix()): model_path,
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
        print(f"  ✓ Config: {config_path}")
        print(f"  ✓ Model: {cfg['model']['experiment_name']}")
        print(f"  ✓ Freeze: {cfg['model']['freeze_layers']} layers")
        print(f"  ✓ Dataset: {cfg['dataset']['test_size']} test images")
        print(f"  ✓ Device: {cfg['inference']['device']}")
        print(f"  ✓ Train val mAP50: {cfg['train_val_metrics']['mAP50']}")

    except Exception as e:
        print(f"  ✗ Error reading config: {e}")
        all_packages_ok = False
    
    print()
    
    # 4. 結論
    if all_exist and all_packages_ok:
        print("✅ All checks passed! You can run eval_test.py\n")
        print("  Usage:")
        print("    python verify_eval_env.py --config eval_configs/exp_b_test.yaml")
        print("    python verify_eval_env.py --config eval_configs/exp_c_test.yaml")
        print("    python eval_test.py --config eval_configs/exp_b_test.yaml --device cuda")
        print("    python eval_test.py --config eval_configs/exp_c_test.yaml --device cuda")
        return True
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        if not all_packages_ok:
            print("\n  Install missing packages with:")
            print("    pip install pyyaml numpy matplotlib ultralytics torch torchvision opencv-python")
        return False

if __name__ == '__main__':
    os.chdir(Path(__file__).parent / 'yolov11_new' if (Path(__file__).parent / 'yolov11_new').exists() else Path(__file__).parent)

    parser = argparse.ArgumentParser(description='Preflight check for eval_test.py')
    parser.add_argument('--config', type=str, default='eval_configs/exp_b_test.yaml', help='Path to eval config YAML')
    args = parser.parse_args()

    sys.exit(0 if verify_environment(Path(args.config)) else 1)
