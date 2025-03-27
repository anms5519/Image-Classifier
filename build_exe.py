import os
import sys
import shutil
import subprocess
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Build executable for the image classifier')
    parser.add_argument('--output_dir', type=str, default='dist', help='Output directory for executable')
    parser.add_argument('--include_model', action='store_true', help='Include pre-trained model in executable')
    parser.add_argument('--name', type=str, default='ImageClassifier', help='Name of the executable')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create spec file dynamically
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

import sys
from PyInstaller.utils.hooks import collect_all

datas = []
binaries = []
hiddenimports = []

# Add PyTorch
torch_collected = collect_all('torch')
datas.extend(torch_collected[0])
binaries.extend(torch_collected[1])
hiddenimports.extend(torch_collected[2])

# Add torchvision
torchvision_collected = collect_all('torchvision')
datas.extend(torchvision_collected[0])
binaries.extend(torchvision_collected[1])
hiddenimports.extend(torchvision_collected[2])

# Add all other dependencies
for pkg in ['numpy', 'PIL', 'cv2', 'tqdm']:
    collected = collect_all(pkg)
    datas.extend(collected[0])
    binaries.extend(collected[1])
    hiddenimports.extend(collected[2])

# Include assets directory 
datas.append(('assets', 'assets'))

# Include model and class labels
datas.append(('assets/model.pth', 'assets'))
datas.append(('assets/class_labels.txt', 'assets'))

a = Analysis(
    ['src/main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='{args.name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
"""
    
    # Write spec file
    spec_path = f"{args.name}.spec"
    with open(spec_path, 'w') as f:
        f.write(spec_content)
    
    print(f"Created PyInstaller spec file: {spec_path}")
    
    # Build executable
    print("Building executable... (this may take a while)")
    cmd = ["pyinstaller", "--clean", spec_path]
    subprocess.run(cmd, check=True)
    
    # Copy to output directory if it's different from 'dist'
    if args.output_dir != 'dist':
        src_exe = os.path.join('dist', f"{args.name}")
        if os.path.exists(src_exe):
            shutil.copy(src_exe, args.output_dir)
            print(f"Executable copied to {args.output_dir}")
    
    print(f"\nBuild completed! Executable is at: {os.path.join(args.output_dir, args.name)}")
    print("You can now copy this executable to a USB drive and run it on any Windows computer.")

if __name__ == "__main__":
    main() 