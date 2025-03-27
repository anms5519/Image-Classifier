
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
    hooksconfig={},
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
    name='ImageClassifier',
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
