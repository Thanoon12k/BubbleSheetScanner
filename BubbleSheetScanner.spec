# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# Collect all submodules for external dependencies
hidden_imports = collect_submodules("cv2") + collect_submodules("tkinter")

# Include local modules explicitly
hidden_imports += [
    'bubbles_caculations',
    'GUI',
    'image_preprocessing',
]

# Collect additional data files if needed
data_files = []

block_cipher = None

a = Analysis(
    ['main.py'],  # Entry-point script
    pathex=['D:\\apps\\bubblesheetScanner'],  # Path to your project folder
    binaries=[],
    datas=data_files,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='bubble_sheet_scanner',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Hide the console
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='bubble_sheet_scanner',
)
