# bubble_sheet_scanner.spec

# Import necessary PyInstaller modules
from PyInstaller.utils.hooks import collect_all

# Add your data files and additional scripts here
datas = [
    ('answers.xlsx', '.'),         # Include your data file
    ('main.py', '.'),              # Include the main.py script
    ('bubbles_caculations.py', '.'),  # Include bubbles_caculations.py script
    ('image_preprocessing.py', '.'),  # Include image_preprocessing.py script
    ('GUI.py', '.'),               # Include GUI.py script
]

# Include any hidden imports
hiddenimports = ['bubbles_caculations', 'image_preprocessing', 'GUI']

a = Analysis(
    ['main.py'],                   # Your main script
    pathex=['D:/apps/bubblesheetScanner'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hooks=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, a.scripts, a.binaries, name='main', debug=False, bootloader_ignore_signals=False, strip=False, upx=True, upx_exclude=[], runtime_hooks=[], cipher=None)

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
    console=True,
    icon=None
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='bubble_sheet_scanner'
)
