# pyinstaller/cli_win.spec
# Windows one-file build for the lean CLI (no heavy scientific deps)

import os

APP_NAME = "odbargo-cli"
# Use the thin entry that imports cli/odbargo_cli.py
ENTRY    = "cli_entry.py"

# Optional: set to your UPX.exe directory to enable UPX compression
#   e.g. set UPX_DIR=C:\tools\upx
UPX_DIR  = "D:/tools/upx" #os.environ.get("UPX_DIR")
STRIP    = True   # strip symbols; safe for release builds
# Remove-Item -Recurse -Force build, dist, __pycache__ -ErrorAction SilentlyContinue
# Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
# Keep the CLI tiny by excluding heavy libs and GUI stacks
excludes = [
    # heavy scientific stacks we do NOT want in the CLI EXE
    "matplotlib", "pandas", "xarray", "h5netcdf", "h5py", "scipy",
    "numpy", "netCDF4", "argopy",
    # GUI toolkits
    "tkinter", "PyQt5", "PySide2", "PySide6", "wx", "gi", "gtk",
    # misc
    "mpl_toolkits.mplot3d", "mpl_toolkits.basemap",
]

block_cipher = None

a = Analysis(
    [ENTRY],
    pathex=[os.path.abspath(".")],
    binaries=[],
    datas=[],
    hiddenimports=[],       # keep empty; CLI shouldn't pull heavy modules indirectly
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],       # no runtime hooks needed for the CLI
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,        # keep default; one-file will still pack everything
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=STRIP,
    upx=bool(UPX_DIR), # enable UPX only if UPX_DIR is set
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,           # CLI app → console window
    version='version.txt',
    icon=['icon.ico']
)
