# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)

# pyinstaller/odbargo-cli.spec
# One-file build; exclude heavy libs

import os

APP_NAME = "odbargo-cli"
ENTRY    = "cli_entry.py"
UPX_DIR  = os.environ.get("UPX_DIR")
STRIP    = True

excludes = [
    "matplotlib", "pandas", "xarray", "h5netcdf", "h5py", "scipy",
    "tkinter", "PyQt5", "PySide2", "PySide6", "wx",
    'numpy', 'netCDF4', 'argopy'
]

block_cipher = None

a = Analysis(
    [ENTRY],
    pathex=[os.path.abspath(".")],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,  # one-file packs these here
    a.zipfiles,
    a.datas,
    [],
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=STRIP,
    upx=bool(UPX_DIR),
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    onefile=True,
    version='version.txt',
    icon=['icon.ico']
)

