# -*- mode: python ; coding: utf-8 -*-
import sys
sys.setrecursionlimit(5000)
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.building.datastruct import TOC

# pyinstaller/odbargo-view.spec
# One-file build; Agg backend; exclude GUI toolkits

import os, warnings
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.building.datastruct import TOC

APP_NAME   = "odbargo-view"
ENTRY      = "view_entry.py"   # supports: python -m odbargo_view
UPX_DIR    = "/usr/bin/upx"            # os.environ.get("UPX_DIR") # set if you have UPX; otherwise None
STRIP      = True                      # no-op on Windows; ok on linux/mac

# Force Agg
os.environ.setdefault("MPLBACKEND", "Agg")  # hard-force Agg

# (optional) pick a font we ship
import matplotlib as mpl
mpl.rcParams["font.family"] = "DejaVu Sans"

# Hidden imports
hidden = set()
hidden.update(collect_submodules("matplotlib.backends",
                                 filter=lambda m: m.endswith("backend_agg")))
hidden.update([
    "matplotlib", "matplotlib.pyplot", "matplotlib.backends.backend_agg",
    "xarray", "h5netcdf", "h5py", "numpy",
])
try:
    import pandas  # noqa
    hidden.add("pandas")
except Exception:
    pass

excludes = [
    "tkinter", "PyQt5", "PySide2", "PySide6", "wx", "gi", "gtk",
    "mpl_toolkits.mplot3d", "mpl_toolkits.basemap",
    "scipy", "argopy",
]

# Minimal mpl-data keep-list
ALLOWED_MPL_FILES = {
    "matplotlibrc",
    "classic.mplstyle",
    "DejaVuSans.ttf",
    # drop mono if you like:
    "DejaVuSansMono.ttf",
}

block_cipher = None

a = Analysis(
    [ENTRY],
    pathex=[os.path.abspath(".")],
    binaries=[],
    datas=[],  # let hooks add mpl-data; we filter next
    hiddenimports=sorted(hidden),
    hookspath=[],
    # RUNTIME HOOK suppresses Axes3D warning before mpl import
    runtime_hooks=["conf/rthook_silence_mpl.py"],
    hooksconfig={"matplotlib": {"backend": "Agg"}},
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# prune mpl-data TOC in-place (keep only the small allow-list)
def _keep_mpl_data(toc_entry):
    dest_name, _, typecode = toc_entry
    if typecode != "DATA":
        return True
    dn = dest_name.replace("\\", "/")
    if "matplotlib/mpl-data" not in dn:
        return True
    base = os.path.basename(dn)
    return base in ALLOWED_MPL_FILES

a.datas = TOC([e for e in a.datas if _keep_mpl_data(e)])

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
    upx=bool(UPX_DIR),
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    onefile=True,
    version='version-viewer.txt',
    icon=['icon.ico']
)
