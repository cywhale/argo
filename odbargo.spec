# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['argo_app/app.py'],
    pathex=[],
    binaries=[],
    datas=[('argo_app', 'argo_app'), ('argo_app/src', 'argo_app/src')],
    hiddenimports=['argopy.static', 'tensorboard', 'PySide2', 'PyQt6', 'PySide6'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide2', 'PyQt6', 'PySide6', 'tensorboard'],  # Excluding unwanted packages
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    noarchive=False,
    cipher=None,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='odbargo',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='odbargo',
)
