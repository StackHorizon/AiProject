# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('models/', 'models/'),  # Include i modelli precaricati
    ],
    hiddenimports=[
        'torch',
        'torchvision',
        'PIL',
        'customtkinter',
        'cv2',
        'numpy',
        'psutil',
        'multiprocessing.popen_spawn_posix',  # Necessario per multiprocessing su macOS
    ],
    hookspath=[],
    hooksconfig={},
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
    name='Cartoonizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Cartoonizer',
)

app = BUNDLE(
    coll,
    name='CapitanAcciaioCartoonizer.app',
    icon=None,  # Aggiungi qui un'icona .icns se disponibile
    bundle_identifier='com.capitanacciaio.cartoonizer',
    info_plist={
        'NSCameraUsageDescription': 'Questa app richiede accesso alla webcam per creare effetti cartoon in tempo reale',
        'CFBundleShortVersionString': '1.0.0',
        'NSHighResolutionCapable': 'True',
        'LSApplicationCategoryType': 'public.app-category.photography',
    },
)