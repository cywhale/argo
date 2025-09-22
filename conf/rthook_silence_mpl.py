# pyinstaller/rthook_silence_mpl.py
# Executed by the bootloader before user code is imported.
import os, warnings
os.environ.setdefault("MPLBACKEND", "Agg")
# Silence early Axes3D probe warnings before matplotlib import
warnings.filterwarnings(
    "ignore",
    message=r".*Unable to import Axes3D.*",
    category=UserWarning,
    module=r"matplotlib.*",
)
# Optionally, silence all matplotlib UserWarnings in frozen app:
# warnings.filterwarnings("ignore", category=UserWarning, module=r"matplotlib.*")
