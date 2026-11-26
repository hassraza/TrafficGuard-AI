import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent

os.environ["TRAFFICGUARD_ROOT"] = str(ROOT)

runpy.run_path(
    str(ROOT / "app/streamlit_app.py"),
    run_name="__main__"
)
