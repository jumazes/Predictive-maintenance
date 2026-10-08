"""Optional helper for an internet-enabled machine.

Downloads the official UCI AI4I 2020 zip and extracts ai4i2020.csv into data/.
The project is bundled with an offline replica so this step is not required to run.
"""
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/601/ai4i%2B2020%2Bpredictive%2Bmaintenance%2Bdataset.zip"
ZIP_PATH = DATA_DIR / "ai4i_uci_official.zip"

print("Downloading official UCI AI4I 2020 dataset...")
urlretrieve(URL, ZIP_PATH)
with ZipFile(ZIP_PATH) as zf:
    zf.extractall(DATA_DIR / "official_uci")
print("Extracted to", DATA_DIR / "official_uci")
