"""Dataset download and loading (training only; the API never imports this module).

Source: ComponentesPC_Scraper, Zenodo record 4263453, licensed CC BY 4.0.
"""
import hashlib
import shutil
import ssl
import sys
import urllib.request
from pathlib import Path

import certifi
import pandas as pd

DATA_URL = "https://zenodo.org/records/4263453/files/ComponentesPC_Scraper_DataSet.csv?download=1"
DATA_SHA256 = "f7c289b5ce0e9408a6fa2d11abc0755eb22acb8360ffdffd3df7155f7b187497"
DATA_PATH = Path("data/ComponentesPC_Scraper_DataSet.csv")

CATEGORICAL = ["company_name", "brand_name", "category"]
NUMERIC = ["score", "reviews"]
TEXT = "name"  # free-text product title, e.g. "Gigabyte SSD M.2 512GB 2280 PCIe x2 NVMe"
FEATURES = CATEGORICAL + NUMERIC + [TEXT]
TARGET = "price"


def download(path: Path = DATA_PATH) -> Path:
    """Download the CSV if missing and check its SHA-256."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL}")
        # certifi gives a CA bundle even on Pythons without system certificates.
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(DATA_URL, context=context) as resp, open(path, "wb") as out:
            shutil.copyfileobj(resp, out)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        raise RuntimeError(f"Unexpected checksum for {path}: {digest}")
    return path


def load_dataset(path: Path = DATA_PATH) -> pd.DataFrame:
    """Return one row per product (latest scrape) with the model features and the price."""
    df = pd.read_csv(download(path))
    # The same product is scraped many times; keep the latest observation so that
    # train / calibration / test rows are not near-duplicates of each other.
    df = df.sort_values("timestamp").drop_duplicates(["company_name", "product_number"], keep="last")
    df = df[df[TARGET] > 0].dropna(subset=FEATURES + [TARGET])
    return df[FEATURES + [TARGET]].reset_index(drop=True)


if __name__ == "__main__":
    out = download()
    print(f"OK: {out} ({out.stat().st_size} bytes, sha256 verified)", file=sys.stderr)
