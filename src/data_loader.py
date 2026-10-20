"""Step 1: Get the dataset."""
import urllib.request
import pandas as pd
from src import config


def download_data():
    """Download the Pima dataset if it isn't already in data/raw."""
    if config.RAW_DATA.exists():
        return
    config.RAW_DATA.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading dataset...")
    urllib.request.urlretrieve(config.DATA_URL, config.RAW_DATA)


def load_raw_data() -> pd.DataFrame:
    download_data()
    df = pd.read_csv(config.RAW_DATA, header=None, names=config.COLUMNS)
    print(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df
