"""
download_model.py

Downloads the pretrained BatNet model (~263 MB) to the path set in
config.py (models/detection/basemodel.pt.zip). This is the same file
BatNet's own fetch_pretrained_models.py downloads.

Run from the project folder with:
    python vision/download_model.py
"""

import urllib.request

from config import WEIGHTS_FILE

MODEL_URL = (
    "https://github.com/GabiK-bat/BatNet/releases/download/"
    "assets.basemodel.2022-12-13/basemodel.zip"
)


def show_progress(block_count, block_size, total_size):
    done_mb = block_count * block_size / 1e6
    total_mb = total_size / 1e6
    print(f"\r  {done_mb:6.1f} / {total_mb:.1f} MB", end="", flush=True)


def main():
    if WEIGHTS_FILE.exists():
        print(f"Model already downloaded: {WEIGHTS_FILE}")
        return
    WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading BatNet model to {WEIGHTS_FILE} ...")
    urllib.request.urlretrieve(MODEL_URL, WEIGHTS_FILE, show_progress)
    print("\nDone.")


if __name__ == "__main__":
    main()
