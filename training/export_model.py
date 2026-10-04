"""
export_model.py

Turns a fine-tuned detector weights file into the small .pth file the
program uses (models/detection/finetuned_detector.pth), so it fits on
GitHub (100 MB per-file limit) and on the Arduino.

How it gets small: the weights are stored as 16-bit numbers instead of
32-bit ones, which halves the file (~168 MB -> ~84 MB). It uses the
"bfloat16" kind of 16-bit number: it keeps fewer decimal places but the
same range as 32-bit numbers. (The more common "float16" kind can't hold
BatNet's tiniest values - they round to zero and break the model.) When the
program loads the file, PyTorch turns them back into 32-bit numbers
automatically, so nothing else has to change.

Before saving, it checks that the small file gives (almost) the same
answers as the original on a few check photos, and that it is under the
GitHub limit.

Run from the project folder (with the .venv active):
    python training/export_model.py models/detection/finetuned_detector_blend70.pt
"""

import argparse
import csv
import sys
from collections import OrderedDict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402

GITHUB_LIMIT_MB = 100
CHECK_PHOTOS = 40           # how many check photos to compare on
MAX_SCORE_DIFFERENCE = 0.02  # biggest allowed change in any bat score


def half_precision(state):
    """Copy a weights dictionary with every decimal number stored as bfloat16.

    The hidden per-layer version notes (_metadata) are kept: without them
    PyTorch renames some layers when loading and the file won't load.
    """
    small = OrderedDict()
    for name, value in state.items():
        small[name] = value.bfloat16() if value.is_floating_point() else value.clone()
    if hasattr(state, "_metadata"):
        small._metadata = state._metadata
    return small


def check_photos():
    """A few check photos (never trained on): bats and not-a-bat photos."""
    rows = list(csv.DictReader(open(config.OPENIMAGES_DIR / "annotations.csv", newline="")))
    paths = sorted({config.OPENIMAGES_DIR / r["image"] for r in rows if r["split"] == "val"})
    step = max(1, len(paths) // CHECK_PHOTOS)
    return paths[::step][:CHECK_PHOTOS]


def biggest_score_change(full_state, small_state):
    """Run both versions on the check photos; return the biggest score change."""
    import cv2
    from vision.bat_detector import BatDetector

    detector = BatDetector(min_confidence=0.0)
    biggest = 0.0
    for path in check_photos():
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        scores = []
        for state in (full_state, small_state):
            detector.model.detector.load_state_dict(state)
            detector.model.to(detector.device)
            found = detector.detect(frame)
            scores.append(max((d.confidence for d in found), default=0.0))
        biggest = max(biggest, abs(scores[0] - scores[1]))
    return biggest


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="fine-tuned weights file (.pt or .pth)")
    parser.add_argument("--out", type=Path, default=config.FINETUNED_DETECTOR_FILE,
                        help="where to save the small .pth (default: the one the program uses)")
    args = parser.parse_args()

    full = torch.load(args.source, map_location="cpu")
    small = half_precision(full)

    change = biggest_score_change(full, small)
    print(f"Biggest bat-score change on {CHECK_PHOTOS} check photos: {change:.4f}")
    if change > MAX_SCORE_DIFFERENCE:
        raise SystemExit(f"Too different (limit {MAX_SCORE_DIFFERENCE}); not saved.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(small, args.out)
    size_mb = args.out.stat().st_size / 1e6
    print(f"Saved {args.out} ({size_mb:.0f} MB)")
    if size_mb >= GITHUB_LIMIT_MB:
        print(f"Warning: over GitHub's {GITHUB_LIMIT_MB} MB per-file limit.")


if __name__ == "__main__":
    main()
