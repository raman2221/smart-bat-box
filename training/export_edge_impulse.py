"""
export_edge_impulse.py

Packs a balanced set of our labelled photos for Edge Impulse, to train a
small, fast FOMO bat detector that runs on the Arduino UNO Q.

What goes in the zip (all photos shrunk so the longest side is MAX_SIDE,
which keeps the upload small and fast):
  training/  - bat photos with boxes (hand-labelled Open Images bats, a
               sample of the cleaned iNaturalist bats, some synthetic bats
               pasted into rooms) + "not a bat" photos (people, lights,
               birds, animals, ...)
  testing/   - photos the training folder does NOT contain, for an honest
               accuracy score: the hand-labelled check bats, a separate
               sample of iNaturalist bats, and "not a bat" check photos
Each folder has a bounding_boxes.labels file (Edge Impulse's own label
format): every bat box in pixels, label "bat". "Not a bat" photos are
listed with no boxes.

Never used: the held-back iNaturalist test photos and anything in
data/qc_exclude.txt.

Run from the project folder (with the .venv active):
    python training/export_edge_impulse.py
Then upload training/ and testing/ in Edge Impulse (see the printed steps).
"""

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402

MAX_SIDE = 320
LABEL = "bat"
OUT_DIR = Path.home() / "Downloads" / "batbox_edge_impulse"

# How many photos of each kind (the hand-labelled ones are all used).
TRAIN_INAT_BATS = 1200
TRAIN_SYNTHETIC_BATS = 300
TRAIN_NOT_A_BAT = 1800
TEST_INAT_BATS = 200
TEST_NOT_A_BAT = 400


def read_boxes(csv_path, path_base):
    """{image path: [box fractions]} from a labels CSV (auto or synthetic)."""
    boxes = {}
    if not csv_path.exists():
        return boxes
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            box = [float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
            boxes.setdefault(path_base / row["image"], []).append(box)
    return boxes


def lines(path):
    return set(path.read_text().split()) if path.exists() else set()


def save(path, boxes, folder, labels, index):
    """Shrink one photo into the export folder and record its boxes in pixels."""
    image = cv2.imread(str(path))
    if image is None:
        return False
    h, w = image.shape[:2]
    scale = min(1.0, MAX_SIDE / max(h, w))
    if scale < 1.0:
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    name = f"{index:05d}_{path.stem}.jpg"
    cv2.imwrite(str(folder / name), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    entries = []
    for x1, y1, x2, y2 in boxes:
        x, y = round(x1 * w), round(y1 * h)
        bw, bh = round((x2 - x1) * w), round((y2 - y1) * h)
        if bw >= 2 and bh >= 2:   # skip boxes too tiny to learn from
            entries.append({"label": LABEL, "x": x, "y": y, "width": bw, "height": bh})
    labels[name] = entries
    return True


def write_folder(folder, items):
    """Save every (path, boxes) item and the folder's bounding_boxes.labels."""
    folder.mkdir(parents=True, exist_ok=True)
    labels = {}
    saved = sum(save(p, b, folder, labels, i) for i, (p, b) in enumerate(items))
    (folder / "bounding_boxes.labels").write_text(json.dumps(
        {"version": 1, "type": "bounding-box-labels", "boundingBoxes": labels}))
    bats = sum(1 for v in labels.values() if v)
    return saved, bats, saved - bats


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    random.seed(args.seed)

    root = config.PROJECT_ROOT
    excluded = {root / p for p in lines(config.QC_EXCLUDE_FILE)}
    held_back = {root / p for p in lines(config.INATURALIST_DIR / "test_images.txt")}
    skip = excluded | held_back

    # Open Images: hand-labelled bats + "not a bat" photos, by split.
    oi = {"train": {}, "val": {}}
    with open(config.OPENIMAGES_DIR / "annotations.csv", newline="") as f:
        for row in csv.DictReader(f):
            path = config.OPENIMAGES_DIR / row["image"]
            if path in skip:
                continue
            boxes = oi[row["split"]].setdefault(path, [])
            if row["is_bat"] == "1":
                boxes.append([float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")])

    def pick(pool, n):
        pool = sorted(pool)
        return random.sample(pool, min(n, len(pool)))

    inat = {p: b for p, b in read_boxes(config.AUTO_LABELS_FILE, root).items() if p not in skip}
    synthetic = {p: b for p, b in read_boxes(config.SYNTHETIC_LABELS_FILE, root).items() if p not in skip}
    inat_paths = pick(inat, TRAIN_INAT_BATS + TEST_INAT_BATS)
    inat_train, inat_test = inat_paths[:TRAIN_INAT_BATS], inat_paths[TRAIN_INAT_BATS:]

    train = [(p, b) for p, b in oi["train"].items() if b]                       # hand-labelled bats
    train += [(p, inat[p]) for p in inat_train]
    train += [(p, synthetic[p]) for p in pick(synthetic, TRAIN_SYNTHETIC_BATS)]
    train += [(p, []) for p in pick([p for p, b in oi["train"].items() if not b], TRAIN_NOT_A_BAT)]

    test = [(p, b) for p, b in oi["val"].items() if b]                          # hand-labelled check bats
    test += [(p, inat[p]) for p in inat_test]
    test += [(p, []) for p in pick([p for p, b in oi["val"].items() if not b], TEST_NOT_A_BAT)]

    random.shuffle(train)
    random.shuffle(test)
    if args.out.exists():
        shutil.rmtree(args.out)
    for name, items in (("training", train), ("testing", test)):
        saved, bats, empty = write_folder(args.out / name, items)
        print(f"{name}: {saved} photos ({bats} with bats, {empty} not a bat)")

    archive = shutil.make_archive(str(args.out), "zip", args.out)
    size_mb = Path(archive).stat().st_size / 1e6
    print(f"Saved {args.out} and {archive} ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
