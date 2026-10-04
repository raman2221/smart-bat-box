"""
auto_label.py

Draws bat boxes automatically on the iNaturalist bat photos, so nobody
has to draw thousands of boxes by hand.

The current detector looks at each photo, and we only keep labels we can
trust:
  - iNaturalist photos always contain a bat. A photo is used if the
    detector draws exactly one box, confidently (>= ACCEPT_SCORE), and no
    "maybe" boxes (between MAYBE_SCORE and ACCEPT_SCORE). Photos where
    it finds no bat are listed in data/auto_label_missed.csv: those are
    the hard ones, worth labelling by hand later.
  - TEST_FRACTION of the iNaturalist photos are held back (never trained
    on) in data/inaturalist/test_images.txt, to fairly measure how many
    real bat photos each model catches.

Output: data/auto_labels_raw.csv (the detector's boxes, one row per box,
in 0-1 coordinates). These are then cleaned up by training/refine_labels.py
(it checks each box really holds ONE whole bat, and tightens loose boxes),
which writes data/auto_labels.csv, the file training/finetune_detector.py
reads. Use --no-refine to skip that step. With --sheets N it also saves N
contact sheets (grids of labelled photos) to data/audit/ for checking by eye.

Run from the project folder (with the .venv active):
    python training/auto_label.py --sheets 4
"""

import argparse
import csv
import hashlib
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402
from vision.bat_detector import BatDetector  # noqa: E402
from training.refine_labels import RAW_LABELS_FILE, refine  # noqa: E402

ACCEPT_SCORE = 0.8
MAYBE_SCORE = 0.4
TEST_FRACTION = 0.10
AUDIT_DIR = config.PROJECT_ROOT / "data" / "audit"
TEST_LIST_FILE = config.INATURALIST_DIR / "test_images.txt"
MISSED_FILE = config.PROJECT_ROOT / "data" / "auto_label_missed.csv"


def is_test_photo(path) -> bool:
    """Decide from the file name alone whether a photo is held back for testing.

    Using the name (not a random draw) means a photo always lands on the
    same side, even when more photos are downloaded later, so a photo the
    model trained on can never turn into a test photo.
    """
    digest = hashlib.md5(path.name.encode()).hexdigest()
    return int(digest, 16) % 100 < TEST_FRACTION * 100


def find_images():
    """Return [(source, path)] for all iNaturalist photos."""
    return [("inaturalist", p) for p in sorted((config.INATURALIST_DIR / "images").glob("*.jpg"))]


def make_contact_sheets(accepted, count, per_sheet=24, columns=6, tile=220):
    """Save grids of numbered, labelled photos for checking by eye."""
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(1)
    sample = random.sample(accepted, min(len(accepted), count * per_sheet))
    index_rows = []
    for s in range(count):
        chunk = sample[s * per_sheet:(s + 1) * per_sheet]
        if not chunk:
            break
        rows = (len(chunk) + columns - 1) // columns
        sheet = np.full((rows * tile, columns * tile, 3), 40, np.uint8)
        for i, (source, path, boxes) in enumerate(chunk):
            image = cv2.imread(str(path))
            h, w = image.shape[:2]
            for (x1, y1, x2, y2), _ in boxes:
                cv2.rectangle(image, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)),
                              (0, 255, 0), max(2, w // 150))
            scale = tile / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))
            r, c = divmod(i, columns)
            y0, x0 = r * tile, c * tile
            sheet[y0:y0 + image.shape[0], x0:x0 + image.shape[1]] = image
            number = s * per_sheet + i + 1
            cv2.putText(sheet, str(number), (x0 + 5, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 0, 255), 2)
            index_rows.append([number, f"sheet_{s + 1:02d}.jpg", source,
                               path.relative_to(config.PROJECT_ROOT).as_posix()])
        cv2.imwrite(str(AUDIT_DIR / f"sheet_{s + 1:02d}.jpg"), sheet)
    with open(AUDIT_DIR / "index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["number", "sheet", "source", "image"])
        writer.writerows(index_rows)
    print(f"Saved contact sheets to {AUDIT_DIR}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sheets", type=int, default=0, help="contact sheets to save for auditing")
    parser.add_argument("--no-refine", action="store_true",
                        help="don't run training/refine_labels.py afterwards")
    args = parser.parse_args()

    # Low threshold so we can also see the "maybe" boxes. The detector also
    # has its own built-in cut-off (0.5) that would hide them, so lower it too.
    detector = BatDetector(min_confidence=MAYBE_SCORE)
    detector.model.detector.basemodel.roi_heads.score_thresh = MAYBE_SCORE

    images = find_images()
    print(f"Labelling {len(images)} images on {detector.device}...")

    # Hold back part of the iNaturalist photos for testing (fixed per photo).
    test_set = {p for source, p in images if source == "inaturalist" and is_test_photo(p)}
    TEST_LIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    TEST_LIST_FILE.write_text("\n".join(
        p.relative_to(config.PROJECT_ROOT).as_posix() for p in sorted(test_set)))

    accepted, missed = [], []
    stats = {"accepted": 0, "unsure": 0, "no bat found": 0, "held back for testing": len(test_set)}
    for n, (source, path) in enumerate(images, 1):
        if path in test_set:
            continue
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        detections = detector.detect(frame)
        sure = [d for d in detections if d.confidence >= ACCEPT_SCORE]
        maybe = [d for d in detections if d.confidence < ACCEPT_SCORE]

        # Exactly one confident box and nothing else. Photos where it drew
        # several boxes were often messy (two boxes on one bat, one box
        # over two bats, tiny stray boxes), so they're left out.
        if len(sure) == 1 and not maybe:
            boxes = [((d.bbox[0] / w, d.bbox[1] / h, d.bbox[2] / w, d.bbox[3] / h), d.confidence)
                     for d in sure]
            accepted.append((source, path, boxes))
            stats["accepted"] += 1
        elif not detections:
            stats["no bat found"] += 1
            if source == "inaturalist":
                missed.append(path)
        else:
            stats["unsure"] += 1
        if n % 500 == 0:
            print(f"  {n}/{len(images)}", flush=True)

    with open(RAW_LABELS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "image", "xmin", "ymin", "xmax", "ymax", "score"])
        for source, path, boxes in accepted:
            for (x1, y1, x2, y2), score in boxes:
                writer.writerow([source, path.relative_to(config.PROJECT_ROOT).as_posix(),
                                 f"{x1:.5f}", f"{y1:.5f}", f"{x2:.5f}", f"{y2:.5f}", f"{score:.3f}"])
    with open(MISSED_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image"])
        writer.writerows([[p.relative_to(config.PROJECT_ROOT).as_posix()] for p in missed])

    by_source = {}
    for source, _, _ in accepted:
        by_source[source] = by_source.get(source, 0) + 1
    print("Results:", stats)
    print("Accepted by source:", by_source)
    print(f"Detector boxes saved to {RAW_LABELS_FILE}")

    if args.sheets:
        make_contact_sheets(accepted, args.sheets)

    if not args.no_refine:
        # Free the GPU memory used by the detector before the next model loads.
        del detector
        print("Refining the boxes with BatNet's segmentation network...")
        refine(RAW_LABELS_FILE, sheets=bool(args.sheets))


if __name__ == "__main__":
    main()
