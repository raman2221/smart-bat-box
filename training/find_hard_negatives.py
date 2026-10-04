"""
find_hard_negatives.py

Which "not a bat" photos does the detector come closest to calling a bat?

Runs the detector at a low threshold (default 0.3) over the not-a-bat photos
in data/openimages/annotations.csv, then looks up what each photo shows in
Open Images' own box annotations, so it can rank Open Images CLASSES by how
often they trigger it. Use the top classes with:

    python training/download_data.py negatives "<class>" ... --per-class N

Outputs (in data/audit/ by default):
    hard_negatives.csv     one row per photo: best score and what it hit
    hard_negatives.jpg     contact sheet of the most bat-like photos
    oi_boxes_cache.csv     Open Images boxes of our photos (so the 2.2 GB
                           train list is only read once)

Each false detection is credited to the Open Images box it overlaps most
(IoU >= 0.1), or "(no labelled object)" if it overlaps none.

Examples (run from the project folder with the .venv active):
    python training/find_hard_negatives.py
    python training/find_hard_negatives.py --split val --skip-listed data/audit/hard_negatives.csv \\
        --out-prefix data/audit/hard_negatives_new
"""

import argparse
import csv
import io
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402

OI_CLASS_LIST_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
OI_BOX_URLS = {  # our split -> Open Images box lists it was drawn from
    "val": ["https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
            "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"],
    "train": ["https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"],
}
NO_OBJECT = "(no labelled object)"
SCORE_BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)
REPORT_THRESHOLDS = (0.3, 0.5, 0.7)
MIN_ATTRIBUTION_IOU = 0.1


@dataclass
class Photo:
    split: str
    image: str            # relative to data/openimages/
    image_id: str
    max_score: float = 0.0
    hits: list = field(default_factory=list)      # (score, (x1,y1,x2,y2) as 0-1, class)
    classes: set = field(default_factory=set)     # every class Open Images labels in it


# ---------------------------------------------------------------------------
# Pure helpers (no GPU, no network: unit tested)
# ---------------------------------------------------------------------------

def image_id_of(image_path):
    """images/train/negative/0000bee39176697a.jpg -> 0000bee39176697a"""
    return Path(image_path).stem


def read_negatives(rows, splits=("train", "val"), skip=()):
    """Not-a-bat photos from annotations.csv rows (dicts), in file order, no duplicates."""
    photos, seen = [], set(skip)
    for row in rows:
        if row["split"] not in splits or str(row["is_bat"]).strip() != "0":
            continue
        if row["image"] in seen:
            continue
        seen.add(row["image"])
        photos.append(Photo(row["split"], row["image"], image_id_of(row["image"])))
    return photos


def parse_box_line(line):
    """One Open Images box CSV line -> (image_id, label, (xmin, ymin, xmax, ymax)), or None."""
    parts = line.rstrip("\r\n").split(",")
    if len(parts) < 8 or parts[0] == "ImageID":
        return None
    try:
        xmin, xmax, ymin, ymax = (float(v) for v in parts[4:8])
    except ValueError:
        return None
    return parts[0], parts[2], (xmin, ymin, xmax, ymax)


def filter_box_lines(lines, wanted_ids):
    """Boxes of only the wanted images from a stream of CSV lines: {id: [(label, box), ...]}."""
    found = defaultdict(list)
    for line in lines:
        if line[:16] not in wanted_ids:        # cheap check first: IDs are 16 characters
            continue
        parsed = parse_box_line(line)
        if parsed and parsed[0] in wanted_ids:
            found[parsed[0]].append((parsed[1], parsed[2]))
    return found


def iou(a, b):
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def attribute(det_box, labelled_boxes, min_iou=MIN_ATTRIBUTION_IOU):
    """Class name of the labelled box that overlaps a detection most, or NO_OBJECT."""
    best, best_iou = NO_OBJECT, min_iou
    for name, box in labelled_boxes:
        overlap = iou(det_box, box)
        if overlap >= best_iou:
            best, best_iou = name, overlap
    return best


def score_histogram(scores, bins=SCORE_BINS):
    """Counts of scores in [bins[i], bins[i+1])."""
    counts = [0] * (len(bins) - 1)
    for s in scores:
        for i in range(len(counts)):
            if bins[i] <= s < bins[i + 1]:
                counts[i] += 1
                break
    return counts


def rank_classes(photos, thresholds=REPORT_THRESHOLDS):
    """
    Per class: how many photos contain it, and how many of those photos have
    a false detection >= each threshold that is credited to that class.

    Returns rows sorted from most to least confusing (by credited hits at
    each threshold, highest threshold first).
    """
    stats = defaultdict(lambda: {"photos": 0, **{t: 0 for t in thresholds}})
    for p in photos:
        for name in p.classes:
            stats[name]["photos"] += 1
        for t in thresholds:
            for name in {cls for score, _, cls in p.hits if score >= t}:
                stats[name][t] += 1
    rows = [{"class": name, **s} for name, s in stats.items()]
    rows.sort(key=lambda r: tuple(-r[t] for t in sorted(thresholds, reverse=True)) + (r["class"],))
    return rows


def read_class_names(lines):
    """Open Images class list lines ('/m/xxx,Name') -> {label id: name}."""
    names = {}
    for parts in csv.reader(lines):
        if len(parts) >= 2:
            names[parts[0]] = parts[1]
    return names


# ---------------------------------------------------------------------------
# Open Images class lookup (network, cached)
# ---------------------------------------------------------------------------

def load_cache(path):
    boxes = defaultdict(list)
    if path.exists():
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                boxes[row["image_id"]]  # an image with no boxes still counts as looked up
                if row["label"]:
                    box = tuple(float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
                    boxes[row["image_id"]].append((row["label"], box))
    return boxes


def save_cache(path, boxes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part")
    with open(temp, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "label", "xmin", "ymin", "xmax", "ymax"])
        for image_id in sorted(boxes):
            if not boxes[image_id]:
                writer.writerow([image_id, "", "", "", "", ""])
            for label, box in boxes[image_id]:
                writer.writerow([image_id, label, *(f"{v:.5f}" for v in box)])
    temp.replace(path)


def lookup_boxes(photos, cache_path):
    """{image_id: [(label id, box)]} for every photo, streaming Open Images lists only when needed."""
    boxes = load_cache(cache_path)
    for split, urls in OI_BOX_URLS.items():
        missing = {p.image_id for p in photos if p.split == split and p.image_id not in boxes}
        if not missing:
            continue
        for url in urls:
            print(f"Looking up {len(missing)} {split} photos in {url.rsplit('/', 1)[-1]} "
                  "(the train list is 2.2 GB: a few minutes)...", flush=True)
            with urllib.request.urlopen(url) as resp:
                found = filter_box_lines(io.TextIOWrapper(resp, encoding="utf-8"), missing)
            boxes.update(found)
            missing -= set(found)
            if not missing:
                break
        for image_id in missing:        # not found anywhere: remember, don't search again
            boxes[image_id] = []
        save_cache(cache_path, boxes)
    return boxes


# ---------------------------------------------------------------------------
# Detector + contact sheet
# ---------------------------------------------------------------------------

def score_photos(photos, weights, threshold):
    import cv2
    from vision.bat_detector import BatDetector

    config.FINETUNED_DETECTOR_FILE = Path(weights)   # BatDetector loads this file
    detector = BatDetector(min_confidence=0.0)
    # BatNet's Faster R-CNN drops boxes below 0.5 by itself; lower that so we
    # also see the near misses (this only changes the model in this process).
    roi_heads = getattr(getattr(detector.model.detector, "basemodel", None), "roi_heads", None)
    if roi_heads is not None:
        roi_heads.score_thresh = min(roi_heads.score_thresh, threshold, 0.05)
    for i, p in enumerate(photos, 1):
        frame = cv2.imread(str(config.OPENIMAGES_DIR / p.image))
        if frame is None:
            print(f"  unreadable: {p.image}")
            continue
        h, w = frame.shape[:2]
        detections = detector.detect(frame)
        p.max_score = max((d.confidence for d in detections), default=0.0)
        p.hits = [(d.confidence, (d.bbox[0] / w, d.bbox[1] / h, d.bbox[2] / w, d.bbox[3] / h), None)
                  for d in detections if d.confidence >= threshold]
        if i % 250 == 0:
            print(f"  scored {i}/{len(photos)}", flush=True)


def contact_sheet(photos, path, tile=320, columns=6):
    import cv2
    import numpy as np

    rows = (len(photos) + columns - 1) // columns
    sheet = np.full((rows * tile, columns * tile, 3), 255, np.uint8)
    for i, p in enumerate(photos):
        img = cv2.imread(str(config.OPENIMAGES_DIR / p.image))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = tile / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        for score, (x1, y1, x2, y2), _ in p.hits:
            color = (0, 0, 255) if score >= 0.7 else (0, 200, 255) if score >= 0.5 else (255, 160, 0)
            cv2.rectangle(img, (int(x1 * img.shape[1]), int(y1 * img.shape[0])),
                          (int(x2 * img.shape[1]), int(y2 * img.shape[0])), color, 2)
        r, c = divmod(i, columns)
        sheet[r * tile:r * tile + img.shape[0], c * tile:c * tile + img.shape[1]] = img
        top = max(p.hits, key=lambda hit: hit[0])[2] if p.hits else ""
        label = f"{p.max_score:.2f} {top}"[:34]
        org = (c * tile + 4, r * tile + 18)
        cv2.putText(sheet, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(sheet, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), sheet)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="lowest score that counts as 'tempted' (default 0.3)")
    parser.add_argument("--split", choices=["train", "val", "all"], default="all")
    parser.add_argument("--weights", default=str(config.FINETUNED_DETECTOR_FILE),
                        help="fine-tuned detector weights (default: the current ones)")
    parser.add_argument("--annotations", default=str(config.OPENIMAGES_DIR / "annotations.csv"),
                        help="annotations CSV to read (default: data/openimages/annotations.csv)")
    parser.add_argument("--skip-listed", metavar="CSV",
                        help="skip photos listed in an earlier hard_negatives.csv (to score only new ones)")
    parser.add_argument("--out-prefix", default=str(config.PROJECT_ROOT / "data" / "audit" / "hard_negatives"),
                        help="writes <prefix>.csv and <prefix>.jpg")
    parser.add_argument("--top", type=int, default=24, help="photos on the contact sheet")
    parser.add_argument("--no-classes", action="store_true", help="skip the Open Images class lookup")
    args = parser.parse_args(argv)

    skip = set()
    if args.skip_listed:
        with open(args.skip_listed, newline="", encoding="utf-8") as f:
            skip = {row["image"] for row in csv.DictReader(f)}
    splits = ("train", "val") if args.split == "all" else (args.split,)
    with open(args.annotations, newline="") as f:
        photos = read_negatives(csv.DictReader(f), splits, skip)
    print(f"{len(photos)} not-a-bat photos ({', '.join(splits)}) to score, threshold {args.threshold}")
    if not photos:
        return

    score_photos(photos, args.weights, args.threshold)

    names = {}
    if not args.no_classes:
        with urllib.request.urlopen(OI_CLASS_LIST_URL) as resp:
            names = read_class_names(io.TextIOWrapper(resp, encoding="utf-8"))
        boxes = lookup_boxes(photos, config.PROJECT_ROOT / "data" / "audit" / "oi_boxes_cache.csv")
        for p in photos:
            labelled = [(names.get(label, label), box) for label, box in boxes.get(p.image_id, [])]
            p.classes = {name for name, _ in labelled}
            p.hits = [(s, b, attribute(b, labelled)) for s, b, _ in p.hits]
    else:
        for p in photos:
            p.hits = [(s, b, NO_OBJECT) for s, b, _ in p.hits]

    # --- Score distribution ---
    print("\nBest score per photo:")
    counts = score_histogram([p.max_score for p in photos])
    for lo, hi, n in zip(SCORE_BINS, SCORE_BINS[1:], counts):
        print(f"  {lo:.1f}-{min(hi, 1.0):.1f}: {n:5d} {'#' * round(60 * n / len(photos))}")
    for split in splits:
        group = [p for p in photos if p.split == split]
        if group:
            rates = ", ".join(f">={t}: {sum(p.max_score >= t for p in group)} ({100 * sum(p.max_score >= t for p in group) / len(group):.1f}%)"
                              for t in REPORT_THRESHOLDS)
            print(f"  {split}: {len(group)} photos; false alarms {rates}")

    # --- Classes ---
    if names:
        print(f"\nClasses credited with false detections (photos; IoU >= {MIN_ATTRIBUTION_IOU}):")
        print(f"  {'class':28s} {'in photos':>9s} " + " ".join(f"{'>=' + str(t):>6s}" for t in REPORT_THRESHOLDS)
              + "  rate>=0.5")
        for r in rank_classes(photos)[:30]:
            if r[REPORT_THRESHOLDS[0]] == 0:
                break
            rate = f"{100 * r[0.5] / r['photos']:.0f}%" if r["photos"] else "-"
            print(f"  {r['class'][:28]:28s} {r['photos']:9d} "
                  + " ".join(f"{r[t]:6d}" for t in REPORT_THRESHOLDS) + f"  {rate:>8s}")

    # --- Top photos, CSV, contact sheet ---
    photos.sort(key=lambda p: -p.max_score)
    print(f"\nMost bat-like photos:")
    for p in photos[:args.top]:
        top = max(p.hits, key=lambda h: h[0])[2] if p.hits else ""
        print(f"  {p.max_score:.3f}  {p.image}  [{top}]  ({'; '.join(sorted(p.classes))[:70]})")

    prefix = Path(args.out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with open(prefix.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "image", "max_score", "hits", "hit_classes", "classes"])
        for p in photos:
            writer.writerow([p.split, p.image, f"{p.max_score:.4f}", len(p.hits),
                             "; ".join(f"{c}:{s:.2f}" for s, _, c in sorted(p.hits, key=lambda h: -h[0])),
                             "; ".join(sorted(p.classes))])
    contact_sheet(photos[:args.top], prefix.with_suffix(".jpg"))
    print(f"\nWrote {prefix.with_suffix('.csv')} and {prefix.with_suffix('.jpg')}")


if __name__ == "__main__":
    main()
