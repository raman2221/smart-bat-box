"""Compare bat detectors and write data/reports/report_card.md.

    python training/evaluate.py MODEL.pt [MODEL.pt ...] [--threshold 0.7]

Each photo is checked once per model. Saved scores supply the whole sweep.
--save-examples adds contact sheets; --reuse-predictions reuses unchanged inputs.
"""

import argparse
import csv
import io
import json
import math
import sys
import time
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import torch.package
import torchvision
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.finetune_detector import count_matches  # noqa: E402
from vision import config  # noqa: E402

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
REPORT_DIR = config.PROJECT_ROOT / "data" / "reports"
CLASS_NAMES_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
BOX_URLS = (
    "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
)
PEOPLE_LINES = range(1609, 2857)  # fallback for the original people download
PEOPLE_CLASSES = {
    "Person", "Man", "Woman", "Boy", "Girl", "Human face", "Human head",
    "Human body", "Human hand", "Human arm", "Human hair", "Clothing", "Glasses",
}
UNKNOWN_CLASS = "(no class available)"


def is_people_photo(path, classes):
    """Multi-class photos still count as just one person photo."""
    return bool(PEOPLE_CLASSES.intersection(classes.get(Path(path).stem, [])))


def load_checks(classes=None):
    """Snapshot complete val rows, including newly appended ones."""
    bats, negatives, old_people = {}, set(), set()
    skipped = 0
    with open(config.OPENIMAGES_DIR / "annotations.csv", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(io.StringIO(f.read())))
    for row_number, row in enumerate(rows, start=2):
        if row.get("split") != "val":
            continue
        try:
            if not row.get("image") or row.get("is_bat") not in ("0", "1"):
                raise ValueError("unfinished row")
            path = config.OPENIMAGES_DIR / row["image"]
            if not path.is_file():
                raise ValueError("missing photo")
            if row["is_bat"] == "1":
                box = [float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                if not all(math.isfinite(v) and 0 <= v <= 1 for v in box):
                    raise ValueError("invalid box")
                if box[0] >= box[2] or box[1] >= box[3]:
                    raise ValueError("empty box")
                if box not in bats.setdefault(path, []):
                    bats[path].append(box)
            else:
                negatives.add(path)
                if row_number in PEOPLE_LINES:
                    old_people.add(path)
        except (TypeError, KeyError, ValueError):
            skipped += 1
    negatives = sorted(negatives - bats.keys())
    people = ([p for p in negatives if is_people_photo(p, classes)]
              if classes is not None else sorted(old_people.intersection(negatives)))
    inat_list = config.INATURALIST_DIR / "test_images.txt"
    names = inat_list.read_text(encoding="utf-8-sig").splitlines() if inat_list.exists() else []
    inat = sorted({config.PROJECT_ROOT / p.strip() for p in names if p.strip()})
    missing = [p for p in inat if not p.is_file()]
    inat = [p for p in inat if p.is_file()]
    if skipped or missing:
        print(f"Skipped {skipped} incomplete/missing val rows and {len(missing)} missing iNaturalist photos.",
              flush=True)
    return bats, negatives, people, inat


def filter_class_lines(lines, wanted_ids, names):
    """Stream box CSV rows; keep only classes for our image IDs."""
    found = {image_id: set() for image_id in wanted_ids}
    for line in lines:
        image_id = line.partition(",")[0]
        if image_id not in found:
            continue
        parts = line.rstrip("\r\n").split(",", 3)
        if len(parts) >= 3:
            found[image_id].add(names.get(parts[2], parts[2]))
    return {key: sorted(value) for key, value in found.items()}


def download_classes(wanted_ids):
    """Keep just the small mapping, not the large Open Images CSVs."""
    with urllib.request.urlopen(CLASS_NAMES_URL, timeout=30) as response:
        names = dict(csv.reader(io.TextIOWrapper(response, encoding="utf-8-sig")))

    def read_boxes(url):
        print(f"Streaming classes from {Path(url).name}...", flush=True)
        with urllib.request.urlopen(url, timeout=60) as response:
            return filter_class_lines(io.TextIOWrapper(response, encoding="utf-8-sig"), wanted_ids, names)

    combined = {image_id: set() for image_id in wanted_ids}
    with ThreadPoolExecutor(max_workers=2) as pool:
        for found in pool.map(read_boxes, BOX_URLS):
            for image_id, labels in found.items():
                combined[image_id].update(labels)
    return {key: sorted(value) for key, value in combined.items()}


def read_class_reports(files, wanted_ids):
    """Reuse another worker's completed Open Images lookups when offline."""
    found = defaultdict(set)
    for filename in files:
        if not filename.is_file():
            continue
        with filename.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("split") != "val" or not row.get("image") or row.get("classes") is None:
                    continue
                image_id = Path(row["image"]).stem
                if image_id in wanted_ids:
                    found[image_id].update(x.strip() for x in row["classes"].split(";") if x.strip())
    return {key: sorted(value) for key, value in found.items()}


def load_classes(paths, cache_file=None, offline=False):
    """Only fetch again when newly appended images need a class lookup."""
    cache_file = Path(cache_file or REPORT_DIR / "val_classes.json")
    cached = {}
    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text(encoding="utf-8"))
            cached = data.get("images", data)
            if not isinstance(cached, dict) or not all(
                    isinstance(v, list) and all(isinstance(x, str) for x in v)
                    for v in cached.values()):
                raise ValueError("expected image IDs mapped to class-name lists")
        except (OSError, ValueError, TypeError, AttributeError) as error:
            print(f"Class cache could not be read: {error}", flush=True)
            cached = {}
    wanted = {Path(p).stem for p in paths}
    missing = wanted - cached.keys()
    if missing and not offline:
        recovered = {}
        reports = [config.PROJECT_ROOT / "data/audit" / name
                   for name in ("hard_negatives.csv", "hard_negatives_new.csv")]
        reused_reports = False
        try:
            recovered = download_classes(missing)
        except (OSError, ValueError, csv.Error) as error:
            print(f"Class download unavailable: {error}.", flush=True)
            recovered = read_class_reports(reports, missing)
            reused_reports = bool(recovered)
            print(f"Recovered {len(recovered)} image class lists from existing Open Images reports.", flush=True)
        if recovered:
            cached.update(recovered)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            data = {"sources": [CLASS_NAMES_URL, *BOX_URLS], "images": cached}
            if reused_reports:
                data["reused_class_reports"] = [str(p) for p in reports if p.is_file()]
            cache_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
    known = sum(bool(cached.get(key)) for key in wanted)
    print(f"Open Images classes available for {known}/{len(wanted)} non-bat photos.", flush=True)
    return cached or None


def load_detector(weights, device):
    detector = torch.package.PackageImporter(str(config.WEIGHTS_FILE)).load_pickle("model", "model.pkl").detector
    detector.load_state_dict(torch.load(weights, map_location="cpu"))
    return detector.to(device).eval()


def read_image(path, image_size):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read check photo: {path}")
    tensor = torchvision.transforms.functional.to_tensor(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # BatNet itself makes this same square resize. Doing it here lets photos
    # with different original sizes share a batch without padding artifacts.
    tensor = torchvision.transforms.functional.resize(tensor, [image_size, image_size])
    return tensor, (image_size,) * 4


@torch.inference_mode()
def collect_predictions(detector, paths, device, low_threshold=0.3, batch_size=4):
    """One inference per unique photo; retain boxes and scores on the CPU."""
    predictions = {}
    paths = list(dict.fromkeys(paths))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for start in range(0, len(paths), batch_size):
            batch = paths[start:start + batch_size]
            photos = list(pool.map(lambda p: read_image(p, detector.image_size), batch))
            outputs = detector(torch.stack([tensor for tensor, _ in photos]).to(device))
            for path, (_, size), output in zip(batch, photos, outputs):
                keep = output["scores"] >= low_threshold
                boxes = output["boxes"][keep].float().cpu() / torch.tensor(size)
                predictions[path] = {"boxes": boxes.tolist(),
                                     "scores": output["scores"][keep].float().cpu().tolist()}
            done = min(start + batch_size, len(paths))
            if start == 0 or done % 100 < batch_size or done == len(paths):
                print(f"  {done}/{len(paths)} photos checked", flush=True)
    return predictions


def boxes_at(prediction, threshold):
    return [box for box, score in zip(prediction["boxes"], prediction["scores"])
            if score >= threshold]


def fraction(hits, total):
    return hits / total if total else None


def recommendation_score(row):
    """Equal weight for both bat checks, minus the non-bat alarm share."""
    rates = [row["bat_recall"], row["inat_recall"], row["false_alarm_rate"]]
    if any(value is None for value in rates):
        return None
    return (rates[0] + rates[1]) / 2 - rates[2]


def threshold_sweep(checks, predictions, thresholds=THRESHOLDS, matcher=count_matches):
    """Calculate every cutoff from existing scores; never run a detector."""
    bats, negatives, people, inat = checks
    rows = []
    for threshold in sorted(set(thresholds)):
        found = extra = 0
        for path, true in bats.items():
            pred = boxes_at(predictions[path], threshold)
            matched = matcher(true, pred)
            found += matched
            extra += len(pred) - matched
        total = sum(len(boxes) for boxes in bats.values())
        fooled = sum(bool(boxes_at(predictions[p], threshold)) for p in set(negatives))
        people_fooled = sum(bool(boxes_at(predictions[p], threshold)) for p in set(people))
        inat_found = sum(bool(boxes_at(predictions[p], threshold)) for p in set(inat))
        row = {"threshold": threshold, "bat_found": found, "bat_total": total,
               "bat_recall": fraction(found, total), "extra_boxes": extra,
               "false_alarms": fooled, "negative_total": len(set(negatives)),
               "false_alarm_rate": fraction(fooled, len(set(negatives))),
               "people_false_alarms": people_fooled, "people_total": len(set(people)),
               "people_false_alarm_rate": fraction(people_fooled, len(set(people))),
               "inat_found": inat_found, "inat_total": len(set(inat)),
               "inat_recall": fraction(inat_found, len(set(inat)))}
        row["score"] = recommendation_score(row)
        rows.append(row)
    return rows


def recommend_threshold(rows):
    """On a score tie, prefer fewer false alarms, then a higher cutoff."""
    valid = [r for r in rows if r["score"] is not None and r["threshold"] in THRESHOLDS]
    return max(valid, key=lambda r: (round(r["score"], 12), -r["false_alarm_rate"],
                                     r["threshold"])) if valid else None


def group_categories(negatives, predictions, classes, threshold, limit=10):
    """Count photos, not boxes. Multi-class photos contribute to each class."""
    groups = defaultdict(lambda: [0, 0])
    for path in set(negatives):
        labels = set((classes or {}).get(Path(path).stem, [])) or {UNKNOWN_CLASS}
        alarm = bool(boxes_at(predictions[path], threshold))
        for label in labels:
            groups[label][0] += int(alarm)
            groups[label][1] += 1
    rows = [{"category": name, "false_alarms": hits, "total": total, "rate": hits / total}
            for name, (hits, total) in groups.items()]
    return sorted(rows, key=lambda r: (-r["false_alarms"], -r["rate"], r["category"]))[:limit]


@torch.inference_mode()
def measure_speed(detector, device):
    frame = torch.rand(1, 3, 480, 640, device=device)
    for _ in range(3):
        detector(frame)
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(20):
        detector(frame)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / 20 * 1000


def cache_signature(weights, paths, low_threshold):
    """Reject reused scores when the model or an input photo changes."""
    def stamp(path):
        stat = Path(path).stat()
        return [str(Path(path).resolve()), stat.st_size, stat.st_mtime_ns]
    return {"model": stamp(weights), "base": stamp(config.WEIGHTS_FILE),
            "photos": [stamp(p) for p in paths], "low_threshold": low_threshold}


def evaluate(weights, checks, device, threshold, classes=None, batch_size=4, reuse=False):
    bats, negatives, _, inat = checks
    paths = list(dict.fromkeys([*bats, *negatives, *inat]))
    low = min(0.3, threshold)
    signature = cache_signature(weights, paths, low)
    cache = REPORT_DIR / f"predictions_{Path(weights).stem}.json"
    predictions = None
    speed = None
    if reuse and cache.exists():
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if data["signature"] == signature:
                predictions = {Path(p): value for p, value in data["predictions"].items()}
                if data.get("device") == device:
                    speed = data["speed_ms"]
                print("  Reusing unchanged predictions.", flush=True)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            print(f"  Prediction cache unavailable: {error}; checking photos again.", flush=True)
            predictions = None
    if predictions is None or speed is None:
        detector = load_detector(weights, device)
        # Lower the detector's own cutoff before the single pass.
        detector.basemodel.roi_heads.score_thresh = min(detector.basemodel.roi_heads.score_thresh, low)
        if predictions is None:
            predictions = collect_predictions(detector, paths, device, low, batch_size)
        speed = measure_speed(detector, device)
        del detector
        if device == "cuda":
            torch.cuda.empty_cache()
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"signature": signature, "speed_ms": speed, "device": device,
                                     "predictions": {str(p): v for p, v in predictions.items()}}),
                         encoding="utf-8")
    rows = threshold_sweep(checks, predictions, (*THRESHOLDS, threshold))
    recommended = recommend_threshold(rows)
    chosen = next(r for r in rows if r["threshold"] == threshold)
    category_threshold = recommended["threshold"] if recommended else threshold
    return {"model": str(weights), "sweep": rows, "selected": chosen,
            "recommended": recommended, "speed_ms": speed, "predictions": predictions,
            "categories": group_categories(negatives, predictions, classes, category_threshold),
            "category_threshold": category_threshold}


def rate_text(hits, total):
    return f"{hits}/{total} ({hits / total:.1%})" if total else "n/a (no photos)"


def summary_rows(results, recommended=False):
    """The same side-by-side table feeds the console and Markdown."""
    rows = [("Metric", *[Path(r["model"]).name for r in results])]
    metrics = [(r["recommended"] or r["selected"]) if recommended else r["selected"] for r in results]
    fields = [("Check bat recall (boxes)", "bat_found", "bat_total"),
              ("Non-bat photos called a bat", "false_alarms", "negative_total"),
              ("People photos called a bat", "people_false_alarms", "people_total"),
              ("iNaturalist recall (photos)", "inat_found", "inat_total")]
    rows.append(("Confidence threshold", *[f"{m['threshold']:.2f}" for m in metrics]))
    for label, hits, total in fields:
        rows.append((label, *[rate_text(m[hits], m[total]) for m in metrics]))
    rows.append(("Unmatched boxes on bat checks", *[str(m["extra_boxes"]) for m in metrics]))
    rows.append(("Recommendation score", *[f"{m['score']:.4f}" if m["score"] is not None else "n/a"
                                              for m in metrics]))
    rows.append(("Speed (640x480, batch 1)", *[f"{r['speed_ms']:.1f} ms/frame" for r in results]))
    return rows


def console_table(rows):
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    return "\n".join(" | ".join(str(cell).ljust(width) for cell, width in zip(row, widths))
                     for row in rows)


def markdown_table(rows):
    def line(row):
        return "| " + " | ".join(str(cell).replace("|", "\\|").replace("\n", " ") for cell in row) + " |"
    return "\n".join([line(rows[0]), line(["---"] * len(rows[0])), *[line(row) for row in rows[1:]]])


def best_model(results):
    valid = [r for r in results if r["recommended"] is not None]
    return max(valid, key=lambda r: (round(r["recommended"]["score"], 12),
                                     -r["recommended"]["false_alarm_rate"], -r["speed_ms"])) if valid else None


def report_summary(results):
    best = best_model(results)
    if best is None:
        return "There is not enough data to recommend a model: both bat sets and a non-bat set are needed."
    row = best["recommended"]
    return (f"{Path(best['model']).name} is best by the requested recall-minus-false-alarm score "
            f"at confidence {row['threshold']:.2f} (score {row['score']:.4f}). "
            f"It finds {row['bat_recall']:.1%} of hand-labelled bat boxes and detects a bat in "
            f"{row['inat_recall']:.1%} of held-back iNaturalist photos, while calling "
            f"{row['false_alarm_rate']:.1%} of non-bat photos a bat. "
            "This balances missed bats against false alarms; speed breaks model ties after "
            "false alarms. These checks help choose a model and threshold, so fresh test "
            "photos and bat-box webcam footage are needed to measure real-world accuracy.")


def recommendation_reason(result):
    row = result["recommended"]
    if row is None:
        return "n/a: both bat sets and a non-bat set are needed."
    return (f"{row['threshold']:.2f}: ({row['bat_recall']:.4f} + {row['inat_recall']:.4f}) / 2 "
            f"- {row['false_alarm_rate']:.4f} = {row['score']:.4f}, the highest sweep score.")


def sweep_table(results):
    rows = [("Threshold", *[Path(r["model"]).name for r in results])]
    for cutoff in sorted({row["threshold"] for r in results for row in r["sweep"]}):
        cells = []
        for result in results:
            row = next(r for r in result["sweep"] if r["threshold"] == cutoff)
            rates = [f"{row[key]:.1%}" if row[key] is not None else "n/a"
                     for key in ("bat_recall", "inat_recall", "false_alarm_rate")]
            score = f"{row['score']:.4f}" if row["score"] is not None else "n/a"
            cells.append(" / ".join([*rates, score]))
        rows.append((f"{cutoff:.2f}", *cells))
    return rows


def category_table(result):
    return [("Open Images class", "False alarms / photos", "Alarm share"),
            *[(r["category"], f"{r['false_alarms']}/{r['total']}", f"{r['rate']:.1%}")
              for r in result["categories"]]]


def render_markdown(results, checks, classes, device, threshold):
    bats, negatives, people, inat = checks
    coverage = sum(bool((classes or {}).get(p.stem)) for p in negatives)
    people_basis = ("Open Images class names" if classes is not None
                    else "legacy physical lines 1609-2856 (class lookup unavailable)")
    parts = ["# Bat detector accuracy report card", "",
             f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}; device: {device}.", "",
             report_summary(results), "", "## At the requested threshold", "",
             f"Requested confidence: {threshold:.2f}.", "", markdown_table(summary_rows(results)), "",
             "## At each model's recommended threshold", "",
             markdown_table(summary_rows(results, recommended=True)), "",
             "Recommendation = (check-photo bat recall + iNaturalist recall) / 2 "
             "- share of non-bat photos called a bat. Higher is better. "
             "Only 0.5, 0.6, 0.7, 0.8 and 0.9 compete. Ties prefer fewer false alarms, "
             "then the higher threshold.", ""]
    parts.extend(f"- **{Path(r['model']).name}**: {recommendation_reason(r)}" for r in results)
    parts.extend(["", "## Threshold sweep (models side by side)", "",
                  "Each cell: check bat recall / iNaturalist recall / non-bat alarm share / score.", "",
                  markdown_table(sweep_table(results)), "", "## Top 10 false-alarm categories", "",
                  "Ranked by number of non-bat photos called a bat, then alarm share. "
                  "These are classes present in the photo, not the object the detector hit. "
                  "Photos with several classes count in each; category totals overlap.", ""])
    for result in results:
        parts.extend([f"### {Path(result['model']).name} (confidence {result['category_threshold']:.2f})", "",
                      markdown_table(category_table(result)), ""])
    parts.extend(["## What was measured", "",
                  f"- One snapshot of annotations.csv: {len(bats)} bat photos / "
                  f"{sum(map(len, bats.values()))} boxes, {len(negatives)} unique non-bat val photos; "
                  f"{len(inat)} unique held-back iNaturalist photos.",
                  "- Hand-labelled recall uses imported count_matches: one-to-one matching "
                  "at IoU >= 0.5. Unmatched detection boxes are reported separately.",
                  "- iNaturalist has no hand-drawn boxes here: its recall means any bat "
                  "detection in a known bat photo, not proof of correct localization.",
                  "- The iNaturalist list is evaluated as supplied. The missed-photo "
                  "contact sheet includes some sound plots, so list quality affects this recall.",
                  f"- People: {len(people)} non-bat photos selected by {people_basis}. "
                  "Class-based selection includes Person, Man, Woman, Boy, Girl, human "
                  "face/head/body/hand/arm/hair, Clothing and Glasses.",
                  f"- Class coverage: {coverage}/{len(negatives)} negatives. Missing classes "
                  "appear as '(no class available)'; incomplete caches restrict the people "
                  "metric to known people classes.",
                  "- Class lists are cached in data/reports/val_classes.json. If downloads "
                  "are unavailable, completed Open Images class lookups in "
                  "data/audit/hard_negatives.csv and hard_negatives_new.csv can supply the cache; "
                  "the cache records this provenance.",
                  "- One detector pass per unique photo at confidence <= 0.3; thresholds "
                  "filter saved scores. Speed averages 20 warmed-up 640x480 frames in "
                  "full precision; sharing the GPU may change timings.",
                  "- Base model: " + str(config.WEIGHTS_FILE),
                  "- Weights: " + ", ".join(r["model"] for r in results), "",
                  "Class sources: " + ", ".join(f"[{Path(url).name}]({url})"
                                                for url in [CLASS_NAMES_URL, *BOX_URLS]) + ".", ""])
    return "\n".join(parts)


def worst_examples(result, checks, limit=12):
    """Highest-confidence alarms and largest missed bat counts come first."""
    bats, negatives, _, inat = checks
    predictions = result["predictions"]
    cutoff = result["category_threshold"]
    alarms = [(p, max(predictions[p]["scores"], default=0)) for p in negatives
              if boxes_at(predictions[p], cutoff)]
    alarms.sort(key=lambda item: (-item[1], str(item[0])))
    misses = []
    for path, true in bats.items():
        missed = len(true) - count_matches(true, boxes_at(predictions[path], cutoff))
        if missed:
            misses.append((path, missed, "check"))
    misses.extend((p, 1, "iNat") for p in inat if not boxes_at(predictions[p], cutoff))
    misses.sort(key=lambda item: (-item[1], max(predictions[item[0]]["scores"], default=0), str(item[0])))
    return ([(p, f"False alarm: score {score:.2f}") for p, score in alarms[:limit]],
            [(p, f"Missed {missed} bat(s), {source}") for p, missed, source in misses[:limit]])


def save_contact_sheet(items, result, checks, filename):
    """Green boxes are hand labels; red boxes are kept detector calls."""
    width, height = 280, 245
    sheet = Image.new("RGB", (width * 4, 40 + height * max(1, (len(items) + 3) // 4)), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), f"{Path(result['model']).name}, confidence {result['category_threshold']:.2f}", fill="black")
    for index, (path, caption) in enumerate(items):
        with Image.open(path) as original:
            image = original.convert("RGB")
        image.thumbnail((width - 12, height - 50))
        w, h = image.size
        boxes = ImageDraw.Draw(image)
        for box in checks[0].get(path, []):
            boxes.rectangle([box[0] * w, box[1] * h, box[2] * w, box[3] * h], outline="lime", width=2)
        for box in boxes_at(result["predictions"][path], result["category_threshold"]):
            boxes.rectangle([box[0] * w, box[1] * h, box[2] * w, box[3] * h], outline="red", width=2)
        x, y = index % 4 * width + 6, 40 + index // 4 * height
        sheet.paste(image, (x, y))
        draw.text((x, y + height - 44), path.name, fill="black")
        draw.text((x, y + height - 25), caption, fill="black")
    if not items:
        draw.text((10, 60), "No errors of this kind at the recommended threshold.", fill="black")
    sheet.save(filename, quality=90)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="detector weight files (.pt) to compare")
    parser.add_argument("--threshold", type=float, default=config.DETECTION_CONFIDENCE_THRESHOLD)
    parser.add_argument("--save-examples", action="store_true", help="save best-model error contact sheets")
    parser.add_argument("--batch-size", type=int, default=4, help="photos per GPU batch (lower if GPU is busy)")
    parser.add_argument("--reuse-predictions", action="store_true", help="reuse scores for unchanged models and photos")
    parser.add_argument("--offline-classes", action="store_true", help="use only the existing class cache")
    args = parser.parse_args()
    if not math.isfinite(args.threshold) or not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    device = "cuda" if torch.cuda.is_available() and not config.FORCE_CPU else "cpu"
    checks = load_checks()
    classes = load_classes(checks[1], offline=args.offline_classes)
    if classes is not None:
        checks = (checks[0], checks[1], [p for p in checks[1] if is_people_photo(p, classes)], checks[3])
    print(f"Device {device}; {len(checks[0])} bat check photos, {len(checks[1])} negatives, "
          f"{len(checks[2])} people photos, {len(checks[3])} iNaturalist photos.", flush=True)
    results = []
    for weights in args.models:
        print(f"\nChecking {weights}...", flush=True)
        results.append(evaluate(weights, checks, device, args.threshold, classes,
                                args.batch_size, args.reuse_predictions))
    print("\nAt requested threshold:\n" + console_table(summary_rows(results)), flush=True)
    print("\nAt recommended thresholds:\n" + console_table(summary_rows(results, recommended=True)), flush=True)
    print("\nSweep cells: check recall / iNaturalist recall / false-alarm share / score.")
    print(console_table(sweep_table(results)))
    print("\nScore = (check bat recall + iNaturalist recall) / 2 - non-bat false-alarm share.")
    print("Ties: fewer false alarms, then higher threshold.")
    for result in results:
        print(f"\n{Path(result['model']).name}: recommended {recommendation_reason(result)}")
        print(f"Top false-alarm classes at {result['category_threshold']:.2f}:")
        print(console_table(category_table(result)))
    print("\n" + report_summary(results), flush=True)
    report_file = REPORT_DIR / "report_card.md"
    report_file.write_text(render_markdown(results, checks, classes, device, args.threshold), encoding="utf-8")
    if args.save_examples and best_model(results):
        best = best_model(results)
        alarms, misses = worst_examples(best, checks)
        for name, items in (("worst_false_alarms.jpg", alarms), ("missed_bats.jpg", misses)):
            save_contact_sheet(items, best, checks, REPORT_DIR / name)
    print(f"\nReport saved to {report_file}", flush=True)


if __name__ == "__main__":
    main()
