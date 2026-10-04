"""
refine_labels.py

Cleans up the automatic bat boxes made by training/auto_label.py.

A visual check of the automatic labels found that every box contains a
bat, but about 30% had a bad shape: one box around a whole group of bats,
a box around only the head, or a box with lots of glove / tree / cave in
it. Those teach the detector the wrong thing, so this script looks at the
bat's actual pixels and keeps a label only if it looks like ONE whole bat.

How it works, for every box:
  1. Cut out the box plus some extra room around it (CONTEXT), so we can
     see whether the bat carries on past the box edge.
  2. Ask BatNet's segmentation network (a "U-Net" that colours in the bat's
     pixels; BatNet uses it to remove the background before naming the
     species) which pixels are bat. It was trained on infrared photos, but
     it also works on most colour photos.
  3. Judge the mask (judge_mask below):
       - no clear bat pixels                      -> reject ("no mask")
       - several separate, similar-sized blobs    -> reject ("several bats")
       - the bat runs off the edge of the extra room, i.e. the box only
         covered part of it                       -> reject ("partial")
       - the blob is tiny compared with the box   -> reject ("mask too small")
     Otherwise keep it, and if the mask's own box is clearly tighter or
     more complete than the old box, use the mask's box instead.

Input:  data/auto_labels_raw.csv (the detector's boxes, from auto_label.py)
Output: data/auto_labels.csv (same columns), plus data/audit/refine_report.csv
        (one row per box with the decision and the reason) and, with
        --sheets, contact sheets data/audit/refined_kept.jpg and
        data/audit/refined_rejected.jpg for checking by eye.

Run from the project folder (with the .venv active):
    python training/refine_labels.py --sheets
"""

import argparse
import csv
import random
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402

LABEL_COLUMNS = ["source", "image", "xmin", "ymin", "xmax", "ymax", "score"]
RAW_LABELS_FILE = config.PROJECT_ROOT / "data" / "auto_labels_raw.csv"
AUDIT_DIR = config.PROJECT_ROOT / "data" / "audit"
REPORT_FILE = AUDIT_DIR / "refine_report.csv"

# The segmentation network looks at square 256 x 256 crops (like in BatNet).
SEG_SIZE = 256
# Extra room around each box, as a fraction of the box size on every side.
CONTEXT = 0.25
# Mask blobs smaller than this fraction of the crop are just specks.
MIN_PART = 0.01
# A second blob at least this big (compared with the biggest) is another bat.
SECOND_BAT = 0.30
# If more than this fraction of an open edge of the crop is bat pixels,
# the bat carries on past what we can see: the box was only on part of it.
EDGE_TOUCH = 0.10
# At most this fraction of the bat blob may lie outside the box. More means
# the box missed part of the bat, or the mask leaked onto a hand or rock.
MAX_OUTSIDE = 0.25
# The bat blob must fill at least this much of the original box.
MIN_FILL = 0.10
# A second blob this big (compared with the biggest) means the mask came
# out in pieces: keep the label, but don't trust the mask's box.
FRAGMENTED = 0.08
# If the mask's box is less than this fraction of the old box's area, reject:
# checked by eye, that was more often a mask failure than a loose box.
MIN_SHRINK = 0.60
# Switch to the mask's box when it overlaps the old box less than this
# (IoU, 1.0 = identical), i.e. when it is clearly different.
TIGHTEN_BELOW_IOU = 0.85
# Padding added around the mask's box (fraction of its size), because the
# mask tends to miss the faint outermost pixels of wings and ears.
MASK_BOX_PAD = 0.03


@dataclass
class Verdict:
    """The decision for one box, plus the measurements it was based on."""

    keep: bool
    reason: str               # "ok", "tightened", or why it was rejected
    box: tuple                # (x1, y1, x2, y2) in window pixels (the new box if kept)
    fill: float = 0.0         # main bat blob pixels inside the box / box area
    outside: float = 0.0      # fraction of the main blob lying outside the box
    second: float = 0.0       # size of the next-biggest blob / main blob
    edge: float = 0.0         # most of any open window side covered by the blob
    blob: np.ndarray = None   # True on the main bat's pixels (window-sized)


def box_area_of(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def box_iou(a, b):
    """Overlap of two boxes: shared area / combined area (0 to 1)."""
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def context_window(box, width, height, context=CONTEXT):
    """The box (in pixels) grown by `context` on each side, kept inside the photo.

    Also returns which sides of the window are "open", i.e. are not the
    photo's own edge. A bat touching the photo's edge is just cut off by
    the photo (fine); a bat touching an open side runs past the window.
    """
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * context, (y2 - y1) * context
    left, top = max(0, int(x1 - mx)), max(0, int(y1 - my))
    right, bottom = min(width, int(np.ceil(x2 + mx))), min(height, int(np.ceil(y2 + my)))
    open_sides = {"left": left > 0, "top": top > 0,
                  "right": right < width, "bottom": bottom < height}
    return (left, top, right, bottom), open_sides


def mask_parts(mask):
    """Split a mask into blobs, after joining nearby pieces and dropping specks.

    Thin wings often come out as separate pieces, so a small "closing"
    (grow then shrink) first glues pieces that almost touch.
    Returns (labels image, blob statistics, list of blob numbers to use).
    """
    mask = mask.astype(np.uint8)
    size = max(3, int(round(min(mask.shape) * 0.03)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = MIN_PART * mask.size
    parts = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    return labels, stats, parts


def judge_mask(mask, box, open_sides):
    """Decide whether a bat mask looks like ONE whole bat inside `box`.

    mask: True where the bat is, for the context window (height x width).
    box: the original box (x1, y1, x2, y2) in the same window's pixels.
    open_sides: from context_window().
    """
    height, width = mask.shape
    labels, stats, parts = mask_parts(mask)
    bx1, by1 = max(0, int(round(box[0]))), max(0, int(round(box[1])))
    bx2, by2 = min(width, int(round(box[2]))), min(height, int(round(box[3])))
    box_area = max(1, (bx2 - bx1) * (by2 - by1))
    inside = {i: int((labels[by1:by2, bx1:bx2] == i).sum()) for i in parts}
    if not parts or max(inside.values()) == 0:
        return Verdict(False, "no mask", tuple(box))

    # --- Measurements -------------------------------------------------------
    # The main bat is the blob with the most pixels inside the original box.
    main = max(parts, key=lambda i: inside[i])
    blob = labels == main
    main_area = int(stats[main, cv2.CC_STAT_AREA])
    fill = inside[main] / box_area
    outside = 1 - inside[main] / main_area
    # Other blobs that are mostly inside the box could be more bats.
    others = [stats[i, cv2.CC_STAT_AREA] for i in parts
              if i != main and inside[i] >= 0.5 * stats[i, cv2.CC_STAT_AREA]]
    second = max(others, default=0) / main_area
    # How much of each open side of the window does the bat touch?
    sides = {"left": blob[:, 0], "right": blob[:, -1], "top": blob[0, :], "bottom": blob[-1, :]}
    edge = max([float(pixels.mean()) for side, pixels in sides.items() if open_sides[side]],
               default=0.0)
    measured = dict(fill=fill, outside=outside, second=second, edge=edge, blob=blob)

    # --- Decisions ------------------------------------------------------------
    if second >= SECOND_BAT:
        return Verdict(False, "several bats", tuple(box), **measured)
    if edge > EDGE_TOUCH:
        return Verdict(False, "partial", tuple(box), **measured)
    if outside > MAX_OUTSIDE:
        return Verdict(False, "spills past box", tuple(box), **measured)
    if fill < MIN_FILL:
        return Verdict(False, "mask too small", tuple(box), **measured)

    # The mask's own box, padded a little. It may only shrink the old box,
    # never grow it: growing tended to swallow hands, gloves and feeders.
    x, y = stats[main, cv2.CC_STAT_LEFT], stats[main, cv2.CC_STAT_TOP]
    w, h = stats[main, cv2.CC_STAT_WIDTH], stats[main, cv2.CC_STAT_HEIGHT]
    px, py = w * MASK_BOX_PAD, h * MASK_BOX_PAD
    tight = (max(box[0], x - px), max(box[1], y - py),
             min(box[2], x + w + px), min(box[3], y + h + py))
    shrink = box_area_of(tight) / box_area_of(box)
    if shrink < MIN_SHRINK:
        # Either a very loose box or (more often, when checked by eye) a mask
        # that only found the face / one bat of a group. Too unsure to keep.
        return Verdict(False, "mask much smaller", tuple(box), **measured)
    if second >= FRAGMENTED:
        # The mask came out in pieces (e.g. wing apart from body), so its box
        # can't be trusted; the detector's box was usually fine in these.
        return Verdict(True, "ok", tuple(box), **measured)
    if box_iou(tight, box) < TIGHTEN_BELOW_IOU:
        return Verdict(True, "tightened", tight, **measured)
    return Verdict(True, "ok", tuple(box), **measured)


# ---------------------------------------------------------------------------
# The segmentation network (only needed when running for real, not in tests)
# ---------------------------------------------------------------------------

class Segmenter:
    """BatNet's bat-vs-background U-Net: crop (BGR) -> True where the bat is."""

    def __init__(self, device=None):
        import torch
        import torch.package

        self.torch = torch
        importer = torch.package.PackageImporter(str(config.WEIGHTS_FILE))
        model = importer.load_pickle("model", "model.pkl")
        self.device = device or ("cuda" if torch.cuda.is_available() and not config.FORCE_CPU else "cpu")
        self.unet = model.segmentation_model.eval().to(self.device)

    def __call__(self, crop_bgr):
        torch = self.torch
        height, width = crop_bgr.shape[:2]
        rgb = cv2.cvtColor(cv2.resize(crop_bgr, (SEG_SIZE, SEG_SIZE), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        # The U-Net expects RGB values from 0 to 1 and normalises them itself.
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.unet(x)[0, 0].float().cpu().numpy()
        # Scale the answer back to the crop's own size; above 0 means "bat".
        return cv2.resize(logits, (width, height), interpolation=cv2.INTER_LINEAR) > 0


def refine_box(image, box_fraction, segmenter):
    """Judge one box (0-1 coordinates) in one photo.

    Returns (verdict, new box in 0-1 coordinates, window, mask) so callers
    can draw what happened.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box_fraction
    box = (x1 * width, y1 * height, x2 * width, y2 * height)
    window, open_sides = context_window(box, width, height)
    left, top, right, bottom = window
    mask = segmenter(image[top:bottom, left:right])
    local_box = (box[0] - left, box[1] - top, box[2] - left, box[3] - top)
    verdict = judge_mask(mask, local_box, open_sides)
    nx1, ny1, nx2, ny2 = verdict.box
    new_box = (min(1.0, max(0.0, (nx1 + left) / width)), min(1.0, max(0.0, (ny1 + top) / height)),
               min(1.0, max(0.0, (nx2 + left) / width)), min(1.0, max(0.0, (ny2 + top) / height)))
    return verdict, new_box, window, mask


def read_labels(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def draw_example(image, row, verdict, new_box, window, tile=200):
    """One contact-sheet tile, zoomed in on the window around the box.

    Red outline = the bat's mask, yellow = old box, green = new box (kept only).
    """
    height, width = image.shape[:2]
    left, top, right, bottom = window
    crop = image[top:bottom, left:right]
    scale = tile / max(crop.shape[:2])
    size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
    out = np.full((tile, tile, 3), 40, np.uint8)
    out[:size[1], :size[0]] = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    if verdict.blob is not None:
        blob = cv2.resize(verdict.blob.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 0, 255), 1)

    def rect(box, colour):
        x1, y1, x2, y2 = [(v * size - offset) * scale for v, size, offset in
                          zip(box, (width, height, width, height), (left, top, left, top))]
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2) - 1, int(y2) - 1), colour, 1)

    rect([float(row[k]) for k in LABEL_COLUMNS[2:6]], (0, 220, 255))
    if verdict.keep:
        rect(new_box, (0, 255, 0))
    return out


def make_sheet(tiles, labels, path, columns=8):
    tile = tiles[0].shape[0]
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.full((rows * tile, columns * tile, 3), 40, np.uint8)
    for i, (image, text) in enumerate(zip(tiles, labels)):
        r, c = divmod(i, columns)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = image
        cv2.putText(sheet, text, (c * tile + 4, r * tile + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3)
        cv2.putText(sheet, text, (c * tile + 4, r * tile + 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1)
    cv2.imwrite(str(path), sheet)


def refine(input_file=RAW_LABELS_FILE, output_file=None, sheets=False, segmenter=None, seed=1):
    """Refine every box in input_file and write the kept ones to output_file."""
    output_file = output_file or config.AUTO_LABELS_FILE
    rows = read_labels(input_file)
    segmenter = segmenter or Segmenter()
    kept, report = [], []
    counts = {}
    examples = {True: [], False: []}
    for n, row in enumerate(rows, 1):
        image = cv2.imread(str(config.PROJECT_ROOT / row["image"]))
        if image is None:
            counts["unreadable"] = counts.get("unreadable", 0) + 1
            continue
        box = tuple(float(row[k]) for k in LABEL_COLUMNS[2:6])
        verdict, new_box, window, mask = refine_box(image, box, segmenter)
        counts[verdict.reason] = counts.get(verdict.reason, 0) + 1
        report.append([row["image"], verdict.reason, *[f"{v:.3f}" for v in (
            verdict.fill, verdict.outside, verdict.second, verdict.edge)],
            *[f"{v:.5f}" for v in new_box]])
        if verdict.keep:
            kept.append({**row, **dict(zip(LABEL_COLUMNS[2:6], [f"{v:.5f}" for v in new_box]))})
        if sheets:
            # Remember just enough to redraw it (masks are big, so not the mask).
            examples[verdict.keep].append((row, replace(verdict, blob=None), new_box, window))
        if n % 500 == 0:
            print(f"  {n}/{len(rows)}", flush=True)

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LABEL_COLUMNS)
        writer.writeheader()
        writer.writerows(kept)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "decision", "fill", "outside", "second", "edge", "xmin", "ymin", "xmax", "ymax"])
        writer.writerows(report)
    print(f"Kept {len(kept)} of {len(rows)} boxes. Decisions: {counts}")
    print(f"Labels saved to {output_file}; report in {REPORT_FILE}")

    if sheets:
        # Redraw a random sample of each group (masks are recomputed for these few).
        rng = random.Random(seed)
        for keep, name in [(True, "refined_kept.jpg"), (False, "refined_rejected.jpg")]:
            sample = rng.sample(examples[keep], min(48, len(examples[keep])))
            tiles, texts = [], []
            for row, verdict, new_box, window in sample:
                image = cv2.imread(str(config.PROJECT_ROOT / row["image"]))
                box = tuple(float(row[k]) for k in LABEL_COLUMNS[2:6])
                verdict, new_box, window, _ = refine_box(image, box, segmenter)
                tiles.append(draw_example(image, row, verdict, new_box, window))
                texts.append(verdict.reason)
            if tiles:
                make_sheet(tiles, texts, AUDIT_DIR / name)
        print(f"Contact sheets: {AUDIT_DIR / 'refined_kept.jpg'}, {AUDIT_DIR / 'refined_rejected.jpg'}")
    return counts, len(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=RAW_LABELS_FILE,
                        help="detector boxes to refine (default: data/auto_labels_raw.csv)")
    parser.add_argument("--sheets", action="store_true",
                        help="save kept/rejected contact sheets to data/audit/")
    parser.add_argument("--seed", type=int, default=1, help="which examples go on the sheets")
    args = parser.parse_args()
    refine(args.input, sheets=args.sheets, seed=args.seed)


if __name__ == "__main__":
    main()
