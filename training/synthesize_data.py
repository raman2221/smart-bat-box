"""Make extra training photos by pasting real bat cutouts onto "not a bat" photos.

Each cutout is ONLY the bat's own pixels: BatNet's segmentation network
(see training/refine_labels.py) colours in which pixels are bat, and we
keep a cutout only if that mask looks like one whole bat (not a group,
not cut off by the photo's edge). The mask edge is softened by a pixel or
so, so the pasted bat blends in instead of sitting in a visible patch of
its old background.

Where the bats come from (training photos only, never test photos):
  - Open Images training photos with bat boxes (data/openimages/annotations.csv)
  - the refined automatic labels on iNaturalist photos (data/auto_labels.csv)
Backgrounds: Open Images "not a bat" training photos and our own room
photos (data/room_negatives/).

Every saved box is the bounding box of the pasted bat's pixels, and every
pasted bat is at least MIN_BAT_PIXELS on its shorter side, so there are no
invisible specks labelled as bats.

Output: data/synthetic/images/synth_00000.jpg ..., data/synthetic/labels.csv
(same columns as data/auto_labels.csv) and data/synthetic/preview.jpg
(the first 12 photos with their boxes drawn, for checking by eye).

Run from the project folder (uses the GPU if there is one, ~1 minute):
    .venv\\Scripts\\python.exe training/synthesize_data.py --count 1500
"""

import argparse
import csv
import math
import random
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402
from training.refine_labels import context_window, judge_mask  # noqa: E402

LABEL_COLUMNS = ["source", "image", "xmin", "ymin", "xmax", "ymax", "score"]
LONGEST_SIDE = 640
# Pasted bats are at least this many pixels on their shorter side.
MIN_BAT_PIXELS = 12
# Bats smaller than this in the source photo are too blurry to cut out.
MIN_SOURCE_PIXELS = 24
# Cutouts are stored at most this big (bigger than any paste we make).
CUTOUT_SIDE = 256
# At most this many cutouts from the iNaturalist auto labels (they are
# plentiful; this keeps loading time and memory small).
MAX_AUTO_CUTOUTS = 1500
# Cutouts are never enlarged more than this (enlarged pixels look blurry).
MAX_ENLARGE = 1.5
# The pasted bat must differ from what it covers by at least this much on
# average (grey levels, 0-255), or it would be practically invisible.
MIN_CONTRAST = 20
# Pasted bats may overlap each other at most this much (IoU).
MAX_OVERLAP = 0.1


@dataclass
class Cutout:
    """One bat, cut out of a training photo."""

    path: Path            # the photo it came from
    pixels: np.ndarray    # colour pixels (BGR), just around the bat
    alpha: np.ndarray     # 1.0 on the bat, 0.0 elsewhere (same height/width)


def read_image(path):
    """Read colour pixels; returns None for unreadable photos."""
    # imdecode also works when a Windows filename contains non-ASCII letters.
    try:
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error, ValueError):
        return None
    return image


def resize_background(image):
    height, width = image.shape[:2]
    scale = LONGEST_SIDE / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)


def excluded_paths(rows):
    """Exclude entire validation/test photos, including duplicate training rows.

    Also skips the photos training/quality_check.py listed (e.g. exact copies
    of test photos), just like training does.
    """
    excluded = {(config.OPENIMAGES_DIR / row["image"]).resolve()
                for row in rows if row["split"] == "val"}
    for photo_list in (config.INATURALIST_DIR / "test_images.txt", config.QC_EXCLUDE_FILE):
        if photo_list.exists():
            for line in photo_list.read_text(encoding="utf-8-sig").splitlines():
                if line.strip():
                    excluded.add((config.PROJECT_ROOT / line.strip()).resolve())
    return excluded


def read_box(row):
    """The (xmin, ymin, xmax, ymax) of a CSV row as 0-1 numbers, or None if invalid."""
    try:
        box = tuple(float(row[key]) for key in LABEL_COLUMNS[2:6])
    except (ValueError, TypeError, KeyError):
        return None
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):  # also False for NaN
        return None
    return box


def make_cutout(image, box, segmenter):
    """Cut the bat in `box` (0-1 coordinates) out of `image`, or return None.

    Kept only when the mask looks like one whole bat (refine_labels.judge_mask)
    that doesn't touch the edge of the area we looked at: a bat cut off by
    the photo's edge would look sliced in half when pasted somewhere else.
    """
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box[0] * width, box[1] * height, box[2] * width, box[3] * height
    if min(x2 - x1, y2 - y1) < MIN_SOURCE_PIXELS or (box[2] - box[0]) * (box[3] - box[1]) > 0.6:
        return None
    (left, top, right, bottom), open_sides = context_window((x1, y1, x2, y2), width, height)
    crop = image[top:bottom, left:right]
    mask = segmenter(crop)
    verdict = judge_mask(mask, (x1 - left, y1 - top, x2 - left, y2 - top), open_sides)
    if not verdict.keep:
        return None
    blob = verdict.blob
    if blob[0].any() or blob[-1].any() or blob[:, 0].any() or blob[:, -1].any():
        return None
    # The bat's own pixels: the network's mask, limited to the main blob
    # (judge_mask's blob also has small gaps filled in, which may be background).
    alpha = (mask & blob).astype(np.uint8)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    ys, xs = np.nonzero(alpha)
    if len(xs) == 0:
        return None
    bx1, bx2, by1, by2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    if min(bx2 - bx1, by2 - by1) < MIN_SOURCE_PIXELS:
        return None
    # One pixel of room around the bat, so the softened edge has space.
    bx1, by1 = max(0, bx1 - 1), max(0, by1 - 1)
    bx2, by2 = min(crop.shape[1], bx2 + 1), min(crop.shape[0], by2 + 1)
    pixels = crop[by1:by2, bx1:bx2].copy()
    alpha = alpha[by1:by2, bx1:bx2].astype(np.float32)
    scale = CUTOUT_SIDE / max(pixels.shape[:2])
    if scale < 1:
        size = (max(1, round(pixels.shape[1] * scale)), max(1, round(pixels.shape[0] * scale)))
        pixels = cv2.resize(pixels, size, interpolation=cv2.INTER_AREA)
        alpha = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    return pixels, alpha


def read_csv(path):
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as file:
        # Rows with missing columns (e.g. a file that is still being written)
        # come back with None values and are skipped later.
        return list(csv.DictReader(file))


def load_sources(segmenter, seed=0):
    """Return (bat cutouts, background paths, skipped bat box count)."""
    rows = read_csv(config.OPENIMAGES_DIR / "annotations.csv")
    rows = [row for row in rows if row.get("split") and row.get("image") and row.get("is_bat")]
    excluded = excluded_paths(rows)
    # A photo with any bat annotation must never become a negative background.
    positive_paths = {(config.OPENIMAGES_DIR / row["image"]).resolve()
                      for row in rows if row["is_bat"] == "1"}

    # Where the bats come from: Open Images training boxes, then auto labels.
    bat_boxes, background_paths = [], set()
    for row in rows:
        path = (config.OPENIMAGES_DIR / row["image"]).resolve()
        if row["split"] != "train" or path in excluded:
            continue
        if row["is_bat"] == "1":
            bat_boxes.append((path, read_box(row)))
        elif row["is_bat"] == "0" and path not in positive_paths:
            background_paths.add(path)
    auto_rows = [row for row in read_csv(config.AUTO_LABELS_FILE) if row.get("image")]
    random.Random(seed).shuffle(auto_rows)
    auto_boxes = [((config.PROJECT_ROOT / row["image"]).resolve(), read_box(row))
                  for row in auto_rows]
    auto_boxes = [(path, box) for path, box in auto_boxes if path not in excluded]

    @lru_cache(maxsize=8)
    def read_source(path):
        return read_image(path)

    cutouts, skipped = [], 0
    auto_used = 0
    for index, (path, box) in enumerate(bat_boxes + auto_boxes):
        from_auto = index >= len(bat_boxes)
        if from_auto and auto_used >= MAX_AUTO_CUTOUTS:
            break
        image = read_source(path) if box is not None else None
        cut = make_cutout(image, box, segmenter) if image is not None else None
        if cut is None:
            skipped += 1
            continue
        cutouts.append(Cutout(path, *cut))
        auto_used += from_auto

    background_paths.update(path.resolve() for path in config.ROOM_NEGATIVES_DIR.glob("*.jpg"))
    # Only check that the files exist here (fast); unreadable ones are
    # skipped later, when they are picked.
    backgrounds = [path for path in sorted(background_paths - excluded - positive_paths)
                   if path.is_file()]
    return cutouts, backgrounds, skipped


def box_iou(a, b):
    """Fraction of the combined box area shared by two pasted bats."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    intersection *= max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1)


def motion_blur(image, rng, length):
    """Smear the pixels along a random direction, like a fast-moving bat."""
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1
    rotation = cv2.getRotationMatrix2D(((length - 1) / 2, (length - 1) / 2), rng.uniform(0, 180), 1)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel /= max(float(kernel.sum()), 1e-6)
    return cv2.filter2D(image, -1, kernel, borderType=cv2.BORDER_CONSTANT)


def transform_cutout(cutout, longest, rng):
    """Resize, flip and slightly rotate a cutout; returns (pixels, alpha 0-1)."""
    pixels, alpha = cutout.pixels, cutout.alpha
    scale = longest / max(pixels.shape[:2])
    size = (max(1, round(pixels.shape[1] * scale)), max(1, round(pixels.shape[0] * scale)))
    method = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    pixels, alpha = cv2.resize(pixels, size, interpolation=method), cv2.resize(alpha, size, interpolation=method)
    if rng.random() < 0.5:
        pixels, alpha = cv2.flip(pixels, 1), cv2.flip(alpha, 1)
    if rng.random() < 0.5:
        # Rotate a little, on a canvas big enough that no part is cut off.
        angle = rng.uniform(-20, 20)
        height, width = alpha.shape
        side = int(math.ceil(math.hypot(height, width))) + 2
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        matrix[:, 2] += ((side - width) / 2, (side - height) / 2)
        pixels = cv2.warpAffine(pixels, matrix, (side, side), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        alpha = cv2.warpAffine(alpha, matrix, (side, side), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # Soften the mask edge by about one pixel so the paste blends in.
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.8)
    if longest >= 40 and rng.random() < 0.2:
        # Only a short smear, so small bats don't turn into a smudge.
        length = rng.choice([3, 5] if longest < 100 else [3, 5, 7])
        pixels, alpha = motion_blur(pixels, rng, length), motion_blur(alpha, rng, length)
    # Trim to the bat itself (plus its soft edge).
    ys, xs = np.nonzero(alpha > 0.02)
    if len(xs) == 0:
        return None
    y1, y2, x1, x2 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    return pixels[y1:y2, x1:x2], alpha[y1:y2, x1:x2]


def paste_bat(canvas, cutout, boxes, rng):
    """Paste one bat if it fits; return its pixel box (x1, y1, x2, y2), or None.

    The box is the bounding box of the pasted bat's pixels (alpha >= 0.5).
    """
    height, width = canvas.shape[:2]
    # Bat size: log-uniform, from small (far away) to large (close up).
    smallest, largest = 20, 0.45 * min(height, width)
    if largest <= smallest:
        return None
    # Don't blow small cutouts up much: enlarged pixels look blurry and fake.
    largest = min(largest, MAX_ENLARGE * max(cutout.alpha.shape))
    if largest <= smallest:
        return None
    longest = round(math.exp(rng.uniform(math.log(smallest), math.log(largest))))
    result = transform_cutout(cutout, longest, rng)
    if result is None:
        return None
    pixels, alpha = result
    bat_height, bat_width = alpha.shape
    if bat_width > width or bat_height > height:
        return None
    ys, xs = np.nonzero(alpha >= 0.5)
    if len(xs) == 0:
        return None
    inner = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    if min(inner[2] - inner[0], inner[3] - inner[1]) < MIN_BAT_PIXELS:
        return None
    weights = alpha[..., None]
    bat_gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bat_mean = float((bat_gray * alpha).sum() / max(float(alpha.sum()), 1e-6))
    solid = alpha >= 0.5
    for _ in range(50):
        x = rng.randint(0, width - bat_width)
        y = rng.randint(0, height - bat_height)
        box = (x + inner[0], y + inner[1], x + inner[2], y + inner[3])
        if any(box_iou(box, other) >= MAX_OVERLAP for other in boxes):
            continue
        area = canvas[y:y + bat_height, x:x + bat_width]
        # Match the bat's brightness gently to the spot it lands on, keeping
        # dark bats recognisably dark.
        local_mean = float(cv2.cvtColor(area, cv2.COLOR_BGR2GRAY).mean())
        gain = np.clip((local_mean + 30) / (bat_mean + 30), 0.7, 1.4)
        bat = np.clip(pixels.astype(np.float32) * gain, 0, 255)
        blended = np.clip(bat * weights + area * (1 - weights), 0, 255).astype(np.uint8)
        # A bat that looks just like what it covers would be an invisible
        # "bat" label, which teaches the detector to see bats everywhere.
        change = np.abs(cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY).astype(np.float32)
                        - cv2.cvtColor(area, cv2.COLOR_BGR2GRAY))
        if float(change[solid].mean()) < MIN_CONTRAST:
            continue
        area[:] = blended
        return box
    return None


def compose_image(background, cutouts, rng):
    """Paste 1-3 bats onto a copy of the background; return (photo, 0-1 boxes).

    The list of boxes is empty if no bat could be pasted (e.g. a very dark
    background where every dark bat would be invisible).
    """
    canvas = background.copy()
    boxes = []
    wanted = rng.choice([1, 1, 1, 2, 2, 3])
    # Retries help with very tall cutouts and already crowded backgrounds.
    for _ in range(40):
        box = paste_bat(canvas, rng.choice(cutouts), boxes, rng)
        if box is not None:
            boxes.append(box)
        if len(boxes) == wanted:
            break
    if not boxes:
        return canvas, []
    if rng.random() < 0.15:
        # Some webcams (and night mode) give grey pictures.
        canvas = cv2.cvtColor(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    height, width = canvas.shape[:2]
    return canvas, [(x1 / width, y1 / height, x2 / width, y2 / height)
                    for x1, y1, x2, y2 in boxes]


def write_jpeg(path, image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise OSError(f"Could not encode {path}")
    encoded.tofile(path)


def save_preview(samples):
    """Draw a 4-by-3 contact sheet, using the same boxes written to the CSV."""
    grid = np.full((3 * 240, 4 * 320, 3), 30, np.uint8)
    for index, (image, boxes) in enumerate(samples):
        marked = image.copy()
        height, width = marked.shape[:2]
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(marked, (round(x1 * width), round(y1 * height)),
                          (min(width - 1, round(x2 * width) - 1),
                           min(height - 1, round(y2 * height) - 1)), (0, 255, 0), 1)
        scale = min(320 / width, 216 / height)
        tile = cv2.resize(marked, (max(1, round(width * scale)), max(1, round(height * scale))),
                          interpolation=cv2.INTER_AREA)
        top, left = (index // 4) * 240 + 24, (index % 4) * 320
        cv2.putText(grid, f"synth_{index:05d} ({len(boxes)} bats)", (left + 5, top - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (240, 240, 240), 1)
        grid[top:top + tile.shape[0], left:left + tile.shape[1]] = tile
    write_jpeg(config.SYNTHETIC_DIR / "preview.jpg", grid)


def synthesize(count=1500, seed=0, segmenter=None):
    """Generate photos and one label per pasted bat. Return summary counts.

    segmenter: function crop (BGR) -> True where the bat is. Defaults to
    BatNet's segmentation network (tests pass a simple fake one).
    """
    if count < 1:
        raise ValueError("--count must be at least 1")
    output = config.SYNTHETIC_DIR.resolve()
    labels = config.SYNTHETIC_LABELS_FILE.resolve()
    if not output.is_relative_to(config.PROJECT_ROOT.resolve()) or output == config.PROJECT_ROOT.resolve():
        raise ValueError("Synthetic output must be inside the project, in its own directory.")
    if not labels.is_relative_to(output):
        raise ValueError("Synthetic labels must be inside SYNTHETIC_DIR.")
    if segmenter is None:
        from training.refine_labels import Segmenter
        segmenter = Segmenter()
    cutouts, backgrounds, skipped = load_sources(segmenter, seed)
    if not cutouts or not backgrounds:
        raise ValueError("Need usable training bat cutouts and negative backgrounds; no output was changed.")
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    samples = []
    box_count = 0

    @lru_cache(maxsize=16)
    def background_pixels(path):
        image = read_image(path)
        if image is None or min(image.shape[:2]) < 32:
            return None
        return resize_background(image)

    def pick_background():
        for _ in range(100):
            pixels = background_pixels(rng.choice(backgrounds))
            if pixels is not None:
                return pixels
        raise ValueError("Could not read the background photos.")

    with labels.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(LABEL_COLUMNS)
        for index in range(count):
            for _ in range(20):
                canvas, boxes = compose_image(pick_background(), cutouts, rng)
                if boxes:
                    break
            else:
                raise ValueError("Could not paste a visible bat onto any background.")
            path = images_dir / f"synth_{index:05d}.jpg"
            write_jpeg(path, canvas)
            relative = path.relative_to(config.PROJECT_ROOT.resolve()).as_posix()
            for box in boxes:
                writer.writerow(["synthetic", relative, *[f"{v:.5f}" for v in box], "1.0"])
            box_count += len(boxes)
            if index < 12:
                samples.append((canvas, boxes))
            if (index + 1) % 500 == 0:
                print(f"Created {index + 1}/{count} images", flush=True)
    save_preview(samples)
    # Only retire old generated photos; leave them in the synthetic directory.
    # This avoids deleting anything, while keeping the active images folder clean.
    current_names = {f"synth_{index:05d}.jpg" for index in range(count)}
    for path in sorted(images_dir.glob("synth_*.jpg")):
        if path.name not in current_names and path.stem[6:].isdigit():
            archive = output / "previous_images"
            archive.mkdir(exist_ok=True)
            target = archive / path.name
            suffix = 1
            while target.exists():
                target = archive / f"{path.stem}_{suffix}.jpg"
                suffix += 1
            path.rename(target)
    result = dict(images=count, boxes=box_count, cutouts=len(cutouts),
                  backgrounds=len(backgrounds), skipped_boxes=skipped)
    print(f"Saved {count} images / {box_count} bats from {len(cutouts)} cutouts and "
          f"{len(backgrounds)} backgrounds ({skipped} bat boxes gave no clean cutout).")
    print(f"Labels: {labels}\nPreview: {output / 'preview.jpg'}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--count", type=int, default=1500, help="number of photos (default: 1500)")
    parser.add_argument("--seed", type=int, default=0, help="repeatable random seed (default: 0)")
    args = parser.parse_args()
    try:
        synthesize(args.count, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Could not synthesize data: {error}\n")


if __name__ == "__main__":
    main()
