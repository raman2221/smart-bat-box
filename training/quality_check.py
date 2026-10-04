"""Check the data before training, without loading a detector or using a GPU.

Run from the project folder:
    .venv\\Scripts\\python.exe training/quality_check.py --write-exclude

Counts describe the input lists, before exclusions. Problems count bad rows or
images, not every possible duplicate pair. Possible near duplicates (64-bit
average hash, at most four different bits) are only reported as warnings, not
excluded: on this data most matches were different photos with similar
brightness layouts, e.g. two dark close-ups. Very thin boxes (aspect ratio over 50:1)
are considered absurd; large boxes are allowed because bats can fill a photo.
Optional sources may be absent. An incomplete CSV is reported, never ignored.
"""

import argparse
import csv
import hashlib
import io
import math
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402

BOX_COLUMNS = ("xmin", "ymin", "xmax", "ymax")
OPEN_COLUMNS = ("split", "image", "is_bat", *BOX_COLUMNS)
LABEL_COLUMNS = ("source", "image", *BOX_COLUMNS, "score")
SOURCE_NAMES = (
    "Open Images train", "Open Images val", "Auto labels", "Synthetic labels",
    "Room train", "Room val", "iNaturalist test",
)


@dataclass
class SourceCounts:
    bats: set = field(default_factory=set)
    negatives: set = field(default_factory=set)
    boxes: int = 0
    status: str = "present"


class Report:
    """Keep only five examples per problem, even for a large dataset."""

    def __init__(self):
        self.root = Path(config.PROJECT_ROOT).resolve()
        self.sources = {name: SourceCounts() for name in SOURCE_NAMES}
        self.splits = defaultdict(set)
        self.problems = defaultdict(int)
        self.examples = defaultdict(list)
        self.bad_images = set()
        self.warnings = defaultdict(int)
        self.warning_examples = defaultdict(list)

    def relative(self, path):
        return path.relative_to(self.root).as_posix()

    def problem(self, kind, example, bad_path=None):
        self.problems[kind] += 1
        if len(self.examples[kind]) < 5:
            self.examples[kind].append(str(example))
        if bad_path is not None:
            self.bad_images.add(bad_path)

    def warning(self, kind, example):
        """Something worth a look by eye, but not excluded from training."""
        self.warnings[kind] += 1
        if len(self.warning_examples[kind]) < 5:
            self.warning_examples[kind].append(str(example))

    def add_photo(self, path, source, split, is_bat):
        self.splits[path].add(split)
        counts = self.sources[source]
        (counts.bats if is_bat else counts.negatives).add(path)
        if path in counts.bats and path in counts.negatives:
            # A photo cannot be labelled both bat and not-a-bat.
            self.problem("conflicting_labels", self.relative(path), path)

    @property
    def exclusions(self):
        # Even if a path is also held back, this list only affects training.
        return {path for path in self.bad_images if "train" in self.splits[path]}

    @property
    def exit_code(self):
        return int(bool(self.problems))


def image_path(report, text, base, context):
    """Keep dataset paths inside the project, and normalise aliases."""
    if not text or not text.strip():
        report.problem("invalid_image_path", context + ": empty image path")
        return None
    raw = Path(text.strip().replace("\\", "/"))
    path = (base / raw).resolve()
    if raw.is_absolute() or not path.is_relative_to(report.root):
        report.problem("invalid_image_path", context + ": " + text)
        return None
    return path


def read_csv(report, path, columns, sources, required=False):
    """Take a snapshot so a writer cannot change rows midway through reading."""
    try:
        before = path.stat()
        content = path.read_text(encoding="utf-8-sig")
        after = path.stat()
    except FileNotFoundError:
        for source in sources:
            report.sources[source].status = "missing"
        if required:
            report.problem("missing_source", str(path))
        return
    except (OSError, UnicodeError) as error:
        report.problem("unreadable_source", f"{path}: {error}")
        return
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        report.problem("source_changed", f"{path}: changed during reading; rerun QC")
    reader = csv.DictReader(io.StringIO(content, newline=""), strict=True)
    try:
        fields = reader.fieldnames or []
        if not set(columns).issubset(fields) or len(fields) != len(set(fields)):
            report.problem("malformed_csv", f"{path}: missing or repeated column names")
            return
        rows = 0
        for row in reader:
            rows += 1
            yield row, f"{path}:{reader.line_num}"
        if rows == 0:
            for source in sources:
                report.sources[source].status = "empty"
            if required:
                report.problem("empty_source", str(path))
    except csv.Error as error:
        report.problem("malformed_csv", f"{path}:{reader.line_num}: {error}")


def complete_row(report, row, context, path):
    if None in row or any(value is None for value in row.values()):
        report.problem("malformed_csv", context + ": incomplete or extra fields", path)
        return False
    return True


def check_box(report, row, path, context):
    """Validate fractions, then check their size at the detector's 320 pixels."""
    example = f"{report.relative(path)} ({context})"
    values = [row.get(name) for name in BOX_COLUMNS]
    if any(value is None or not value.strip() for value in values):
        report.problem("missing_box", example, path)
        return False
    try:
        x1, y1, x2, y2 = [float(value) for value in values]
    except ValueError:
        report.problem("non_numeric_box", example, path)
        return False
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x1, y1, x2, y2)):
        report.problem("out_of_range_box", example, path)
        return False
    if x1 >= x2 or y1 >= y2:
        report.problem("reversed_or_empty_box", example, path)
        return False
    width, height = x2 - x1, y2 - y1
    if min(width, height) * 320 < 2:
        report.problem("tiny_box", example, path)
        return False
    if max(width, height) / min(width, height) > 50:
        report.problem("absurd_box", example, path)
        return False
    return True


def read_open_images(report):
    sources = SOURCE_NAMES[:2]
    path = Path(config.OPENIMAGES_DIR) / "annotations.csv"
    for row, context in read_csv(report, path, OPEN_COLUMNS, sources, required=True):
        photo = image_path(report, row.get("image"), path.parent, context)
        split = row.get("split")
        if split not in ("train", "val"):
            report.problem("invalid_split", context)
            continue
        if photo is None:
            continue
        source = "Open Images " + split
        # Register the split even for a malformed label, so it can be excluded.
        report.splits[photo].add(split)
        if not complete_row(report, row, context, photo):
            continue
        if row["is_bat"] not in ("0", "1"):
            report.problem("invalid_is_bat", report.relative(photo), photo)
            continue
        is_bat = row["is_bat"] == "1"
        report.add_photo(photo, source, split, is_bat)
        if is_bat:
            report.sources[source].boxes += int(check_box(report, row, photo, context))
        elif any(row[name].strip() for name in BOX_COLUMNS):
            report.problem("negative_has_box", report.relative(photo), photo)


def read_labels(report, path, source):
    for row, context in read_csv(report, Path(path), LABEL_COLUMNS, (source,)):
        photo = image_path(report, row.get("image"), report.root, context)
        if photo is None:
            continue
        report.add_photo(photo, source, "train", True)
        if complete_row(report, row, context, photo):
            report.sources[source].boxes += int(check_box(report, row, photo, context))


def read_room_and_test(report):
    frames = sorted(Path(config.ROOM_NEGATIVES_DIR).glob("*.jpg"))
    for i, path in enumerate(frames):
        # Match the training loader: every fifth room frame is held back.
        photo = image_path(report, path.name, path.parent, "Room frames")
        if photo is not None:
            split = "val" if i % 5 == 0 else "train"
            report.add_photo(photo, "Room " + split, split, False)
    if not frames:
        for source in ("Room train", "Room val"):
            report.sources[source].status = "missing or empty"
    test_file = Path(config.INATURALIST_DIR) / "test_images.txt"
    try:
        lines = test_file.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        report.sources["iNaturalist test"].status = "missing"
        return
    except (OSError, UnicodeError) as error:
        report.problem("unreadable_source", f"{test_file}: {error}")
        return
    for line in lines:
        if line.strip():
            photo = image_path(report, line, report.root, str(test_file))
            if photo is not None:
                report.add_photo(photo, "iNaturalist test", "test", True)


def inspect_image(path):
    """Verify the file, fully decode it, and compute its two hashes once."""
    try:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()  # verify alone does not catch every truncated JPEG.
            small = ImageOps.exif_transpose(image).convert("L").resize(
                (8, 8), Image.Resampling.LANCZOS)
            pixels = list(small.get_flattened_data())
        average = sum(pixels) / 64
        phash = sum(1 << i for i, value in enumerate(pixels) if value > average)
        return path, digest, phash, None
    except FileNotFoundError:
        return path, None, None, "missing_image"
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        return path, None, None, "broken_image"


def hash_buckets(value):
    # If at most four bits differ, at least one of these five pieces matches.
    # This finds all close hashes without comparing every train/check pair.
    pieces = ((0, 13), (13, 13), (26, 13), (39, 13), (52, 12))
    for part, (shift, bits) in enumerate(pieces):
        yield part, (value >> shift) & ((1 << bits) - 1)


def check_leakage(report, images):
    checking = {path for path, splits in report.splits.items() if splits & {"val", "test"}}
    training = {path for path, splits in report.splits.items() if "train" in splits}
    for path in sorted(training & checking):
        kind = "test_photo_leak" if "test" in report.splits[path] else "val_photo_leak"
        report.problem(kind, report.relative(path), path)

    by_digest = {}
    by_hash = {}
    buckets = defaultdict(set)
    for path in sorted(checking):
        if path not in images:
            continue
        digest, phash = images[path]
        by_digest.setdefault(digest, path)
        by_hash.setdefault(phash, path)
        for key in hash_buckets(phash):
            buckets[key].add(phash)
    for path in sorted(training):
        if path not in images:
            continue
        digest, phash = images[path]
        if digest in by_digest:
            other = by_digest[digest]
            report.problem("exact_duplicate_leak",
                           f"{report.relative(path)} -> {report.relative(other)}", path)
            continue  # Count near duplicates only when the file bytes differ.
        candidates = set()
        for key in hash_buckets(phash):
            candidates.update(buckets.get(key, ()))
        for candidate in candidates:
            distance = (phash ^ candidate).bit_count()
            if distance <= 4:
                other = by_hash[candidate]
                report.warning("possible_near_duplicate",
                               f"{report.relative(path)} -> {report.relative(other)}"
                               f" (hash distance {distance})")
                break  # One held-back match is enough to report this photo.


def check_data(progress=False):
    report = Report()
    read_open_images(report)
    read_labels(report, config.AUTO_LABELS_FILE, "Auto labels")
    read_labels(report, config.SYNTHETIC_LABELS_FILE, "Synthetic labels")
    read_room_and_test(report)
    paths = sorted(report.splits)
    if progress:
        print(f"Checking {len(paths)} unique images on CPU...", flush=True)
    images = {}
    # A few readers overlap disk work; no models, tensors, or GPU are needed.
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (path, digest, phash, problem) in enumerate(pool.map(inspect_image, paths), 1):
            if problem:
                report.problem(problem, report.relative(path), path)
            else:
                images[path] = digest, phash
            if progress and i % 500 == 0:
                print(f"  Decoded {i}/{len(paths)}", flush=True)
    check_leakage(report, images)
    return report


def print_report(report):
    print("\nSource counts (unique photos, valid boxes, before exclusions):")
    for name, counts in report.sources.items():
        print(f"  {name}: {len(counts.bats)} bat photos, {counts.boxes} boxes, "
              f"{len(counts.negatives)} not-a-bat photos [{counts.status}]")
    for split in ("train", "val", "test"):
        photos = {p for p, splits in report.splits.items() if split in splits}
        bats, negatives = set(), set()
        for counts in report.sources.values():
            bats.update(counts.bats & photos)
            negatives.update(counts.negatives & photos)
        fraction = 100 * len(bats) / len(photos) if photos else 0
        print(f"  {split} balance: {len(photos)} unique photos; {len(bats)} bat, "
              f"{len(negatives)} not-a-bat ({fraction:.1f}% bat)")
    print("\nProblems (counts can overlap; leakage counts training images):")
    if not report.problems:
        print("  None. Data checks passed.")
    for kind, count in sorted(report.problems.items()):
        print(f"  {kind}: {count}")
        for example in report.examples[kind]:
            print(f"    {example}")
    if report.warnings:
        print("\nWarnings (check by eye; NOT excluded):")
        for kind, count in sorted(report.warnings.items()):
            print(f"  {kind}: {count}")
            for example in report.warning_examples[kind]:
                print(f"    {example}")
    print(f"Training images to skip: {len(report.exclusions)}")


def write_exclusions(report):
    path = Path(config.QC_EXCLUDE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = sorted(report.relative(photo) for photo in report.exclusions)
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write-exclude", action="store_true",
                        help="write training-side exclusions to config.QC_EXCLUDE_FILE")
    args = parser.parse_args(argv)
    report = check_data(progress=True)
    print_report(report)
    if args.write_exclude:
        print(f"Wrote exclusions to {write_exclusions(report)}")
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
