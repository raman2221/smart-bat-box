"""
finetune_detector.py

Fine-tunes BatNet's bat *detector* (the part that draws boxes) so it can
find bats in normal colour camera images, not just infrared camera-trap
photos.

Training data:
  - data/openimages/annotations.csv   bat photos with hand-drawn boxes, and
                                      "not a bat" photos (people, lights, ...)
  - data/auto_labels.csv              iNaturalist bat photos boxed by
                                      training/auto_label.py
  - data/synthetic/labels.csv         bats pasted onto non-bat backgrounds
  - data/room_negatives/*.jpg         webcam frames with no bats

Balanced epochs: there are far more bat photos than "not a bat" photos.
Showing them all every epoch taught the detector to see bats everywhere
(lots of false alarms). Instead, every epoch is a fixed number of photos
(--epoch-size, default 4000), half bat photos and half "not a bat" photos
(--bat-fraction, default 0.5). Each half is dealt from its own shuffled
pile, like cards, so every photo is shown before any photo repeats, and
over a few epochs the model sees all the data.

Speed: every photo is decoded and shrunk to 320x320 once and kept in
data/cache/ (see training/cache_images.py), so epochs don't decode JPEGs.

Checked on photos it never trains on:
  - Open Images check photos: bats found and false alarms (hand-labelled),
    after every epoch. These alone choose which epoch is kept.
  - held-back iNaturalist bat photos: how many it finds a bat in. For
    information only, before training, every --inat-every epochs and at
    the end.

By default it continues from the last fine-tuned model (faster). Use
--from-scratch to start again from BatNet's original detector. The best
epoch is saved to models/detection/finetuned_detector.pth only if it beats
the starting model (use --save-to or --no-save for test runs).

Run from the project folder (with the .venv active):
    python training/finetune_detector.py
"""

import argparse
import csv
import os
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.package
import torchvision
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training import cache_images  # noqa: E402
from training.export_model import half_precision  # noqa: E402
from vision import config  # noqa: E402

IMAGE_SIZE = cache_images.DEFAULT_SIZE   # BatNet's detector works on 320x320 images
BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16   # checking needs no gradients, so bigger batches fit

# Each epoch shows EPOCH_SIZE photos, BAT_FRACTION of them bat photos.
EPOCH_SIZE = 4000
BAT_FRACTION = 0.5

# The held-back iNaturalist test photos are checked every this many epochs
# (and before training and after the last epoch).
INAT_EVERY = 5

# Starting from BatNet's original detector needs more epochs and a higher
# learning rate than continuing from our last fine-tuned one.
FROM_SCRATCH = {"epochs": 20, "lr": 0.005, "milestones": [14, 18]}
CONTINUE = {"epochs": 10, "lr": 0.002, "milestones": [7, 9]}

# Open Images bat photos have hand-drawn boxes, so each one goes into the
# pile of bat photos this many times (automatic and synthetic labels go in
# once). They are dealt this many times more often.
HAND_LABELLED_BAT_REPEATS = 3


def read_photo_list(path):
    """Read optional path lists, keeping spaces inside photo names."""
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError):
        return set()
    return {(config.PROJECT_ROOT / line.strip()).resolve()
            for line in lines if line.strip()}


def load_extra_labels(path, label, excluded):
    """Read labels that another worker may still be writing."""
    photos = {}
    skipped = 0
    if not path.exists():
        print(f"{label}: using 0 photos, skipped 0 rows (file not present)")
        return []
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                try:
                    name = row.get("image")
                    if not name:
                        raise ValueError("missing image")
                    photo = (config.PROJECT_ROOT / name).resolve()
                    box = [float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                    valid = (all(np.isfinite(x) and 0 <= x <= 1 for x in box)
                             and box[0] < box[2] and box[1] < box[3])
                    if photo in excluded or not photo.is_file() or not valid:
                        skipped += 1
                        continue
                except (KeyError, TypeError, ValueError, OSError):
                    skipped += 1   # ignore an unfinished or malformed CSV row
                    continue
                photos.setdefault(photo, []).append(box)
    except (OSError, UnicodeError, csv.Error):
        skipped += 1   # keep complete rows if the file is temporarily incomplete
    print(f"{label}: using {len(photos)} photos, skipped {skipped} rows "
          "(held-back, QC, missing or incomplete)")
    return list(photos.items())


def inaturalist_test_photos():
    """The held-back iNaturalist test photos (never trained on), minus QC rejects."""
    test_photos = read_photo_list(config.INATURALIST_DIR / "test_images.txt")
    return test_photos - read_photo_list(config.QC_EXCLUDE_FILE)


def load_annotations():
    """Return {"train": [(path, boxes)], "val": [(path, boxes)]}."""
    test_photos = read_photo_list(config.INATURALIST_DIR / "test_images.txt")
    qc_photos = read_photo_list(config.QC_EXCLUDE_FILE)
    excluded = test_photos | qc_photos
    print(f"Exclusions: {len(qc_photos)} QC photos, {len(test_photos)} held-back test photos")
    boxes_by_image = {}
    split_of = {}
    skipped = set()
    with open(config.OPENIMAGES_DIR / "annotations.csv", newline="") as f:
        for row in csv.DictReader(f):
            try:
                path = (config.OPENIMAGES_DIR / row["image"]).resolve()
                if row["split"] not in ("train", "val"):
                    raise ValueError("unknown split")
                box = ([float(row[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
                       if row["is_bat"] == "1" else None)
            except (KeyError, TypeError, ValueError):
                continue   # a row another program is still writing
            if path in excluded:
                skipped.add(path)
                continue
            split_of[path] = row["split"]
            boxes_by_image.setdefault(path, [])   # negatives have no boxes
            if box is not None:
                boxes_by_image[path].append(box)

    data = {"train": [], "val": []}
    for path, boxes in boxes_by_image.items():
        data[split_of[path]].append((path, boxes))
    print(f"Open Images: using {len(boxes_by_image)} photos, skipped {len(skipped)} photos")

    # Hand-labelled bat photos count extra (see HAND_LABELLED_BAT_REPEATS).
    hand_bats = [item for item in data["train"] if item[1]]
    data["train"] += hand_bats * (HAND_LABELLED_BAT_REPEATS - 1)

    # Extra labels are training only. A validation photo must never sneak
    # into training through a second labels file.
    room_frames = sorted(config.ROOM_NEGATIVES_DIR.glob("*.jpg"))
    validation_photos = {path for path, _ in data["val"]}
    validation_photos.update(path.resolve() for i, path in enumerate(room_frames) if i % 5 == 0)
    extra_excluded = excluded | validation_photos
    data["train"] += load_extra_labels(config.AUTO_LABELS_FILE, "Automatic labels", extra_excluded)
    data["train"] += load_extra_labels(config.SYNTHETIC_LABELS_FILE, "Synthetic labels", extra_excluded)

    # Frames of a room with no bats (training/capture_room_negatives.py).
    # Every 5th one is kept aside to check for false alarms.
    room_skipped = 0
    for i, path in enumerate(room_frames):
        if path.resolve() in excluded:
            room_skipped += 1
            continue
        data["val" if i % 5 == 0 else "train"].append((path, []))
    print(f"Room frames: using {len(room_frames) - room_skipped} photos, "
          f"skipped {room_skipped} photos")
    return data


# ---------------------------------------------------------------------------
# Augmentation: change each training photo a little, randomly, so the
# detector learns bats under many conditions and doesn't memorise photos.
# ---------------------------------------------------------------------------

MIN_BAT_PIXELS = 4   # zoom_out never shrinks a bat smaller than this


def zoom_out(image, boxes):
    """Shrink the photo onto a bigger background, so bats look far away/small."""
    # Don't shrink so far that a bat becomes too small to keep its box.
    smallest_side = float((boxes[:, 2:] - boxes[:, :2]).min()) if len(boxes) else IMAGE_SIZE
    lowest = max(0.3, MIN_BAT_PIXELS / max(smallest_side, 1e-6))
    if lowest >= 0.8:
        return image, boxes   # bats already tiny: leave the photo as it is
    scale = random.uniform(lowest, 0.8)
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    # Background: a heavily blurred, darker copy of the photo itself.
    canvas = (cv2.blur(image, (51, 51)) * random.uniform(0.3, 0.8)).astype(np.uint8)
    h, w = small.shape[:2]
    y0 = random.randint(0, IMAGE_SIZE - h)
    x0 = random.randint(0, IMAGE_SIZE - w)
    canvas[y0:y0 + h, x0:x0 + w] = small
    boxes = boxes * scale + torch.tensor([x0, y0, x0, y0], dtype=torch.float32)
    return canvas, boxes


def motion_blur(image):
    """Smear the photo in a random direction, like a fast-moving bat."""
    length = random.randint(5, 15)
    kernel = np.zeros((length, length), np.float32)
    kernel[length // 2, :] = 1.0 / length
    angle = random.uniform(0, 180)
    rotation = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle, 1.0)
    kernel = cv2.warpAffine(kernel, rotation, (length, length))
    kernel /= max(kernel.sum(), 1e-6)
    return cv2.filter2D(image, -1, kernel)


def boxes_to_pixels(boxes):
    """Boxes are stored as fractions (0-1) of the photo's size -> pixels."""
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4) * IMAGE_SIZE


def drop_thin_boxes(boxes):
    """Drop boxes too thin to learn from (the detector needs width/height > 0)."""
    keep = (boxes[:, 2] - boxes[:, 0] >= 2) & (boxes[:, 3] - boxes[:, 1] >= 2)
    return boxes[keep]


def to_image_tensor(pixels):
    """height x width x 3 pixels -> 3 x height x width uint8 tensor.

    Pixels stay as whole numbers (0-255) until they reach the GPU, which is
    4x less data to copy than decimals. See as_float_images().
    """
    return torch.from_numpy(np.ascontiguousarray(pixels)).permute(2, 0, 1).contiguous()


def as_float_images(images, device):
    """Move a batch to the device and scale 0-255 whole numbers to 0-1."""
    images = images.to(device, non_blocking=True)
    if images.dtype == torch.uint8:
        # Gives exactly the same numbers as torchvision's to_tensor().
        images = images.float().div_(255)
    return images


class BatPhotos(torch.utils.data.Dataset):
    """Gives (image tensor, {"boxes", "labels"}) pairs to the training loop.

    Each item is (photo, boxes). The photo can be a cache file made by
    training/cache_images.py (fast) or a normal photo file (decoded here).
    """

    def __init__(self, items, augment):
        self.items = items
        self.augment = augment
        self.color_jitter = torchvision.transforms.ColorJitter(0.3, 0.3, 0.2, 0.02)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, boxes = self.items[i]
        pixels = cache_images.read_pixels(path, IMAGE_SIZE)
        boxes = boxes_to_pixels(boxes)

        if self.augment:
            image = self.color_jitter(Image.fromarray(pixels))
            if random.random() < 0.5:   # mirror left-right
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                boxes[:, [0, 2]] = IMAGE_SIZE - boxes[:, [2, 0]]
            if random.random() < 0.3:   # sometimes black-and-white, like infrared
                image = image.convert("L").convert("RGB")

            pixels = np.array(image)
            if random.random() < 0.3:
                pixels, boxes = zoom_out(pixels, boxes)
            if random.random() < 0.2:
                pixels = motion_blur(pixels)
            elif random.random() < 0.15:   # out of focus
                pixels = cv2.GaussianBlur(pixels, (0, 0), random.uniform(0.8, 2.0))

        boxes = drop_thin_boxes(boxes)
        # The detector only needs to know "bat", so every label is 1.
        target = {"boxes": boxes, "labels": torch.ones(len(boxes), dtype=torch.int64)}
        return to_image_tensor(pixels), target


class BalancedSampler(torch.utils.data.Sampler):
    """Chooses which photos make up each epoch.

    Every epoch has `epoch_size` photos: `bat_fraction` of them from the
    pile of bat photos, the rest from the pile of "not a bat" photos. Each
    pile is shuffled and dealt like a deck of cards: when a pile runs out it
    is reshuffled, so every photo is shown before any photo is shown again,
    and the deal carries on where it stopped in the next epoch.

    A photo that appears several times in `items` (the hand-labelled bat
    photos) is simply several cards in the pile, so it is dealt more often.
    """

    def __init__(self, items, epoch_size=EPOCH_SIZE, bat_fraction=BAT_FRACTION, seed=None):
        if epoch_size < 1 or not 0 <= bat_fraction <= 1:
            raise ValueError("epoch_size must be positive and bat_fraction between 0 and 1")
        self.piles = {"bat": [i for i, (_, boxes) in enumerate(items) if len(boxes)],
                      "other": [i for i, (_, boxes) in enumerate(items) if not len(boxes)]}
        self.epoch_size = epoch_size if items else 0
        bats = round(self.epoch_size * bat_fraction)
        # If one pile is empty, take everything from the other one.
        if not self.piles["other"]:
            bats = self.epoch_size
        elif not self.piles["bat"]:
            bats = 0
        self.counts = {"bat": bats, "other": self.epoch_size - bats}
        self.random = random.Random(seed)
        self.decks = {"bat": [], "other": []}

    def __len__(self):
        return self.epoch_size

    def _deal(self, name, count):
        deck = self.decks[name]
        dealt = []
        while len(dealt) < count:
            if not deck:   # pile used up: reshuffle all its photos
                deck.extend(self.random.sample(self.piles[name], len(self.piles[name])))
            dealt.append(deck.pop())
        return dealt

    def __iter__(self):
        chosen = self._deal("bat", self.counts["bat"]) + self._deal("other", self.counts["other"])
        self.random.shuffle(chosen)   # mix bats and non-bats within the epoch
        return iter(chosen)


def collate(batch):
    """Stack images into one tensor; keep targets as a list (box counts differ)."""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def prepare_worker(worker_id):
    """Each loader worker needs only one CPU thread for these small photos."""
    # Otherwise every worker can start 24 threads and compete with the GPU.
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def make_loader(items, augment, batch_size, workers, prefetch_factor, shuffle=False,
                sampler=None):
    """Keep workers alive between epochs and prepare batches ahead of time."""
    options = {"num_workers": workers, "pin_memory": True}
    if workers > 0:
        # These options need worker processes; --workers 0 is useful for debugging.
        options.update(persistent_workers=True, prefetch_factor=prefetch_factor)
    return torch.utils.data.DataLoader(
        BatPhotos(items, augment), batch_size=batch_size, shuffle=shuffle and sampler is None,
        sampler=sampler, collate_fn=collate, worker_init_fn=prepare_worker, **options)


class QuickBoxSampler:
    """Faster drop-in for torchvision's BalancedPositiveNegativeSampler.

    During training the detector picks, for every photo, a random handful
    of candidate boxes to learn from (some on a bat, most on background).
    torchvision does this by shuffling *all* ~25,000 candidates with
    torch.randperm, which is very slow on the GPU: measured at 29% of all
    GPU time per training step. Here each candidate gets a random number
    and the `count` highest are kept. Every handful is still equally
    likely, so training learns the same thing, about 30% faster overall.
    """

    def __init__(self, batch_size_per_image, positive_fraction):
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    @staticmethod
    def pick(indices, count):
        """A random `count` of these indices (all of them if there are fewer)."""
        if count >= indices.numel():
            return indices
        scores = torch.rand(indices.numel(), device=indices.device)
        return indices[scores.topk(count).indices]

    def __call__(self, matched_idxs):
        # Same inputs and outputs as torchvision's version: for each photo,
        # labels (-1 ignore, 0 background, 1+ bat) in; two 0/1 masks out.
        positive_masks, negative_masks = [], []
        for matched in matched_idxs:
            positive = torch.where(matched >= 1)[0]
            negative = torch.where(matched == 0)[0]
            num_pos = min(positive.numel(), int(self.batch_size_per_image * self.positive_fraction))
            num_neg = min(negative.numel(), self.batch_size_per_image - num_pos)
            positive_mask = torch.zeros_like(matched, dtype=torch.uint8)
            negative_mask = torch.zeros_like(matched, dtype=torch.uint8)
            positive_mask[self.pick(positive, num_pos)] = 1
            negative_mask[self.pick(negative, num_neg)] = 1
            positive_masks.append(positive_mask)
            negative_masks.append(negative_mask)
        return positive_masks, negative_masks


def use_quick_box_sampler(detector):
    """Swap in QuickBoxSampler (same settings) wherever the detector samples boxes."""
    for part in (detector.basemodel.rpn, detector.basemodel.roi_heads):
        old = part.fg_bg_sampler
        part.fg_bg_sampler = QuickBoxSampler(old.batch_size_per_image, old.positive_fraction)


def mixed_precision(device):
    """Use fast half precision on CUDA; keep normal precision on CPU."""
    if device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


def count_matches(true, pred, min_iou=0.5):
    """How many real bats are matched by a detection, one-to-one.

    Pairs are taken best-overlap first, and each detection can match only
    one bat (so one big box over two bats counts as one found, one missed).
    """
    if len(true) == 0 or len(pred) == 0:
        return 0
    ious = torchvision.ops.box_iou(torch.as_tensor(true), torch.as_tensor(pred))
    matched = 0
    while True:
        best = ious.max()
        if best < min_iou:
            return matched
        t, p = divmod(int(ious.argmax()), ious.shape[1])
        matched += 1
        ious[t, :] = -1   # this bat and this detection are now used up
        ious[:, p] = -1


class CheckPhotos:
    """Photos for checking the model, kept in memory as ready 320x320 pixels.

    They are the same every time, so there's no point decoding them again
    (or starting loader processes) after every epoch. Looping over it gives
    (images, targets) batches, just like a DataLoader.
    """

    def __init__(self, items, batch_size=EVAL_BATCH_SIZE):
        self.images = [to_image_tensor(cache_images.read_pixels(path, IMAGE_SIZE))
                       for path, _ in items]
        self.targets = [{"boxes": drop_thin_boxes(boxes_to_pixels(boxes))} for _, boxes in items]
        self.batch_size = batch_size

    def __len__(self):
        return len(self.images)

    def __iter__(self):
        for start in range(0, len(self.images), self.batch_size):
            end = start + self.batch_size
            yield torch.stack(self.images[start:end]), self.targets[start:end]


@torch.no_grad()
def detections(detector, batches, device):
    """Yield (detected boxes, target) for every photo, above the confidence threshold."""
    detector.eval()
    for images, targets in batches:
        with mixed_precision(device):
            outputs = detector(as_float_images(images, device))
        for out, target in zip(outputs, targets):
            keep = out["scores"] >= config.DETECTION_CONFIDENCE_THRESHOLD
            yield out["boxes"][keep].float().cpu(), target


def evaluate(detector, batches, device):
    """Count bats found, bats missed and false alarms on the validation photos."""
    found = missed = false_alarms = 0
    for pred, target in detections(detector, batches, device):
        true = target["boxes"]
        matched = count_matches(true, pred)
        found += matched
        missed += len(true) - matched
        false_alarms += max(len(pred) - matched, 0)
    return found, missed, false_alarms


def photos_with_bat_found(detector, batches, device):
    """How many of these photos (all known to contain a bat) get a detection."""
    return sum(int(len(pred) > 0) for pred, _ in detections(detector, batches, device))


def check_inaturalist_after(epoch, epochs, every):
    """Are the iNaturalist test photos checked after this epoch?

    Epoch 0 means "before training". They are also checked after the last
    epoch, and after every `every`-th epoch (0: never in between).
    """
    return epoch == 0 or epoch == epochs or (every > 0 and epoch % every == 0)


def training_step(detector, optimizer, images, targets, device, scaler=None):
    """Train on one batch and return its loss."""
    images = as_float_images(images, device)
    targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
    with mixed_precision(device):
        losses = detector(images, targets)
        loss = sum(losses.values())
    optimizer.zero_grad()
    if scaler is None:
        loss.backward()
        optimizer.step()
    else:
        # Scaling protects tiny half-precision gradients from rounding to zero.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    return loss.item()


def benchmark(detector, optimizer, loader, device, iterations, scaler=None):
    """Time real training batches after warm-up, without saving weights."""
    detector.train()
    batches = iter(loader)
    image_count = 0
    measured_losses = []
    for step in range(5 + iterations):
        if step == 5:
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
        try:
            images, targets = next(batches)
        except StopIteration:
            batches = iter(loader)
            images, targets = next(batches)
        loss = training_step(detector, optimizer, images, targets, device, scaler)
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at iteration {step + 1}: {loss}")
        if step >= 5:
            image_count += len(images)
            measured_losses.append(loss)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"Benchmark: {iterations} iterations after 5 warm-up iterations", flush=True)
    print(f"  {elapsed / iterations:.4f} seconds/iteration; "
          f"{image_count / elapsed:.2f} images/second", flush=True)
    print(f"  Losses finite: {min(measured_losses):.4f} to {max(measured_losses):.4f}", flush=True)


def use_cache(items, cached):
    """Swap each photo for its cache file, leaving out photos that couldn't be cached."""
    return [(cached[path], boxes) for path, boxes in items if path in cached]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-scratch", action="store_true",
                        help="start from BatNet's original detector instead of our last one")
    parser.add_argument("--epochs", type=int,
                        help="number of epochs (default: 10, or 20 with --from-scratch)")
    parser.add_argument("--epoch-size", type=int, default=EPOCH_SIZE,
                        help=f"photos shown per epoch (default: {EPOCH_SIZE})")
    parser.add_argument("--bat-fraction", type=float, default=BAT_FRACTION,
                        help=f"share of each epoch that is bat photos, 0-1 (default: {BAT_FRACTION})")
    parser.add_argument("--inat-every", type=int, default=INAT_EVERY,
                        help="check the iNaturalist test photos every N epochs (0: only "
                             f"before training and at the end; default: {INAT_EVERY})")
    parser.add_argument("--save-to", type=Path, default=config.FINETUNED_DETECTOR_FILE,
                        help="where to save the best epoch (default: the model the app uses, "
                             "models/detection/finetuned_detector.pth)")
    parser.add_argument("--no-save", action="store_true",
                        help="don't save any weights (for timing or test runs)")
    parser.add_argument("--benchmark", type=int, metavar="N",
                        help="time N training iterations after 5 warm-ups, then exit without saving")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1),
                        help="data-loading processes (0 runs loading in the main process; default: 4)")
    parser.add_argument("--prefetch-factor", type=int, default=4,
                        help="batches prepared ahead of time by each worker (default: 4)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="photos per training batch (default: 8; learning rate unchanged)")
    parser.add_argument("--eval-batch-size", type=int, default=EVAL_BATCH_SIZE,
                        help=f"photos per batch when checking (default: {EVAL_BATCH_SIZE})")
    parser.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1),
                        help="CPU threads in the training process (default: up to 4)")
    args = parser.parse_args(argv)
    if args.benchmark is not None and args.benchmark < 1:
        parser.error("--benchmark must be at least 1")
    if args.workers < 0 or min(args.prefetch_factor, args.batch_size, args.cpu_threads,
                               args.eval_batch_size, args.epoch_size) < 1:
        parser.error("--workers must be nonnegative; batch sizes, epoch size, prefetch "
                     "and CPU threads must be positive")
    if not 0 <= args.bat_fraction <= 1:
        parser.error("--bat-fraction must be between 0 and 1")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.inat_every < 0:
        parser.error("--inat-every must be 0 or more")
    return args


def main():
    args = parse_args()
    start_time = time.perf_counter()

    # Fixed 320x320 inputs let cuDNN choose its fastest convolution routines.
    torch.backends.cudnn.benchmark = True
    # Too many CPU threads compete with the loader workers on this laptop.
    torch.set_num_threads(args.cpu_threads)
    cv2.setNumThreads(1)

    data = load_annotations()
    inat_photos = sorted(inaturalist_test_photos())

    # Decode + shrink every photo once (only new or changed photos take time).
    every_photo = [path for path, _ in data["train"] + data["val"]] + inat_photos
    cached = cache_images.ensure_cached(every_photo, IMAGE_SIZE)
    train_items = use_cache(data["train"], cached)
    val_items = use_cache(data["val"], cached)
    inat_items = use_cache([(path, []) for path in inat_photos], cached)
    print(f"Train: {len(train_items)} images, val: {len(val_items)} images, "
          f"iNaturalist test: {len(inat_items)} images")
    if not train_items:
        sys.exit("No training photos found.")

    sampler = BalancedSampler(train_items, args.epoch_size, args.bat_fraction)
    print(f"Each epoch: {args.epoch_size} photos = {sampler.counts['bat']} bat + "
          f"{sampler.counts['other']} not-a-bat, dealt from piles of "
          f"{len(sampler.piles['bat'])} bat and {len(sampler.piles['other'])} not-a-bat "
          f"(hand-labelled bats counted {HAND_LABELLED_BAT_REPEATS}x)")
    train_loader = make_loader(train_items, True, args.batch_size, args.workers,
                               args.prefetch_factor, sampler=sampler)
    val_photos = CheckPhotos(val_items, args.eval_batch_size)
    inat_test = CheckPhotos(inat_items, args.eval_batch_size) if inat_items else None
    print(f"Data ready in {time.perf_counter() - start_time:.0f} s", flush=True)

    # Load BatNet and take out just its detector.
    importer = torch.package.PackageImporter(str(config.WEIGHTS_FILE))
    detector = importer.load_pickle("model", "model.pkl").detector
    settings = FROM_SCRATCH
    if not args.from_scratch and config.FINETUNED_DETECTOR_FILE.exists():
        detector.load_state_dict(torch.load(config.FINETUNED_DETECTOR_FILE, map_location="cpu"))
        settings = CONTINUE
        print("Continuing from the last fine-tuned detector")
    use_quick_box_sampler(detector)
    epochs = args.epochs or settings["epochs"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    detector.to(device)
    print(f"Training on {device} for {epochs} epochs")
    print(f"Batch size {args.batch_size}, workers {args.workers}, "
          f"prefetch {args.prefetch_factor}, CPU threads {torch.get_num_threads()}, "
          f"mixed precision {device == 'cuda'}", flush=True)

    # Train everything except the very first layers (they detect simple
    # edges and colours, which don't need to change).
    frozen = ("basemodel.backbone.body.conv1", "basemodel.backbone.body.layer1")
    for name, param in detector.named_parameters():
        param.requires_grad = not name.startswith(frozen)
    params = [p for p in detector.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=settings["lr"], momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, settings["milestones"], gamma=0.1)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    if args.benchmark is not None:
        benchmark(detector, optimizer, train_loader, device, args.benchmark, scaler)
        return

    def check(label, epoch):
        started = time.perf_counter()
        found, missed, fa = evaluate(detector, val_photos, device)
        line = f"{label}  bats found {found}/{found + missed}  false alarms {fa}"
        if inat_test and check_inaturalist_after(epoch, epochs, args.inat_every):
            # For information only: the held-back test photos must not
            # decide which epoch is kept, or the final test score would be
            # too optimistic.
            inat_hits = photos_with_bat_found(detector, inat_test, device)
            line += f"  (iNaturalist test photos with a bat found {inat_hits}/{len(inat_test)})"
        print(f"{line}  [checked in {time.perf_counter() - started:.0f} s]", flush=True)
        # Balance of bats found against false alarms, on the check photos.
        return found - fa

    best_score = check("Before training:", 0)
    best_epoch = None
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        detector.train()
        total_loss = 0.0
        for images, targets in train_loader:
            total_loss += training_step(detector, optimizer, images, targets, device, scaler)
        scheduler.step()
        trained = time.perf_counter() - started

        score = check(f"Epoch {epoch}/{epochs}  loss {total_loss / len(train_loader):.3f}"
                      f"  [trained in {trained:.0f} s]", epoch)
        # Keep the epoch with the best balance (only if it beats the start).
        if score > best_score:
            best_score, best_epoch = score, epoch
            if args.no_save:
                print("  best so far (not saved: --no-save)")
            else:
                args.save_to.parent.mkdir(parents=True, exist_ok=True)
                # Saved as 16-bit numbers (~84 MB instead of ~168 MB) so the
                # .pth fits on GitHub; it loads back as normal 32-bit weights.
                torch.save(half_precision(detector.state_dict()), args.save_to)
                print(f"  saved -> {args.save_to}")

    if best_epoch is None:
        print("No epoch beat the starting model, so nothing was saved.")
    else:
        print(f"Best epoch: {best_epoch} (bats found minus false alarms = {best_score})")
    print(f"Total time {time.perf_counter() - start_time:.0f} s")


if __name__ == "__main__":
    main()
