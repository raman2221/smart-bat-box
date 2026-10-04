"""
cache_images.py

Decoding a JPEG and shrinking it to 320x320 takes a few milliseconds per
photo. Training used to do that again for every photo in every epoch.
This module does it once and keeps the small 320x320 copy on disk in
data/cache/, so later epochs (and later runs) just read the ready pixels.

Why exactly 320x320 and not bigger: training always shrinks every photo
to 320x320 *first* and only then applies its random changes (colour,
mirror, zoom-out, blur). None of those changes needs pixels from outside
the 320x320 picture, so a bigger copy would only cost disk space.

The copies are saved as .npy files (raw pixels, no JPEG compression), so
they give back exactly the same pixels as decoding the photo again.
About 300 KB per photo, so roughly 3 GB for ~10,000 photos.

Each cached file name contains a fingerprint of the source photo's size
and modification time. If a photo is replaced (for example the synthetic
photos are regenerated), its fingerprint changes, so it is cached again and
the stale copy is deleted.

The training script builds the cache automatically. You can also build it
ahead of time (run from the project folder, with the .venv active):
    python training/cache_images.py
"""

import hashlib
import os
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402

CACHE_DIR = config.PROJECT_ROOT / "data" / "cache"
DEFAULT_SIZE = 320   # BatNet's detector works on 320x320 images


def decode_photo(source, size=DEFAULT_SIZE):
    """Open a photo the way training always has: RGB, squashed to size x size."""
    with Image.open(source) as photo:
        return np.asarray(photo.convert("RGB").resize((size, size)))


def _fingerprint(source):
    """Short text that changes whenever the photo file is replaced or edited."""
    info = os.stat(source)
    return hashlib.sha1(f"{info.st_size}:{info.st_mtime_ns}".encode()).hexdigest()[:12]


def _name_key(source, size):
    """Short name for this photo's path (the same path always gives the same key)."""
    source = Path(source).resolve()
    try:
        # A path inside the project is stored relative to it, so moving the
        # whole project folder keeps the cache valid.
        text = source.relative_to(config.PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        text = source.as_posix()
    return hashlib.sha1(f"{size}:{text.lower()}".encode()).hexdigest()[:20]


def cache_file_for(source, size=DEFAULT_SIZE, cache_dir=None):
    """Where the cached pixels of this photo (as it is right now) belong."""
    key = _name_key(source, size)
    # Spread over 256 small folders: a folder with 10,000 files is slow to search.
    folder = Path(cache_dir or CACHE_DIR) / f"{size}px" / key[:2]
    return folder / f"{key}-{_fingerprint(source)}.npy"


def cache_one(source, size=DEFAULT_SIZE, cache_dir=None):
    """Make sure one photo is cached. Returns (cache file, was it new?).

    Returns (None, False) when the photo is missing, unreadable, or is
    being rewritten by another program right now.
    """
    try:
        target = cache_file_for(source, size, cache_dir)
        if target.is_file():
            return target, False
        pixels = decode_photo(source, size)
        if cache_file_for(source, size, cache_dir) != target:
            return None, False   # the photo changed while we were reading it
        target.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temporary name first, then rename: a crash can never
        # leave a half-written cache file behind under the real name.
        temporary = target.with_name(f"{target.stem}.{uuid.uuid4().hex}.tmp")
        with open(temporary, "wb") as f:
            np.save(f, pixels)
        os.replace(temporary, target)
        # Delete copies made from an older version of the same photo.
        for old in target.parent.glob(f"{target.name.split('-')[0]}-*.npy"):
            if old != target:
                try:
                    old.unlink()
                except OSError:
                    pass
        return target, True
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        # Missing, truncated (still being written) or not a photo at all.
        return None, False


def ensure_cached(sources, size=DEFAULT_SIZE, cache_dir=None, threads=8, quiet=False):
    """Cache every photo in `sources`. Returns {source: cache file}.

    Photos that could not be read are left out of the result, so the
    caller can simply skip them.
    """
    sources = list(dict.fromkeys(Path(s) for s in sources))   # unique, keep order
    with ThreadPoolExecutor(max(1, threads)) as pool:
        results = list(pool.map(lambda s: cache_one(s, size, cache_dir), sources))
    ready = {source: target for source, (target, _) in zip(sources, results) if target}
    new = sum(made for _, made in results)
    if not quiet:
        print(f"Image cache: {len(ready)} photos ready ({new} newly cached, "
              f"{len(sources) - len(ready)} missing or unreadable skipped)", flush=True)
    return ready


def read_pixels(path, size=DEFAULT_SIZE):
    """Pixels (height x width x 3, uint8) from a cache file or a normal photo."""
    if Path(path).suffix == ".npy":
        return np.load(path)
    return decode_photo(path, size)


def main():
    # Imported here so this file can be imported without a circular import.
    from training import finetune_detector as trainer

    data = trainer.load_annotations()
    photos = [path for path, _ in data["train"] + data["val"]]
    photos += sorted(trainer.inaturalist_test_photos())
    ensure_cached(photos, trainer.IMAGE_SIZE)


if __name__ == "__main__":
    main()
