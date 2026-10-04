"""
download_data.py

Downloads training photos. Two modes:

  negatives    "Not a bat" photos from Google's Open Images. Use it whenever
               the detector mistakes something for a bat: download photos of
               that thing, re-run training, and it learns it isn't a bat.
               Added to data/openimages/annotations.csv with no bat box.

  inaturalist  Bat photos from iNaturalist: research-grade observations of
               North American bats with open licenses (CC0, CC BY, CC BY-NC),
               up to --per-species photos of each species, plus every
               Michigan observation. Saved to data/inaturalist/images/ with
               credits in data/inaturalist/photos.csv. These have no bat
               boxes yet: run training/auto_label.py next.

Examples (run from the project folder with the .venv active):
    python training/download_data.py negatives Person "Human face" Man Woman
    python training/download_data.py negatives Lamp "Light bulb" --per-class 50
    python training/download_data.py inaturalist --per-species 100
"""

import argparse
import csv
import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402

MAX_SIDE = 640   # photos are shrunk so their longest side is at most this


def save_photo(url, dest):
    """Download one photo, shrink it and save it as JPEG. Returns True if OK."""
    if dest.exists():
        return True
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "smart-bat-box"})
        data = urllib.request.urlopen(request, timeout=30).read()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail((MAX_SIDE, MAX_SIDE))
        dest.parent.mkdir(parents=True, exist_ok=True)
        # Save under a temporary name first, then rename: a crash mid-save
        # can't leave a half-written photo with the real name.
        temp = dest.with_suffix(".part")
        image.save(temp, format="JPEG", quality=90)
        temp.replace(dest)
        return True
    except Exception as e:
        print(f"  failed {url}: {e}", flush=True)
        return False


# ===========================================================================
# Mode 1: Open Images "not a bat" photos
# ===========================================================================

OI_CLASS_LIST_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
OI_ANNOTATION_URLS = [  # (our split, Open Images folder, box annotation list)
    ("val", "validation", "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"),
    ("val", "test", "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv"),
    ("train", "train", "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv"),
]
OI_BAT_CLASS = "/m/01h44"


def download_negatives(class_names, per_class_train):
    per_class = {"train": per_class_train, "val": max(1, per_class_train // 5)}
    random.seed(0)

    # 1. Turn class names into Open Images IDs.
    with urllib.request.urlopen(OI_CLASS_LIST_URL) as resp:
        name_to_id = {name: mid for mid, name in csv.reader(io.TextIOWrapper(resp, encoding="utf-8"))}
    wanted = {}
    for name in class_names:
        if name not in name_to_id:
            print(f'Unknown class "{name}". Check the spelling (it is case-sensitive).')
            return
        wanted[name_to_id[name]] = name

    # 2. Photos already in our training data.
    annotations_file = config.OPENIMAGES_DIR / "annotations.csv"
    existing = set()
    with open(annotations_file, newline="") as f:
        for row in csv.DictReader(f):
            existing.add(Path(row["image"]).stem)

    # 3. Find photos of each class (the train list is 2.2 GB, so this takes a few minutes).
    photos = {split: {mid: set() for mid in wanted} for split in per_class}
    has_bat = set()
    folder_of = {}
    for split, folder, url in OI_ANNOTATION_URLS:
        print(f"Reading {folder} annotations...", flush=True)
        with urllib.request.urlopen(url) as resp:
            for line in io.TextIOWrapper(resp, encoding="utf-8"):
                parts = line.split(",", 3)
                if len(parts) < 3:
                    continue
                image_id, label = parts[0], parts[2]
                if label == OI_BAT_CLASS:
                    has_bat.add(image_id)
                elif label in wanted:
                    photos[split][label].add(image_id)
                    folder_of[image_id] = (split, folder)

    # 4. Pick photos: no bats, not already used, no duplicates between classes.
    chosen = []
    for split, count in per_class.items():
        used = set()
        for mid, ids in photos[split].items():
            pool = sorted(ids - has_bat - existing - used)
            pick = random.sample(pool, min(count, len(pool)))
            used.update(pick)
            chosen += pick
            print(f"  {split:5s} {wanted[mid]}: {len(pick)} photos")

    # 5. Download, shrink and add to annotations.csv.
    print(f"Downloading {len(chosen)} photos...", flush=True)

    def download(image_id):
        split, folder = folder_of[image_id]
        dest = config.OPENIMAGES_DIR / "images" / split / "negative" / f"{image_id}.jpg"
        url = f"https://s3.amazonaws.com/open-images-dataset/{folder}/{image_id}.jpg"
        return (split, dest) if save_photo(url, dest) else None

    added = {"train": 0, "val": 0}
    with ThreadPoolExecutor(16) as pool, open(annotations_file, "a", newline="") as f:
        writer = csv.writer(f)
        for result in pool.map(download, chosen):
            if result:
                split, dest = result
                writer.writerow([split, dest.relative_to(config.OPENIMAGES_DIR).as_posix(),
                                 0, "", "", "", ""])
                added[split] += 1
    print(f"Added {added['train']} training and {added['val']} checking photos. "
          "Now re-run: python training/finetune_detector.py")


# ===========================================================================
# Mode 2: iNaturalist bat photos
# ===========================================================================

INAT_API = "https://api.inaturalist.org/v1"
INAT_BATS = 40268              # taxon ID of Chiroptera (all bats)
INAT_NORTH_AMERICA = 97394     # place IDs
INAT_MICHIGAN = 10
INAT_LICENSES = "cc0,cc-by,cc-by-nc"


def inat_get(path, params):
    """Call the iNaturalist API. It asks for at most ~1 request per second."""
    url = f"{INAT_API}/{path}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "smart-bat-box (hackathon project)"})
    with urllib.request.urlopen(request, timeout=60) as resp:
        data = json.load(resp)
    time.sleep(1.1)
    return data


def download_inaturalist(per_species):
    out_dir = config.INATURALIST_DIR
    base = {"quality_grade": "research", "photo_license": INAT_LICENSES, "photos": "true"}

    # 1. Which bat species have open-license photos in North America?
    counts = inat_get("observations/species_counts",
                      {**base, "taxon_id": INAT_BATS, "place_id": INAT_NORTH_AMERICA, "per_page": 500})
    species = [r["taxon"] for r in counts["results"]]
    print(f"{len(species)} North American bat species with open-license photos")

    # 2. Up to `per_species` observations of each species, plus all of Michigan.
    observations = {}
    for taxon in species:
        data = inat_get("observations", {**base, "taxon_id": taxon["id"], "place_id": INAT_NORTH_AMERICA,
                                          "per_page": min(per_species, 200)})
        for obs in data["results"][:per_species]:
            observations[obs["id"]] = (obs, False)
        print(f"  {taxon['name']}: {min(len(data['results']), per_species)}", flush=True)
    page = 1
    while True:
        data = inat_get("observations", {**base, "taxon_id": INAT_BATS, "place_id": INAT_MICHIGAN,
                                          "per_page": 200, "page": page})
        for obs in data["results"]:
            observations[obs["id"]] = (obs, True)
        if page * 200 >= data["total_results"]:
            break
        page += 1
    print(f"{len(observations)} observations in total "
          f"({sum(1 for _, mi in observations.values() if mi)} from Michigan)")

    # 3. Download the first photo of each observation, keeping its credit.
    rows = []
    for obs, in_michigan in observations.values():
        photo = obs["photos"][0]
        if photo.get("license_code") not in INAT_LICENSES.split(","):
            continue
        rows.append({
            "image": f"images/{photo['id']}.jpg",
            "url": photo["url"].replace("/square.", "/medium."),
            "species": obs["taxon"]["name"],
            "common_name": obs["taxon"].get("preferred_common_name", ""),
            "michigan": int(in_michigan),
            "license": photo["license_code"],
            "attribution": photo["attribution"],
            "observation": obs["uri"],
        })

    print(f"Downloading {len(rows)} photos...", flush=True)
    with ThreadPoolExecutor(8) as pool:
        ok = list(pool.map(lambda r: save_photo(r["url"], out_dir / r["image"]), rows))
    rows = [r for r, good in zip(rows, ok) if good]
    if not rows:
        print("No photos could be downloaded.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "photos.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} photos to {out_dir}. Next: python training/auto_label.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    neg = modes.add_parser("negatives", help="Open Images 'not a bat' photos")
    neg.add_argument("classes", nargs="+", help='Open Images class names, e.g. Person "Human face"')
    neg.add_argument("--per-class", type=int, default=100,
                     help="training photos per class (default 100; plus 20%% for checking)")
    inat = modes.add_parser("inaturalist", help="iNaturalist North American bat photos")
    inat.add_argument("--per-species", type=int, default=100,
                      help="max photos per species (default 100)")
    args = parser.parse_args()

    if args.mode == "negatives":
        download_negatives(args.classes, args.per_class)
    else:
        download_inaturalist(args.per_species)


if __name__ == "__main__":
    main()
