"""Rehearse the real entrance pipeline without opening a camera.

Run: python tools/replay_demo.py --clips 6 --seed 42
Only validation photos are used. Nothing is written outside data/rehearsal.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import tempfile
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision import config, entrance_tracker
from vision.bat_detector import BatDetector
from vision.bat_tracker import BatTracker

WIDTH, HEIGHT, VIDEO_FPS = 640, 480, 30
SCENARIOS = ("in", "out", "flypast", "pan", "person", "lamp")
# This validation lamp caused BAT DETECTED in the first GPU rehearsal.
# Keep a separate no-bat regression clip so that failure stays visible.
LAMP_PHOTO = "71b828357f8f3fdf.jpg"
# Visually checked rooms, lamps and a fan. A CSV negative encountered during
# rehearsal actually contained a flying bat, so labels alone are not enough.
BACKGROUND_PHOTOS = (
    "0056ffa38a8489b6.jpg", "02183289fae0c47c.jpg", "040659777cc851a1.jpg",
    "2179f74d04522cd0.jpg", LAMP_PHOTO,
)
# These validation negatives were visually checked: a standing man, skaters,
# and people boxing. The CSV still has to confirm their split and label.
PERSON_PHOTOS = (
    "07b06449065bb322.jpg", "04ccf8c5df047075.jpg",
    "0dc12bf575bcda55.jpg", "0402361d506a21f5.jpg",
)


@dataclass
class Clip:
    name: str
    scenario: str
    frames: list
    boxes: list
    visible: list
    expected_events: list
    sources: dict


def validation_rows(root):
    """Read validation annotations, never training or iNaturalist photos."""
    with (root / "annotations.csv").open(newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] == "val"]
    rows.sort(key=lambda row: (row["image"], row["xmin"], row["ymin"], row["xmax"], row["ymax"]))
    bats = [row for row in rows if row["is_bat"] == "1"]
    negatives = [row for row in rows if row["is_bat"] == "0"]
    if not bats or not negatives:
        raise ValueError("Validation bat and negative annotations are required.")
    return bats, negatives


def read_photo(root, row):
    path = (root / row["image"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Photo path leaves the Open Images directory.")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot read validation photo: {path}")
    return image


def bat_crop(image, row):
    """Turn fractional annotation coordinates into a real photo crop."""
    height, width = image.shape[:2]
    x1, y1 = int(float(row["xmin"]) * width), int(float(row["ymin"]) * height)
    x2 = int(np.ceil(float(row["xmax"]) * width))
    y2 = int(np.ceil(float(row["ymax"]) * height))
    crop = image[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]
    if not crop.size:
        raise ValueError("Empty validation bat box.")
    return crop


def small_patch(image, size=180):
    height, width = image.shape[:2]
    scale = size / max(height, width)
    return cv2.resize(image, (max(1, round(width * scale)),
                              max(1, round(height * scale))))


def paste(frame, photo, x, y):
    """Paste the visible part of a photo and return its clipped box."""
    height, width = photo.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(WIDTH, x + width), min(HEIGHT, y + height)
    if x1 >= x2 or y1 >= y2:
        return None
    frame[y1:y2, x1:x2] = photo[y1-y:y2-y, x1-x:x2-x]
    return (x1, y1, x2, y2)


def make_clip(scenario, background, photo=None, name=None, sources=None):
    """Build a 30 fps script with settling frames and a full missing timeout.

    Edge poses last six frames so confirmation can happen before movement.
    Photo rectangles retain their original context; this is a compositing
    stress test, not a claim that these frames look like a real flying bat.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    if scenario not in ("pan", "lamp") and photo is None:
        raise ValueError("A bat/person photo is required for this scenario.")
    background = cv2.resize(background, (WIDTH, HEIGHT))
    photo = small_patch(photo) if photo is not None else None
    warmup, movement = 30, 48
    tail = config.MAX_FRAMES_MISSING + 20
    frames, boxes, visible = [], [], []
    for index in range(warmup + movement + tail):
        frame = background.copy()
        box = None
        active = warmup <= index < warmup + movement
        if scenario in ("pan", "lamp") and index >= warmup:
            phase = (index - warmup) / VIDEO_FPS
            matrix = cv2.getRotationMatrix2D((WIDTH/2, HEIGHT/2), 0,
                                             1.025 + .015 * np.sin(phase))
            matrix[:, 2] += (5 * np.sin(phase), 4 * np.cos(phase))
            frame = cv2.warpAffine(frame, matrix, (WIDTH, HEIGHT),
                                   borderMode=cv2.BORDER_REFLECT_101)
        elif active:
            progress = float(np.clip((index - warmup - 5) / 37, 0, 1))
            pw, ph = photo.shape[1], photo.shape[0]
            middle = (WIDTH - pw) // 2
            start = middle if scenario in ("out", "person") else 0
            end = middle if scenario == "in" else WIDTH - pw
            x = round(start + (end - start) * progress)
            y = (HEIGHT - ph) // 2
            if scenario == "person":
                x = round(middle + 150 * np.sin(progress * 2 * np.pi))
                y += round(50 * np.sin(progress * 4 * np.pi))
            box = paste(frame, photo, x, y)
        frames.append(frame)
        boxes.append(box)
        visible.append(box is not None and scenario in ("in", "out", "flypast"))
    expected = {"in": ["IN"], "out": ["OUT"]}.get(scenario, [])
    return Clip(name or scenario, scenario, frames, boxes, visible, expected,
                sources or {})


def generate_clips(count=6, seed=42, root=None):
    """Use a seeded photo selection without consulting detector predictions."""
    if count < 1:
        raise ValueError("At least one clip is required.")
    root = Path(root or config.OPENIMAGES_DIR)
    bats, negatives = validation_rows(root)
    people = [row for row in negatives if Path(row["image"]).name in PERSON_PHOTOS]
    lamps = [row for row in negatives if Path(row["image"]).name == LAMP_PHOTO]
    backgrounds = [row for row in negatives if Path(row["image"]).name in BACKGROUND_PHOTOS]
    if not backgrounds:
        raise ValueError("The visually checked validation backgrounds are missing.")
    if not people and count >= 5:
        raise ValueError("The visually checked validation person photos are missing.")
    rng = random.Random(seed)
    for index in range(count):
        scenario = SCENARIOS[index % len(SCENARIOS)]
        negative = rng.choice(backgrounds)
        if scenario == "lamp":
            if not lamps:
                raise ValueError("The validation lamp regression photo is missing.")
            negative = lamps[0]
        sources = {"background": negative["image"], "split": "val"}
        background = read_photo(root, negative)
        photo = None
        if scenario in ("in", "out", "flypast"):
            bat = rng.choice(bats)
            photo = bat_crop(read_photo(root, bat), bat)
            sources["bat"] = dict(bat)
        elif scenario == "person":
            person = rng.choice(people)
            photo = read_photo(root, person)
            sources["person"] = person["image"]
        yield make_clip(scenario, background, photo,
                        f"{index+1:02d}_{scenario}", sources)


def score_clip(clip, statuses, frame_ms, events, snapshots_ok=True):
    """Score the actual on-screen status, including tracker confirmation."""
    if not (len(statuses) == len(frame_ms) == len(clip.frames)):
        raise ValueError("Each frame needs one status and one processing time.")
    indices = [i for i, visible in enumerate(clip.visible) if visible]
    detection = 100 * sum(bool(statuses[i]) for i in indices) / len(indices) if indices else None
    false_alarm = 100 * sum(bool(s) for s in statuses) / len(statuses) if not indices else None
    # Allow the tracker its normal missing-frame grace period after a bat
    # vanishes. Before appearance and after that timeout, status must be quiet.
    quiet = [i for i in range(len(statuses)) if indices and
             (i < indices[0] or i > indices[-1] + config.MAX_FRAMES_MISSING)]
    background_alarm = 100 * sum(bool(statuses[i]) for i in quiet) / len(quiet) if quiet else None
    preexisting_status = any(statuses[i] for i in range(indices[0])) if indices else False
    first_detected = next((i for i in indices if statuses[i]), None)
    latency = first_detected - indices[0] if first_detected is not None else None
    average_ms = sum(frame_ms) / len(frame_ms)
    fps = 1000 / average_ms if average_ms > 0 else 0
    labels_ok = (detection >= 90 and latency is not None and latency <= 6 and background_alarm == 0
                 if indices else false_alarm == 0)
    return {
        "clip": clip.name, "scenario": clip.scenario,
        "frames": len(statuses), "visible_frames": len(indices),
        "detected_visible_frames": sum(bool(statuses[i]) for i in indices),
        "false_alarm_frames": sum(bool(s) for s in statuses) if not indices else None,
        "detection_pct": detection, "false_alarm_pct": false_alarm,
        "background_false_alarm_pct": background_alarm,
        "quiet_frames": len(quiet), "preexisting_status": preexisting_status,
        "latency_frames": latency, "latency_ms": latency * 1000 / VIDEO_FPS if latency is not None else None,
        "average_ms": average_ms, "fps": fps, "realtime": fps >= VIDEO_FPS,
        "events": events, "expected_events": clip.expected_events,
        "snapshots_ok": snapshots_ok,
        "passed": bool(labels_ok and events == clip.expected_events and snapshots_ok),
    }


class MemoryCapture:
    """VideoCapture's small interface, backed by the synthetic frames."""

    def __init__(self, frames):
        self.frames = frames
        self.index = -1
        self.started = 0
        self.released = False

    def isOpened(self):
        return True

    def read(self):
        self.index += 1
        if self.index >= len(self.frames):
            return False, None
        self.started = time.perf_counter()
        return True, self.frames[self.index].copy()

    def release(self):
        self.released = True


def run_clip(clip, detector, output):
    """Run main unchanged: its MOG2 gate, detect, update, status, and finish.

    Only the camera/display and detector construction are replaced. Config
    paths are temporarily redirected and restored even if the run fails.
    """
    output.mkdir(parents=True, exist_ok=True)
    capture = MemoryCapture(clip.frames)
    tracker = BatTracker()
    statuses, frame_ms, samples, records, detected = [], [], {}, [], {}
    original_status = entrance_tracker.draw_status
    original_detect = detector.detect
    indices = [i for i, v in enumerate(clip.visible) if v]
    sample_ids = {0, 30, 33, 54, 77, len(clip.frames)-1}
    if indices:
        sample_ids.update((indices[0], indices[-1]))

    def draw_status(display, present):
        statuses.append(bool(present))
        original_status(display, present)

    def detect(frame):
        detections = original_detect(frame)
        detected[capture.index] = [{"bbox": list(d.bbox), "confidence": d.confidence}
                                  for d in detections]
        return detections

    def show(window, display):
        # synchronize is needed because GPU work can finish asynchronously.
        if detector.device == "cuda":
            import torch
            torch.cuda.synchronize()
        frame_ms.append((time.perf_counter() - capture.started) * 1000)
        records.append([{"id": t.track_id, "bbox": list(t.bbox),
                         "confidence": t.confidence, "missing": t.frames_missing}
                        for t in tracker.tracks.values()])
        if capture.index in sample_ids:
            samples[capture.index] = display.copy()

    with ExitStack() as stack:
        stack.enter_context(patch.object(config, "EVENTS_DIR", output / "events"))
        stack.enter_context(patch.object(config, "ENTRANCE_POINTS_FILE", output / "entrance_points.json"))
        stack.enter_context(patch.object(entrance_tracker, "BatDetector", return_value=detector))
        stack.enter_context(patch.object(detector, "detect", side_effect=detect))
        stack.enter_context(patch.object(entrance_tracker, "BatTracker", return_value=tracker))
        stack.enter_context(patch.object(entrance_tracker, "draw_status", side_effect=draw_status))
        stack.enter_context(patch.object(entrance_tracker.cv2, "VideoCapture", return_value=capture))
        stack.enter_context(patch.object(entrance_tracker.cv2, "imshow", side_effect=show))
        stack.enter_context(patch.object(entrance_tracker.cv2, "waitKey", side_effect=lambda delay:
                                        ord("q") if capture.index == len(clip.frames)-1 else -1))
        stack.enter_context(patch.object(entrance_tracker.cv2, "destroyAllWindows"))
        entrance_tracker.main()

    log = output / "events" / "events.csv"
    rows = []
    if log.exists():
        with log.open(newline="") as f:
            rows = list(csv.DictReader(f))
    snapshots = [output / "events" / row["photo"] for row in rows]
    raw = [output / "events" / "raw" / row["photo"] for row in rows]
    snapshots_ok = all(p.exists() and cv2.imread(str(p)) is not None for p in snapshots + raw)
    result = score_clip(clip, statuses, frame_ms, [row["direction"] for row in rows], snapshots_ok)
    result.update(sources=clip.sources, snapshots=[str(p) for p in snapshots],
                  raw_snapshots=[str(p) for p in raw], output=str(output))
    inference_ms = [frame_ms[i] for i in detected]
    result["detector_frames"] = len(inference_ms)
    result["detector_frame_average_ms"] = sum(inference_ms) / len(inference_ms) if inference_ms else None
    (output / "frame_results.json").write_text(json.dumps({
        "visible": clip.visible, "boxes": clip.boxes, "statuses": statuses,
        "frame_ms": frame_ms, "tracks": records,
        "detections": detected,
    }, indent=2), encoding="utf-8")
    annotated = []
    for index, frame in sorted(samples.items()):
        box = clip.boxes[index]
        if box:
            cv2.rectangle(frame, box[:2], box[2:], (255, 200, 0), 2)
        for track in records[index]:
            bbox = [int(v) for v in track["bbox"]]
            cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), (0, 165, 255), 2)
        cv2.putText(frame, f"{clip.name} frame {index} | expected {clip.expected_events or 'none'}",
                    (10, 465), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
        path = output / f"frame_{index:03d}.jpg"
        if not cv2.imwrite(str(path), frame):
            raise OSError(f"Could not write {path}")
        annotated.append(frame)
    return result, annotated


def write_contact_sheet(rows, path):
    columns = max(len(row) for row in rows)
    sheet = np.full((len(rows)*240, columns*320, 3), 35, np.uint8)
    for y, row in enumerate(rows):
        for x, frame in enumerate(row):
            sheet[y*240:(y+1)*240, x*320:(x+1)*320] = cv2.resize(frame, (320, 240))
    if not cv2.imwrite(str(path), sheet):
        raise OSError(f"Could not write {path}")


def report_table(results):
    lines = ["| Clip | Result | Detection | False alarms | Background FA | Latency (frames) | Events / expected | Snapshots | ms/frame | FPS |",
             "|---|---|---:|---:|---:|---:|---|---:|---:|---:|"]
    for r in results:
        percent = lambda value: "—" if value is None else f"{value:.1f}%"
        latency = r['latency_frames'] if r['latency_frames'] is not None else "—"
        if r["preexisting_status"]:
            latency = f"{latency}*"
        events = ','.join(r['events']) or 'none'
        expected = ','.join(r['expected_events']) or 'none'
        lines.append(f"| {r['clip']} | {'PASS' if r['passed'] else 'FAIL'} | "
                     f"{percent(r['detection_pct'])} | {percent(r['false_alarm_pct'])} | "
                     f"{percent(r['background_false_alarm_pct'])} | "
                     f"{latency} | {events} / {expected} | {len(r['snapshots'])} | "
                     f"{r['average_ms']:.1f} | {r['fps']:.1f} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=config.FINETUNED_DETECTOR_FILE)
    parser.add_argument("--clips", type=int, default=6, help="Total clips; six covers all scenarios including the lamp regression")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.clips < 1 or not args.weights.is_file():
        parser.error("--clips must be positive and --weights must name an existing fine-tuned state dict")
    output = config.PROJECT_ROOT / "data" / "rehearsal"
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=f"seed_{args.seed}_", dir=output))
    with args.weights.open("rb") as f:
        weights_hash = hashlib.file_digest(f, "sha256").hexdigest()
    # The constructor needs the base package; --weights is the fine-tuned
    # state dict, not that package. Restore the setting immediately afterward.
    with patch.object(config, "FINETUNED_DETECTOR_FILE", args.weights.resolve()):
        detector = BatDetector()
    if detector.device != "cuda":
        raise RuntimeError("This rehearsal requires the real CUDA detector.")
    print(f"Rehearsal uses {detector.device}; seed {args.seed}; output {run}", flush=True)
    results, sheets = [], []
    for clip in generate_clips(args.clips, args.seed):
        print(f"Running {clip.name} ({len(clip.frames)} frames)...", flush=True)
        result, images = run_clip(clip, detector, run / clip.name)
        results.append(result)
        sheets.append(images)
        print(report_table([result]), flush=True)
    write_contact_sheet(sheets, output / "contact_sheet.jpg")
    write_contact_sheet(sheets, run / "contact_sheet.jpg")
    metadata = {"seed": args.seed, "device": detector.device,
                "weights": str(args.weights.resolve()),
                "weights_sha256": weights_hash,
                "annotations_sha256": hashlib.sha256((config.OPENIMAGES_DIR / "annotations.csv").read_bytes()).hexdigest(),
                "pipeline_sha256": {name: hashlib.sha256((config.PROJECT_ROOT / "vision" / name).read_bytes()).hexdigest()
                                    for name in ("entrance_tracker.py", "bat_detector.py", "bat_tracker.py", "entrance_monitor.py", "config.py")},
                "run": str(run), "results": results}
    (output / "rehearsal_results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    text = "# Demo rehearsal\n\n" + report_table(results)
    text += (f"\n\nSeed: {args.seed}. Device: {detector.device}. Fine-tuned weights: `{args.weights}`.\n"
             f"\nRun files: `{run}`. Contact sheet: [contact_sheet.jpg](contact_sheet.jpg).\n"
             "\nPASS means at least 90% visible-frame status detection, latency at most six frames "
             "(200 ms at the scripted 30 fps), zero status false alarms on negative clips, "
             "zero background false alarms before bat appearance/after the missing-frame grace period, "
             "exact expected events, and readable annotated/raw event snapshots. Speed is reported separately; "
             "30 FPS is the real-time target. A dash means not applicable or never detected.\n"
             "\nBackground FA scores bat-clip frames with no bat, excluding the normal tracker timeout "
             "after disappearance. An asterisk on latency means BAT DETECTED already appeared before "
             "the bat was pasted; that latency cannot demonstrate successful bat recognition.\n"
             "\nFrames are processed as quickly as possible, with 30 fps scripted motion; there is no sleeping. "
             "Timing includes the real motion gate, detector, tracking, species classification/event writes, "
             "status drawing and CUDA synchronization; excludes model loading, clip generation and rehearsal image writes. "
             "The first inference is included, so startup overhead can affect speed.\n"
             "\nEvery clip starts with fresh tracking and entrance learning in its own directory. "
             "The main loop, including MOG2 and its motion gate, runs unchanged with a memory VideoCapture. "
             "Cyan boxes show pasted-photo bounds; orange boxes show real tracks (including missing ones). "
             "All source paths/boxes and frame-level results are saved in JSON.\n"
             "\nThis small synthetic check cannot prove webcam performance. Rectangular photo crops, "
             "abrupt disappearance and static lighting differ from real flying bats. "
             "Photo selection is seeded and independent of detector scores; event names/timings depend on wall time.\n")
    text += ("\nBackgrounds and person photos come from a visually checked list of validation negatives. "
             "The lamp regression was added after it caused a false positive in an earlier run. "
             "A mislabeled validation negative containing a real flying bat "
             "(`images/val/negative/7ead8a05548b7dd8.jpg`) was found during visual QA and is excluded "
             "by this list. Source annotations are untouched.\n")
    text += "\n## Detector work\n\n"
    for result in results:
        average = result["detector_frame_average_ms"]
        timing = f"{average:.1f} ms" if average is not None else "not called"
        text += (f"- {result['clip']}: detector ran on {result['detector_frames']}/{result['frames']} "
                 f"frames; average full processing time on those frames: {timing}.\n")
    missing = set(SCENARIOS) - {r["scenario"] for r in results}
    if missing:
        text += f"\nIncomplete scenario coverage: {', '.join(sorted(missing))}. Use --clips 6 or more.\n"
    (output / "rehearsal_report.md").write_text(text, encoding="utf-8")
    (run / "rehearsal_report.md").write_text(text, encoding="utf-8")
    (run / "rehearsal_results.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Report saved to {output / 'rehearsal_report.md'}", flush=True)
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
