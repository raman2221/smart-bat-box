"""
entrance_monitor.py

Decides when a tracked bat has gone IN to the bat box or come OUT, saves a
snapshot of it, and writes the event to a log. No setup needed: it works
out where the entrance is by itself.

Camera placement 1: the camera is outside the box, looking at the entry
slot. Bats can leave the picture in only two ways: across the edge of the
frame (flying away) or into the slot (going into the box). So:

    came in across the frame edge, vanished in the middle   ->  IN
    appeared in the middle, left across the frame edge      ->  OUT
    edge to edge (flew past) or middle to middle            ->  ignored

Learning the entrance: every IN/OUT event adds the spot where the bat
vanished/appeared. Once there are ENTRANCE_LEARN_MIN_EVENTS of them, the
middle of those spots is taken as the entrance. From then on, a bat that
vanishes far away from the entrance is ignored (that's usually the
detector briefly losing it, not a bat going in). The learned spots are
saved, so the next run starts already knowing where the entrance is.

The decision is made when the tracker gives up on a bat (it has been
missing for MAX_FRAMES_MISSING frames), because only then do we know
where its path ended.

Each event saves:
  data/events/<time>_<IN|OUT>_bat<id>.jpg      snapshot with the bat boxed
  data/events/raw/<same name>.jpg              same frame without drawings
                                               (useful for future training)
  data/events/events.csv                       one line per event
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime

import cv2

from . import config

SNAPSHOT_COLOR = (0, 165, 255)   # orange, as (Blue, Green, Red)


# ---------------------------------------------------------------------------
# The learned entrance
# ---------------------------------------------------------------------------

class LearnedEntrance:
    """Works out where the entry slot is from where bats vanish and appear.

    Points are stored as fractions of the frame (0-1) so they work at any
    camera resolution.
    """

    def __init__(self):
        self.points = []
        if config.ENTRANCE_POINTS_FILE.exists():
            try:
                points = [(float(x), float(y)) for x, y in
                          json.loads(config.ENTRANCE_POINTS_FILE.read_text())]
                # Keep only real positions inside the frame (0-1). This also
                # drops NaN and infinity, which fail these comparisons.
                self.points = [(x, y) for x, y in points if 0 <= x <= 1 and 0 <= y <= 1]
            except (ValueError, TypeError):
                # e.g. the file was cut off by a crash mid-save: just re-learn.
                print("Entrance file was unreadable; re-learning the entrance.")
                self.points = []

    @property
    def is_learned(self) -> bool:
        return len(self.points) >= config.ENTRANCE_LEARN_MIN_EVENTS

    def add(self, fx, fy):
        """Remember one vanish/appear spot and save the list."""
        self.points.append((fx, fy))
        config.ENTRANCE_POINTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.ENTRANCE_POINTS_FILE.write_text(json.dumps(self.points))

    def center_and_radius(self):
        """The entrance's middle and size, as fractions of the frame.

        The median (middle value) is used instead of the average so a few
        odd points don't drag the entrance away from where it really is.
        """
        cx = statistics.median(p[0] for p in self.points)
        cy = statistics.median(p[1] for p in self.points)
        typical_distance = statistics.median(
            math.hypot(px - cx, py - cy) for px, py in self.points)
        radius = max(3 * typical_distance, config.ENTRANCE_MIN_RADIUS)
        return (cx, cy), radius

    def is_near(self, fx, fy) -> bool:
        """True if the spot is near the entrance (always True while still learning)."""
        if not self.is_learned:
            return True
        (cx, cy), radius = self.center_and_radius()
        return math.hypot(fx - cx, fy - cy) <= radius


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@dataclass
class BatEvent:
    """One bat going in or coming out."""

    time: datetime
    direction: str              # "IN" or "OUT"
    track_id: int
    confidence: float           # best detection confidence for this bat
    species: str                # "" if BatNet couldn't name it
    species_confidence: float
    photo: str                  # snapshot file name (in data/events/)


@dataclass
class _BestShot:
    """The clearest frame we've seen of one bat so far."""

    frame: object               # the full camera frame (no drawings)
    bbox: tuple
    confidence: float


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class EntranceMonitor:
    """Watches the tracker's bats and turns finished paths into IN/OUT events."""

    def __init__(self, detector):
        self.detector = detector        # used for classify_species() on snapshots
        self.entrance = LearnedEntrance()
        self.count_in = 0
        self.count_out = 0

        self._tracks = {}               # track_id -> TrackedBat we are watching
        self._first_bbox = {}           # track_id -> box where the bat was first seen
        self._best_shot = {}            # track_id -> _BestShot
        self._frame_size = None         # (width, height)

        self._raw_dir = config.EVENTS_DIR / "raw"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = config.EVENTS_DIR / "events.csv"

    def update(self, tracks, frame) -> list[BatEvent]:
        """
        Call once per video frame with the tracker's current bats and the
        clean (un-drawn-on) camera frame. Returns any new IN/OUT events.
        """
        if frame is not None:
            height, width = frame.shape[:2]
            self._frame_size = (width, height)

        for track in tracks:
            self._tracks[track.track_id] = track
            # Where was this bat when we first saw it?
            self._first_bbox.setdefault(track.track_id, track.bbox)

            # Remember the clearest frame of each bat (highest confidence).
            if frame is None or track.frames_missing > 0:
                continue
            best = self._best_shot.get(track.track_id)
            if best is None or track.confidence > best.confidence:
                self._best_shot[track.track_id] = _BestShot(
                    frame.copy(), track.bbox, track.confidence)

        # Bats the tracker has dropped are finished: decide IN or OUT.
        active_ids = {track.track_id for track in tracks}
        events = []
        for track_id in list(self._tracks):
            if track_id in active_ids:
                continue
            track = self._tracks.pop(track_id)
            first_bbox = self._first_bbox.pop(track_id)
            shot = self._best_shot.pop(track_id, None)
            direction = self._direction(track, first_bbox)
            if direction is not None and shot is not None:
                events.append(self._record(direction, track, shot))
        return events

    def finish(self) -> list[BatEvent]:
        """Call when shutting down. Bats still being tracked are discarded:
        their paths were cut short by quitting, not by the bat going in or
        out, so judging them would invent events."""
        self._tracks.clear()
        self._first_bbox.clear()
        self._best_shot.clear()
        return []

    @property
    def net_inside(self) -> int:
        """Bats that went in minus bats that came out during this run."""
        return self.count_in - self.count_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _at_frame_edge(self, bbox) -> bool:
        """True if the box touches (or is very close to) the edge of the frame."""
        width, height = self._frame_size
        mx = config.EDGE_MARGIN * width
        my = config.EDGE_MARGIN * height
        x1, y1, x2, y2 = bbox
        return x1 <= mx or y1 <= my or x2 >= width - mx or y2 >= height - my

    def _direction(self, track, first_bbox):
        """Return "IN", "OUT" or None for a finished bat path."""
        if len(track.history) < config.MIN_TRACK_DETECTIONS:
            return None   # too short to trust

        appeared_in_middle = not self._at_frame_edge(first_bbox)
        vanished_in_middle = not self._at_frame_edge(track.bbox)   # last box seen

        if vanished_in_middle and not appeared_in_middle:
            direction, (x, y) = "IN", track.history[-1]    # where it vanished
        elif appeared_in_middle and not vanished_in_middle:
            direction, (x, y) = "OUT", track.history[0]    # where it appeared
        else:
            return None   # flew past, or appeared and vanished mid-frame

        width, height = self._frame_size
        fx, fy = x / width, y / height
        if not self.entrance.is_near(fx, fy):
            print(f"  (ignored bat #{track.track_id}: {direction.lower()} too far "
                  "from the learned entrance)")
            return None
        self.entrance.add(fx, fy)
        return direction

    def _record(self, direction, track, shot) -> BatEvent:
        """Count the event, save the snapshot and add a line to the log."""
        if direction == "IN":
            self.count_in += 1
        else:
            self.count_out += 1

        # Only now do we run the (slower) species classifier, once per event.
        # If it fails, still record the event, just without a species.
        try:
            species, species_confidence = self.detector.classify_species(shot.frame, shot.bbox)
        except Exception as e:
            print(f"  (species check failed: {e})")
            species, species_confidence = "", 0.0

        now = datetime.now()
        # Bat IDs restart at 1 each run, so add _2, _3... if the name is taken.
        stem = f"{now:%Y-%m-%d_%H-%M-%S}_{direction}_bat{track.track_id}"
        name = f"{stem}.jpg"
        copy_number = 2
        while (config.EVENTS_DIR / name).exists():
            name = f"{stem}_{copy_number}.jpg"
            copy_number += 1

        # Clean copy for future training, then a copy with the bat boxed.
        cv2.imwrite(str(self._raw_dir / name), shot.frame)
        photo = shot.frame.copy()
        x1, y1, x2, y2 = (int(v) for v in shot.bbox)
        cv2.rectangle(photo, (x1, y1), (x2, y2), SNAPSHOT_COLOR, 3)
        cv2.putText(photo, f"{direction}  bat #{track.track_id}  {now:%H:%M:%S}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, SNAPSHOT_COLOR, 2)
        cv2.imwrite(str(config.EVENTS_DIR / name), photo)

        event = BatEvent(
            time=now,
            direction=direction,
            track_id=track.track_id,
            confidence=shot.confidence,
            species=species,
            species_confidence=species_confidence,
            photo=name,
        )
        self._append_to_log(event)
        return event

    def _append_to_log(self, event: BatEvent) -> None:
        new_file = not self._log_path.exists()
        with open(self._log_path, "a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["time", "direction", "bat_id", "detection_confidence",
                                 "species", "species_confidence", "photo"])
            writer.writerow([
                event.time.isoformat(timespec="seconds"),
                event.direction,
                event.track_id,
                f"{event.confidence:.2f}",
                event.species.replace("_", " / ") or "unknown",
                f"{event.species_confidence:.2f}" if event.species else "",
                event.photo,
            ])
