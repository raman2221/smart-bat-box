r"""Camera-free review checks. All event files are written to temporary folders.

Run from the project root with PYTHONDONTWRITEBYTECODE=1:
    .venv\Scripts\python.exe -m unittest discover -s tests -t . -v
No detector is constructed and no model, GPU, or webcam is used.
"""

import csv
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vision import config
from vision.bat_detector import BatDetection
from vision.bat_tracker import BatTracker, TrackedBat
from vision.entrance_monitor import EntranceMonitor, LearnedEntrance


class FakeDetector:
    def __init__(self):
        self.calls = []

    def classify_species(self, frame, bbox):
        self.calls.append((frame.copy(), bbox))
        return "", 0.0


class EntranceMonitorReviewTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        settings = {
            "EVENTS_DIR": self.root / "events",
            "ENTRANCE_POINTS_FILE": self.root / "entrance_points.json",
            "EDGE_MARGIN": 0.05,
            "MIN_TRACK_DETECTIONS": 3,
            "ENTRANCE_LEARN_MIN_EVENTS": 5,
            "ENTRANCE_MIN_RADIUS": 0.1,
        }
        for name, value in settings.items():
            override = patch.object(config, name, value)
            override.start()
            self.addCleanup(override.stop)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector = FakeDetector()
        self.monitor = EntranceMonitor(self.detector)

    def feed(self, points, monitor=None, track_id=1):
        monitor = monitor or self.monitor
        history = []
        for x, y in points:
            history.append((x, y))
            track = TrackedBat(
                center_x=x, center_y=y,
                track_id=track_id, bbox=(x - 15, y - 15, x + 15, y + 15),
                confidence=0.9, frames_missing=0, history=list(history),
            )
            self.assertEqual(monitor.update([track], self.frame), [])
        return track

    def complete(self, points, monitor=None, track_id=1):
        monitor = monitor or self.monitor
        track = self.feed(points, monitor, track_id)
        # Model normal tracker expiry; no camera, model, or torch dependency.
        track.frames_missing = config.MAX_FRAMES_MISSING + 1
        return monitor.update([], self.frame)

    def test_completed_entry_is_logged_with_two_photos(self):
        events = self.complete([(20, 240), (110, 240), (200, 240)])
        self.assertEqual([event.direction for event in events], ["IN"])
        self.assertEqual(self.monitor.count_in, 1)
        self.assertEqual(self.monitor.count_out, 0)
        self.assertEqual(self.monitor.net_inside, 1)
        self.assertTrue((config.EVENTS_DIR / events[0].photo).is_file())
        self.assertTrue((config.EVENTS_DIR / "raw" / events[0].photo).is_file())
        with (config.EVENTS_DIR / "events.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["direction"], "IN")
        self.assertEqual(rows[0]["bat_id"], "1")
        self.assertEqual(rows[0]["photo"], events[0].photo)
        self.assertEqual(rows[0]["detection_confidence"], "0.90")
        self.assertEqual(rows[0]["species"], "unknown")
        self.assertEqual(rows[0]["species_confidence"], "")
        self.assertEqual(rows[0]["time"], events[0].time.isoformat(timespec="seconds"))
        self.assertEqual(events[0].species, "")
        self.assertEqual(events[0].species_confidence, 0.0)
        raw = cv2.imread(str(config.EVENTS_DIR / "raw" / events[0].photo))
        snapshot = cv2.imread(str(config.EVENTS_DIR / events[0].photo))
        self.assertIsNotNone(raw)
        self.assertIsNotNone(snapshot)
        np.testing.assert_array_equal(raw, self.frame)
        self.assertEqual(snapshot.shape, self.frame.shape)
        self.assertGreater(np.count_nonzero(snapshot), 0)
        np.testing.assert_array_equal(self.frame, np.zeros_like(self.frame))
        self.assertEqual(len(self.detector.calls), 1)
        # Once retired, this track must never generate a duplicate event.
        self.assertEqual(self.monitor.update([], self.frame), [])

    def test_completed_exit(self):
        events = self.complete([(200, 240), (110, 240), (20, 240)])
        self.assertEqual([event.direction for event in events], ["OUT"])
        self.assertEqual(self.monitor.count_out, 1)
        self.assertEqual(self.monitor.count_in, 0)
        self.assertEqual(self.monitor.net_inside, -1)

    def assert_no_event_side_effects(self):
        self.assertEqual((self.monitor.count_in, self.monitor.count_out), (0, 0))
        self.assertEqual(self.monitor.entrance.points, [])
        self.assertEqual(self.detector.calls, [])
        self.assertEqual(list(config.EVENTS_DIR.rglob("*.jpg")), [])
        self.assertFalse((config.EVENTS_DIR / "events.csv").exists())
        self.assertFalse(config.ENTRANCE_POINTS_FILE.exists())

    def test_flypast_is_ignored(self):
        self.assertEqual(self.complete([(20, 240), (320, 240), (620, 240)]), [])
        self.assert_no_event_side_effects()

    def test_mid_to_mid_is_ignored(self):
        self.assertEqual(self.complete([(200, 240), (260, 240), (320, 240)]), [])
        self.assert_no_event_side_effects()

    def test_short_path_is_ignored(self):
        self.assertEqual(self.complete([(20, 240), (200, 240)]), [])
        self.assert_no_event_side_effects()

    def test_short_exit_is_ignored(self):
        self.assertEqual(self.complete([(200, 240), (20, 240)]), [])
        self.assert_no_event_side_effects()

    def test_learning_then_rejecting_far_entry_and_exit(self):
        for track_id in range(1, config.ENTRANCE_LEARN_MIN_EVENTS + 1):
            self.assertFalse(self.monitor.entrance.is_learned)
            # Both directions teach the vanish/appear point, not the edge.
            points = [(20, 240), (160, 240), (320, 240)]
            direction = "IN"
            if track_id % 2 == 0:
                points.reverse()
                direction = "OUT"
            events = self.complete(points, track_id=track_id)
            self.assertEqual([event.direction for event in events], [direction])
        self.assertTrue(self.monitor.entrance.is_learned)
        self.assertEqual(self.monitor.entrance.center_and_radius(), ((0.5, 0.5), 0.1))
        saved_points = json.loads(config.ENTRANCE_POINTS_FILE.read_text())
        self.assertEqual(saved_points, [[0.5, 0.5]] * config.ENTRANCE_LEARN_MIN_EVENTS)
        restored = LearnedEntrance()
        self.assertTrue(restored.is_learned)
        self.assertEqual(restored.points, self.monitor.entrance.points)

        before_counts = (self.monitor.count_in, self.monitor.count_out)
        before_files = {path.relative_to(self.root): path.read_bytes()
                        for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(self.complete([(20, 80), (100, 80), (180, 80)], track_id=20), [])
        self.assertEqual(self.complete([(180, 80), (100, 80), (20, 80)], track_id=21), [])
        self.assertEqual((self.monitor.count_in, self.monitor.count_out), before_counts)
        self.assertEqual(len(self.detector.calls), config.ENTRANCE_LEARN_MIN_EVENTS)
        self.assertEqual({path.relative_to(self.root): path.read_bytes()
                          for path in self.root.rglob("*") if path.is_file()}, before_files)
        # Learning still permits subsequent events close to the entrance.
        self.assertEqual([event.direction for event in self.complete(
            [(20, 240), (160, 240), (330, 240)], track_id=22)], ["IN"])

    def test_tracker_and_monitor_keep_one_track_through_missing_frames(self):
        tracker = BatTracker(max_distance=100, max_frames_missing=4)

        def step(x=None):
            detections = [] if x is None else [BatDetection(
                bbox=(x - 15, 225, x + 15, 255), confidence=0.9,
                species="", species_confidence=0.0)]
            tracks = tracker.update(detections)
            return tracks, self.monitor.update(tracks, self.frame)

        for x in (20, 100):
            tracks, events = step(x)
            self.assertEqual(events, [])
            self.assertEqual([track.track_id for track in tracks], [1])
        for missing in range(1, 4):
            tracks, events = step()
            self.assertEqual(events, [])
            self.assertEqual(tracks[0].frames_missing, missing)
            self.assertEqual(len(tracks[0].history), 2)
        tracks, events = step(180)
        self.assertEqual(events, [])
        self.assertEqual(tracks[0].track_id, 1)
        self.assertEqual(tracks[0].frames_missing, 0)
        self.assertEqual(tracks[0].history, [(20, 240), (100, 240), (180, 240)])
        for missing in range(1, tracker.max_frames_missing + 1):
            tracks, events = step()
            self.assertEqual(events, [])
            self.assertEqual(tracks[0].frames_missing, missing)
        tracks, events = step()
        self.assertEqual(tracks, [])
        self.assertEqual([(event.direction, event.track_id) for event in events], [("IN", 1)])
        self.assertEqual(self.monitor.count_in, 1)
        self.assertEqual(len(self.detector.calls), 1)
        self.assertEqual(step(), ([], []))

    def test_shutdown_does_not_turn_visible_track_into_entry(self):
        self.feed([(20, 240), (110, 240), (200, 240)])
        self.assertEqual(self.monitor.finish(), [])
        self.assert_no_event_side_effects()
        self.assertEqual(self.monitor.finish(), [])
        self.assertEqual(self.monitor.update([], self.frame), [])

    def test_shutdown_does_not_turn_visible_track_into_exit(self):
        self.feed([(200, 240), (110, 240), (20, 240)])
        self.assertEqual(self.monitor.finish(), [])
        self.assert_no_event_side_effects()
        self.assertEqual(self.monitor.update([], self.frame), [])

    def test_malformed_saved_json_does_not_crash_startup(self):
        config.ENTRANCE_POINTS_FILE.write_text("{unfinished", encoding="utf-8")
        entrance = LearnedEntrance()
        self.assertFalse(entrance.is_learned)
        self.assertEqual(entrance.points, [])
        restarted = EntranceMonitor(FakeDetector())
        self.assertEqual([event.direction for event in self.complete(
            [(20, 240), (110, 240), (200, 240)], monitor=restarted)], ["IN"])
        self.assertEqual(json.loads(config.ENTRANCE_POINTS_FILE.read_text()), [[0.3125, 0.5]])

    # Nonfinite/out-of-range saved points must be dropped, or they could mark
    # an invalid entrance as learned and reject valid events.
    def test_invalid_saved_coordinates_are_not_used(self):
        # All values are float-convertible: a bad string would hide the bug
        # by making the implementation discard the entire list.
        invalid = [[float("nan"), 0.5], [float("inf"), 0.5], [-0.1, 0.5],
                   [1.1, 0.5], [0.5, -0.2]]
        config.ENTRANCE_POINTS_FILE.write_text(json.dumps(invalid), encoding="utf-8")
        entrance = LearnedEntrance()
        self.assertFalse(entrance.is_learned)
        self.assertEqual(entrance.points, [])

    # Bug: vision/entrance_monitor.py:251 ignores imwrite(False); an event is
    # counted/logged and teaches the entrance even though neither photo exists.
    @unittest.expectedFailure
    def test_failed_photo_write_does_not_count_or_teach_entrance(self):
        with patch("vision.entrance_monitor.cv2.imwrite", return_value=False):
            try:
                events = self.complete([(20, 240), (110, 240), (200, 240)])
            except OSError:
                events = []
        self.assertEqual(events, [])
        self.assertEqual(self.monitor.count_in, 0)
        self.assertEqual(self.monitor.entrance.points, [])
        self.assertFalse((config.EVENTS_DIR / "events.csv").exists())

    # Bug: vision/entrance_monitor.py:240 increments the count (and :234 has
    # already learned the point) before the CSV open at :273 can fail.
    @unittest.expectedFailure
    def test_failed_csv_write_does_not_count_or_teach_entrance(self):
        # A directory where the CSV should be makes actual file opening fail.
        (config.EVENTS_DIR / "events.csv").mkdir()
        try:
            self.complete([(20, 240), (110, 240), (200, 240)])
        except OSError:
            pass
        self.assertEqual(self.monitor.count_in, 0)
        self.assertEqual(self.monitor.entrance.points, [])

    # A classifier error must not abort the event; it is recorded without a species.
    def test_classifier_failure_still_records_unknown_species(self):
        with patch.object(self.monitor.detector, "classify_species",
                          side_effect=RuntimeError("classifier unavailable")):
            events = self.complete([(20, 240), (110, 240), (200, 240)])
        self.assertEqual([event.direction for event in events], ["IN"])
        self.assertEqual(events[0].species, "")
        self.assertEqual(self.monitor.count_in, 1)

    # Track IDs reset on restart, so two runs in one second must not
    # overwrite each other's photos.
    def test_restart_same_second_does_not_overwrite_photo(self):
        with patch("vision.entrance_monitor.datetime") as clock:
            clock.now.return_value = datetime(2026, 10, 3, 12, 0, 0)
            first = self.complete([(20, 240), (110, 240), (200, 240)])[0]
            restarted = EntranceMonitor(FakeDetector())
            second = self.complete([(20, 240), (110, 240), (200, 240)], restarted)[0]
        self.assertNotEqual(first.photo, second.photo)
        self.assertTrue((config.EVENTS_DIR / first.photo).exists())
        self.assertTrue((config.EVENTS_DIR / second.photo).exists())


if __name__ == "__main__":
    unittest.main()
