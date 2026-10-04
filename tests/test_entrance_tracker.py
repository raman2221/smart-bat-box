"""Camera-free tracker tests: no BatNet construction, model loading, or GPU.

Run with PYTHONDONTWRITEBYTECODE=1 so imports do not write outside tests/.
All event files and learned entrance points live in temporary folders here.
"""

import ast
import csv
import importlib
import io
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from vision import config, entrance_tracker
from vision.bat_detector import BatDetection
from vision.bat_tracker import BatTracker
from vision.entrance_monitor import EntranceMonitor


def detection(x=200):
    return BatDetection(
        bbox=(x - 15, 225, x + 15, 255), confidence=0.9,
        species="", species_confidence=0.0,
    )


class FakeDetector:
    device = "cpu"

    def __init__(self, script):
        self.script = iter(script)
        self.detect_calls = 0
        self.classify_calls = 0

    def detect(self, frame):
        self.detect_calls += 1
        return next(self.script)

    def classify_species(self, frame, bbox):
        self.classify_calls += 1
        return "", 0.0


class FakeCapture:
    def __init__(self, count):
        self.frames = [np.zeros((480, 640, 3), dtype=np.uint8)
                       for _ in range(count)]
        self.read_calls = 0
        self.release_calls = 0

    def isOpened(self):
        return True

    def read(self):
        self.read_calls += 1
        if self.read_calls <= len(self.frames):
            return True, self.frames[self.read_calls - 1]
        return False, None

    def release(self):
        self.release_calls += 1


class FakeBackgroundSubtractor:
    def apply(self, frame):
        # Force motion deterministically so every scripted detection is used.
        # OpenCV's threshold, morphology, and contour operations remain real.
        return np.full(frame.shape[:2], 255, dtype=np.uint8)


class EntranceTrackerTests(unittest.TestCase):
    def setUp(self):
        minimum = patch.object(config, "MIN_TRACK_DETECTIONS", 3)
        minimum.start()
        self.addCleanup(minimum.stop)
        self.tracker = BatTracker(max_distance=100, max_frames_missing=3)

    def test_import_has_no_camera_or_model_side_effects(self):
        # Reload executes the actual module body even if discovery imported it.
        with ExitStack() as stack:
            guards = [stack.enter_context(patch(target, side_effect=AssertionError(
                "Import must not construct a detector, open a camera, or use GPU/model APIs"
            ))) for target in (
                "vision.bat_detector.BatDetector.__init__",
                "vision.entrance_tracker.cv2.VideoCapture",
                "torch.package.PackageImporter", "torch.load",
                "torch.cuda.is_available", "torch.cuda.init",
            )]
            imported = importlib.reload(importlib.import_module("vision.entrance_tracker"))
            self.assertIs(imported, entrance_tracker)
            self.assertTrue(callable(imported.main))
            self.assertTrue(callable(imported.bat_in_view))
            self.assertTrue(callable(imported.draw_status))
            for guard in guards:
                guard.assert_not_called()

    def test_no_references_to_removed_overlay_or_time_names(self):
        removed = {"draw_overlay", "banner", "fps", "IN_COLOR", "OUT_COLOR",
                   "ZONE_COLOR", "time"}
        tree = ast.parse(Path(entrance_tracker.__file__).read_text(encoding="utf-8"))
        references = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                references.add(node.id)
            elif isinstance(node, ast.Attribute):
                # event.time is the event's datetime, not the removed time module.
                if node.attr != "time":
                    references.add(node.attr)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                references.add(node.name)
            elif isinstance(node, ast.arg):
                references.add(node.arg)
            elif isinstance(node, ast.Import):
                references.update(alias.name.split(".")[0] for alias in node.names)
                references.update(alias.asname for alias in node.names if alias.asname)
            elif isinstance(node, ast.ImportFrom):
                references.add((node.module or "").split(".")[0])
                references.update(alias.name for alias in node.names)
                references.update(alias.asname for alias in node.names if alias.asname)
        self.assertEqual(references & removed, set())
        self.assertEqual(set(vars(entrance_tracker)) & removed, set())

    def test_bat_in_view_is_false_without_tracks(self):
        self.assertIs(entrance_tracker.bat_in_view(self.tracker.update([])), False)

    def test_bat_in_view_is_false_below_detection_threshold(self):
        for count in range(1, config.MIN_TRACK_DETECTIONS):
            tracks = self.tracker.update([detection()])
            self.assertEqual(len(tracks), 1)
            self.assertEqual(len(tracks[0].history), count)
            self.assertIs(entrance_tracker.bat_in_view(tracks), False)

    def test_bat_in_view_becomes_true_at_configured_threshold(self):
        for minimum in (3, 5):
            with self.subTest(minimum=minimum), patch.object(
                    config, "MIN_TRACK_DETECTIONS", minimum):
                tracker = BatTracker()
                for count in range(1, minimum + 1):
                    tracks = tracker.update([detection()])
                    self.assertEqual(len(tracks[0].history), count)
                    self.assertIs(entrance_tracker.bat_in_view(tracks), count >= minimum)

    def test_bat_in_view_stays_true_during_missing_frames_then_expires(self):
        for _ in range(config.MIN_TRACK_DETECTIONS):
            tracks = self.tracker.update([detection()])
        track_id = tracks[0].track_id
        for missing in range(1, self.tracker.max_frames_missing + 1):
            tracks = self.tracker.update([])
            self.assertEqual([track.track_id for track in tracks], [track_id])
            self.assertEqual(tracks[0].frames_missing, missing)
            self.assertEqual(len(tracks[0].history), config.MIN_TRACK_DETECTIONS)
            self.assertIs(entrance_tracker.bat_in_view(tracks), True)
        tracks = self.tracker.update([])
        self.assertEqual(tracks, [])
        self.assertIs(entrance_tracker.bat_in_view(tracks), False)

    def assert_status_drawn(self, present, text):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        before = frame.copy()
        with patch.object(cv2, "putText", wraps=cv2.putText) as put_text:
            entrance_tracker.draw_status(frame, present)
        put_text.assert_called_once()
        self.assertIs(put_text.call_args.args[0], frame)
        self.assertEqual(put_text.call_args.args[1], text)
        self.assertEqual(frame.shape, before.shape)
        self.assertEqual(frame.dtype, before.dtype)
        self.assertTrue(np.any(frame != before))

    def test_draw_status_bat_detected_changes_given_frame(self):
        self.assert_status_drawn(True, "BAT DETECTED")

    def test_draw_status_not_a_bat_changes_given_frame(self):
        self.assert_status_drawn(False, "NOT A BAT")


class EntranceTrackerMainTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        settings = {
            "EVENTS_DIR": self.root / "events",
            "ENTRANCE_POINTS_FILE": self.root / "entrance_points.json",
            "MIN_TRACK_DETECTIONS": 3,
            "MAX_TRACK_DISTANCE": 100,
            "MAX_FRAMES_MISSING": 2,
            "EDGE_MARGIN": 0.05,
            "ENTRANCE_LEARN_MIN_EVENTS": 5,
        }
        for name, value in settings.items():
            override = patch.object(config, name, value)
            override.start()
            self.addCleanup(override.stop)

    def exercise_main(self, script, statuses, quit_on_last_frame=False):
        detector = FakeDetector(script)
        capture = FakeCapture(len(script))
        # Keep the real monitor, including finish() and temporary persistence.
        monitor = EntranceMonitor(detector)
        shown = []
        output = io.StringIO()
        keys = [-1] * len(script)
        if quit_on_last_frame:
            keys[-1] = ord("q")
        with ExitStack() as stack:
            construct = stack.enter_context(patch.object(
                entrance_tracker, "BatDetector", return_value=detector))
            stack.enter_context(patch.object(
                entrance_tracker, "EntranceMonitor", return_value=monitor))
            video = stack.enter_context(patch.object(
                cv2, "VideoCapture", return_value=capture))
            stack.enter_context(patch.object(cv2, "createBackgroundSubtractorMOG2",
                                            return_value=FakeBackgroundSubtractor()))
            imshow = stack.enter_context(patch.object(
                cv2, "imshow", side_effect=lambda name, frame: shown.append(frame.copy())))
            wait_key = stack.enter_context(patch.object(cv2, "waitKey", side_effect=keys))
            destroy = stack.enter_context(patch.object(cv2, "destroyAllWindows"))
            finish = stack.enter_context(patch.object(monitor, "finish", wraps=monitor.finish))
            updates = stack.enter_context(patch.object(monitor, "update", wraps=monitor.update))
            status = stack.enter_context(patch.object(
                entrance_tracker, "draw_status", wraps=entrance_tracker.draw_status))
            stack.enter_context(redirect_stdout(output))
            self.assertIsNone(entrance_tracker.main())

        construct.assert_called_once_with()
        video.assert_called_once_with(entrance_tracker.CAMERA_INDEX)
        finish.assert_called_once_with()
        destroy.assert_called_once_with()
        self.assertEqual(capture.release_calls, 1)
        self.assertEqual(capture.read_calls, len(script) + (not quit_on_last_frame))
        self.assertEqual(detector.detect_calls, len(script))
        self.assertEqual(updates.call_count, len(script))
        self.assertEqual(imshow.call_count, len(script))
        self.assertEqual(wait_key.call_count, len(script))
        self.assertEqual([call.args[1] for call in status.call_args_list], statuses)
        for call, original, display, present in zip(
                imshow.call_args_list, capture.frames, shown, statuses):
            self.assertEqual(call.args[0], entrance_tracker.WINDOW)
            self.assertIsNot(call.args[1], original)
            np.testing.assert_array_equal(original, np.zeros_like(original))
            expected = original.copy()
            entrance_tracker.draw_status(expected, present)
            # The whole displayed frame must contain only the status overlay.
            np.testing.assert_array_equal(display, expected)
        self.assertIn("Tracker stopped.", output.getvalue())
        # Shutdown cannot produce another event, even if called a second time.
        counts = (monitor.count_in, monitor.count_out)
        self.assertEqual(monitor.finish(), [])
        self.assertEqual(monitor.update([], capture.frames[-1]), [])
        self.assertEqual((monitor.count_in, monitor.count_out), counts)
        return monitor, detector, output.getvalue()

    def assert_no_events(self, monitor, detector, output):
        self.assertEqual((monitor.count_in, monitor.count_out), (0, 0))
        self.assertEqual(detector.classify_calls, 0)
        self.assertEqual(monitor.entrance.points, [])
        self.assertEqual(list(config.EVENTS_DIR.rglob("*.jpg")), [])
        self.assertFalse((config.EVENTS_DIR / "events.csv").exists())
        self.assertFalse(config.ENTRANCE_POINTS_FILE.exists())
        self.assertNotIn("Bat went", output)

    def test_main_exits_on_end_of_fake_capture_without_detections(self):
        result = self.exercise_main([[], [], []], [False, False, False])
        self.assert_no_events(*result)

    def test_main_end_of_capture_does_not_count_visible_entry(self):
        script = [[detection(x)] for x in (20, 110, 200)]
        result = self.exercise_main(script, [False, False, True])
        self.assert_no_events(*result)

    def test_main_end_of_capture_does_not_count_visible_exit(self):
        script = [[detection(x)] for x in (200, 110, 20)]
        result = self.exercise_main(script, [False, False, True])
        self.assert_no_events(*result)

    def test_main_q_shutdown_does_not_count_visible_bat(self):
        script = [[detection(x)] for x in (20, 110, 200)]
        result = self.exercise_main(script, [False, False, True], quit_on_last_frame=True)
        self.assert_no_events(*result)

    def test_main_prints_completed_event_and_displays_only_status(self):
        script = [[detection(x)] for x in (20, 110, 200)] + [[], [], []]
        monitor, detector, output = self.exercise_main(
            script, [False, False, True, True, True, False])
        self.assertEqual((monitor.count_in, monitor.count_out), (1, 0))
        self.assertEqual(detector.classify_calls, 1)
        self.assertEqual(output.count("Bat went IN -> photo saved:"), 1)
        self.assertNotIn("Bat went OUT", output)
        with (config.EVENTS_DIR / "events.csv").open(newline="") as file:
            rows = list(csv.DictReader(file))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["direction"], "IN")
        self.assertEqual(len(list(config.EVENTS_DIR.rglob("*.jpg"))), 2)


if __name__ == "__main__":
    unittest.main()
