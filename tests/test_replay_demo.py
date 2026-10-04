"""Camera-free rehearsal tests. No model loading or CUDA is needed."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from tools import replay_demo as replay
from vision import config
from vision.bat_detector import BatDetection


class FakeDetector:
    device = "cpu"

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        # The fixture bat is a bright rectangle; everything else is dark.
        ys, xs = np.where(np.all(frame == 230, axis=2))
        if not len(xs):
            return []
        return [BatDetection((int(xs.min()), int(ys.min()),
                              int(xs.max())+1, int(ys.max())+1), .99, "", 0)]

    def classify_species(self, frame, bbox):
        return "", 0


class ReplayDemoTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((480, 640, 3), 20, np.uint8)
        self.bat = np.full((80, 120, 3), 230, np.uint8)

    def output_folder(self):
        # Keep all output within the authorized rehearsal directory. Do not
        # remove artifacts: other workers may also be using this workspace.
        root = config.PROJECT_ROOT / "data" / "rehearsal"
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix="unittest_", dir=root))

    def test_positions_visibility_and_event_expectations(self):
        expected = {"in": ["IN"], "out": ["OUT"], "flypast": []}
        for scenario, events in expected.items():
            with self.subTest(scenario=scenario):
                clip = replay.make_clip(scenario, self.background, self.bat)
                indices = [i for i, visible in enumerate(clip.visible) if visible]
                self.assertEqual(indices, list(range(30, 78)))
                self.assertEqual(clip.expected_events, events)
                first, last = clip.boxes[30], clip.boxes[77]
                if scenario != "out":
                    self.assertEqual(first[0], 0)
                else:
                    self.assertGreater(first[0], 32)
                    self.assertLess(first[2], 608)
                if scenario != "in":
                    self.assertEqual(last[2], 640)
                else:
                    self.assertGreater(last[0], 32)
                    self.assertLess(last[2], 608)
                xs = [clip.boxes[i][0] for i in indices]
                self.assertEqual(xs, sorted(xs))
                self.assertTrue(all(frame.shape == (480, 640, 3) and
                                    frame.dtype == np.uint8 for frame in clip.frames))
                self.assertTrue(all(box is None for box in clip.boxes[78:]))
                self.assertGreater(len(clip.frames)-78, config.MAX_FRAMES_MISSING)

    def test_negative_labels_and_motion(self):
        for scenario in ("pan", "person", "lamp"):
            clip = replay.make_clip(scenario, self.background, self.bat)
            self.assertFalse(any(clip.visible))
            self.assertEqual(clip.expected_events, [])
            if scenario == "person":
                self.assertNotEqual(clip.boxes[40], clip.boxes[60])
        textured = self.background.copy()
        textured[100:200, 200:300] = 80
        clip = replay.make_clip("pan", textured)
        self.assertFalse(np.array_equal(clip.frames[30], clip.frames[60]))

    def test_fractional_crop(self):
        image = np.arange(100*200*3).reshape(100, 200, 3)
        row = dict(xmin=".25", ymin=".2", xmax=".75", ymax=".8")
        np.testing.assert_array_equal(replay.bat_crop(image, row), image[20:80, 50:150])

    def test_seed_and_validation_only_selection(self):
        root = self.output_folder()
        rows = [
            ["train", "MUST_NOT_BE_READ.jpg", 1, 0, 0, 1, 1],
            ["train", "ALSO_NOT_READ.jpg", 0, "", "", "", ""],
            ["val", "bat.jpg", 1, .1, .1, .9, .9],
            ["val", replay.BACKGROUND_PHOTOS[0], 0, "", "", "", ""],
            ["val", replay.PERSON_PHOTOS[0], 0, "", "", "", ""],
            ["val", replay.LAMP_PHOTO, 0, "", "", "", ""],
        ]
        with (root / "annotations.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "image", "is_bat", "xmin", "ymin", "xmax", "ymax"])
            writer.writerows(rows)
        cv2.imwrite(str(root / "bat.jpg"), self.bat)
        cv2.imwrite(str(root / replay.BACKGROUND_PHOTOS[0]), self.background)
        cv2.imwrite(str(root / replay.PERSON_PHOTOS[0]), self.background)
        cv2.imwrite(str(root / replay.LAMP_PHOTO), self.background)
        first = list(replay.generate_clips(6, 19, root))
        # CSV row order may change while other workers refresh annotations.
        with (root / "annotations.csv").open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["split", "image", "is_bat", "xmin", "ymin", "xmax", "ymax"])
            writer.writerows(reversed(rows))
        second = list(replay.generate_clips(6, 19, root))
        self.assertEqual([c.scenario for c in first], list(replay.SCENARIOS))
        for a, b in zip(first, second):
            self.assertEqual(a.sources, b.sources)
            self.assertEqual(a.sources["split"], "val")
            for frame_a, frame_b in zip(a.frames, b.frames):
                np.testing.assert_array_equal(frame_a, frame_b)

    def test_scoring_math_and_no_detection(self):
        clip = replay.make_clip("in", self.background, self.bat)
        statuses = [False] * len(clip.frames)
        for index in range(32, 78):
            statuses[index] = True
        result = replay.score_clip(clip, statuses, [40] * len(statuses), ["IN"])
        self.assertAlmostEqual(result["detection_pct"], 46 / 48 * 100)
        self.assertEqual(result["latency_frames"], 2)
        self.assertAlmostEqual(result["latency_ms"], 1000 * 2 / 30)
        self.assertEqual(result["average_ms"], 40)
        self.assertEqual(result["fps"], 25)
        self.assertTrue(result["passed"])
        self.assertFalse(result["realtime"])
        statuses = [False] * len(statuses)
        result = replay.score_clip(clip, statuses, [10] * len(statuses), [])
        self.assertEqual(result["detection_pct"], 0)
        self.assertIsNone(result["latency_frames"])
        self.assertFalse(result["passed"])

    def test_false_alarm_denominator_and_event_snapshot_failures(self):
        clip = replay.make_clip("pan", self.background)
        statuses = [False] * len(clip.frames)
        result = replay.score_clip(clip, statuses, [10] * len(statuses), [])
        self.assertTrue(result["passed"])
        statuses[40] = True
        result = replay.score_clip(clip, statuses, [10] * len(statuses), [])
        self.assertAlmostEqual(result["false_alarm_pct"], 100 / len(statuses))
        self.assertFalse(result["passed"])
        statuses[40] = False
        self.assertFalse(replay.score_clip(clip, statuses, [10]*len(statuses), ["IN"])["passed"])
        self.assertFalse(replay.score_clip(clip, statuses, [10]*len(statuses), [], False)["passed"])
        with self.assertRaises(ValueError):
            replay.score_clip(clip, [], [], [])

    def test_preexisting_false_status_cannot_pass_a_bat_clip(self):
        clip = replay.make_clip("flypast", self.background, self.bat)
        statuses = [True] * len(clip.frames)
        result = replay.score_clip(clip, statuses, [10]*len(statuses), [])
        self.assertEqual(result["detection_pct"], 100)
        self.assertEqual(result["latency_frames"], 0)
        self.assertEqual(result["background_false_alarm_pct"], 100)
        self.assertTrue(result["preexisting_status"])
        self.assertFalse(result["passed"])

    def test_normal_tracker_grace_does_not_count_as_background_false_alarm(self):
        clip = replay.make_clip("flypast", self.background, self.bat)
        statuses = [False]*len(clip.frames)
        for index in range(32, 78 + config.MAX_FRAMES_MISSING):
            statuses[index] = True
        result = replay.score_clip(clip, statuses, [10]*len(statuses), [])
        self.assertEqual(result["background_false_alarm_pct"], 0)
        self.assertTrue(result["passed"])

    def test_real_loop_with_fake_detector_and_isolated_outputs(self):
        original_paths = config.EVENTS_DIR, config.ENTRANCE_POINTS_FILE
        root = self.output_folder()
        with patch("torch.cuda.synchronize", side_effect=AssertionError("No GPU in unit tests")):
            for scenario in ("in", "out", "flypast", "pan", "person", "lamp"):
                with self.subTest(scenario=scenario):
                    photo = self.bat if scenario != "person" else np.full_like(self.bat, 60)
                    background = self.background.copy()
                    background[80:160, 80:160] = 90
                    clip = replay.make_clip(scenario, background, photo)
                    detector = FakeDetector()
                    result, samples = replay.run_clip(clip, detector, root / scenario)
                    self.assertTrue(result["passed"], result)
                    self.assertEqual(result["events"], clip.expected_events)
                    self.assertGreater(detector.calls, 0)
                    self.assertGreater(len(samples), 2)
                    self.assertTrue(all(Path(p).is_file() for p in result["snapshots"]))
        self.assertEqual((config.EVENTS_DIR, config.ENTRANCE_POINTS_FILE), original_paths)

    def test_bad_arguments(self):
        with self.assertRaises(ValueError):
            replay.make_clip("unknown", self.background)
        with self.assertRaises(ValueError):
            replay.make_clip("in", self.background)
        with self.assertRaises(ValueError):
            list(replay.generate_clips(0))


if __name__ == "__main__":
    unittest.main()
