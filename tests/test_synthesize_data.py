"""Small fake photos test the synthesizer without touching project data.

A fake "segmenter" (dark pixels = bat) stands in for BatNet's network, so
these tests need no model weights and no GPU.
"""

import csv
import random
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from training import synthesize_data as synth


def fake_segmenter(crop):
    """Pretend every dark pixel is bat."""
    return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 60


class SynthesizeDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        paths = dict(PROJECT_ROOT=self.root, OPENIMAGES_DIR=self.root / "data/openimages",
                     INATURALIST_DIR=self.root / "data/inaturalist",
                     ROOM_NEGATIVES_DIR=self.root / "data/room_negatives",
                     AUTO_LABELS_FILE=self.root / "data/auto_labels.csv",
                     QC_EXCLUDE_FILE=self.root / "data/qc_exclude.txt",
                     SYNTHETIC_DIR=self.root / "data/synthetic",
                     SYNTHETIC_LABELS_FILE=self.root / "data/synthetic/labels.csv")
        settings = patch.multiple(synth.config, **paths)
        settings.start()
        self.addCleanup(settings.stop)
        for key in ("OPENIMAGES_DIR", "INATURALIST_DIR", "ROOM_NEGATIVES_DIR"):
            paths[key].mkdir(parents=True)
        self.bat = self.photo("data/openimages/bat.jpg", (35, 65, 110))
        self.background = self.photo("data/openimages/background.jpg", (100, 120, 140), bat=False)
        self.rows = [["train", "bat.jpg", 1, 0.2, 0.3, 0.8, 0.7],
                     ["train", "background.jpg", 0, "", "", "", ""]]
        self.write_annotations()

    def photo(self, name, colour, bat=True):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = np.full((100, 160, 3), colour, np.uint8)
        if bat:
            # A dark bat-like shape inside the box (0.2, 0.3, 0.8, 0.7).
            cv2.ellipse(pixels, (80, 50), (45, 16), 0, 0, 360, (10, 20, 30), -1)
        synth.write_jpeg(path, pixels)
        return path.resolve()

    def write_annotations(self):
        with (synth.config.OPENIMAGES_DIR / "annotations.csv").open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["split", "image", "is_bat", "xmin", "ymin", "xmax", "ymax"])
            writer.writerows(self.rows)

    def write_auto_labels(self, rows):
        with synth.config.AUTO_LABELS_FILE.open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(synth.LABEL_COLUMNS)
            writer.writerows(rows)

    def labels(self):
        with synth.config.SYNTHETIC_LABELS_FILE.open(newline="") as file:
            return list(csv.DictReader(file))

    def run_synth(self, **kwargs):
        return synth.synthesize(segmenter=fake_segmenter, **kwargs)

    def test_boxes_match_successful_pastes_and_stay_inside_images(self):
        pasted = []
        original = synth.paste_bat

        def record_paste(canvas, cutout, boxes, rng):
            box = original(canvas, cutout, boxes, rng)
            if box is not None:
                height, width = canvas.shape[:2]
                pasted.append(tuple(v / size for v, size in zip(box, [width, height, width, height])))
            return box

        with patch.object(synth, "paste_bat", side_effect=record_paste):
            result = self.run_synth(count=18, seed=7)
        rows = self.labels()
        self.assertEqual(len(rows), len(pasted))
        self.assertEqual(result["boxes"], len(pasted))
        per_image = defaultdict(list)
        for row, expected in zip(rows, pasted):
            box = tuple(float(row[key]) for key in synth.LABEL_COLUMNS[2:6])
            for value, wanted in zip(box, expected):
                self.assertAlmostEqual(value, wanted, places=4)  # CSV keeps 5 decimals
            self.assertTrue(all(0 <= value <= 1 for value in box))
            image = synth.read_image(self.root / row["image"])
            height, width = image.shape[:2]
            self.assertEqual(max(height, width), 640)
            # No specks: every bat is at least MIN_BAT_PIXELS on its shorter side.
            self.assertGreaterEqual(min((box[2] - box[0]) * width, (box[3] - box[1]) * height),
                                    synth.MIN_BAT_PIXELS - 1e-3)
            self.assertEqual(row["source"], "synthetic")
            self.assertEqual(float(row["score"]), 1.0)
            per_image[row["image"]].append(box)
        self.assertEqual(len(per_image), 18)
        for boxes in per_image.values():
            self.assertIn(len(boxes), [1, 2, 3])
            for index, box in enumerate(boxes):
                for other in boxes[:index]:
                    self.assertLess(synth.box_iou([v * 640 for v in box], [v * 640 for v in other]),
                                    synth.MAX_OVERLAP + 0.01)
        preview = synth.read_image(synth.config.SYNTHETIC_DIR / "preview.jpg")
        self.assertEqual(preview.shape[:2], (720, 1280))

    def test_same_seed_produces_identical_jpegs_labels_and_preview(self):
        self.run_synth(count=12, seed=42)
        output = synth.config.SYNTHETIC_DIR
        before = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
        self.run_synth(count=12, seed=42)
        after = {path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        self.run_synth(count=12, seed=43)
        self.assertNotEqual(before[Path("images/synth_00000.jpg")],
                            (output / "images/synth_00000.jpg").read_bytes())

    def test_validation_and_test_photos_are_never_read(self):
        val_bat = self.photo("data/openimages/val_bat.jpg", (1, 2, 3))
        val_background = self.photo("data/openimages/val_background.jpg", (4, 5, 6))
        test_bat = self.photo("data/openimages/test_bat.jpg", (7, 8, 9))
        test_background = self.photo("data/openimages/test background.jpg", (10, 11, 12))
        room_test = self.photo("data/room_negatives/test.jpg", (13, 14, 15))
        room_train = self.photo("data/room_negatives/train.jpg", (16, 17, 18))
        inat_train = self.photo("data/inaturalist/images/1.jpg", (90, 140, 160))
        inat_test = self.photo("data/inaturalist/images/2.jpg", (90, 140, 160))
        inat_copy_of_test = self.photo("data/inaturalist/images/3.jpg", (90, 140, 160))
        self.rows += [["val", "val_bat.jpg", 1, 0.2, 0.3, 0.8, 0.7],
                      ["train", "val_bat.jpg", 1, 0.2, 0.3, 0.8, 0.7],
                      ["val", "val_background.jpg", 0, "", "", "", ""],
                      ["train", "val_background.jpg", 0, "", "", "", ""],
                      ["train", "test_bat.jpg", 1, 0.2, 0.3, 0.8, 0.7],
                      ["train", "test background.jpg", 0, "", "", "", ""]]
        self.write_annotations()
        self.write_auto_labels([["inaturalist", "data/inaturalist/images/1.jpg", 0.2, 0.3, 0.8, 0.7, 0.99],
                                ["inaturalist", "data/inaturalist/images/2.jpg", 0.2, 0.3, 0.8, 0.7, 0.99],
                                ["inaturalist", "data/inaturalist/images/3.jpg", 0.2, 0.3, 0.8, 0.7, 0.99]])
        synth.config.QC_EXCLUDE_FILE.write_text("data/inaturalist/images/3.jpg")
        test_list = synth.config.INATURALIST_DIR / "test_images.txt"
        test_list.write_text("\n".join(path.relative_to(self.root).as_posix()
                                       for path in [test_bat, test_background, room_test, inat_test]))
        with patch.object(synth, "read_image", wraps=synth.read_image) as reader:
            cutouts, backgrounds, _ = synth.load_sources(fake_segmenter)
            self.run_synth(count=12, seed=0)
        # Open Images training bat first, then the (non-test) auto-labelled photo.
        self.assertEqual([cutout.path for cutout in cutouts], [self.bat, inat_train])
        self.assertEqual(set(backgrounds), {self.background, room_train})
        read_paths = {Path(call.args[0]).resolve() for call in reader.call_args_list}
        self.assertTrue(read_paths.isdisjoint(
            {val_bat, val_background, test_bat, test_background, room_test, inat_test,
             inat_copy_of_test}))

    def test_cutout_keeps_only_bat_pixels_and_rejects_bad_boxes(self):
        image = synth.read_image(self.bat)
        pixels, alpha = synth.make_cutout(image, (0.2, 0.3, 0.8, 0.7), fake_segmenter)
        self.assertEqual(pixels.shape[:2], alpha.shape)
        # Trimmed to the ellipse (90 x 32 px) plus a 1-pixel border.
        self.assertTrue(88 <= alpha.shape[1] <= 94 and 30 <= alpha.shape[0] <= 36, alpha.shape)
        self.assertEqual(float(alpha[alpha.shape[0] // 2, alpha.shape[1] // 2]), 1.0)
        self.assertEqual(float(alpha[0, 0]), 0.0)  # corner of an ellipse = background
        for box in [(0, 0, 0.05, 0.1), (0, 0, 1, 1), (-0.1, 0, 0.3, 0.5)]:
            with self.subTest(box=box):
                self.assertIsNone(synth.make_cutout(image, box, fake_segmenter))
        for box in [(0.5, 0.2, 0.3, 0.6), (0, 0, float("nan"), 0.5)]:
            with self.subTest(box=box):
                self.assertIsNone(synth.read_box(dict(zip(synth.LABEL_COLUMNS[2:6], box))))
        # A bat cut off by the photo's edge would look sliced when pasted.
        edge = np.full((100, 160, 3), 150, np.uint8)
        cv2.ellipse(edge, (20, 50), (45, 16), 0, 0, 360, (10, 20, 30), -1)
        self.assertIsNone(synth.make_cutout(edge, (0.0, 0.3, 0.45, 0.7), fake_segmenter))

    def test_paste_changes_only_pixels_around_its_tight_box(self):
        canvas = np.full((200, 300, 3), 200, np.uint8)
        before = canvas.copy()
        alpha = np.zeros((40, 80), np.float32)
        alpha[5:35, 10:70] = 1
        cutout = synth.Cutout(Path("x.jpg"), np.full((40, 80, 3), 10, np.uint8), alpha)
        box = synth.paste_bat(canvas, cutout, [], random.Random(9))
        self.assertIsNotNone(box)
        x1, y1, x2, y2 = box
        changed = np.any(np.abs(canvas.astype(int) - before) > 3, axis=2)
        dark = np.any(canvas < 100, axis=2)
        ys, xs = np.nonzero(dark)
        # The box is the bat's own pixels (within a pixel), not a padded patch.
        self.assertLessEqual(abs(xs.min() - x1), 1)
        self.assertLessEqual(abs(xs.max() + 1 - x2), 1)
        self.assertLessEqual(abs(ys.min() - y1), 1)
        self.assertLessEqual(abs(ys.max() + 1 - y2), 1)
        # Only the softened (and maybe motion-blurred) edge may spill past the box.
        changed[max(0, y1 - 5):y2 + 5, max(0, x1 - 5):x2 + 5] = False
        self.assertFalse(changed.any())

    def test_invisible_paste_is_refused(self):
        canvas = np.full((200, 300, 3), 120, np.uint8)
        before = canvas.copy()
        cutout = synth.Cutout(Path("x.jpg"), np.full((40, 80, 3), 120, np.uint8),
                              np.ones((40, 80), np.float32))
        self.assertIsNone(synth.paste_bat(canvas, cutout, [], random.Random(1)))
        self.assertTrue(np.array_equal(canvas, before))

    def test_smaller_rerun_keeps_only_current_images_without_deleting_files(self):
        self.run_synth(count=12, seed=0)
        old = (synth.config.SYNTHETIC_DIR / "images/synth_00011.jpg").read_bytes()
        note = synth.config.SYNTHETIC_DIR / "keep.txt"
        note.write_text("leave unrelated output alone")
        self.run_synth(count=3, seed=0)
        self.assertEqual(len(list((synth.config.SYNTHETIC_DIR / "images").glob("synth_*.jpg"))), 3)
        self.assertEqual((synth.config.SYNTHETIC_DIR / "previous_images/synth_00011.jpg").read_bytes(), old)
        self.assertEqual(note.read_text(), "leave unrelated output alone")
        self.assertEqual(len({row["image"] for row in self.labels()}), 3)

    def test_invalid_count_or_missing_sources_leaves_output_untouched(self):
        synth.config.SYNTHETIC_DIR.mkdir()
        note = synth.config.SYNTHETIC_DIR / "keep.txt"
        note.write_text("keep me")
        with self.assertRaises(ValueError):
            self.run_synth(count=0)
        self.rows = [self.rows[1]]
        self.write_annotations()
        with self.assertRaises(ValueError):
            self.run_synth(count=1)
        self.assertEqual(note.read_text(), "keep me")
        self.assertFalse(synth.config.SYNTHETIC_LABELS_FILE.exists())


if __name__ == "__main__":
    unittest.main()
