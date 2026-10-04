"""Small, temporary datasets for the CPU-only quality checker.

Run only these tests from the project folder:
    .venv\\Scripts\\python.exe -B -m unittest tests.test_quality_check -v
"""

import csv
import io
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from training import quality_check as qc
from vision import config


class QualityCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        settings = {
            "PROJECT_ROOT": self.root,
            "OPENIMAGES_DIR": self.root / "data/openimages",
            "INATURALIST_DIR": self.root / "data/inaturalist",
            "AUTO_LABELS_FILE": self.root / "data/auto_labels.csv",
            "SYNTHETIC_LABELS_FILE": self.root / "data/synthetic/labels.csv",
            "ROOM_NEGATIVES_DIR": self.root / "data/room_negatives",
            "QC_EXCLUDE_FILE": self.root / "data/qc_exclude.txt",
        }
        for name, value in settings.items():
            override = patch.object(config, name, value)
            override.start()
            self.addCleanup(override.stop)
        self.train = self.make_image("data/openimages/train.png", 1)
        self.val = self.make_image("data/openimages/val.png", 2)
        self.negative = self.make_image("data/openimages/negative.png", 3)
        self.rows = [self.open_row(self.train, "train"),
                     self.open_row(self.val, "val"),
                     self.open_row(self.negative, "train", is_bat="0")]
        self.save_open_rows()

    def make_image(self, relative, seed):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pixels = [20] * 32 + [220] * 32
        random.Random(seed).shuffle(pixels)
        image = Image.new("L", (8, 8))
        image.putdata(pixels)
        image.save(path)
        return path

    def open_row(self, path, split, is_bat="1", box=None):
        if box is None:
            box = (0.1, 0.2, 0.7, 0.8) if is_bat == "1" else ("", "", "", "")
        return dict(zip(qc.OPEN_COLUMNS,
                        (split, path.relative_to(config.OPENIMAGES_DIR).as_posix(),
                         is_bat, *box)))

    def write_csv(self, path, columns, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def save_open_rows(self):
        self.write_csv(config.OPENIMAGES_DIR / "annotations.csv", qc.OPEN_COLUMNS, self.rows)

    def save_labels(self, path, images):
        rows = [dict(zip(qc.LABEL_COLUMNS,
                         ("inaturalist", image.relative_to(self.root).as_posix(),
                          0.1, 0.2, 0.7, 0.8, 0.95))) for image in images]
        self.write_csv(path, qc.LABEL_COLUMNS, rows)

    def save_test_list(self, images):
        config.INATURALIST_DIR.mkdir(parents=True, exist_ok=True)
        (config.INATURALIST_DIR / "test_images.txt").write_text(
            "\n".join(image.relative_to(self.root).as_posix() for image in images),
            encoding="utf-8")

    def test_clean_data_passes(self):
        auto = self.make_image("data/inaturalist/auto.png", 4)
        synthetic = self.make_image("data/synthetic/bat.png", 5)
        held_back = self.make_image("data/inaturalist/test.png", 6)
        self.save_labels(config.AUTO_LABELS_FILE, [auto, auto])
        self.save_labels(config.SYNTHETIC_LABELS_FILE, [synthetic])
        self.save_test_list([held_back])
        report = qc.check_data()
        self.assertEqual(report.exit_code, 0, report.examples)
        self.assertEqual(report.exclusions, set())
        self.assertEqual(len(report.sources["Auto labels"].bats), 1)
        self.assertEqual(report.sources["Auto labels"].boxes, 2)
        self.assertEqual(len(report.sources["Open Images train"].negatives), 1)

    def test_corrupt_image_is_excluded(self):
        self.train.write_bytes(b"not an image")
        report = qc.check_data()
        self.assertEqual(report.problems["broken_image"], 1)
        self.assertEqual(report.exclusions, {self.train})
        self.assertEqual(report.exit_code, 1)

    def test_truncated_jpeg_is_excluded(self):
        # Its header opens successfully, but full decoding must fail.
        photo = self.root / "data/openimages/truncated.jpg"
        Image.new("RGB", (64, 64), "red").save(photo)
        photo.write_bytes(photo.read_bytes()[:-20])
        self.rows[0] = self.open_row(photo, "train")
        self.save_open_rows()
        report = qc.check_data()
        self.assertEqual(report.problems["broken_image"], 1)
        self.assertIn(photo, report.exclusions)

    def test_bad_boxes_are_excluded(self):
        cases = [
            ((-0.1, 0.2, 0.7, 0.8), "out_of_range_box"),
            ((0.8, 0.2, 0.1, 0.8), "reversed_or_empty_box"),
            ((0.1, 0.8, 0.7, 0.2), "reversed_or_empty_box"),
            ((0.1, 0.2, 0.1, 0.8), "reversed_or_empty_box"),
            ((0.1, 0.2, 0.101, 0.8), "tiny_box"),
            ((0, 0, 1, 0.01), "absurd_box"),
            (("nan", 0.2, 0.7, 0.8), "out_of_range_box"),
            ((0.1, 0.2, "inf", 0.8), "out_of_range_box"),
            (("oops", 0.2, 0.7, 0.8), "non_numeric_box"),
            (("", "", "", ""), "missing_box"),
        ]
        for box, kind in cases:
            with self.subTest(box=box):
                self.rows[0] = self.open_row(self.train, "train", box=box)
                self.save_open_rows()
                report = qc.check_data()
                self.assertEqual(report.problems[kind], 1)
                self.assertEqual(report.exclusions, {self.train})

    def test_negative_must_not_have_boxes(self):
        self.rows[2] = self.open_row(self.negative, "train", "0", (0.1, 0.2, 0.7, 0.8))
        self.save_open_rows()
        report = qc.check_data()
        self.assertEqual(report.problems["negative_has_box"], 1)
        self.assertEqual(report.exclusions, {self.negative})

    def test_test_photo_leak_in_auto_and_synthetic_labels(self):
        held_back = self.make_image("data/inaturalist/test.png", 4)
        self.save_test_list([held_back])
        for labels in (config.AUTO_LABELS_FILE, config.SYNTHETIC_LABELS_FILE):
            with self.subTest(labels=labels):
                self.save_labels(labels, [held_back])
                report = qc.check_data()
                self.assertEqual(report.problems["test_photo_leak"], 1)
                self.assertIn(held_back, report.exclusions)
                self.assertEqual(report.exit_code, 1)

    def test_val_photo_in_training_labels_is_excluded(self):
        self.save_labels(config.AUTO_LABELS_FILE, [self.val])
        report = qc.check_data()
        self.assertEqual(report.problems["val_photo_leak"], 1)
        self.assertEqual(report.exclusions, {self.val})

    def test_exact_duplicate_across_splits_excludes_training_copy(self):
        self.train.write_bytes(self.val.read_bytes())
        report = qc.check_data()
        self.assertEqual(report.problems["exact_duplicate_leak"], 1)
        self.assertEqual(report.warnings.get("possible_near_duplicate", 0), 0)
        self.assertEqual(report.exclusions, {self.train})
        self.assertNotIn(self.val, report.exclusions)

    def test_exact_duplicate_of_test_photo_in_synthetic_labels(self):
        held_back = self.make_image("data/inaturalist/test.png", 4)
        copy = self.root / "data/synthetic/copy.png"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(held_back.read_bytes())
        self.save_test_list([held_back])
        self.save_labels(config.SYNTHETIC_LABELS_FILE, [copy])
        report = qc.check_data()
        self.assertEqual(report.problems["exact_duplicate_leak"], 1)
        self.assertEqual(report.exclusions, {copy})

    def test_near_duplicate_at_four_bit_threshold(self):
        with Image.open(self.val) as image:
            pixels = list(image.get_flattened_data())
        # Two light and two dark pixels change sides of the average.
        flipped = [i for i, value in enumerate(pixels) if value == 20][:2]
        flipped += [i for i, value in enumerate(pixels) if value == 220][:2]
        for i in flipped:
            pixels[i] = 240 - pixels[i]
        image = Image.new("L", (8, 8))
        image.putdata(pixels)
        image.save(self.train)
        report = qc.check_data()
        # Near duplicates are only warnings: they're reported, not excluded.
        self.assertEqual(report.warnings["possible_near_duplicate"], 1)
        self.assertEqual(report.exclusions, set())
        self.assertIn("hash distance 4", report.warning_examples["possible_near_duplicate"][0])

    def test_bucket_search_does_not_miss_four_changed_bits(self):
        original = 0xDEADBEEF01234567
        # Changes spread over four pieces still leave the fifth unchanged.
        changed = original ^ sum(1 << bit for bit in (0, 13, 26, 39))
        self.assertTrue(set(qc.hash_buckets(original)) & set(qc.hash_buckets(changed)))

    def test_broken_validation_image_is_reported_but_not_excluded(self):
        self.val.write_bytes(b"corrupt checking image")
        report = qc.check_data()
        self.assertEqual(report.problems["broken_image"], 1)
        self.assertEqual(report.exclusions, set())
        self.assertEqual(report.exit_code, 1)

    def test_missing_training_image_is_excluded(self):
        missing = self.root / "data/openimages/missing.png"
        self.rows[0] = self.open_row(missing, "train")
        self.save_open_rows()
        report = qc.check_data()
        self.assertEqual(report.problems["missing_image"], 1)
        self.assertEqual(report.exclusions, {missing})

    def test_missing_optional_sources_are_allowed(self):
        report = qc.check_data()
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(report.sources["Synthetic labels"].status, "missing")

    def test_partial_synthetic_row_does_not_crash(self):
        self.save_labels(config.SYNTHETIC_LABELS_FILE, [])
        with config.SYNTHETIC_LABELS_FILE.open("a", encoding="utf-8") as stream:
            stream.write("inaturalist,data/synthetic/unfinished.png,0.1,0.2")
        report = qc.check_data()
        self.assertEqual(report.problems["malformed_csv"], 1)
        self.assertEqual(report.exit_code, 1)
        self.assertIn(self.root / "data/synthetic/unfinished.png", report.exclusions)

    def test_partial_synthetic_header_does_not_crash(self):
        config.SYNTHETIC_LABELS_FILE.parent.mkdir(parents=True, exist_ok=True)
        config.SYNTHETIC_LABELS_FILE.write_text("source,image,xmin,ymin", encoding="utf-8")
        report = qc.check_data()
        self.assertEqual(report.problems["malformed_csv"], 1)
        self.assertEqual(report.exit_code, 1)

    def test_unfinished_quoted_csv_does_not_crash(self):
        self.save_labels(config.SYNTHETIC_LABELS_FILE, [])
        with config.SYNTHETIC_LABELS_FILE.open("a", encoding="utf-8") as stream:
            stream.write('inaturalist,"unfinished image')
        report = qc.check_data()
        self.assertEqual(report.problems["malformed_csv"], 1)

    def test_room_split_matches_training_loader(self):
        frames = [self.make_image(f"data/room_negatives/{i:02d}.jpg", 10 + i)
                  for i in range(6)]
        report = qc.check_data()
        self.assertEqual(report.splits[frames[0]], {"val"})
        self.assertEqual(report.splits[frames[1]], {"train"})
        self.assertEqual(report.splits[frames[5]], {"val"})
        self.assertEqual(len(report.sources["Room train"].negatives), 4)

    def test_cli_writes_only_training_copy_relative_to_root(self):
        self.train.write_bytes(self.val.read_bytes())
        output = io.StringIO()
        with redirect_stdout(output):
            code = qc.main(["--write-exclude"])
        self.assertEqual(code, 1)
        self.assertEqual(config.QC_EXCLUDE_FILE.read_text().splitlines(),
                         ["data/openimages/train.png"])
        self.assertIn("exact_duplicate_leak: 1", output.getvalue())

    def test_clean_cli_replaces_stale_exclusions(self):
        config.QC_EXCLUDE_FILE.write_text("stale/path.jpg\n", encoding="utf-8")
        with redirect_stdout(io.StringIO()):
            code = qc.main(["--write-exclude"])
        self.assertEqual(code, 0)
        self.assertEqual(config.QC_EXCLUDE_FILE.read_text(), "")


if __name__ == "__main__":
    unittest.main()
