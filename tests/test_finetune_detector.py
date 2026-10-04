"""Small CPU-only checks for the training helpers; no model weights needed."""

import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from training import cache_images
from training import finetune_detector as trainer


class CountMatchesTests(unittest.TestCase):
    def test_one_detection_cannot_count_as_two_bats(self):
        true = [[0., 0., 10., 10.], [0., 0., 10., 10.]]
        self.assertEqual(trainer.count_matches(true, true[:1]), 1)

    def test_two_detections_cannot_count_the_same_bat_twice(self):
        box = [[0., 0., 10., 10.]]
        self.assertEqual(trainer.count_matches(box, box * 2), 1)

    def test_separate_matches_and_unmatched_detection(self):
        true = [[0., 0., 10., 10.], [20., 20., 30., 30.]]
        pred = true + [[50., 50., 60., 60.]]
        self.assertEqual(trainer.count_matches(true, pred), 2)

    def test_best_overlap_is_used_first(self):
        true = [[0., 0., 10., 10.], [5., 0., 15., 10.]]
        pred = [[0., 0., 10., 10.], [0., 0., 9., 10.]]
        self.assertEqual(trainer.count_matches(true, pred, min_iou=0.3), 1)

    def test_threshold_is_inclusive_and_empty_boxes_work(self):
        box = [[0., 0., 10., 10.]]
        half = [[0., 0., 5., 10.]]
        self.assertEqual(trainer.count_matches(box, half), 1)
        self.assertEqual(trainer.count_matches(box, half, min_iou=0.6), 0)
        self.assertEqual(trainer.count_matches([], box), 0)
        self.assertEqual(trainer.count_matches(box, []), 0)


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        paths = {
            "PROJECT_ROOT": self.root,
            "OPENIMAGES_DIR": self.root / "data/openimages",
            "INATURALIST_DIR": self.root / "data/inaturalist",
            "AUTO_LABELS_FILE": self.root / "data/auto_labels.csv",
            "SYNTHETIC_LABELS_FILE": self.root / "data/synthetic/labels.csv",
            "QC_EXCLUDE_FILE": self.root / "data/qc_exclude.txt",
            "ROOM_NEGATIVES_DIR": self.root / "data/room_negatives",
        }
        config_patch = patch.multiple(trainer.config, **paths)
        config_patch.start()
        self.addCleanup(config_patch.stop)
        for path in paths.values():
            if path.suffix:
                path.parent.mkdir(parents=True, exist_ok=True)
            else:
                path.mkdir(parents=True, exist_ok=True)
        self.open_fields = ["split", "image", "is_bat", "xmin", "ymin", "xmax", "ymax"]
        self.extra_fields = ["source", "image", "xmin", "ymin", "xmax", "ymax", "score"]

    def photo(self, name):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path)
        return path.resolve()

    def write_csv(self, path, fields, rows):
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            writer.writerows(rows)

    def load(self):
        with redirect_stdout(io.StringIO()) as output:
            data = trainer.load_annotations()
        self.assertIn("skipped", output.getvalue())
        return data

    def test_filters_every_source_and_synthetic_is_training_only(self):
        config = trainer.config
        bat = self.photo("data/openimages/bat.jpg")
        val = self.photo("data/openimages/val.jpg")
        qc = self.photo("data/openimages/qc.jpg")
        held = self.photo("data/openimages/held.jpg")
        negative = self.photo("data/openimages/negative.jpg")
        auto = self.photo("data/inaturalist/auto.jpg")
        synthetic = self.photo("data/synthetic/bat with space.jpg")
        excluded_synthetic = self.photo("data/synthetic/qc.jpg")
        rooms = [self.photo(f"data/room_negatives/{i}.jpg") for i in range(6)]
        self.write_csv(config.OPENIMAGES_DIR / "annotations.csv", self.open_fields, [
            ["train", "bat.jpg", 1, .1, .2, .4, .5],
            ["val", "val.jpg", 1, .1, .2, .4, .5],
            ["train", "qc.jpg", 1, .1, .2, .4, .5],
            ["train", "held.jpg", 1, .1, .2, .4, .5],
            ["train", "negative.jpg", 0, "", "", "", ""],
        ])
        config.QC_EXCLUDE_FILE.write_text(
            "data/openimages/./qc.jpg\ndata/synthetic/qc.jpg\ndata/room_negatives/0.jpg\n")
        (config.INATURALIST_DIR / "test_images.txt").write_text(
            "data/openimages/held.jpg\ndata/room_negatives/1.jpg\n\n")
        self.write_csv(config.AUTO_LABELS_FILE, self.extra_fields, [
            ["auto", "data/inaturalist/auto.jpg", .1, .2, .4, .5, .9],
            ["auto", "data/openimages/held.jpg", .1, .2, .4, .5, .9],
            ["auto", "data/openimages/qc.jpg", .1, .2, .4, .5, .9],
            ["auto", "data/openimages/val.jpg", .1, .2, .4, .5, .9],
        ])
        self.write_csv(config.SYNTHETIC_LABELS_FILE, self.extra_fields, [
            ["synthetic", "data/synthetic/bat with space.jpg", .1, .2, .4, .5, 1],
            ["synthetic", "data/synthetic/bat with space.jpg", .5, .6, .8, .9, 1],
            ["synthetic", "data/synthetic/qc.jpg", .1, .2, .4, .5, 1],
            ["synthetic", "data/openimages/held.jpg", .1, .2, .4, .5, 1],
            ["synthetic", "data/openimages/val.jpg", .1, .2, .4, .5, 1],
            ["synthetic", "data/room_negatives/5.jpg", .1, .2, .4, .5, 1],
        ])
        data = self.load()
        train_paths = [path for path, _ in data["train"]]
        val_paths = [path for path, _ in data["val"]]
        self.assertEqual(train_paths.count(bat), trainer.HAND_LABELLED_BAT_REPEATS)
        self.assertIn(negative, train_paths)
        self.assertIn(auto, train_paths)
        self.assertIn(synthetic, train_paths)
        self.assertEqual(len(dict(data["train"])[synthetic]), 2)
        self.assertEqual(set(val_paths), {val, rooms[5]})
        self.assertFalse(set(train_paths) & set(val_paths))
        self.assertFalse({qc, held, excluded_synthetic, rooms[0], rooms[1]} &
                         set(train_paths + val_paths))

    def test_missing_optional_files_and_unfinished_rows(self):
        config = trainer.config
        self.write_csv(config.OPENIMAGES_DIR / "annotations.csv", self.open_fields, [])
        self.assertEqual(self.load(), {"train": [], "val": []})
        photo = self.photo("data/synthetic/ready.jpg")
        self.write_csv(config.SYNTHETIC_LABELS_FILE, self.extra_fields, [
            ["synthetic", "data/synthetic/ready.jpg", .1, .2, .4, .5, 1],
            ["synthetic", "data/synthetic/missing.jpg", .1, .2, .4, .5, 1],
            ["synthetic", "data/synthetic/ready.jpg", "nan", .2, .4, .5, 1],
            ["synthetic", "data/synthetic/ready.jpg", .4, .2, .1, .5, 1],
            ["synthetic", "data/synthetic/ready.jpg", .1],
        ])
        self.assertEqual(self.load(), {"train": [(photo, [[.1, .2, .4, .5]])], "val": []})
        config.SYNTHETIC_LABELS_FILE.write_text("source,image,xmin")
        self.assertEqual(self.load(), {"train": [], "val": []})


class ZoomOutTests(unittest.TestCase):
    def test_boxes_stay_inside_and_at_least_four_pixels(self):
        image = np.zeros((trainer.IMAGE_SIZE, trainer.IMAGE_SIZE, 3), dtype=np.uint8)
        boxes = torch.tensor([[0., 0., 8., 12.], [270., 280., 320., 320.]])
        # Always choose the strongest permitted zoom, including its boundary.
        with patch.object(trainer.random, "uniform", side_effect=lambda low, high: low):
            for _ in range(30):
                result, zoomed = trainer.zoom_out(image, boxes)
                self.assertEqual(result.shape, image.shape)
                self.assertTrue((zoomed >= 0).all())
                self.assertTrue((zoomed <= trainer.IMAGE_SIZE).all())
                self.assertTrue(((zoomed[:, 2:] - zoomed[:, :2]) >= trainer.MIN_BAT_PIXELS).all())
        self.assertEqual(boxes[0, 2].item(), 8.)   # input boxes aren't changed

    def test_already_tiny_bats_are_not_shrunk(self):
        image = np.zeros((trainer.IMAGE_SIZE, trainer.IMAGE_SIZE, 3), dtype=np.uint8)
        boxes = torch.tensor([[10., 10., 14., 14.]])
        result, zoomed = trainer.zoom_out(image, boxes)
        self.assertIs(result, image)
        self.assertTrue(torch.equal(zoomed, boxes))

    def test_negative_image_keeps_empty_boxes(self):
        image = np.zeros((trainer.IMAGE_SIZE, trainer.IMAGE_SIZE, 3), dtype=np.uint8)
        result, boxes = trainer.zoom_out(image, torch.empty((0, 4)))
        self.assertEqual(result.shape, image.shape)
        self.assertEqual(boxes.shape, (0, 4))


class CPUTrainingTests(unittest.TestCase):
    def test_cpu_training_step_does_not_use_autocast_or_scaler(self):
        class TinyDetector(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, images, targets):
                return {"loss": (self.weight * images.mean()) ** 2}

        detector = TinyDetector()
        optimizer = torch.optim.SGD(detector.parameters(), lr=.1)
        targets = [{"boxes": torch.empty((0, 4)), "labels": torch.empty(0, dtype=torch.int64)}]
        with patch.object(torch, "autocast", side_effect=AssertionError("CPU autocast")), \
                patch.object(torch.amp, "GradScaler", side_effect=AssertionError("CPU scaler")):
            loss = trainer.training_step(detector, optimizer, torch.ones((1, 3, 8, 8)), targets, "cpu")
        self.assertEqual(loss, 1.)
        self.assertLess(detector.weight.item(), 1.)

    def test_zero_worker_loader_omits_worker_only_options(self):
        loader = trainer.make_loader([], False, batch_size=8, workers=0, prefetch_factor=2)
        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.persistent_workers)
        self.assertIsNone(loader.prefetch_factor)
        self.assertTrue(loader.pin_memory)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cache = self.root / "cache"
        rng = np.random.default_rng(0)
        self.photo = self.root / "photo.jpg"
        Image.fromarray(rng.integers(0, 255, (50, 70, 3), dtype=np.uint8)).save(self.photo)

    def test_cached_pixels_match_the_old_decoding(self):
        from torchvision.transforms.functional import to_tensor
        cached = cache_images.ensure_cached([self.photo], cache_dir=self.cache, quiet=True)
        old = Image.open(self.photo).convert("RGB").resize((320, 320))
        self.assertTrue(np.array_equal(cache_images.read_pixels(cached[self.photo]), np.array(old)))
        image, _ = trainer.BatPhotos([(cached[self.photo], [])], augment=False)[0]
        self.assertEqual(image.dtype, torch.uint8)
        self.assertTrue(torch.equal(trainer.as_float_images(image, "cpu"), to_tensor(old)))

    def test_changed_photo_is_cached_again_and_old_copy_removed(self):
        first, made = cache_images.cache_one(self.photo, cache_dir=self.cache)
        self.assertTrue(made)
        self.assertEqual(cache_images.cache_one(self.photo, cache_dir=self.cache), (first, False))
        Image.new("RGB", (30, 30), "red").save(self.photo)
        os.utime(self.photo, ns=(1, 2_000_000_000))
        second, made = cache_images.cache_one(self.photo, cache_dir=self.cache)
        self.assertTrue(made)
        self.assertNotEqual(first, second)
        self.assertFalse(first.exists())
        self.assertTrue(np.array_equal(cache_images.read_pixels(second),
                                       cache_images.decode_photo(self.photo)))
        self.assertGreater(cache_images.read_pixels(second)[0, 0, 0], 250)   # now red

    def test_missing_or_broken_photos_are_skipped(self):
        broken = self.root / "broken.jpg"
        broken.write_bytes(b"not a jpeg")
        cached = cache_images.ensure_cached([self.photo, broken, self.root / "gone.jpg"],
                                            cache_dir=self.cache, quiet=True)
        self.assertEqual(list(cached), [self.photo])
        items = [(self.photo, [[.1, .1, .5, .5]]), (broken, [])]
        self.assertEqual(trainer.use_cache(items, cached),
                         [(cached[self.photo], [[.1, .1, .5, .5]])])


class BalancedSamplerTests(unittest.TestCase):
    def items(self, bats, others):
        return ([(f"bat{i}", [[.1, .1, .5, .5]]) for i in range(bats)] +
                [(f"other{i}", []) for i in range(others)])

    def test_epoch_size_and_ratio(self):
        items = self.items(300, 50)
        for fraction in (0.5, 0.25):
            sampler = trainer.BalancedSampler(items, epoch_size=200, bat_fraction=fraction, seed=1)
            for _ in range(3):
                chosen = list(sampler)
                self.assertEqual(len(chosen), 200)
                self.assertEqual(len(sampler), 200)
                self.assertEqual(sum(1 for i in chosen if items[i][1]), round(200 * fraction))

    def test_every_photo_dealt_before_repeats(self):
        items = self.items(30, 20)
        sampler = trainer.BalancedSampler(items, epoch_size=20, bat_fraction=0.5, seed=2)
        three_epochs = [i for _ in range(3) for i in sampler]   # 30 bats, 30 others
        self.assertEqual(sorted(i for i in three_epochs if items[i][1]), list(range(30)))
        others = [i for i in three_epochs if not items[i][1]]
        self.assertEqual(set(others), set(range(30, 50)))

    def test_repeated_hand_labelled_photos_are_dealt_more_often(self):
        items = self.items(10, 10) + [("hand", [[.1, .1, .5, .5]])] * 3
        sampler = trainer.BalancedSampler(items, epoch_size=26, bat_fraction=0.5, seed=3)
        chosen = list(sampler)   # exactly one full deal of the 13 bat cards
        self.assertEqual(sum(items[i][0] == "hand" for i in chosen), 3)
        self.assertEqual(sum(items[i][0] == "bat0" for i in chosen), 1)

    def test_one_empty_pile_uses_the_other(self):
        sampler = trainer.BalancedSampler(self.items(5, 0), epoch_size=12, bat_fraction=0.5, seed=0)
        self.assertEqual(len(list(sampler)), 12)
        self.assertEqual(len(trainer.BalancedSampler([], epoch_size=12)), 0)
        with self.assertRaises(ValueError):
            trainer.BalancedSampler(self.items(1, 1), epoch_size=10, bat_fraction=1.5)

    def test_loader_uses_sampler_for_epoch_length(self):
        items = self.items(3, 3)
        sampler = trainer.BalancedSampler(items, epoch_size=10, seed=0)
        loader = trainer.make_loader(items, False, batch_size=4, workers=0,
                                     prefetch_factor=2, sampler=sampler)
        self.assertIs(loader.sampler, sampler)
        self.assertEqual(len(loader), 3)


class QuickBoxSamplerTests(unittest.TestCase):
    def test_counts_and_classes_match_torchvision_rules(self):
        from torchvision.models.detection._utils import BalancedPositiveNegativeSampler
        matched = torch.tensor([-1] * 50 + [0] * 1000 + [1] * 40 + [2] * 30)
        for labels in (matched, matched[:1060]):
            ours = trainer.QuickBoxSampler(256, 0.25)([labels])
            theirs = BalancedPositiveNegativeSampler(256, 0.25)([labels])
            for mine, reference in zip(ours, theirs):
                self.assertEqual(int(mine[0].sum()), int(reference[0].sum()))
            positive, negative = ours[0][0].bool(), ours[1][0].bool()
            self.assertTrue((labels[positive] >= 1).all())
            self.assertTrue((labels[negative] == 0).all())

    def test_pick_keeps_everything_when_too_few(self):
        indices = torch.arange(5)
        self.assertTrue(torch.equal(trainer.QuickBoxSampler.pick(indices, 9), indices))
        picked = trainer.QuickBoxSampler.pick(torch.arange(100), 10).tolist()
        self.assertEqual(len(set(picked)), 10)


class EvaluationScheduleTests(unittest.TestCase):
    def test_inaturalist_checked_before_every_n_and_at_end(self):
        checked = [e for e in range(0, 11) if trainer.check_inaturalist_after(e, 10, 4)]
        self.assertEqual(checked, [0, 4, 8, 10])
        checked = [e for e in range(0, 8) if trainer.check_inaturalist_after(e, 7, 0)]
        self.assertEqual(checked, [0, 7])

    def test_check_photos_batches_and_counts(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        photo = Path(temp.name) / "p.jpg"
        Image.new("RGB", (40, 40)).save(photo)
        items = [(photo, [[.1, .1, .5, .5]])] * 5 + [(photo, [[.1, .1, .1001, .5]])]
        photos = trainer.CheckPhotos(items, batch_size=4)
        batches = list(photos)
        self.assertEqual([len(images) for images, _ in batches], [4, 2])
        self.assertEqual(batches[0][0].dtype, torch.uint8)
        self.assertEqual(batches[1][1][1]["boxes"].shape, (0, 4))   # too-thin box dropped

        class FakeDetector(torch.nn.Module):
            def forward(self, images):
                box = torch.tensor([[32., 32., 160., 160.]])
                return [{"boxes": box, "scores": torch.tensor([.9])} for _ in images]

        self.assertEqual(trainer.evaluate(FakeDetector(), photos, "cpu"), (5, 0, 1))
        self.assertEqual(trainer.photos_with_bat_found(FakeDetector(), photos, "cpu"), 6)


class ArgumentTests(unittest.TestCase):
    def test_defaults_and_validation(self):
        args = trainer.parse_args([])
        self.assertEqual((args.epoch_size, args.bat_fraction), (trainer.EPOCH_SIZE, 0.5))
        self.assertEqual(args.save_to, trainer.config.FINETUNED_DETECTOR_FILE)
        self.assertFalse(args.no_save)
        args = trainer.parse_args(["--save-to", "x.pt", "--no-save", "--epochs", "1"])
        self.assertEqual((args.save_to, args.no_save, args.epochs), (Path("x.pt"), True, 1))
        for bad in (["--bat-fraction", "1.2"], ["--epoch-size", "0"], ["--inat-every", "-1"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                trainer.parse_args(bad)


if __name__ == "__main__":
    unittest.main()
