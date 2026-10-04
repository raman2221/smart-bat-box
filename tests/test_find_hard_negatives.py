"""Pure-Python parts of training/find_hard_negatives.py (no GPU, no network).

Run only these tests from the project folder:
    .venv\\Scripts\\python.exe -B -m unittest tests.test_find_hard_negatives -v
"""

import tempfile
import unittest
from pathlib import Path

from training import find_hard_negatives as fhn


def row(split, image, is_bat):
    return {"split": split, "image": image, "is_bat": str(is_bat),
            "xmin": "", "ymin": "", "xmax": "", "ymax": ""}


class ReadNegativesTest(unittest.TestCase):
    def test_keeps_only_negatives_of_wanted_splits_once(self):
        rows = [row("train", "images/train/negative/aaaaaaaaaaaaaaaa.jpg", 0),
                row("train", "images/train/bat/bbbbbbbbbbbbbbbb.jpg", 1),
                row("val", "images/val/negative/cccccccccccccccc.jpg", 0),
                row("train", "images/train/negative/aaaaaaaaaaaaaaaa.jpg", 0)]
        photos = fhn.read_negatives(rows, splits=("train",))
        self.assertEqual([p.image_id for p in photos], ["aaaaaaaaaaaaaaaa"])
        self.assertEqual(len(fhn.read_negatives(rows)), 2)

    def test_skip_list(self):
        rows = [row("val", "images/val/negative/cccccccccccccccc.jpg", 0),
                row("val", "images/val/negative/dddddddddddddddd.jpg", 0)]
        photos = fhn.read_negatives(rows, skip={"images/val/negative/cccccccccccccccc.jpg"})
        self.assertEqual([p.image_id for p in photos], ["dddddddddddddddd"])


class BoxLinesTest(unittest.TestCase):
    HEADER = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded\n"

    def test_parse_reorders_to_xyxy(self):
        parsed = fhn.parse_box_line("0123456789abcdef,xclick,/m/01g317,1,0.1,0.5,0.2,0.6,0\n")
        self.assertEqual(parsed, ("0123456789abcdef", "/m/01g317", (0.1, 0.2, 0.5, 0.6)))

    def test_parse_rejects_header_and_junk(self):
        self.assertIsNone(fhn.parse_box_line(self.HEADER))
        self.assertIsNone(fhn.parse_box_line("short,line\n"))
        self.assertIsNone(fhn.parse_box_line("0123456789abcdef,x,/m/1,1,a,b,c,d\n"))

    def test_filter_keeps_only_wanted_ids(self):
        lines = [self.HEADER,
                 "0123456789abcdef,x,/m/a,1,0,1,0,1,0\n",
                 "ffffffffffffffff,x,/m/b,1,0,1,0,1,0\n",
                 "0123456789abcdef,x,/m/c,1,0,0.5,0,0.5,0\n"]
        found = fhn.filter_box_lines(lines, {"0123456789abcdef"})
        self.assertEqual([label for label, _ in found["0123456789abcdef"]], ["/m/a", "/m/c"])
        self.assertNotIn("ffffffffffffffff", found)

    def test_class_names(self):
        names = fhn.read_class_names(["/m/011k07,Tortoise\n", '/m/0abc,"Jaguar (Animal)"\n'])
        self.assertEqual(names, {"/m/011k07": "Tortoise", "/m/0abc": "Jaguar (Animal)"})


class GeometryTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(fhn.iou((0, 0, 1, 1), (0, 0, 1, 1)), 1.0)
        self.assertAlmostEqual(fhn.iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)
        self.assertAlmostEqual(fhn.iou((0, 0, 2, 1), (1, 0, 3, 1)), 1 / 3)
        self.assertEqual(fhn.iou((0, 0, 0, 0), (0, 0, 0, 0)), 0.0)

    def test_attribute_picks_best_overlap(self):
        labelled = [("Person", (0, 0, 1, 1)), ("Human hair", (0.0, 0.0, 0.3, 0.3))]
        self.assertEqual(fhn.attribute((0.0, 0.0, 0.3, 0.3), labelled), "Human hair")
        self.assertEqual(fhn.attribute((0.0, 0.0, 0.9, 0.9), labelled), "Person")

    def test_attribute_none_when_no_overlap(self):
        self.assertEqual(fhn.attribute((0.9, 0.9, 1, 1), [("Lamp", (0, 0, 0.2, 0.2))]), fhn.NO_OBJECT)
        self.assertEqual(fhn.attribute((0, 0, 1, 1), []), fhn.NO_OBJECT)


class ReportTest(unittest.TestCase):
    def test_histogram_includes_one(self):
        self.assertEqual(fhn.score_histogram([0.0, 0.05, 0.35, 0.7, 1.0]), [2, 0, 1, 0, 1, 1])

    def test_rank_classes_counts_photos_not_boxes(self):
        a = fhn.Photo("val", "a.jpg", "a", 0.9, [(0.9, (0, 0, 1, 1), "Bird"), (0.8, (0, 0, 1, 1), "Bird")],
                      {"Bird", "Tree"})
        b = fhn.Photo("val", "b.jpg", "b", 0.4, [(0.4, (0, 0, 1, 1), "Tree")], {"Tree"})
        c = fhn.Photo("val", "c.jpg", "c", 0.0, [], {"Lamp"})
        ranked = fhn.rank_classes([a, b, c])
        self.assertEqual([r["class"] for r in ranked], ["Bird", "Tree", "Lamp"])
        self.assertEqual((ranked[0]["photos"], ranked[0][0.3], ranked[0][0.7]), (1, 1, 1))
        self.assertEqual((ranked[1]["photos"], ranked[1][0.3], ranked[1][0.5]), (2, 1, 0))


class CacheTest(unittest.TestCase):
    def test_round_trip_keeps_images_without_boxes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.csv"
            fhn.save_cache(path, {"a": [("/m/1", (0.1, 0.2, 0.3, 0.4))], "b": []})
            loaded = fhn.load_cache(path)
        self.assertEqual(set(loaded), {"a", "b"})
        self.assertEqual(loaded["b"], [])
        self.assertEqual(loaded["a"][0][0], "/m/1")
        for got, want in zip(loaded["a"][0][1], (0.1, 0.2, 0.3, 0.4)):
            self.assertAlmostEqual(got, want)


if __name__ == "__main__":
    unittest.main()
