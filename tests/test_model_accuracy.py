"""Permanent held-out accuracy gate. Fast suite: SBB_SKIP_ACCURACY=1.

Run the GPU gate with: python -m unittest tests.test_model_accuracy -v
Benchmark boxes and predictions use normalized coordinates; repeated CSV rows
represent multiple bats in one photo, which is inferred only once.
"""

import csv
import json
import os
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data/benchmarks/accuracy_set.csv"
LIMITS = ROOT / "data/benchmarks/thresholds.json"
BOX_FIELDS = ("xmin", "ymin", "xmax", "ymax")


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def matched_labels(labels, boxes, min_iou=0.3):
    """Maximum one-to-one matching; a detection cannot recall two bats."""
    edges = [[j for j, box in enumerate(boxes) if box_iou(label, box) >= min_iou]
             for label in labels]
    owners = {}

    def assign(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in owners or assign(owners[j], seen):
                owners[j] = i
                return True
        return False

    for i in range(len(labels)):
        assign(i, set())
    return {i for i in owners.values()}


def load_benchmark(path=BENCHMARK):
    records = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            image = row["image"]
            record = records.setdefault(image, {"expected": row["expected"],
                                                "group": row["group"], "boxes": []})
            if (record["expected"], record["group"]) != (row["expected"], row["group"]):
                raise ValueError(f"Conflicting benchmark labels for {image}")
            if row["xmin"]:
                record["boxes"].append([float(row[k]) for k in BOX_FIELDS])
    return records


def score_predictions(records, predictions, threshold, min_iou=0.3):
    """Score hand boxes, unlocalized iNat recall, and per-photo alarms."""
    counts = {key: [0, 0] for key in (
        "bat_recall", "inaturalist_recall", "false_alarm_rate", "people_false_alarm_rate")}
    for image, record in records.items():
        prediction = predictions[image]  # Missing predictions must fail, not count as clean.
        boxes = [box for box, score in zip(prediction["boxes"], prediction["scores"])
                 if score >= threshold]
        if record["expected"] == "not_bat":
            counts["false_alarm_rate"][0] += bool(boxes)
            counts["false_alarm_rate"][1] += 1
            if record["group"] == "people":
                counts["people_false_alarm_rate"][0] += bool(boxes)
                counts["people_false_alarm_rate"][1] += 1
        elif record["group"] == "hand_labelled":
            counts["bat_recall"][0] += len(matched_labels(record["boxes"], boxes, min_iou))
            counts["bat_recall"][1] += len(record["boxes"])
        elif record["group"] == "inaturalist":
            counts["inaturalist_recall"][0] += bool(boxes)
            counts["inaturalist_recall"][1] += 1
        else:
            raise ValueError(f"Unknown benchmark group: {record['group']}")
    if any(total == 0 for _, total in counts.values()):
        raise ValueError("Benchmark must contain hand-labelled bats, iNat bats, and people negatives")
    return {key: {"hits": hits, "total": total, "rate": hits / total}
            for key, (hits, total) in counts.items()}


class ScoringTests(unittest.TestCase):
    def test_matching_thresholds_and_photo_denominators(self):
        full = [0, 0, 1, 1]
        records = {
            "hand": {"expected": "bat", "group": "hand_labelled", "boxes": [full, full]},
            "inat": {"expected": "bat", "group": "inaturalist", "boxes": []},
            "person": {"expected": "not_bat", "group": "people", "boxes": []},
            "bird": {"expected": "not_bat", "group": "birds", "boxes": []},
        }
        predictions = {
            "hand": {"boxes": [[0, 0, 0.3, 1]], "scores": [0.7]},
            "inat": {"boxes": [full], "scores": [0.699]},
            "person": {"boxes": [full, full], "scores": [0.8, 0.9]},
            "bird": {"boxes": [], "scores": []},
        }
        scores = score_predictions(records, predictions, 0.7)
        self.assertEqual(scores["bat_recall"], {"hits": 1, "total": 2, "rate": 0.5})
        self.assertEqual(scores["inaturalist_recall"]["rate"], 0)
        self.assertEqual(scores["false_alarm_rate"], {"hits": 1, "total": 2, "rate": 0.5})
        self.assertEqual(scores["people_false_alarm_rate"]["rate"], 1)
        self.assertEqual(score_predictions(records, predictions, 0.8)["bat_recall"]["hits"], 0)
        with self.assertRaises(KeyError):
            score_predictions(records, {}, 0.7)

    def test_matching_reassigns_boxes_instead_of_greedy_undercount(self):
        labels = [[0, 0, 0.5, 1], [0, 0, 0.25, 1]]
        boxes = [[0, 0, 0.5, 1], [0.25, 0, 0.5, 1]]
        self.assertEqual(matched_labels(labels, boxes), {0, 1})
        self.assertEqual(box_iou([0, 0, 0, 0], boxes[0]), 0)


@unittest.skipIf(os.environ.get("SBB_SKIP_ACCURACY") == "1",
                 "SBB_SKIP_ACCURACY=1: skipping slow held-out GPU accuracy gate")
class ModelAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Lazy imports keep the scoring tests independent of torch and CUDA.
        from vision import config
        missing = [str(p) for p in (config.WEIGHTS_FILE, config.FINETUNED_DETECTOR_FILE)
                   if not p.is_file()]
        if missing:
            raise unittest.SkipTest("Accuracy gate model files missing: " + ", ".join(missing))
        import torch
        if not torch.cuda.is_available() or config.FORCE_CPU:
            raise unittest.SkipTest("Accuracy gate requires available CUDA and SBB_FORCE_CPU != 1")
        import cv2
        from vision.bat_detector import BatDetector

        torch.set_num_threads(2)
        limits = json.loads(LIMITS.read_text(encoding="utf-8"))
        records = load_benchmark()
        detector = BatDetector(min_confidence=config.DETECTION_CONFIDENCE_THRESHOLD)
        predictions = {}
        for image in records:
            frame = cv2.imread(str(ROOT / image))
            if frame is None:
                raise ValueError(f"Cannot read benchmark image: {image}")
            height, width = frame.shape[:2]
            detections = detector.detect(frame)
            predictions[image] = {
                "boxes": [[x1 / width, y1 / height, x2 / width, y2 / height]
                          for x1, y1, x2, y2 in (d.bbox for d in detections)],
                "scores": [d.confidence for d in detections],
            }
        cls.scores = score_predictions(records, predictions, config.DETECTION_CONFIDENCE_THRESHOLD,
                                       limits["min_iou"])
        cls.limits = limits
        print("\nAccuracy gate at confidence " + str(config.DETECTION_CONFIDENCE_THRESHOLD) + ": " +
              "; ".join(f"{key}={value['hits']}/{value['total']} ({value['rate']:.1%})"
                        for key, value in cls.scores.items()), flush=True)
        del detector
        torch.cuda.empty_cache()

    def test_production_model_meets_fixed_accuracy_limits(self):
        for key, value in self.scores.items():
            with self.subTest(benchmark_denominator=key):
                self.assertEqual(value["total"], self.limits["baseline"][key]["total"],
                                 "Fixed benchmark denominator changed")
        for key in ("bat_recall", "inaturalist_recall"):
            with self.subTest(metric=key):
                self.assertGreaterEqual(self.scores[key]["rate"], self.limits["minimum"][key])
        for key in ("false_alarm_rate", "people_false_alarm_rate"):
            with self.subTest(metric=key):
                self.assertLessEqual(self.scores[key]["rate"], self.limits["maximum"][key])


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(ROOT))
    unittest.main()
