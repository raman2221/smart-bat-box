"""Small report-card checks: fake detections, no GPU and no downloads."""

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from training import evaluate as report
from training.finetune_detector import count_matches


BOX = [0.0, 0.0, 0.4, 0.4]
OTHER_BOX = [0.6, 0.6, 1.0, 1.0]


def prediction(*scores):
    return {"boxes": [BOX[:] for _ in scores], "scores": list(scores)}


def exact_matches(true, pred):
    """Fake matcher for the sweep's arithmetic, using identical tiny boxes."""
    return sum(box in pred for box in true)


def tiny_checks():
    bat, negative, clean, inat = map(Path, ("bat.jpg", "person.jpg", "bird.jpg", "inat.jpg"))
    checks = ({bat: [BOX, OTHER_BOX]}, [negative, clean], [negative], [inat])
    predictions = {bat: {"boxes": [BOX, OTHER_BOX, BOX], "scores": [0.9, 0.6, 0.55]},
                   negative: prediction(0.65, 0.6), clean: prediction(), inat: prediction(0.75)}
    return checks, predictions


def tiny_result(name="test.pt", speed=12.5):
    checks, predictions = tiny_checks()
    sweep = report.threshold_sweep(checks, predictions, matcher=exact_matches)
    recommended = report.recommend_threshold(sweep)
    return {"model": name, "sweep": sweep, "selected": sweep[2], "recommended": recommended,
            "speed_ms": speed, "predictions": predictions,
            "category_threshold": recommended["threshold"],
            "categories": report.group_categories(checks[1], predictions,
                                                  {"person": ["Person"], "bird": ["Bird"]},
                                                  recommended["threshold"])}


class SweepTests(unittest.TestCase):
    def test_sweep_counts_boxes_and_photos_separately(self):
        checks, predictions = tiny_checks()
        rows = report.threshold_sweep(checks, predictions, matcher=exact_matches)
        self.assertEqual([row["threshold"] for row in rows], list(report.THRESHOLDS))
        self.assertEqual(rows[0]["bat_found"], 2)
        self.assertEqual(rows[0]["bat_total"], 2)
        self.assertEqual(rows[0]["extra_boxes"], 1)
        self.assertEqual(rows[0]["false_alarms"], 1)  # two boxes, just one photo
        self.assertEqual(rows[0]["people_false_alarms"], 1)
        self.assertEqual(rows[0]["inat_found"], 1)
        self.assertEqual(rows[0]["score"], 0.5)
        self.assertEqual(rows[2]["score"], 0.75)
        self.assertEqual(rows[-1]["score"], 0.25)

    def test_cutoffs_are_inclusive_and_duplicate_cutoffs_removed(self):
        checks, predictions = tiny_checks()
        rows = report.threshold_sweep(checks, predictions, (0.6, 0.6), exact_matches)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["bat_found"], 2)
        self.assertEqual(report.boxes_at(prediction(0.5, 0.49), 0.5), [BOX])

    def test_sweep_does_not_load_or_run_a_detector(self):
        checks, predictions = tiny_checks()
        with patch.object(report, "load_detector", side_effect=AssertionError("no model allowed")):
            report.threshold_sweep(checks, predictions, matcher=exact_matches)

    def test_repeated_negative_and_inat_paths_count_only_once(self):
        checks, predictions = tiny_checks()
        bats, negatives, people, inat = checks
        row = report.threshold_sweep((bats, negatives * 2, people * 2, inat * 2),
                                     predictions, (0.5,), exact_matches)[0]
        self.assertEqual((row["negative_total"], row["people_total"], row["inat_total"]), (2, 1, 1))

    def test_recommendation_maximizes_the_requested_formula(self):
        result = tiny_result()
        self.assertEqual(result["recommended"]["threshold"], 0.7)
        self.assertAlmostEqual(report.recommendation_score(
            {"bat_recall": 0.6, "inat_recall": 0.8, "false_alarm_rate": 0.2}), 0.5)

    def test_ties_prefer_fewer_alarms_then_higher_cutoff(self):
        rows = [{"threshold": t, "score": 0.5, "false_alarm_rate": alarm}
                for t, alarm in ((0.5, 0.2), (0.6, 0.1), (0.7, 0.1))]
        self.assertEqual(report.recommend_threshold(rows)["threshold"], 0.7)

    def test_custom_cli_cutoff_does_not_compete_in_fixed_sweep(self):
        rows = [{"threshold": 0.7, "score": 0.5, "false_alarm_rate": 0.1},
                {"threshold": 0.42, "score": 0.9, "false_alarm_rate": 0.0}]
        self.assertEqual(report.recommend_threshold(rows)["threshold"], 0.7)

    def test_empty_sets_are_not_reported_as_zero_accuracy(self):
        rows = report.threshold_sweep(({}, [], [], []), {}, matcher=exact_matches)
        self.assertIsNone(rows[0]["bat_recall"])
        self.assertIsNone(rows[0]["score"])
        self.assertIsNone(report.recommend_threshold(rows))
        self.assertEqual(report.rate_text(0, 0), "n/a (no photos)")

    def test_the_real_matching_helper_is_imported(self):
        self.assertIs(report.count_matches, count_matches)
        self.assertIs(report.threshold_sweep.__defaults__[-1], count_matches)


class CategoryTests(unittest.TestCase):
    def test_grouping_deduplicates_photos_and_labels_but_keeps_multi_class(self):
        paths = list(map(Path, ("a.jpg", "b.jpg", "c.jpg")))
        predictions = {paths[0]: prediction(0.9, 0.8), paths[1]: prediction(0.6), paths[2]: prediction()}
        classes = {"a": ["Person", "Person", "Clothing"], "b": ["Person"]}
        rows = report.group_categories(paths + paths[:1], predictions, classes, 0.7)
        groups = {r["category"]: r for r in rows}
        self.assertEqual(groups["Person"]["false_alarms"], 1)
        self.assertEqual(groups["Person"]["total"], 2)
        self.assertEqual(groups["Clothing"]["rate"], 1)
        self.assertEqual(groups[report.UNKNOWN_CLASS]["total"], 1)
        self.assertEqual(rows[0]["category"], "Clothing")

    def test_categories_rank_by_alarm_count_before_alarm_share(self):
        paths = [Path(f"{n}.jpg") for n in range(5)]
        predictions = {p: prediction(0.8) if n < 3 else prediction() for n, p in enumerate(paths)}
        classes = {p.stem: ["Common" if n != 2 else "Rare"] for n, p in enumerate(paths)}
        rows = report.group_categories(paths, predictions, classes, 0.7, limit=1)
        self.assertEqual(rows[0]["category"], "Common")
        self.assertEqual(rows[0]["rate"], 0.5)

    def test_people_classes_cover_body_parts_and_clothing(self):
        for label in report.PEOPLE_CLASSES:
            self.assertTrue(report.is_people_photo(Path("new.jpg"), {"new": [label]}))
        self.assertFalse(report.is_people_photo(Path("new.jpg"), {"new": ["Bird", "Lamp"]}))

    def test_stream_filters_image_ids_and_merges_classes(self):
        lines = ["ImageID,Source,LabelName,Confidence\n", "a,xclick,/person,1\n",
                 "a,xclick,/person,1\n", "a,xclick,/lamp,1\n", "other,xclick,/bird,1\n"]
        found = report.filter_class_lines(iter(lines), {"a", "missing"},
                                          {"/person": "Person", "/lamp": "Lamp"})
        self.assertEqual(found, {"a": ["Lamp", "Person"], "missing": []})

    def test_cache_downloads_only_missing_ids_and_never_repeats(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "val_classes.json"
            cache.write_text(json.dumps({"images": {"old": ["Person"]}}), encoding="utf-8")
            with patch.object(report, "download_classes", return_value={"new": ["Bird"]}) as download:
                with redirect_stdout(io.StringIO()):
                    found = report.load_classes([Path("old.jpg"), Path("new.jpg")], cache)
                    again = report.load_classes([Path("old.jpg"), Path("new.jpg")], cache)
            download.assert_called_once_with({"new"})
            self.assertEqual(found, again)
            self.assertEqual(found["old"], ["Person"])

    def test_download_streams_both_sources_and_merges_the_names(self):
        responses = {report.CLASS_NAMES_URL: b"/person,Person\n/lamp,Lamp\n",
                     report.BOX_URLS[0]: b"a,xclick,/person,1\nother,xclick,/lamp,1\n",
                     report.BOX_URLS[1]: b"a,xclick,/lamp,1\n"}
        with patch.object(report.urllib.request, "urlopen",
                          side_effect=lambda url, **kwargs: io.BytesIO(responses[url])) as download:
            with redirect_stdout(io.StringIO()):
                found = report.download_classes({"a"})
        self.assertEqual(found, {"a": ["Lamp", "Person"]})
        self.assertEqual(download.call_count, 3)

    def test_network_failure_leaves_existing_cache_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "val_classes.json"
            text = json.dumps({"images": {"old": ["Person"]}})
            cache.write_text(text, encoding="utf-8")
            with patch.object(report, "download_classes", side_effect=OSError("offline")):
                with redirect_stdout(io.StringIO()):
                    found = report.load_classes([Path("old.jpg"), Path("new.jpg")], cache)
            self.assertEqual(found, {"old": ["Person"]})
            self.assertEqual(cache.read_text(encoding="utf-8"), text)

    def test_offline_report_import_keeps_only_val_and_wanted_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "classes.csv"
            file.write_text("split,image,classes\nval,a.jpg,Person; Clothing; Person\n"
                            "train,b.jpg,Bird\nval,other.jpg,Lamp\n", encoding="utf-8")
            found = report.read_class_reports([file, Path(directory) / "absent.csv"], {"a", "b"})
            self.assertEqual(found, {"a": ["Clothing", "Person"]})

    def test_offline_report_fallback_saves_cache_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "val_classes.json"
            with patch.object(report, "download_classes", side_effect=OSError("offline")):
                with patch.object(report, "read_class_reports", return_value={"new": ["Bird"]}):
                    with redirect_stdout(io.StringIO()):
                        found = report.load_classes([Path("new.jpg")], cache)
            self.assertEqual(found, {"new": ["Bird"]})
            self.assertIn("reused_class_reports", json.loads(cache.read_text(encoding="utf-8")))


class ReportTests(unittest.TestCase):
    def test_markdown_has_side_by_side_metrics_sweep_reason_and_categories(self):
        checks, _ = tiny_checks()
        results = [tiny_result("one.pt"), tiny_result("two.pt", speed=10)]
        text = report.render_markdown(results, checks, {"person": ["Person"]}, "cpu", 0.7)
        self.assertIn("| Metric | one.pt | two.pt |", text)
        self.assertIn("| Threshold | one.pt | two.pt |", text)
        self.assertIn("1/2 (50.0%)", text)
        self.assertIn("## Top 10 false-alarm categories", text)
        self.assertIn("highest sweep score", text)
        self.assertIn("two.pt is best", text)
        self.assertIn("IoU >= 0.5", text)
        self.assertIn("Class coverage: 1/2", text)
        self.assertIn("Open Images class names", text)

    def test_markdown_escapes_pipes_and_newlines(self):
        text = report.markdown_table([("class", "rate"), ("A|B\nC", "50%")])
        self.assertIn("A\\|B C", text)
        self.assertIn("| --- | --- |", text)

    def test_missing_data_explains_why_there_is_no_recommendation(self):
        result = tiny_result()
        result["recommended"] = None
        text = report.render_markdown([result], tiny_checks()[0], None, "cpu", 0.7)
        self.assertIn("not enough data", text)
        self.assertIn("legacy physical lines", text)


class AnnotationTests(unittest.TestCase):
    def test_new_class_based_people_rows_duplicates_and_incomplete_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oi = root / "openimages"
            oi.mkdir()
            for name in ("bat.jpg", "new.jpg", "bird.jpg"):
                (oi / name).touch()
            with open(oi / "annotations.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["split", "image", "is_bat", "xmin", "ymin", "xmax", "ymax"])
                writer.writerow(["val", "bat.jpg", 1, *BOX])
                writer.writerow(["val", "bat.jpg", 1, *BOX])
                writer.writerow(["val", "new.jpg", 0, "", "", "", ""])
                writer.writerow(["val", "new.jpg", 0, "", "", "", ""])
                writer.writerow(["val", "bird.jpg", 0, "", "", "", ""])
                writer.writerow(["val", "missing.jpg", 0, "", "", "", ""])
                writer.writerow(["val", "bat.jpg", 1, "unfinished"])
            with patch.multiple(report.config, OPENIMAGES_DIR=oi, INATURALIST_DIR=root / "inat"):
                with redirect_stdout(io.StringIO()):
                    bats, negatives, people, inat = report.load_checks({"new": ["Human hand"]})
            self.assertEqual(bats, {oi / "bat.jpg": [BOX]})
            self.assertEqual(negatives, [oi / "bird.jpg", oi / "new.jpg"])
            self.assertEqual(people, [oi / "new.jpg"])
            self.assertEqual(inat, [])


class PredictionCacheTests(unittest.TestCase):
    def test_reusing_predictions_never_loads_a_detector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, base = root / "test.pt", root / "base.zip"
            weights.touch()
            base.touch()
            negative = root / "negative.jpg"
            negative.touch()
            checks = ({}, [negative], [], [])
            with patch.object(report, "REPORT_DIR", root), patch.object(report.config, "WEIGHTS_FILE", base):
                signature = report.cache_signature(weights, [negative], 0.3)
                (root / "predictions_test.json").write_text(json.dumps({
                    "signature": signature, "device": "cpu", "speed_ms": 10,
                    "predictions": {str(negative): prediction(0.8)}}), encoding="utf-8")
                with patch.object(report, "load_detector", side_effect=AssertionError("no detector allowed")):
                    with redirect_stdout(io.StringIO()):
                        result = report.evaluate(weights, checks, "cpu", 0.7, reuse=True)
            self.assertEqual(result["selected"]["false_alarms"], 1)
            self.assertEqual(result["speed_ms"], 10)

    def test_image_change_invalidates_the_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights, base, photo = root / "test.pt", root / "base.zip", root / "photo.jpg"
            for path in (weights, base, photo):
                path.touch()
            with patch.object(report.config, "WEIGHTS_FILE", base):
                before = report.cache_signature(weights, [photo], 0.3)
                photo.write_bytes(b"changed photo")
                after = report.cache_signature(weights, [photo], 0.3)
            self.assertNotEqual(before, after)


if __name__ == "__main__":
    unittest.main()
