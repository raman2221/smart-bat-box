"""Tests for training/refine_labels.py with hand-drawn masks (no model, no GPU)."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from training import refine_labels as refine

ALL_OPEN = {"left": True, "top": True, "right": True, "bottom": True}
BOX = (50, 50, 150, 150)   # the original box, inside a 200 x 200 window


def blank():
    return np.zeros((200, 200), bool)


def ellipse(mask, centre, axes):
    pixels = mask.astype(np.uint8)
    cv2.ellipse(pixels, centre, axes, 0, 0, 360, 1, -1)
    return pixels.astype(bool)


class JudgeMaskTests(unittest.TestCase):
    def test_one_whole_bat_is_kept(self):
        mask = ellipse(blank(), (100, 100), (48, 46))
        verdict = refine.judge_mask(mask, BOX, ALL_OPEN)
        self.assertTrue(verdict.keep)
        self.assertEqual(verdict.reason, "ok")
        self.assertEqual(verdict.box, BOX)

    def test_empty_mask_is_rejected(self):
        verdict = refine.judge_mask(blank(), BOX, ALL_OPEN)
        self.assertFalse(verdict.keep)
        self.assertEqual(verdict.reason, "no mask")

    def test_two_bats_in_one_box_are_rejected(self):
        mask = ellipse(ellipse(blank(), (75, 100), (20, 40)), (125, 100), (20, 40))
        verdict = refine.judge_mask(mask, BOX, ALL_OPEN)
        self.assertFalse(verdict.keep)
        self.assertEqual(verdict.reason, "several bats")

    def test_bat_running_past_an_open_side_is_partial(self):
        mask = blank()
        mask[60:140, 60:200] = True   # carries on off the right-hand side
        self.assertEqual(refine.judge_mask(mask, BOX, ALL_OPEN).reason, "partial")
        # ...but if that side is the photo's own edge, the bat is just cut off
        # by the photo; the box can still be right (here it spills, though).
        closed = dict(ALL_OPEN, right=False)
        self.assertNotEqual(refine.judge_mask(mask, BOX, closed).reason, "partial")

    def test_mask_spilling_far_past_the_box_is_rejected(self):
        mask = blank()
        mask[60:140, 60:185] = True   # 35 of 125 columns lie outside the box
        self.assertEqual(refine.judge_mask(mask, BOX, ALL_OPEN).reason, "spills past box")

    def test_tiny_mask_and_much_smaller_mask_are_rejected(self):
        tiny = ellipse(blank(), (100, 100), (13, 13))  # just over the speck size
        self.assertEqual(refine.judge_mask(tiny, BOX, ALL_OPEN).reason, "mask too small")
        corner = blank()
        corner[52:100, 52:100] = True  # fills only a quarter of the box
        self.assertEqual(refine.judge_mask(corner, BOX, ALL_OPEN).reason, "mask much smaller")

    def test_loose_box_is_tightened_but_never_grown(self):
        mask = blank()
        mask[60:140, 55:130] = True
        verdict = refine.judge_mask(mask, BOX, ALL_OPEN)
        self.assertTrue(verdict.keep)
        self.assertEqual(verdict.reason, "tightened")
        x1, y1, x2, y2 = verdict.box
        # Holds the whole mask (plus a little padding) ...
        self.assertTrue(x1 <= 55 and y1 <= 60 and x2 >= 130 and y2 >= 140)
        self.assertLess(x2, 140)
        # ... and stays inside the old box.
        self.assertTrue(x1 >= 50 and y1 >= 50 and x2 <= 150 and y2 <= 150)

    def test_mask_in_pieces_keeps_the_original_box(self):
        mask = ellipse(blank(), (90, 100), (35, 46))
        mask[55:115, 138:149] = True  # a separate wing piece, 13% of the body
        verdict = refine.judge_mask(mask, BOX, ALL_OPEN)
        self.assertTrue(verdict.keep)
        self.assertEqual(verdict.box, BOX)

    def test_context_window_marks_photo_edges_as_closed(self):
        window, open_sides = refine.context_window((0, 10, 50, 50), 100, 60)
        self.assertEqual(window, (0, 0, 63, 60))
        self.assertEqual(open_sides, {"left": False, "top": False, "right": True, "bottom": False})


class RefineFileTests(unittest.TestCase):
    def test_refine_writes_only_kept_boxes_and_a_report(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        settings = [patch.object(refine.config, "PROJECT_ROOT", root),
                    patch.object(refine, "AUDIT_DIR", root / "audit"),
                    patch.object(refine, "REPORT_FILE", root / "audit/report.csv")]
        for setting in settings:
            setting.start()
            self.addCleanup(setting.stop)
        one = np.full((100, 160, 3), 200, np.uint8)
        cv2.ellipse(one, (80, 50), (40, 20), 0, 0, 360, (20, 20, 20), -1)
        two = np.full((100, 160, 3), 200, np.uint8)
        cv2.ellipse(two, (60, 50), (15, 20), 0, 0, 360, (20, 20, 20), -1)
        cv2.ellipse(two, (100, 50), (15, 20), 0, 0, 360, (20, 20, 20), -1)
        cv2.imwrite(str(root / "one.png"), one)
        cv2.imwrite(str(root / "two.png"), two)
        raw = root / "raw.csv"
        with raw.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(refine.LABEL_COLUMNS)
            writer.writerow(["inaturalist", "one.png", 0.2, 0.2, 0.8, 0.8, 0.99])
            writer.writerow(["inaturalist", "two.png", 0.2, 0.2, 0.8, 0.8, 0.95])
        out = root / "auto_labels.csv"

        def segmenter(crop):
            return cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) < 100

        counts, kept = refine.refine(raw, out, segmenter=segmenter)
        self.assertEqual(kept, 1)
        self.assertEqual(counts.get("several bats"), 1)
        rows = list(csv.DictReader(out.open()))
        self.assertEqual([row["image"] for row in rows], ["one.png"])
        self.assertEqual(list(rows[0]), refine.LABEL_COLUMNS)
        self.assertEqual(rows[0]["score"], "0.99")
        report = list(csv.DictReader((root / "audit/report.csv").open()))
        self.assertEqual(len(report), 2)


if __name__ == "__main__":
    unittest.main()
