"""Hardware-free FOMO tests using only Python's standard library."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from uno_q import run_fomo


def bat_box(value=0.87, label="bat"):
    return {"label": label, "value": value, "x": 10, "y": 20,
            "width": 8, "height": 6}


class FakeFrame:
    """Enough image behavior to check the loop without NumPy or OpenCV."""

    def __init__(self, color="BGR"):
        self.color = color
        self.shape = (96, 96) if color == "GRAY" else (96, 96, 3)

    def copy(self):
        return FakeFrame(self.color)


def fake_cv2(camera):
    cv2 = SimpleNamespace(
        COLOR_BGR2RGB="BGR2RGB", COLOR_RGB2BGR="RGB2BGR",
        COLOR_BGR2GRAY="BGR2GRAY", COLOR_GRAY2RGB="GRAY2RGB",
        COLOR_GRAY2BGR="GRAY2BGR", FONT_HERSHEY_SIMPLEX=0,
        VideoCapture=Mock(return_value=camera),
        circle=Mock(), putText=Mock(), imshow=Mock(), waitKey=Mock(return_value=-1),
        destroyAllWindows=Mock(),
        imencode=Mock(return_value=(True, SimpleNamespace(tobytes=lambda: b"fake JPEG"))),
    )

    def convert(frame, conversion):
        original, target = conversion.split("2")
        if frame.color != original:
            raise AssertionError(f"Wrong color order: {frame.color}, expected {original}")
        return FakeFrame(target)

    cv2.cvtColor = Mock(side_effect=convert)
    return cv2


def fake_runner(script, grayscale=False, legacy=False):
    runner = SimpleNamespace(
        init=Mock(return_value={
            "project": {"name": "Test bats"},
            "model_parameters": {"image_input_width": 96, "image_input_height": 96,
                                 "image_channel_count": 1 if grayscale else 3,
                                 "labels": ["bat"]},
        }),
        classify=Mock(side_effect=[{"result": {"bounding_boxes": boxes}}
                                   for boxes in script]),
        stop=Mock(),
    )

    def extract(image):
        if image.color != "RGB":
            raise AssertionError("SDK must receive RGB")
        return [123], FakeFrame("GRAY" if grayscale else "RGB")

    runner.get_features_from_image = Mock(side_effect=extract)
    if not legacy:
        runner.get_features_from_image_auto_studio_settings = Mock(side_effect=extract)
    return runner


class LogicTests(unittest.TestCase):
    def test_default_needs_three_positive_frames(self):
        self.assertFalse(run_fomo.smoothed_status([], 3))
        self.assertFalse(run_fomo.smoothed_status([True, True], 3))
        self.assertTrue(run_fomo.smoothed_status([True, True, True], 3))

    def test_hits_need_not_be_consecutive(self):
        self.assertTrue(run_fomo.smoothed_status([True, False, True, False, True], 3))

    def test_old_hits_expire_after_five_frames(self):
        self.assertTrue(run_fomo.smoothed_status([True, True, True, False, False], 3))
        self.assertFalse(run_fomo.smoothed_status([True, True, True, False, False, False], 3))
        self.assertFalse(run_fomo.smoothed_status([True] * 5 + [False] * 5, 3))

    def test_configurable_frame_count(self):
        self.assertTrue(run_fomo.smoothed_status([True], 1))
        self.assertFalse(run_fomo.smoothed_status([True] * 4, 5))
        self.assertTrue(run_fomo.smoothed_status([True] * 5, 5))

    def test_threshold_is_inclusive_and_label_must_be_bat(self):
        boxes = [bat_box(0.59), bat_box(0.6), bat_box(0.9), bat_box(1, "bird"),
                 {"label": "bat"}, {"value": 0.9}]
        self.assertEqual(run_fomo.filter_bat_boxes(boxes, 0.6), boxes[1:3])
        self.assertEqual(run_fomo.filter_bat_boxes([], 0.6), [])

    def test_cli_defaults_and_validation(self):
        args = run_fomo.parse_args(["--model", "model.eim"])
        self.assertEqual((args.camera, args.threshold, args.frames_needed), (0, 0.6, 3))
        self.assertFalse(args.no_window)
        self.assertEqual(args.save_dir, Path("snapshots"))
        for option, value in [("--threshold", "-0.1"), ("--threshold", "1.1"),
                              ("--threshold", "nan"), ("--frames-needed", "0"),
                              ("--frames-needed", "6"), ("--camera", "-1")]:
            with self.subTest(option=option, value=value), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    run_fomo.parse_args(["--model", "model.eim", option, value])
                self.assertEqual(caught.exception.code, 2)


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cv2 = fake_cv2(Mock())
        self.when = datetime(2026, 10, 4, 21, 4, 13, 123456)

    def test_timestamp_name(self):
        self.assertEqual(run_fomo.snapshot_name(self.when), "bat_20261004_210413_123456.jpg")
        self.assertEqual(run_fomo.snapshot_name(self.when, 2), "bat_20261004_210413_123456_2.jpg")

    def test_existing_snapshot_is_never_overwritten(self):
        original = self.root / run_fomo.snapshot_name(self.when)
        original.write_bytes(b"existing photo")
        first = run_fomo.save_snapshot(self.cv2, FakeFrame("RGB"), [bat_box()], self.root, self.when)
        second = run_fomo.save_snapshot(self.cv2, FakeFrame("RGB"), [bat_box()], self.root, self.when)
        self.assertEqual(original.read_bytes(), b"existing photo")
        self.assertEqual(first.name, run_fomo.snapshot_name(self.when, 1))
        self.assertEqual(second.name, run_fomo.snapshot_name(self.when, 2))
        self.assertEqual(first.read_bytes(), b"fake JPEG")
        self.assertEqual(second.read_bytes(), b"fake JPEG")

    def test_snapshot_marks_center_in_model_coordinates_and_converts_rgb(self):
        image = FakeFrame("RGB")
        saved = run_fomo.save_snapshot(self.cv2, image, [bat_box()], self.root / "new", self.when)
        self.assertTrue(saved.is_file())
        self.cv2.circle.assert_called_once()
        call = self.cv2.circle.call_args.args
        self.assertEqual(call[1:], ((14, 23), 3, run_fomo.BAT_COLOR, 1))
        self.assertEqual(call[0].color, "BGR")
        self.assertIsNot(call[0], image)
        self.cv2.imencode.assert_called_once_with(".jpg", call[0])

    def test_grayscale_snapshot_is_converted_for_green_marks(self):
        run_fomo.save_snapshot(self.cv2, FakeFrame("GRAY"), [bat_box()], self.root, self.when)
        self.assertEqual(self.cv2.cvtColor.call_args.args[1], self.cv2.COLOR_GRAY2BGR)

    def test_encoding_failure_does_not_create_a_photo(self):
        self.cv2.imencode.return_value = (False, None)
        with self.assertRaisesRegex(RuntimeError, "encode"):
            run_fomo.save_snapshot(self.cv2, FakeFrame("RGB"), [], self.root, self.when)
        self.assertEqual(list(self.root.iterdir()), [])


class LoopTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / "fake.eim"
        self.model.touch()  # Only main's existence check reads this file.

    def exercise(self, script, window=False, grayscale=False, legacy=False,
                 configure=None, ticks=None):
        frames = [FakeFrame() for _ in script]
        camera = SimpleNamespace(isOpened=Mock(return_value=True), release=Mock(),
                                 read=Mock(side_effect=[(True, frame) for frame in frames]
                                           + [KeyboardInterrupt()]))
        cv2 = fake_cv2(camera)
        runner = fake_runner(script, grayscale, legacy)
        factory = Mock(return_value=runner)
        if window:
            cv2.waitKey.side_effect = [-1] * (len(script) - 1) + [ord("q")]
        if configure:
            configure(cv2, runner, camera)
        output, errors = io.StringIO(), io.StringIO()
        argv = ["--model", str(self.model), "--save-dir", str(self.root / "photos")]
        if not window:
            argv.append("--no-window")
        with patch.object(run_fomo, "load_dependencies", return_value=(cv2, factory)), \
                patch.object(run_fomo.time, "monotonic", side_effect=ticks or
                             [0] + [2 * (i + 1) for i in range(len(script))]), \
                patch.dict(run_fomo.os.environ, {"DISPLAY": ":0"}), \
                redirect_stdout(output), redirect_stderr(errors):
            code = run_fomo.main(argv)
        return SimpleNamespace(code=code, cv2=cv2, runner=runner, camera=camera,
                               factory=factory, output=output.getvalue(), errors=errors.getvalue())

    def test_no_window_never_calls_any_gui_function_even_when_saving(self):
        script = [[], [bat_box()], [bat_box()], [bat_box()]] + [[]] * 5 + [[bat_box()]] * 3
        result = self.exercise(script)
        self.assertEqual(result.code, 0, result.errors)
        result.cv2.imshow.assert_not_called()
        result.cv2.waitKey.assert_not_called()
        result.cv2.destroyAllWindows.assert_not_called()
        result.cv2.putText.assert_not_called()
        result.camera.release.assert_called_once()
        result.runner.stop.assert_called_once()
        result.factory.assert_called_once_with(str(self.model.resolve()))
        self.assertEqual(result.output.count("BAT DETECTED ("), 2)
        self.assertEqual(result.output.count("NOT A BAT ("), 2)
        self.assertEqual(result.output.count("Snapshot saved:"), 2)
        self.assertEqual(result.output.count("FPS:"), 2)
        self.assertEqual(len(list((self.root / "photos").glob("*.jpg"))), 2)
        self.assertEqual(result.runner.classify.call_count, len(script))
        result.runner.get_features_from_image.assert_not_called()

    def test_many_bat_boxes_in_one_frame_do_not_satisfy_three_frame_rule(self):
        result = self.exercise([[bat_box()] * 5, [], [], [], []])
        self.assertNotIn("BAT DETECTED (", result.output)
        result.cv2.imencode.assert_not_called()

    def test_low_confidence_and_other_labels_do_not_count(self):
        result = self.exercise([[bat_box(0.59), bat_box(1, "bird")]] * 5)
        self.assertNotIn("BAT DETECTED (", result.output)
        result.cv2.imencode.assert_not_called()

    def test_window_has_only_status_text_and_quits_on_q(self):
        result = self.exercise([[bat_box()]] * 3, window=True)
        self.assertEqual(result.code, 0)
        self.assertEqual(result.cv2.imshow.call_count, 3)
        self.assertEqual(result.cv2.waitKey.call_count, 3)
        self.assertEqual([call.args[1] for call in result.cv2.putText.call_args_list],
                         ["NOT A BAT", "NOT A BAT", "BAT DETECTED"])
        self.assertEqual([call.args[5] for call in result.cv2.putText.call_args_list],
                         [run_fomo.NO_BAT_COLOR, run_fomo.NO_BAT_COLOR, run_fomo.BAT_COLOR])
        result.cv2.destroyAllWindows.assert_called_once()
        result.camera.release.assert_called_once()
        result.runner.stop.assert_called_once()
        # Only the snapshot is marked with a centroid, not the displayed frame.
        result.cv2.circle.assert_called_once()
        shown = [call.args[1] for call in result.cv2.imshow.call_args_list]
        self.assertNotIn(result.cv2.circle.call_args.args[0], shown)

    def test_older_sdk_falls_back_to_rgb_center_crop_method(self):
        result = self.exercise([[], []], legacy=True)
        self.assertEqual(result.code, 0, result.errors)
        self.assertEqual(result.runner.get_features_from_image.call_count, 2)
        self.assertEqual(result.cv2.cvtColor.call_args.args[1], result.cv2.COLOR_BGR2RGB)

    def test_grayscale_model_uses_gray_channels_and_saves(self):
        result = self.exercise([[bat_box()]] * 3, grayscale=True)
        self.assertEqual(result.code, 0, result.errors)
        conversions = [call.args[1] for call in result.cv2.cvtColor.call_args_list]
        self.assertEqual(conversions[:2], [result.cv2.COLOR_BGR2GRAY, result.cv2.COLOR_GRAY2RGB])
        self.assertEqual(conversions[-1], result.cv2.COLOR_GRAY2BGR)

    def test_camera_open_failure_releases_camera_and_stops_runner(self):
        result = self.exercise([], configure=lambda cv, runner, cam:
                               setattr(cam.isOpened, "return_value", False))
        self.assertEqual(result.code, 1)
        self.assertIn("--camera 1", result.errors)
        self.assertIn("ls /dev/video*", result.errors)
        result.camera.release.assert_called_once()
        result.runner.stop.assert_called_once()
        result.cv2.imshow.assert_not_called()

    def test_failed_model_initialization_still_stops_runner(self):
        result = self.exercise([], configure=lambda cv, runner, cam:
                               setattr(runner.init, "side_effect", RuntimeError("not executable")))
        self.assertEqual(result.code, 1)
        self.assertIn("chmod +x", result.errors)
        result.runner.stop.assert_called_once()
        result.cv2.VideoCapture.assert_not_called()

    def test_inference_error_still_cleans_up(self):
        result = self.exercise([[]], configure=lambda cv, runner, cam:
                               setattr(runner.classify, "side_effect", RuntimeError("inference failed")))
        self.assertEqual(result.code, 1)
        self.assertIn("inference failed", result.errors)
        result.camera.release.assert_called_once()
        result.runner.stop.assert_called_once()

    def test_read_failure_still_cleans_up(self):
        result = self.exercise([[]], configure=lambda cv, runner, cam:
                               setattr(cam.read, "side_effect", [(False, None)]))
        self.assertEqual(result.code, 1)
        self.assertIn("read a frame", result.errors)
        result.camera.release.assert_called_once()
        result.runner.stop.assert_called_once()

    def test_classification_model_is_rejected(self):
        result = self.exercise([[]], configure=lambda cv, runner, cam:
                               setattr(runner.classify, "side_effect", [{"result": {"classification": {"bat": 1}}}]))
        self.assertEqual(result.code, 1)
        self.assertIn("FOMO", result.errors)
        result.runner.stop.assert_called_once()

    def test_cleanup_failure_does_not_skip_runner_stop(self):
        result = self.exercise([], configure=lambda cv, runner, cam:
                               setattr(cam.release, "side_effect", RuntimeError("release failed")))
        self.assertEqual(result.code, 0)
        self.assertIn("could not close camera", result.errors)
        result.runner.stop.assert_called_once()


class SetupErrorTests(unittest.TestCase):
    def test_missing_model_is_reported_before_importing_dependencies(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch.object(run_fomo, "load_dependencies") as dependencies, \
                redirect_stderr(io.StringIO()) as errors:
            code = run_fomo.main(["--model", str(Path(folder) / "missing.eim"), "--no-window"])
        self.assertEqual(code, 1)
        self.assertIn("model file not found", errors.getvalue())
        dependencies.assert_not_called()

    def test_missing_sdk_has_install_instructions(self):
        # Block the SDK import regardless of the machine's installed packages.
        with patch.dict(sys.modules, {"cv2": SimpleNamespace(), "edge_impulse_linux": None}):
            with self.assertRaisesRegex(RuntimeError, "pip install edge_impulse_linux"):
                run_fomo.load_dependencies()

    def test_libgl_error_has_both_fixes(self):
        message = run_fomo.error_message(ImportError("libGL.so.1: cannot open shared object file"))
        self.assertIn("sudo apt install -y libgl1 libglib2.0-0", message)
        self.assertIn("opencv-python-headless", message)
        self.assertIn("--no-window", message)

    def test_linux_without_display_gets_headless_instructions(self):
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "fake.eim"
            model.touch()
            with patch.object(run_fomo.sys, "platform", "linux"), \
                    patch.dict(run_fomo.os.environ, {}, clear=True), \
                    patch.object(run_fomo, "load_dependencies") as dependencies, \
                    redirect_stderr(io.StringIO()) as errors:
                code = run_fomo.main(["--model", str(model)])
            self.assertEqual(code, 1)
            self.assertIn("--no-window", errors.getvalue())
            dependencies.assert_not_called()


if __name__ == "__main__":
    unittest.main()
