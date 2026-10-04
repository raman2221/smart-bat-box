"""Watch a USB webcam for bats with an Edge Impulse FOMO .eim model.

Run on the UNO Q's Linux side, for example:
    python uno_q/run_fomo.py --model model.eim --no-window
"""

import argparse
from collections import deque
from datetime import datetime
import os
from pathlib import Path
import sys
import time


WINDOW = "Bat Detector"
BAT_COLOR = (0, 255, 0)       # OpenCV colors are (Blue, Green, Red).
NO_BAT_COLOR = (0, 0, 255)
HISTORY_SIZE = 5


def filter_bat_boxes(boxes, threshold):
    """Keep only bat detections at or above the chosen confidence."""
    return [box for box in boxes
            if box.get("label") == "bat" and box.get("value", 0) >= threshold]


def smoothed_status(history, frames_needed):
    """Count positive frames in the last five; they need not be consecutive.

    During startup, unseen frames count as misses. Three positive frames
    can therefore turn the message on as early as the third camera frame.
    """
    return sum(bool(hit) for hit in list(history)[-HISTORY_SIZE:]) >= frames_needed


def snapshot_name(when, suffix=0):
    """Make a timestamped JPEG name, with a counter for timestamp collisions."""
    stem = when.strftime("bat_%Y%m%d_%H%M%S_%f")
    return f"{stem}{'_' + str(suffix) if suffix else ''}.jpg"


def save_snapshot(cv2, image, boxes, save_dir, when):
    """Mark centroids on the model input image and save without overwriting."""
    marked = image.copy()
    # The SDK returns RGB for color models, or a 2-D image for grayscale.
    conversion = cv2.COLOR_RGB2BGR if len(marked.shape) == 3 else cv2.COLOR_GRAY2BGR
    marked = cv2.cvtColor(marked, conversion)
    for box in boxes:
        center = (round(box["x"] + box["width"] / 2),
                  round(box["y"] + box["height"] / 2))
        cv2.circle(marked, center, 3, BAT_COLOR, 1)

    ok, encoded = cv2.imencode(".jpg", marked)
    if not ok:
        raise RuntimeError("Could not encode the snapshot as a JPEG.")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        path = save_dir / snapshot_name(when, suffix)
        try:
            # 'xb' creates a NEW file. Even two simultaneous saves cannot
            # replace an existing photo, unlike a normal OpenCV imwrite.
            with path.open("xb") as photo:
                photo.write(encoded.tobytes())
            return path
        except FileExistsError:
            suffix += 1


def image_features(cv2, runner, frame, grayscale):
    """Convert the webcam image to the format expected by the SDK."""
    if grayscale:
        # Convert using the camera's BGR order first. Replicated gray channels
        # also work with SDK releases that use BGR weights internally for gray.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # New SDKs can follow Studio's crop/resize settings. Older SDKs offer
    # get_features_from_image, which uses a center crop instead.
    extract = getattr(runner, "get_features_from_image_auto_studio_settings", None)
    if extract is None:
        extract = runner.get_features_from_image
    return extract(rgb)


def draw_status(cv2, display, present):
    """Show just the same large status message as entrance_tracker.py."""
    text, color = ("BAT DETECTED", BAT_COLOR) if present else ("NOT A BAT", NO_BAT_COLOR)
    cv2.putText(display, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)


def run_detector(args, cv2, runner_factory):
    """Run the camera loop; dependencies are passed in so tests need no camera."""
    runner = None
    camera = None
    window_started = False
    try:
        runner = runner_factory(str(args.model.resolve()))
        model_info = runner.init()
        parameters = model_info["model_parameters"]
        width = parameters["image_input_width"]
        height = parameters["image_input_height"]
        if width <= 0 or height <= 0 or "bat" not in parameters.get("labels", []):
            raise ValueError("Use an image FOMO model trained with the label 'bat'.")
        grayscale = parameters.get("image_channel_count", 3) == 1
        project = model_info.get("project", {}).get("name", "Edge Impulse model")
        print(f"Loaded {project} ({width}x{height}).", flush=True)

        camera = cv2.VideoCapture(args.camera)
        if not camera.isOpened():
            raise RuntimeError(
                f"Could not open camera {args.camera}. Try --camera 1 or --camera 2; "
                "run `ls /dev/video*` to find cameras. Close other camera apps "
                "and check permission to access the device.")
        print("Watching for bats. " + ("Stop with Ctrl+C." if args.no_window
                                      else "Press 'q' or Ctrl+C to stop."), flush=True)

        history = deque(maxlen=HISTORY_SIZE)
        previous_status = None
        fps_start = time.monotonic()
        fps_frames = 0
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the webcam. "
                                   "Check its connection and try another --camera index.")

            features, model_image = image_features(cv2, runner, frame, grayscale)
            prediction = runner.classify(features)
            result = prediction.get("result", {})
            if "bounding_boxes" not in result:
                raise RuntimeError("The model did not return bounding_boxes. "
                                   "Deploy an object detection (FOMO) model.")
            boxes = filter_bat_boxes(result["bounding_boxes"], args.threshold)
            history.append(bool(boxes))  # Several boxes still count as ONE frame.
            present = smoothed_status(history, args.frames_needed)
            confidence = max((box["value"] for box in boxes), default=0.0)

            fps_frames += 1
            now = time.monotonic()
            elapsed = now - fps_start
            fps = fps_frames / elapsed if elapsed > 0 else 0.0
            when = datetime.now()
            if present != previous_status:
                status = "BAT DETECTED" if present else "NOT A BAT"
                print(f"[{when:%H:%M:%S}] {status} ({confidence:.2f}, {fps:.1f} fps)",
                      flush=True)
                if present:
                    path = save_snapshot(cv2, model_image, boxes, args.save_dir, when)
                    print(f"Snapshot saved: {path}", flush=True)
                previous_status = present

            if elapsed >= 10:
                print(f"[{when:%H:%M:%S}] FPS: {fps:.1f}", flush=True)
                fps_start = now
                fps_frames = 0

            # Headless mode never calls any OpenCV window function.
            if not args.no_window:
                display = frame.copy()
                draw_status(cv2, display, present)
                window_started = True
                cv2.imshow(WINDOW, display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        # Try every cleanup step even if one fails. This also runs if model
        # startup fails, the camera disconnects, or the user presses Ctrl+C.
        cleanup = []
        if camera is not None:
            cleanup.append(("camera", camera.release))
        if runner is not None:
            cleanup.append(("model runner", runner.stop))
        if window_started:
            cleanup.append(("window", cv2.destroyAllWindows))
        for name, close in cleanup:
            try:
                close()
            except Exception as error:
                print(f"Warning: could not close {name}: {error}", file=sys.stderr, flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Detect bats with a FOMO .eim model.")
    parser.add_argument("--model", type=Path, required=True, help="Path to the Linux .eim model")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Minimum confidence (0 to 1)")
    parser.add_argument("--no-window", action="store_true", help="Run over SSH / App Lab without a window")
    parser.add_argument("--frames-needed", type=int, default=3, help="Positive frames needed out of the last 5")
    parser.add_argument("--save-dir", type=Path, default=Path("snapshots"), help="Folder for bat JPEGs")
    args = parser.parse_args(argv)
    if not 0 <= args.threshold <= 1:
        parser.error("--threshold must be between 0 and 1")
    if not 1 <= args.frames_needed <= HISTORY_SIZE:
        parser.error("--frames-needed must be between 1 and 5")
    if args.camera < 0:
        parser.error("--camera must be 0 or a positive index")
    return args


def load_dependencies():
    """Delay imports so help and unit tests work without OpenCV or the SDK."""
    import cv2
    try:
        from edge_impulse_linux.image import ImageImpulseRunner
    except ModuleNotFoundError as error:
        if (error.name or "").startswith("edge_impulse_linux"):
            raise RuntimeError("edge_impulse_linux is not installed. Activate your venv, "
                               "then run `pip install edge_impulse_linux`.") from error
        raise
    return cv2, ImageImpulseRunner


def error_message(error):
    """Add practical fixes to common Linux setup errors."""
    detail = str(error)
    if "libgl" in detail.lower():
        return (f"{detail}\nOpenCV could not load libGL. Run "
                "`sudo apt install -y libgl1 libglib2.0-0`, or replace opencv-python "
                "with opencv-python-headless in your venv and use --no-window.")
    if isinstance(error, ModuleNotFoundError) and error.name == "cv2":
        return "OpenCV is not installed. Run `pip install -r uno_q/requirements.txt` in your venv."
    if "not executable" in detail.lower() or isinstance(error, PermissionError):
        return (f"{detail}\nFor the model, run `chmod +x path/to/model.eim`. "
                "Also check camera permissions and that --save-dir is writable.")
    if "function is not implemented" in detail.lower() or "could not connect to display" in detail.lower():
        return (f"{detail}\nUse --no-window over SSH / App Lab. Window mode needs "
                "opencv-python (replace opencv-python-headless) and a working monitor/display.")
    return detail


def main(argv=None):
    args = parse_args(argv)
    if not args.model.is_file():
        print(f"Error: model file not found: {args.model}. Download a Linux (AARCH64) .eim "
              "and pass its path with --model.", file=sys.stderr, flush=True)
        return 1
    # Qt-backed OpenCV can abort the process before Python can catch an error.
    if (not args.no_window and sys.platform.startswith("linux")
            and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))):
        print("Error: no display found. Use --no-window over SSH / App Lab. "
              "Window mode needs opencv-python and a working monitor/display.",
              file=sys.stderr, flush=True)
        return 1
    try:
        cv2, runner_factory = load_dependencies()
        run_detector(args, cv2, runner_factory)
    except KeyboardInterrupt:
        print("\nDetector stopped.", flush=True)
        return 0
    except Exception as error:
        print(f"Error: {error_message(error)}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
