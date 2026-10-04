"""
export_onnx.py

Turns our fine-tuned bat detector into an ONNX file
(models/detection/bat_detector.onnx).

Why ONNX: an .onnx file holds the WHOLE detector - its structure and its
learned numbers - and runs with ONNX Runtime, a small library (~20 MB).
So a device like the Arduino UNO Q needs neither PyTorch (gigabytes) nor
BatNet's base model to run it.

What goes in / comes out of the ONNX file:
  in:  "image"  - one RGB camera frame, shape (1, 3, height, width),
                  numbers 0-1 (any size; it resizes to 320x320 itself)
  out: "boxes"  - (N, 4) bat boxes x1, y1, x2, y2 in the frame's pixels
       "scores" - (N,) confidence 0-1 for each box
       "labels" - (N,) always 1 ("bat")
Boxes scoring under 0.5 are already dropped inside the model (BatNet's
built-in floor); the program applies its own cut-off (0.8) on top.

Before saving it checks the ONNX file gives the same answers as PyTorch
on check photos (photos never trained on).

Run from the project folder (with the .venv active):
    pip install onnx onnxruntime
    python training/export_onnx.py
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.package

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402

ONNX_FILE = config.WEIGHTS_DIR / "bat_detector.onnx"
CHECK_PHOTOS = 40
MAX_SCORE_DIFFERENCE = 0.01   # biggest allowed change in any bat score
MIN_BOX_OVERLAP = 0.98        # boxes must land in (almost) the same place


class DetectorForOnnx(torch.nn.Module):
    """BatNet's detector with a plain tensors-in, tensors-out interface."""

    def __init__(self, detector):
        super().__init__()
        self.detector = detector

    def forward(self, image):
        out = self.detector(image)[0]   # resizes, normalises, detects, rescales boxes
        return out["boxes"], out["scores"], out["labels"]


def load_detector(weights):
    """BatNet's detector with our fine-tuned weights, on the CPU, in eval mode."""
    detector = torch.package.PackageImporter(str(config.WEIGHTS_FILE)).load_pickle("model", "model.pkl").detector
    detector.load_state_dict(torch.load(weights, map_location="cpu"))
    return detector.float().eval()


def to_input(frame_bgr):
    """OpenCV frame -> (1, 3, H, W) RGB float array, numbers 0-1."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def check_photos():
    rows = list(csv.DictReader(open(config.OPENIMAGES_DIR / "annotations.csv", newline="")))
    paths = sorted({config.OPENIMAGES_DIR / r["image"] for r in rows if r["split"] == "val"})
    bats = sorted({config.OPENIMAGES_DIR / r["image"] for r in rows if r["split"] == "val" and r["is_bat"] == "1"})
    step = max(1, len(paths) // CHECK_PHOTOS)
    return bats + paths[::step][:CHECK_PHOTOS]   # all bat photos + a spread of others


def box_overlap(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 1.0


def compare(detector, onnx_path):
    """Run PyTorch and ONNX Runtime on the check photos; return the worst differences."""
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = DetectorForOnnx(detector).eval()
    worst_score, worst_overlap, photos = 0.0, 1.0, 0
    for path in check_photos():
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        x = to_input(frame)
        with torch.no_grad():
            t_boxes, t_scores, _ = model(torch.from_numpy(x))
        o_boxes, o_scores, _ = session.run(None, {"image": x})
        photos += 1
        if len(t_scores) != len(o_scores):
            worst_score = max(worst_score, 1.0)
            continue
        for tb, ts, ob, os_ in zip(t_boxes.numpy(), t_scores.numpy(), o_boxes, o_scores):
            worst_score = max(worst_score, abs(float(ts) - float(os_)))
            worst_overlap = min(worst_overlap, box_overlap(tb, ob))
    return worst_score, worst_overlap, photos


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, default=config.FINETUNED_DETECTOR_FILE,
                        help="fine-tuned weights to export (default: the production .pth)")
    parser.add_argument("--out", type=Path, default=ONNX_FILE)
    args = parser.parse_args()

    detector = load_detector(args.weights)
    example = torch.from_numpy(to_input(np.zeros((480, 640, 3), np.uint8) + 128))

    # eval() matters: the exporter puts the model back in the mode it found it
    # in, and a model left in training mode expects training labels.
    model = DetectorForOnnx(detector).eval()
    print("Exporting to ONNX...")
    torch.onnx.export(
        model, (example,), str(args.out),
        input_names=["image"], output_names=["boxes", "scores", "labels"],
        dynamic_axes={"image": {2: "height", 3: "width"},
                      "boxes": {0: "n"}, "scores": {0: "n"}, "labels": {0: "n"}},
        opset_version=17, dynamo=False,
    )
    print(f"Saved {args.out} ({args.out.stat().st_size / 1e6:.0f} MB)")

    worst_score, worst_overlap, photos = compare(detector, args.out)
    print(f"Checked {photos} photos: biggest score change {worst_score:.4f}, "
          f"smallest box overlap {worst_overlap:.3f}")
    if worst_score > MAX_SCORE_DIFFERENCE or worst_overlap < MIN_BOX_OVERLAP:
        raise SystemExit("ONNX answers differ from PyTorch - don't use this file.")
    print("ONNX file matches PyTorch.")


if __name__ == "__main__":
    main()
