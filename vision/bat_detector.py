"""
bat_detector.py

A small wrapper around the BatNet model (https://github.com/GabiK-bat/BatNet)
so the rest of the project can just ask: "are there bats in this frame?"

BatNet has two parts, and we use them separately:
  1. The detector finds bats in the image (draws boxes around them).
     It is fast on a GPU, so we run it on every video frame: detect().
  2. The species classifier guesses the species of one bat (13 European
     species / species groups). It is slower, so we only run it once per
     entrance event, on the snapshot: classify_species().

BatNet was trained on infrared camera-trap photos of bats flying through
roost entrances. We fine-tuned the detector on colour photos as well
(training/finetune_detector.py); the species classifier is unchanged, so
it often can't name the species in colour images.

BatNet is licensed CC BY-NC-SA 4.0 (non-commercial use only).
Citation: Krivek et al. (2023), Remote Sensing in Ecology and Conservation.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import torch
import torch.package
import torchvision

from . import config

# BatNet's own app flags species guesses below 70% as "unsure".
SPECIES_UNSURE_BELOW = 0.70


# ---------------------------------------------------------------------------
# Data class for one detection (this is what bat_tracker.py consumes)
# ---------------------------------------------------------------------------

@dataclass
class BatDetection:
    """One bat found in one frame."""

    bbox: tuple                 # (x1, y1, x2, y2) in pixels
    confidence: float           # how sure BatNet is that this is a bat (0-1)
    species: str                # e.g. "Myotis nattereri"
    species_confidence: float   # how sure BatNet is about the species (0-1)

    @property
    def center_x(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2

    @property
    def species_unsure(self) -> bool:
        return self.species_confidence < SPECIES_UNSURE_BELOW


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class BatDetector:
    """Loads BatNet once and runs it on OpenCV frames."""

    def __init__(self, model_path=None, min_confidence=None):
        model_path = model_path or config.WEIGHTS_FILE
        if min_confidence is None:
            min_confidence = config.DETECTION_CONFIDENCE_THRESHOLD
        self.min_confidence = min_confidence

        if not model_path.exists():
            raise FileNotFoundError(
                f"BatNet model not found at {model_path}\n"
                "Download it with:  python vision/download_model.py"
            )

        # BatNet models are saved as a "torch package": a zip file that holds
        # both the trained weights AND the Python code that defines the model.
        # Only load model files from sources you trust.
        importer = torch.package.PackageImporter(str(model_path))
        self.model = importer.load_pickle("model", "model.pkl")
        self.model.eval()  # switch off training-only behaviour

        # If we've fine-tuned the detector (training/finetune_detector.py),
        # swap in those weights.
        if config.FINETUNED_DETECTOR_FILE.exists():
            state = torch.load(config.FINETUNED_DETECTOR_FILE, map_location="cpu")
            self.model.detector.load_state_dict(state)
            print("Using fine-tuned detector.")

        # Use the NVIDIA GPU if PyTorch can see one (much faster), else the CPU.
        use_gpu = torch.cuda.is_available() and not config.FORCE_CPU
        self.device = "cuda" if use_gpu else "cpu"
        self.model.to(self.device)

    def _to_tensor(self, frame_bgr):
        """OpenCV frame (BGR, 0-255) -> batch of one RGB image (0-1) on the device."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = torchvision.transforms.functional.to_tensor(frame_rgb)
        return x.unsqueeze(0).to(self.device)

    @torch.no_grad()
    def detect(self, frame_bgr) -> list[BatDetection]:
        """Find the bats in one frame using only BatNet's (fast) detector."""
        output = self.model.detector(self._to_tensor(frame_bgr))[0]

        detections = []
        for box, score in zip(output["boxes"].tolist(), output["scores"].tolist()):
            if score < self.min_confidence:
                continue
            x1, y1, x2, y2 = (int(v) for v in box)
            detections.append(BatDetection(
                bbox=(x1, y1, x2, y2),
                confidence=score,
                species="",              # filled in later by classify_species()
                species_confidence=0.0,
            ))
        return detections

    @torch.no_grad()
    def classify_species(self, frame_bgr, bbox) -> tuple[str, float]:
        """
        Guess the species of the bat inside `bbox`.

        Returns (species, confidence). species is "" when BatNet thinks it
        is not a bat it knows (its "Not-A-Bat" class).
        """
        x = self._to_tensor(frame_bgr)
        # roi_align cuts the box out of the image and resizes it to the
        # square size the classifier expects. The leading 0 means "image 0
        # of the batch".
        box = torch.tensor([[0, *bbox]], dtype=torch.float32, device=self.device)
        crop = torchvision.ops.roi_align(
            x, box, self.model.classifier.image_size, sampling_ratio=1
        )
        probabilities = self.model.classifier(crop, T=0.75)[0]
        best = int(probabilities.argmax())
        return self.model.class_list[best], float(probabilities[best])
