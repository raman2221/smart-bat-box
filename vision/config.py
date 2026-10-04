"""
config.py

Centralised configuration for the Smart Bat Box monitoring system.

All values can be overridden by setting environment variables with the
prefix ``SBB_`` (e.g. ``SBB_CAMERA_INDEX=1``).  Defaults are sensible
for a Windows laptop with an NVIDIA RTX 4070 and a USB webcam.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Root of the smart-bat-box project (two levels up from this file).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where BatNet pretrained weights are stored after download.
WEIGHTS_DIR = PROJECT_ROOT / "models" / "detection"
WEIGHTS_FILE = WEIGHTS_DIR / "basemodel.pt.zip"

# Directory for event log files.
EVENTS_DIR = PROJECT_ROOT / "data" / "events"

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

# OpenCV camera index.  0 = built-in webcam, 1 = first USB camera, etc.
CAMERA_INDEX = int(os.environ.get("SBB_CAMERA_INDEX", "0"))

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

# Minimum confidence score (0.0 – 1.0) to keep a detection.
DETECTION_CONFIDENCE_THRESHOLD = float(
    os.environ.get("SBB_DETECTION_CONFIDENCE", "0.8")
)

# Force CPU even when a GPU is available (useful for debugging).
FORCE_CPU = os.environ.get("SBB_FORCE_CPU", "0") == "1"

# ---------------------------------------------------------------------------
# Entrance Region of Interest (ROI)
#
# The ROI is defined as a horizontal band across the frame.  Anything
# above ``roi_y_top`` is the OUTSIDE zone; anything below
# ``roi_y_bottom`` is the INSIDE zone.  The band between them is the
# entrance zone.
#
# Values are *fractions* of the frame height (0.0 = top, 1.0 = bottom)
# so they work for any resolution.
# ---------------------------------------------------------------------------

ROI_Y_TOP = float(os.environ.get("SBB_ROI_Y_TOP", "0.35"))
ROI_Y_BOTTOM = float(os.environ.get("SBB_ROI_Y_BOTTOM", "0.65"))

# ---------------------------------------------------------------------------
# Automatic entrance detection (camera placement 1: camera outside the box,
# looking at the entry slot). See entrance_monitor.py for how it works:
#   came in across the frame edge, vanished in the middle  -> IN
#   appeared in the middle, left across the frame edge     -> OUT
# ---------------------------------------------------------------------------

# How close (fraction of the frame) a bat's box must be to the frame's
# border to count as "at the edge".
EDGE_MARGIN = float(os.environ.get("SBB_EDGE_MARGIN", "0.05"))

# After this many IN/OUT events, the entrance position is considered
# learned, and vanishings far from it are ignored.
ENTRANCE_LEARN_MIN_EVENTS = int(os.environ.get("SBB_ENTRANCE_LEARN_MIN_EVENTS", "5"))

# Smallest allowed radius of the learned entrance (fraction of the frame).
ENTRANCE_MIN_RADIUS = float(os.environ.get("SBB_ENTRANCE_MIN_RADIUS", "0.1"))

# Where the learned vanish/appear spots are saved between runs.
# Delete this file if the camera is moved, so it re-learns.
ENTRANCE_POINTS_FILE = PROJECT_ROOT / "data" / "entrance_points.json"

# A bat must be detected in at least this many frames before an IN/OUT
# event counts (filters out one-off false detections).
MIN_TRACK_DETECTIONS = int(os.environ.get("SBB_MIN_TRACK_DETECTIONS", "3"))

# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

# Maximum pixel distance between detections in consecutive frames to be
# considered the same bat.
MAX_TRACK_DISTANCE = int(os.environ.get("SBB_MAX_TRACK_DISTANCE", "100"))

# Number of consecutive frames a track can be missing before it is
# dropped.
MAX_FRAMES_MISSING = int(os.environ.get("SBB_MAX_FRAMES_MISSING", "15"))

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

# Whether to show the debug overlay windows.
SHOW_DEBUG_DISPLAY = os.environ.get("SBB_SHOW_DEBUG", "1") == "1"

# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

# Open Images training photos + annotations.csv.
OPENIMAGES_DIR = PROJECT_ROOT / "data" / "openimages"

# iNaturalist bat photos (training/download_data.py inaturalist).
INATURALIST_DIR = PROJECT_ROOT / "data" / "inaturalist"

# Bat boxes drawn automatically by training/auto_label.py.
AUTO_LABELS_FILE = PROJECT_ROOT / "data" / "auto_labels.csv"

# Synthetic training photos (real bat cutouts pasted onto "not a bat"
# backgrounds) made by training/synthesize_data.py. The labels file has
# the same columns as AUTO_LABELS_FILE: source,image,xmin,ymin,xmax,ymax,score
# (image relative to PROJECT_ROOT, boxes as 0-1 fractions).
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
SYNTHETIC_LABELS_FILE = SYNTHETIC_DIR / "labels.csv"

# Photos training must skip (one path per line, relative to PROJECT_ROOT),
# written by training/quality_check.py.
QC_EXCLUDE_FILE = PROJECT_ROOT / "data" / "qc_exclude.txt"

# Webcam frames of our own room with no bats in view, saved by
# training/capture_room_negatives.py. Used as "not a bat" examples.
ROOM_NEGATIVES_DIR = PROJECT_ROOT / "data" / "room_negatives"

# Fine-tuned detector weights, created by training/finetune_detector.py.
FINETUNED_DETECTOR_FILE = WEIGHTS_DIR / "finetuned_detector.pt"
