"""
entrance_tracker.py

Watches the Smart Bat Box entrance, tracks bats, and records when they go
IN or come OUT (with a snapshot of each one).

What it does:
  1. Opens the webcam.
  2. Uses background subtraction to notice when something is moving
     (nothing is drawn for this; it just decides when to run BatNet).
  3. While something is moving (or a bat is being tracked), runs BatNet's
     detector on every frame to find bats.
  4. Gives each bat an ID and follows it across frames (bat_tracker.py).
  5. Shows just one thing on screen: "BAT DETECTED" or "NOT A BAT".
  6. When a bat's path ends, decides IN or OUT (entrance_monitor.py),
     saves a snapshot to data/events/ and adds a line to events.csv.
     No setup needed: it learns where the entrance is by itself.

Press "q" in the video window to quit.

Run it from the project folder with:
    python vision/entrance_tracker.py
"""

import sys
from pathlib import Path

import cv2

# Let Python find the "vision" package when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision import config  # noqa: E402
from vision.bat_detector import BatDetector  # noqa: E402
from vision.bat_tracker import BatTracker  # noqa: E402
from vision.entrance_monitor import EntranceMonitor  # noqa: E402

# ---------------------------------------------------------------------------
# Settings you can tweak
# ---------------------------------------------------------------------------

# Index of the camera to use. 0 = the default webcam.
CAMERA_INDEX = 1

# Moving blobs smaller than this (in pixels) are ignored.
# Raise it if you see boxes around tiny flickers; lower it to catch small objects.
MIN_CONTOUR_AREA = 50

WINDOW = "Entrance Tracker"

# Colors are (Blue, Green, Red).
BAT_COLOR = (0, 255, 0)        # green
NO_BAT_COLOR = (0, 0, 255)     # red


def bat_in_view(tracks):
    """True if a bat is being tracked right now.

    A bat must have been seen in a few frames (MIN_TRACK_DETECTIONS) before
    we say "BAT DETECTED", so a one-frame mistake doesn't flash it on.
    The tracker keeps a bat for a moment after a missed frame, so the
    message doesn't flicker either.
    """
    return any(len(track.history) >= config.MIN_TRACK_DETECTIONS for track in tracks)


def draw_status(display, bat_present):
    """Write "BAT DETECTED" or "NOT A BAT" in the top-left corner."""
    text, color = ("BAT DETECTED", BAT_COLOR) if bat_present else ("NOT A BAT", NO_BAT_COLOR)
    cv2.putText(display, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)


def main():
    # -----------------------------------------------------------------------
    # 1. Load BatNet (takes a few seconds)
    # -----------------------------------------------------------------------
    print("Loading BatNet model...")
    try:
        detector = BatDetector()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    print(f"BatNet loaded (running on {detector.device}).")
    if detector.device == "cpu":
        print("Warning: no GPU found, video will be slow (a few frames per second).")

    tracker = BatTracker()
    monitor = EntranceMonitor(detector)

    # -----------------------------------------------------------------------
    # 2. Open the webcam
    # -----------------------------------------------------------------------
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open the webcam.")
        return

    # -----------------------------------------------------------------------
    # 3. Create the background subtractor
    #
    # MOG2 "learns" what the background looks like over the last `history`
    # frames. Anything that differs from that learned background is marked
    # as foreground (white) in a black-and-white "mask" image.
    #   - varThreshold: higher = less sensitive to small changes
    #   - detectShadows: shadows are marked grey (127) so we can remove them
    # -----------------------------------------------------------------------
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )

    # A small round "brush" used to clean up noise in the mask (see step 5).
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    print("Tracking started. Press 'q' to quit.")

    try:
        while True:
            # ---------------------------------------------------------------
            # 4. Grab one frame from the camera
            # ---------------------------------------------------------------
            ok, frame = cap.read()
            if not ok:
                print("Error: could not read a frame from the webcam.")
                break

            # ---------------------------------------------------------------
            # 5. Is anything moving? (background subtraction)
            # ---------------------------------------------------------------
            # White = moving, black = background, grey = shadow.
            fg_mask = back_sub.apply(frame)

            # Keep only the pure white pixels, which throws away the grey shadows.
            _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

            # "Opening" removes small specks of noise.
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

            # "Dilating" grows the white areas so one object isn't split into pieces.
            fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)

            contours, _ = cv2.findContours(
                fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            motion = any(cv2.contourArea(c) >= MIN_CONTOUR_AREA for c in contours)

            # ---------------------------------------------------------------
            # 6. Find bats with BatNet's detector
            #
            # Only while something is moving, or while we're still following
            # a bat (so a bat that stops moving isn't lost).
            # ---------------------------------------------------------------
            if motion or tracker.tracks:
                detections = detector.detect(frame)
            else:
                detections = []

            # ---------------------------------------------------------------
            # 7. Update the tracker (once per frame, even with no detections,
            #    so bats that disappear are eventually dropped)
            # ---------------------------------------------------------------
            tracks = tracker.update(detections)

            # ---------------------------------------------------------------
            # 8. Check for bats going IN or OUT
            #
            # The monitor keeps each bat's clearest frame, and when a bat's
            # path ends it decides IN/OUT and saves the snapshot. Events are
            # only printed in the terminal, not drawn on the video.
            # ---------------------------------------------------------------
            for event in monitor.update(tracks, frame):
                print(f"[{event.time:%H:%M:%S}] Bat went {event.direction} "
                      f"-> photo saved: data/events/{event.photo}")

            # ---------------------------------------------------------------
            # 9. Show the video with just "BAT DETECTED" / "NOT A BAT"
            # ---------------------------------------------------------------
            display = frame.copy()
            draw_status(display, bat_in_view(tracks))
            cv2.imshow(WINDOW, display)

            # ---------------------------------------------------------------
            # 10. Quit when "q" is pressed
            #
            # waitKey(1) waits 1 ms for a key press and also lets OpenCV
            # refr "& 0xFF" keeps just the key code.
            # ---------------------------------------------------------------
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        # -------------------------------------------------------------------
        # 11. Clean up: drop bats still being tracked, release the camera
        #     and close the window, even if the program crashes or you
        #     press Ctrl+C.
        # -------------------------------------------------------------------
        monitor.finish()
        cap.release()
        cv2.destroyAllWindows()
        print(f"Tracker stopped. IN: {monitor.count_in}  OUT: {monitor.count_out}")


if __name__ == "__main__":
    main()
