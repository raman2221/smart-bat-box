"""
capture_room_negatives.py

Saves webcam frames of the room (with NO bats in view) so the detector can
learn that lights, windows, people, screens etc. in *our* room are not bats.

How to use:
  1. Make sure no bat pictures are on screen or in view.
  2. Run from the project folder (with the .venv active):
         python training/capture_room_negatives.py
  3. Slowly point the camera around the room: lights, windows, the ceiling,
     laptops, anything that was mistaken for a bat. Get yourself (and your
     teammates) in the picture too: move around, wave, turn your head, come
     close and step back. The detector was built from a person detector,
     so frames of you are the most useful "not a bat" examples there are.
  4. It saves a frame every half second and stops by itself after
     MAX_FRAMES frames (or press "q" to stop early).
  5. Re-run training:  python training/finetune_detector.py
"""

import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from vision import config  # noqa: E402
from vision.entrance_tracker import CAMERA_INDEX  # noqa: E402

MAX_FRAMES = 200
SECONDS_BETWEEN_FRAMES = 0.5


def main():
    out_dir = config.ROOM_NEGATIVES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # Continue numbering after any frames saved in earlier runs.
    start = len(list(out_dir.glob("room_*.jpg")))

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("Error: could not open the webcam.")
        return

    print(f"Saving up to {MAX_FRAMES} frames to {out_dir}. Press 'q' to stop.")
    saved = 0
    last_save = 0.0
    try:
        while saved < MAX_FRAMES:
            ok, frame = cap.read()
            if not ok:
                print("Error: could not read a frame from the webcam.")
                break

            if time.time() - last_save >= SECONDS_BETWEEN_FRAMES:
                path = out_dir / f"room_{start + saved:04d}.jpg"
                cv2.imwrite(str(path), frame)
                saved += 1
                last_save = time.time()

            preview = frame.copy()
            cv2.putText(preview, f"Saved {saved}/{MAX_FRAMES}  (q = stop)", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Capture room (no bats!)", preview)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved {saved} frames.")


if __name__ == "__main__":
    main()
