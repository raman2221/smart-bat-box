"""
bat_tracker.py — Simple centroid-based multi-object tracker.

This module assigns persistent **track IDs** to bats detected across
consecutive video frames.  It is intentionally simple: no Kalman filter,
no deep-learning re-identification — just nearest-neighbour matching on
the centre point of each bounding box.

Why centroid tracking works here
--------------------------------
Bats at a bat-box entrance are relatively few (single digits), move at
moderate speed between frames, and rarely overlap.  In this scenario,
matching each new detection to the nearest existing track is reliable
enough.  If the scene becomes more crowded we can swap in a more
sophisticated tracker (e.g. SORT / DeepSORT) later because the
``TrackedBat`` interface would stay the same.

Algorithm
---------
Each frame:

1. Receive a list of ``BatDetection`` from the detector.
2. Compute a distance matrix between every existing track's *last
   known position* and every new detection's centre.
3. Greedily match pairs by shortest distance, subject to a maximum
   distance threshold (``MAX_TRACK_DISTANCE``).
4. For matched pairs → update the track with the new position.
5. For unmatched detections → create a brand-new track.
6. For unmatched tracks → increment their "missing" counter.  If a
   track has been missing for too many frames
   (``MAX_FRAMES_MISSING``) it is retired.

Public API
----------
>>> tracker = BatTracker()
>>> tracked = tracker.update(detections)   # returns list[TrackedBat]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bat_detector import BatDetection


# ---------------------------------------------------------------------------
# Data class representing one tracked bat
# ---------------------------------------------------------------------------

@dataclass
class TrackedBat:
    """A bat that is being tracked across frames.

    Attributes
    ----------
    track_id : int
        Unique integer identifying this track for its lifetime.
    center_x : float
        Current horizontal centre (pixels).
    center_y : float
        Current vertical centre (pixels).
    bbox : tuple[float, float, float, float]
        Current bounding box ``(x1, y1, x2, y2)`` in pixels.
    confidence : float
        Detection confidence from the most recent frame.
    frames_missing : int
        How many consecutive frames this bat has gone undetected.
        Zero means it was seen in the current frame.
    history : list[tuple[float, float]]
        Time-ordered list of ``(center_x, center_y)`` positions.
        This is used by the entrance monitor to decide crossing
        direction.
    """

    track_id: int
    center_x: float
    center_y: float
    bbox: tuple
    confidence: float
    frames_missing: int = 0
    history: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

class BatTracker:
    """Centroid-based multi-object tracker for detected bats.

    Parameters
    ----------
    max_distance : int
        Maximum Euclidean distance (in pixels) between a track and a
        new detection for them to be considered the same bat.  If a
        bat moves further than this between two frames it will be
        treated as a new track.
    max_frames_missing : int
        Number of consecutive frames a track can go unmatched before
        it is deleted.  This adds robustness against brief detection
        failures (e.g. motion blur).

    Attributes
    ----------
    tracks : dict[int, TrackedBat]
        Currently active tracks, keyed by track ID.
    """

    def __init__(
        self,
        max_distance: int | None = None,
        max_frames_missing: int | None = None,
    ):
        # Import defaults from config if not overridden.
        from . import config

        self.max_distance = (
            max_distance if max_distance is not None else config.MAX_TRACK_DISTANCE
        )
        self.max_frames_missing = (
            max_frames_missing
            if max_frames_missing is not None
            else config.MAX_FRAMES_MISSING
        )

        # Active tracks, keyed by their unique ID.
        self.tracks: dict[int, TrackedBat] = {}

        # Monotonically increasing counter used to assign unique IDs.
        self._next_id: int = 1

    # ------------------------------------------------------------------
    # Core update step — called once per frame
    # ------------------------------------------------------------------

    def update(self, detections: list[BatDetection]) -> list[TrackedBat]:
        """Match new detections to existing tracks and return results.

        Parameters
        ----------
        detections : list[BatDetection]
            Detections in the *current* frame (may be empty).

        Returns
        -------
        list[TrackedBat]
            All *active* tracks (including those that were not matched
            this frame, as long as they haven't timed out yet).

        Step-by-step
        ------------
        1. If there are no existing tracks, every detection starts a
           new track.
        2. Otherwise, build a distance matrix between existing track
           centres and new detection centres.
        3. Greedily assign the closest pairs (below ``max_distance``).
        4. Create new tracks for any leftover detections.
        5. Age out tracks that have been missing too long.
        """

        # --- Shortcut: no existing tracks → just create new ones -------
        if not self.tracks:
            for det in detections:
                self._create_track(det)
            return list(self.tracks.values())

        # --- Shortcut: no detections this frame → age all tracks -------
        if not detections:
            self._age_and_prune()
            return list(self.tracks.values())

        # --- Build distance matrix (tracks × detections) ---------------
        track_ids = list(self.tracks.keys())
        track_centres = [
            (self.tracks[tid].center_x, self.tracks[tid].center_y)
            for tid in track_ids
        ]

        det_centres = [(d.center_x, d.center_y) for d in detections]

        # ``distances[i][j]`` = Euclidean distance between track *i*
        # and detection *j*.
        distances: list[list[float]] = []
        for tx, ty in track_centres:
            row = []
            for dx, dy in det_centres:
                row.append(((tx - dx) ** 2 + (ty - dy) ** 2) ** 0.5)
            distances.append(row)

        # --- Greedy matching -------------------------------------------
        #
        # We repeatedly pick the globally shortest distance, assign
        # that pair, and remove both from further consideration.
        # This is O(N*M) which is fine for a handful of bats.

        matched_tracks: set[int] = set()     # indices into track_ids
        matched_dets: set[int] = set()       # indices into detections

        n_tracks = len(track_ids)
        n_dets = len(detections)

        for _ in range(min(n_tracks, n_dets)):
            # Find the minimum distance among all unmatched pairs.
            best_dist = float("inf")
            best_t = -1
            best_d = -1
            for t_idx in range(n_tracks):
                if t_idx in matched_tracks:
                    continue
                for d_idx in range(n_dets):
                    if d_idx in matched_dets:
                        continue
                    if distances[t_idx][d_idx] < best_dist:
                        best_dist = distances[t_idx][d_idx]
                        best_t = t_idx
                        best_d = d_idx

            # If the best remaining pair is too far apart, stop.
            if best_dist > self.max_distance:
                break

            # Assign: update the track with the new detection.
            tid = track_ids[best_t]
            det = detections[best_d]
            self._update_track(tid, det)

            matched_tracks.add(best_t)
            matched_dets.add(best_d)

        # --- Create new tracks for unmatched detections ----------------
        for d_idx in range(n_dets):
            if d_idx not in matched_dets:
                self._create_track(detections[d_idx])

        # --- Mark unmatched tracks as missing & prune old ones ---------
        for t_idx in range(n_tracks):
            if t_idx not in matched_tracks:
                tid = track_ids[t_idx]
                self.tracks[tid].frames_missing += 1

        self._prune()

        return list(self.tracks.values())

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_track(self, det: BatDetection) -> TrackedBat:
        """Start tracking a new bat.

        Assigns a unique ID and records the initial position in the
        history list.
        """
        track = TrackedBat(
            track_id=self._next_id,
            center_x=det.center_x,
            center_y=det.center_y,
            bbox=det.bbox,
            confidence=det.confidence,
            history=[(det.center_x, det.center_y)],
        )
        self.tracks[self._next_id] = track
        self._next_id += 1
        return track

    def _update_track(self, track_id: int, det: BatDetection) -> None:
        """Update an existing track with a new detection.

        Resets the ``frames_missing`` counter and appends the new
        position to the history.
        """
        track = self.tracks[track_id]
        track.center_x = det.center_x
        track.center_y = det.center_y
        track.bbox = det.bbox
        track.confidence = det.confidence
        track.frames_missing = 0
        track.history.append((det.center_x, det.center_y))

    def _age_and_prune(self) -> None:
        """Increment missing counter on *all* tracks, then prune."""
        for track in self.tracks.values():
            track.frames_missing += 1
        self._prune()

    def _prune(self) -> None:
        """Remove tracks that have been missing for too long."""
        stale = [
            tid
            for tid, track in self.tracks.items()
            if track.frames_missing > self.max_frames_missing
        ]
        for tid in stale:
            del self.tracks[tid]
