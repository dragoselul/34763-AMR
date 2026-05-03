"""
Track Manager — corrected to use the project's EKFTracker and
CoordinateFrameManager instead of the local stub helpers.

Key changes vs. the original file
──────────────────────────────────
1.  Removed the duplicated _h / _H / _R / SENSOR_POSITIONS / SENSOR_NOISE
    stubs.  All measurement-model calls now go through frame_manager.

2.  Track now owns an EKFTracker instance instead of raw x/P arrays.
    All read/write of track.x and track.P are proxied through track.ekf.

3.  process_scan predict/update stubs are replaced with real EKF calls:
      • predict  → track.ekf.motion_model dt update  +  track.ekf.predict()
      • update   → track.ekf.update(sensor_id, detection, time=timestamp)
      • AIS      → track.ekf.update_ais(z_ned, time=timestamp)

4.  Gating / cost matrix now call frame_manager (via _h / _H / _R wrappers
    that delegate to the manager) so range-gate health checks, vessel-position
    updates and noise covariances are all consistent with the EKF.

5.  AIS measurements arrive as Cartesian NED [pN, pE], not polar.  The AIS
    Mahalanobis distance uses the linear H = [[1,0,0,0],[0,1,0,0]] from
    EKFTracker.update_ais, matching update_ais exactly.

6.  init_velocity_from_second_hit now uses frame_manager for the polar→NED
    conversion instead of the removed SENSOR_POSITIONS dict.

7.  try_confirm now also resets status "coasting" → "confirmed" so a track
    that went coasting and then gets M hits in the last N scans recovers
    its confirmed status (the original only promoted tentative tracks).

8.  CVMotionModel is included here so the file is self-contained; replace
    sigma_a with a project-wide constant or inject it from outside.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

# Gate threshold: chi²(n_z=2, P_G=0.99) ≈ 9.210
DEFAULT_GATE_THRESHOLD = chi2.ppf(0.99, df=2)


# ============================================================
#   MOTION MODEL  (constant-velocity, matches test_EKFTracker)
# ============================================================

class CVMotionModel:
    """Constant-velocity process model."""

    def __init__(self, dt: float, sigma_a: float = 0.05) -> None:
        self.dt = float(dt)
        self.sigma_a = float(sigma_a)
        self._rebuild()

    def _rebuild(self) -> None:
        dt = self.dt
        q  = self.sigma_a ** 2
        self._F = np.array(
            [[1.0, 0.0, dt,  0.0],
             [0.0, 1.0, 0.0, dt ],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]],
            dtype=float,
        )
        self._Q = q * np.array(
            [[dt**4/4, 0,       dt**3/2, 0      ],
             [0,       dt**4/4, 0,       dt**3/2],
             [dt**3/2, 0,       dt**2,   0      ],
             [0,       dt**3/2, 0,       dt**2  ]],
            dtype=float,
        )

    @property
    def F(self) -> np.ndarray:
        return self._F.copy()

    @property
    def Q(self) -> np.ndarray:
        return self._Q.copy()

    def set_dt(self, dt: float) -> None:
        """Update the time step and recompute F and Q in place."""
        self.dt = float(dt)
        self._rebuild()


# ============================================================
#   ANGLE HELPER
# ============================================================

def _wrap(a: float) -> float:
    """Wrap angle to [-π, π]."""
    return float((a + np.pi) % (2 * np.pi) - np.pi)


# ============================================================
#   TRACK CLASS
# ============================================================

class Track:
    """
    A single tracked target.  State and covariance live inside
    the EKFTracker instance (self.ekf).  Accessors .x and .P
    delegate to it for convenience.
    """

    def __init__(
        self,
        track_id: int,
        initial_detection: np.ndarray,
        sensor_id: str,
        timestamp: float,
        frame_manager,          # CoordinateFrameManager
        sigma_a: float = 0.05,
        dt_init: float = 1.0,   # placeholder until first predict
    ) -> None:
        from .EKFTracker import EKFTracker   # local import avoids circular deps

        self.id            = track_id
        self.origin_sensor = sensor_id
        self.frame_manager = frame_manager

        # ── Initialise position from first polar detection ──────────────
        sensor_pos = frame_manager.get_sensor_position(sensor_id)   # (pN, pE)
        r, phi     = float(initial_detection[0]), float(initial_detection[1])
        pN = sensor_pos[0] + r * np.cos(phi)
        pE = sensor_pos[1] + r * np.sin(phi)
        x0 = np.array([pN, pE, 0.0, 0.0], dtype=float)

        # ── Initial covariance ──────────────────────────────────────────
        R        = frame_manager.get_noise_covariance(sensor_id)   # 2×2 polar
        sigma_r  = float(np.sqrt(R[0, 0]))
        sigma_phi = float(np.sqrt(R[1, 1]))
        sigma_pos = max(sigma_r, r * sigma_phi)
        P0 = np.diag([sigma_pos**2, sigma_pos**2, 100.0**2, 100.0**2]).astype(float)

        motion_model = CVMotionModel(dt=dt_init, sigma_a=sigma_a)
        self.ekf = EKFTracker(
            x0=x0,
            P0=P0,
            motion_model=motion_model,
            frame_manager=frame_manager,
            initial_time=timestamp,
        )

        # ── Track lifecycle ─────────────────────────────────────────────
        self.status  = "tentative"
        self.hits    = 0
        self.misses  = 0
        self.history: list[str] = []

        # ── Velocity initialisation via finite difference ───────────────
        self.last_detection               = np.array(initial_detection, dtype=float)
        self.last_timestamp               = timestamp
        self._velocity_initialised        = False

    # ── Convenience proxies ────────────────────────────────────────────

    @property
    def x(self) -> np.ndarray:
        return self.ekf.x

    @x.setter
    def x(self, value: np.ndarray) -> None:
        self.ekf.x = value

    @property
    def P(self) -> np.ndarray:
        return self.ekf.P

    @P.setter
    def P(self, value: np.ndarray) -> None:
        self.ekf.P = value

    # ── Velocity initialisation ────────────────────────────────────────

    def init_velocity_from_second_hit(
        self,
        detection: np.ndarray,
        timestamp: float,
    ) -> None:
        """
        Estimate velocity via finite difference between the NED positions
        implied by two consecutive polar measurements from the same sensor.
        Uses frame_manager for sensor position (replaces removed SENSOR_POSITIONS).
        """
        if self._velocity_initialised:
            return
        dt = timestamp - self.last_timestamp
        if dt <= 0.0:
            return

        sensor_pos = self.frame_manager.get_sensor_position(self.origin_sensor)
        r1, phi1 = float(self.last_detection[0]), float(self.last_detection[1])
        r2, phi2 = float(detection[0]),           float(detection[1])

        pN1 = sensor_pos[0] + r1 * np.cos(phi1)
        pE1 = sensor_pos[1] + r1 * np.sin(phi1)
        pN2 = sensor_pos[0] + r2 * np.cos(phi2)
        pE2 = sensor_pos[1] + r2 * np.sin(phi2)

        self.ekf.x[2] = (pN2 - pN1) / dt
        self.ekf.x[3] = (pE2 - pE1) / dt
        self._velocity_initialised = True


# ============================================================
#   TRACK MANAGER
# ============================================================

class TrackManager:
    def __init__(self, M: int = 3, N: int = 5, K_del: int = 5) -> None:
        self.tracks:   list[Track] = []
        self.next_id:  int         = 0
        self.M     = M
        self.N     = N
        self.K_del = K_del

    def initiate_track(
        self,
        detection: np.ndarray,
        sensor_id: str,
        timestamp: float,
        frame_manager,
        sigma_a: float = 0.05,
    ) -> None:
        """Create a new tentative track from an unmatched detection."""
        new_track = Track(
            track_id=self.next_id,
            initial_detection=detection,
            sensor_id=sensor_id,
            timestamp=timestamp,
            frame_manager=frame_manager,
            sigma_a=sigma_a,
        )
        self.tracks.append(new_track)
        self.next_id += 1

    def try_confirm(self, track: Track) -> None:
        """
        Promote tentative → confirmed  OR  coasting → confirmed
        if the last N scans contain at least M hits.
        """
        if track.status not in ("tentative", "coasting"):
            return
        if len(track.history) >= self.N:
            recent = track.history[-self.N:]
            if recent.count("hit") >= self.M:
                track.status = "confirmed"

    def delete_old_tracks(self) -> None:
        self.tracks = [
            t for t in self.tracks
            if not (t.misses >= self.K_del or t.status == "deleted")
        ]

    def merge_duplicates(self, merge_threshold: float = 9.21) -> None:
        """
        Merge confirmed/coasting track pairs whose position estimates are
        within merge_threshold (Mahalanobis² in the 2-D position subspace).
        The track with fewer hits is marked deleted.
        """
        active = [t for t in self.tracks if t.status in ("confirmed", "coasting")]
        to_delete: set[int] = set()

        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                ta, tb = active[i], active[j]
                if ta.id in to_delete or tb.id in to_delete:
                    continue
                dx = ta.x[:2] - tb.x[:2]
                S  = ta.P[:2, :2] + tb.P[:2, :2]
                try:
                    d2 = float(dx @ np.linalg.inv(S) @ dx)
                except np.linalg.LinAlgError:
                    continue
                if d2 < merge_threshold:
                    loser = tb.id if ta.hits >= tb.hits else ta.id
                    to_delete.add(loser)

        for track in self.tracks:
            if track.id in to_delete:
                track.status = "deleted"


# ============================================================
#   GATING  (Mahalanobis distance via frame_manager)
# ============================================================

def compute_mahalanobis(
    track: Track,
    detection: np.ndarray,
    sensor_id: str,
) -> float | None:
    """
    Mahalanobis distance d² = y^T S^{-1} y  where
      y = z − h(x⁻)
      S = H P⁻ H^T + R
    computed via frame_manager so it is consistent with the EKF update.

    AIS measurements are Cartesian [pN, pE], not polar.  The linear
    observation model H = [[1,0,0,0],[0,1,0,0]] is used, matching
    EKFTracker.update_ais exactly.
    """
    fm    = track.frame_manager
    x     = track.x
    P     = track.P
    sid   = sensor_id.lower() if isinstance(sensor_id, str) else sensor_id

    if sid == "ais":
        # AIS: Cartesian NED measurement
        z      = np.asarray(detection, dtype=float)
        z_pred = x[:2]
        H      = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=float)
        R      = fm.get_noise_covariance("ais")
        y      = z - z_pred
    else:
        # Radar / camera: polar measurement via frame_manager
        z_pred = fm.compute_measurement(sid, x)
        H      = fm.compute_jacobian(sid, x)
        R      = fm.get_noise_covariance(sid)

        if z_pred is None or H is None:
            return None

        z = np.asarray(detection, dtype=float)
        y = z - z_pred
        y[1] = _wrap(y[1])   # wrap bearing innovation

    S = H @ P @ H.T + R
    try:
        d2 = float(y @ np.linalg.inv(S) @ y)
    except np.linalg.LinAlgError:
        return None

    return d2


def gating_matrix(
    tracks: list[Track],
    detections: list[np.ndarray],
    sensor_id: str,
    gate_threshold: float,
) -> list[list[bool]]:
    """Boolean matrix (n_tracks × n_detections): True = inside gate."""
    return [
        [
            (lambda d2: d2 is not None and d2 <= gate_threshold)(
                compute_mahalanobis(track, det, sensor_id)
            )
            for det in detections
        ]
        for track in tracks
    ]


# ============================================================
#   DATA ASSOCIATION  (Hungarian / GNN)
# ============================================================

def build_cost_matrix(
    tracks: list[Track],
    detections: list[np.ndarray],
    gating: list[list[bool]],
    sensor_id: str,
) -> np.ndarray:
    """
    Cost matrix (n_tracks × n_detections).
    Entry (i,j) = Mahalanobis d²  if inside gate, else 1e9.

    sensor_id is now an explicit parameter (removed track.origin_sensor
    look-up which was wrong for multi-sensor scans).
    """
    LARGE  = 1e9
    n_t, n_d = len(tracks), len(detections)
    if n_t == 0 or n_d == 0:
        return np.full((n_t, n_d), LARGE)

    cost = np.full((n_t, n_d), LARGE)
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            if not gating[i][j]:
                continue
            d2 = compute_mahalanobis(track, det, sensor_id)
            if d2 is not None:
                cost[i, j] = d2
    return cost


def assign_tracks(
    cost_matrix: np.ndarray,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Hungarian algorithm.  Pairs whose cost == 1e9 are unmatched.

    Returns
    -------
    matched_pairs        : [(track_idx, det_idx), ...]
    unmatched_tracks     : [track_idx, ...]
    unmatched_detections : [det_idx, ...]
    """
    LARGE = 1e9
    if cost_matrix is None or cost_matrix.size == 0:
        n_t = cost_matrix.shape[0] if cost_matrix is not None else 0
        n_d = cost_matrix.shape[1] if cost_matrix is not None else 0
        return [], list(range(n_t)), list(range(n_d))

    n_t, n_d   = cost_matrix.shape
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_pairs  = []
    matched_tracks = set()
    matched_dets   = set()

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] < LARGE:
            matched_pairs.append((r, c))
            matched_tracks.add(r)
            matched_dets.add(c)

    unmatched_tracks     = [i for i in range(n_t) if i not in matched_tracks]
    unmatched_detections = [j for j in range(n_d) if j not in matched_dets]
    return matched_pairs, unmatched_tracks, unmatched_detections


# ============================================================
#   MAIN UPDATE LOOP  (called every scan)
# ============================================================

def process_scan(
    track_manager: TrackManager,
    detections: list[np.ndarray],
    timestamp: float,
    sensor_id: str,
    frame_manager,
    gate_threshold: float = DEFAULT_GATE_THRESHOLD,
    sigma_a: float = 0.05,
) -> None:
    """
    Full pipeline for one sensor scan.

    Parameters
    ----------
    track_manager  : TrackManager instance
    detections     : list of measurement arrays for this scan.
                     Radar/camera: [range_m, bearing_rad]
                     AIS:          [p_north_m, p_east_m]
    timestamp      : current time in seconds
    sensor_id      : "radar" | "camera" | "ais"
    frame_manager  : CoordinateFrameManager instance
    gate_threshold : Mahalanobis² gate threshold (default χ²(2, 0.99) ≈ 9.21)
    sigma_a        : process-noise std dev for newly created tracks
    """
    sid = sensor_id.lower() if isinstance(sensor_id, str) else sensor_id

    # ── 1. Predict every track forward to current timestamp ────────────
    for track in track_manager.tracks:
        dt = timestamp - track.last_timestamp
        if dt > 0.0:
            track.ekf.motion_model.set_dt(dt)
            track.ekf.predict()
            # last_timestamp is updated on a hit (step 5) or kept here for
            # coasting tracks so the next predict uses the right dt.
            track.last_timestamp = timestamp

    # ── 2. Gating ──────────────────────────────────────────────────────
    gate = gating_matrix(track_manager.tracks, detections, sid, gate_threshold)

    # ── 3. Cost matrix ─────────────────────────────────────────────────
    cost_matrix = build_cost_matrix(track_manager.tracks, detections, gate, sid)

    # ── 4. Hungarian assignment ────────────────────────────────────────
    matches, unmatched_tracks, unmatched_dets = assign_tracks(cost_matrix)

    # ── 5. Update matched tracks ───────────────────────────────────────
    for t_idx, d_idx in matches:
        track     = track_manager.tracks[t_idx]
        detection = np.asarray(detections[d_idx], dtype=float)

        # Velocity finite-difference on the second hit
        if not track._velocity_initialised and track.hits == 0 and sid != "ais":
            track.init_velocity_from_second_hit(detection, timestamp)

        # EKF measurement update via EKFTracker
        if sid == "ais":
            # AIS: measurement is already Cartesian NED [pN, pE]
            track.ekf.update_ais(detection, time=timestamp)
        else:
            # Radar / camera: polar measurement [range_m, bearing_rad]
            track.ekf.update(sid, detection, time=timestamp)

        track.hits   += 1
        track.misses  = 0
        track.last_detection = detection.copy()
        track.last_timestamp = timestamp
        track.history.append("hit")
        track_manager.try_confirm(track)

    # ── 6. Unmatched tracks → coasting ────────────────────────────────
    for t_idx in unmatched_tracks:
        track = track_manager.tracks[t_idx]
        track.misses += 1
        track.history.append("miss")
        if track.status == "confirmed":
            track.status = "coasting"

    # ── 7. Unmatched detections → new tentative tracks ─────────────────
    for d_idx in unmatched_dets:
        track_manager.initiate_track(
            detection=np.asarray(detections[d_idx], dtype=float),
            sensor_id=sid,
            timestamp=timestamp,
            frame_manager=frame_manager,
            sigma_a=sigma_a,
        )

    # ── 8. Delete tracks exceeding K_del consecutive misses ────────────
    track_manager.delete_old_tracks()

    # ── 9. Merge duplicate confirmed/coasting tracks ───────────────────
    track_manager.merge_duplicates()
