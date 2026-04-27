# ============================================================
#   TRACK CLASS (state, covariance, counters, status)
# ============================================================

class Track:
    def __init__(self, track_id, initial_detection, sensor_id, timestamp):
        self.id = track_id

        # EKF state & covariance (filled later by EKF group)
        self.x = None
        self.P = None

        # Track lifecycle
        self.status = "tentative"   # tentative / confirmed / deleted
        self.hits = 0               # number of successful associations
        self.misses = 0             # consecutive missed detections

        # For M-of-N confirmation
        self.history = []           # store hit/miss history

        # For velocity initialization (finite difference)
        self.last_detection = initial_detection
        self.last_timestamp = timestamp

        # Sensor origin (needed for Jacobian)
        self.origin_sensor = sensor_id


# ============================================================
#   TRACK MANAGER (creates, updates, deletes tracks)
# ============================================================

class TrackManager:
    def __init__(self, M=3, N=5, K_del=5):
        self.tracks = []
        self.next_id = 0

        # Confirmation & deletion parameters
        self.M = M
        self.N = N
        self.K_del = K_del

    # ------------------------------
    # Create a new tentative track
    # ------------------------------
    def initiate_track(self, detection, sensor_id, timestamp):
        new_track = Track(self.next_id, detection, sensor_id, timestamp)
        self.tracks.append(new_track)
        self.next_id += 1

    # ------------------------------
    # Promote tentative → confirmed
    # ------------------------------
    def try_confirm(self, track):
        if len(track.history) >= self.N:
            recent = track.history[-self.N:]
            if recent.count("hit") >= self.M:
                track.status = "confirmed"

    # ------------------------------
    # Delete tracks with too many misses
    # ------------------------------
    def delete_old_tracks(self):
        self.tracks = [
            t for t in self.tracks
            if not (t.misses >= self.K_del or t.status == "deleted")
        ]

    # ------------------------------
    # Merge duplicate tracks
    # ------------------------------
    def merge_duplicates(self): # Kyriakos
        # Placeholder — real logic uses Mahalanobis distance
        pass


# ============================================================
#   GATING (Mahalanobis distance)
# ============================================================

def compute_mahalanobis(track, detection, sensor_id): # Kyriakos
    """
    Returns the Mahalanobis distance d^2 between a track and a detection.
    EKF group will provide:
        - h(x)
        - H
        - R
        - S = H P H^T + R
    """
    # Placeholder
    return None


def gating_matrix(tracks, detections, sensor_id, gate_threshold):
    """
    Returns a matrix (tracks x detections) with:
        True  = detection is inside gate
        False = detection is outside gate
    """
    gate = []
    for track in tracks:
        row = []
        for det in detections:
            d2 = compute_mahalanobis(track, det, sensor_id)
            row.append(d2 is not None and d2 <= gate_threshold)
        gate.append(row)
    return gate


# ============================================================
#   DATA ASSOCIATION (Hungarian / GNN)
# ============================================================

def build_cost_matrix(tracks, detections, gating): # Chiara
    """
    Build a cost matrix using Mahalanobis distance.
    If gating[i][j] is False → cost = large number.
    """
    # Placeholder
    return None


def assign_tracks(cost_matrix): # Chiara
    """
    Run Hungarian algorithm.
    Returns:
        matched_pairs = [(track_idx, det_idx), ...]
        unmatched_tracks = [...]
        unmatched_detections = [...]
    """
    # Placeholder
    return [], [], []


# ============================================================
#   MAIN UPDATE LOOP (called every scan)
# ============================================================

def process_scan(track_manager, detections, timestamp, sensor_id, gate_threshold):
    """
    Full T6 + T7 pipeline for one sensor scan.
    """

    # 1. Predict step (done by EKF group)
    # for track in track_manager.tracks:
    #     EKF.predict(track, timestamp)

    # 2. Gating
    gate = gating_matrix(track_manager.tracks, detections, sensor_id, gate_threshold)

    # 3. Cost matrix
    cost_matrix = build_cost_matrix(track_manager.tracks, detections, gate)

    # 4. Hungarian assignment
    matches, unmatched_tracks, unmatched_dets = assign_tracks(cost_matrix)

    # 5. Update matched tracks
    for t_idx, d_idx in matches:
        track = track_manager.tracks[t_idx]
        detection = detections[d_idx]

        # EKF update (done by EKF group)
        # EKF.update(track, detection, sensor_id)

        track.hits += 1
        track.misses = 0
        track.history.append("hit")
        track_manager.try_confirm(track)

    # 6. Handle unmatched tracks (coasting)
    for t_idx in unmatched_tracks:
        track = track_manager.tracks[t_idx]
        track.misses += 1
        track.history.append("miss")

    # 7. Create new tracks from unmatched detections
    for d_idx in unmatched_dets:
        detection = detections[d_idx]
        track_manager.initiate_track(detection, sensor_id, timestamp)

    # 8. Delete old tracks
    track_manager.delete_old_tracks()

    # 9. Merge duplicates
    track_manager.merge_duplicates()
