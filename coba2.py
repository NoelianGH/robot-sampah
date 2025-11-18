# multi_fall_mediapipe_tasks.py
import os
import time
from collections import deque, defaultdict
import cv2
import numpy as np

import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# =======================
# CONFIG (tweakable)
# =======================
MODELS_DIR = "models"
POSE_MODEL = os.path.join(MODELS_DIR, "pose_landmarker_full.task")
FACE_MODEL = os.path.join(MODELS_DIR, "face_landmarker.task")

MAX_POSES = 6                    # deteksi hingga 6 orang
VEL_BUFFER = 6                   # frame buffer utk hitung velocity
VEL_THRESHOLD = 0.015            # semakin kecil -> semakin sensitif jatuh
HORIZONTAL_ANGLE_THRESH = 25.0   # derajat ke horizontal; makin besar -> lebih mudah dianggap mendatar
ASPECT_RATIO_THRESH = 0.70       # bbox_h / bbox_w ; kecil -> kemungkinan telentang/terlentang
ASSOC_DIST_THRESH = 0.12         # ambang jarak untuk asosiakan pose ke track (normalized)
FALL_DEBOUNCE_SEC = 1.0          # jarak minimal antar alert per track
DRAW_FACE = True                 # gambar face mesh multi-face
SAVE_SNAPSHOT = True
SNAPSHOT_DIR = "fall_snapshots"

# Warna utility
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
WHITE = (255, 255, 255)

os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# =======================
# Utility
# =======================
def put_text_bg(img, text, org, fg=WHITE, bg=(0,0,0), scale=0.6, thick=2):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x, y = org
    cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), bg, -1)
    cv2.putText(img, text, (x + 3, y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)

def angle_to_horizontal(p1, p2):
    # p1, p2: np.array [x,y] normalized
    v = p2 - p1
    ang = abs(np.degrees(np.arctan2(v[1], v[0])))  # 0=horizontal, 90=vertical
    return ang

def bbox_from_points(points_xy):
    xs = [p[0] for p in points_xy]
    ys = [p[1] for p in points_xy]
    return (min(xs), min(ys), max(xs), max(ys))

def draw_pose_skeleton(img, landmarks, w, h, color=CYAN, radius=3, thick=2):
    # landmarks: list of (x,y) normalized
    # Minimal connections utk torso & kaki & tangan
    C = [
        (11,12), (11,23), (12,24), (23,24),       # shoulders/hips box
        (11,13), (13,15), (12,14), (14,16),       # arms
        (23,25), (25,27), (24,26), (26,28),       # legs (upper to lower)
        (27,29), (29,31), (28,30), (30,32)        # ankles to feet
    ]
    # draw lines
    for a,b in C:
        if 0 <= a < len(landmarks) and 0 <= b < len(landmarks):
            ax, ay = int(landmarks[a][0]*w), int(landmarks[a][1]*h)
            bx, by = int(landmarks[b][0]*w), int(landmarks[b][1]*h)
            cv2.line(img, (ax,ay), (bx,by), color, thick)
    # draw keypoints
    for x,y in landmarks:
        cv2.circle(img, (int(x*w), int(y*h)), radius, color, -1)

def draw_face_mesh(img, face_landmarks, w, h, color=GREEN, radius=1):
    # Gambar titik-titik (tanpa tesselation penuh agar cepat)
    for lm in face_landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(img, (x,y), radius, color, -1)

# =======================
# Simple Multi-Target Tracker (by hip midpoint)
# =======================
class PoseTracker:
    def __init__(self, maxlen=VEL_BUFFER):
        self.next_id = 1
        self.tracks = {}  # id -> dict
        self.maxlen = maxlen

    def _new_track(self, hip_xy):
        tid = self.next_id
        self.next_id += 1
        self.tracks[tid] = {
            "hip_y": deque(maxlen=self.maxlen),
            "time": deque(maxlen=self.maxlen),
            "last_xy": np.array(hip_xy, dtype=np.float32),
            "last_alert": 0.0,
        }
        return tid

    def associate(self, hips_list):
        """
        hips_list: list of np.array([x,y]) for each detected person
        returns: list of (track_id, hip_xy)
        """
        assigned = []
        unmatched_tracks = set(self.tracks.keys())
        results = []

        for hip in hips_list:
            # find nearest track within threshold
            best_id, best_dist = None, 1e9
            for tid in unmatched_tracks:
                dist = np.linalg.norm(self.tracks[tid]["last_xy"] - hip)
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is not None and best_dist <= ASSOC_DIST_THRESH:
                # assign
                unmatched_tracks.remove(best_id)
                assigned.append(best_id)
                self.tracks[best_id]["last_xy"] = hip
                results.append((best_id, hip))
            else:
                # create new
                tid = self._new_track(hip)
                assigned.append(tid)
                results.append((tid, hip))

        # (Optional) could drop very stale tracks if needed
        return results

    def update_measure(self, tid, hip_y, tstamp):
        self.tracks[tid]["hip_y"].append(hip_y)
        self.tracks[tid]["time"].append(tstamp)

    def velocity(self, tid):
        buf_y = self.tracks[tid]["hip_y"]
        buf_t = self.tracks[tid]["time"]
        if len(buf_y) < 2:
            return 0.0
        dy = buf_y[-1] - buf_y[0]  # + means going down
        dt = buf_t[-1] - buf_t[0]
        if dt <= 1e-6: return 0.0
        return dy / dt

# =======================
# Main
# =======================
def main():
    # --- Prepare camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    # --- Pose Landmarker (multi-pose)
    base_opts = mp_python.BaseOptions(model_asset_path=POSE_MODEL)
    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=MAX_POSES,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    pose_detector = mp_vision.PoseLandmarker.create_from_options(pose_opts)

    # --- Face Landmarker (multi-face)
    if DRAW_FACE:
        face_base = mp_python.BaseOptions(model_asset_path=FACE_MODEL)
        face_opts = mp_vision.FaceLandmarkerOptions(
            base_options=face_base,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        face_detector = mp_vision.FaceLandmarker.create_from_options(face_opts)
    else:
        face_detector = None

    tracker = PoseTracker(maxlen=VEL_BUFFER)

    prev_ts_ms = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        ts_ms = int(time.time() * 1000)

        # Pose inference
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame)       # BENAR
        pose_result = pose_detector.detect_for_video(mp_image, ts_ms)

        annotated = frame.copy()

        hips_xy = []
        per_person_data = []  # [(landmarks_xy, torso_angle, aspect_ratio)]

        if pose_result and pose_result.pose_landmarks:
            for lm_list in pose_result.pose_landmarks:
                # Convert to simple xy list (normalized)
                pts = [(lm.x, lm.y) for lm in lm_list]
                pts_np = np.array(pts, dtype=np.float32)

                # indices
                L_SH, R_SH = 11, 12
                L_HP, R_HP = 23, 24
                if max(L_HP, R_HP, L_SH, R_SH) < len(pts_np):
                    sh_mid = (pts_np[L_SH] + pts_np[R_SH]) / 2.0
                    hp_mid = (pts_np[L_HP] + pts_np[R_HP]) / 2.0
                else:
                    # skip if incomplete
                    continue

                # torso orientation relative to horizontal
                torso_ang = angle_to_horizontal(sh_mid, hp_mid)

                # bbox & aspect ratio
                bx0, by0, bx1, by1 = bbox_from_points(pts_np)
                bw, bh = (bx1 - bx0), (by1 - by0)
                aspect = (bh / bw) if bw > 1e-6 else 999.0

                hips_xy.append(hp_mid)
                per_person_data.append((pts_np, torso_ang, aspect))

        # Associate to tracks
        assigned = tracker.associate(hips_xy)

        # Draw and check FALL per person
        for (tid, hip_xy), person in zip(assigned, per_person_data):
            pts_np, torso_ang, aspect = person

            # update velocity buffer
            tracker.update_measure(tid, float(hip_xy[1]), time.time())
            vel = tracker.velocity(tid)  # + => turun

            # Heuristic conditions
            cond_vel = vel > VEL_THRESHOLD
            cond_horizontal = torso_ang < HORIZONTAL_ANGLE_THRESH
            cond_aspect = aspect < ASPECT_RATIO_THRESH

            # Draw skeleton & info
            draw_pose_skeleton(annotated, pts_np, w, h, color=CYAN)
            bx0, by0, bx1, by1 = bbox_from_points(pts_np)
            cv2.rectangle(annotated, (int(bx0*w), int(by0*h)), (int(bx1*w), int(by1*h)), YELLOW, 2)

            info = f"ID {tid} | v:{vel:.3f} ang:{torso_ang:.1f} ar:{aspect:.2f}"
            put_text_bg(annotated, info, (int(bx0*w), max(20, int(by0*h)-8)), fg=WHITE, bg=(20,20,20))

            # FALL logic with debounce
            fall_flag = False
            if cond_vel and (cond_horizontal or cond_aspect):
                now = time.time()
                if now - tracker.tracks[tid]["last_alert"] > FALL_DEBOUNCE_SEC:
                    fall_flag = True
                    tracker.tracks[tid]["last_alert"] = now

            if fall_flag:
                cx = int((bx0 + bx1)/2 * w)
                cy = int(by0 * h) - 10
                put_text_bg(annotated, f"FALL DETECTED (ID {tid})", (max(10, cx-120), max(30, cy)),
                            fg=WHITE, bg=(0,0,255), scale=0.7, thick=2)
                if SAVE_SNAPSHOT:
                    fname = os.path.join(SNAPSHOT_DIR, f"fall_id{tid}_{time.strftime('%Y%m%d_%H%M%S')}.jpg")
                    cv2.imwrite(fname, annotated)
                    print(f"[ALERT] Fall detected for ID {tid} -> saved {fname}")

        # Face mesh (multi-face; independen dari pose)
        if DRAW_FACE and face_detector is not None:
            face_result = face_detector.detect_for_video(mp_image, ts_ms)
            if face_result and face_result.face_landmarks:
                for fl in face_result.face_landmarks:
                    draw_face_mesh(annotated, fl, w, h, color=GREEN, radius=1)

        # HUD
        put_text_bg(annotated, f"People: {len(assigned)}", (10, 30), fg=WHITE, bg=(0,0,0))
        put_text_bg(annotated, "ESC to quit", (10, 58), fg=WHITE, bg=(0,0,0))

        cv2.imshow("Multi-person Fall Detection (MediaPipe Tasks)", annotated)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Cek model files
    missing = [p for p in [POSE_MODEL, FACE_MODEL] if not os.path.isfile(p)]
    if missing:
        print("Model file(s) missing:")
        for m in missing: print(" -", m)
        print("Silakan download .task model MediaPipe dan letakkan di folder 'models/' sesuai nama di atas.")
    main()