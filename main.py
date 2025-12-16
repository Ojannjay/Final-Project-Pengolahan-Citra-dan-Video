"""
VTuber Tracking System - VSeeFace
Full body(dari kepala sampai kaki)
"""

import time
import math
import cv2
import numpy as np
import mediapipe as mp
from pythonosc import udp_client

# --- KONFIGURASI UTAMA ---
OSC_IP = "10.125.186.121"  # IP VSeeFace Perangkat kamu
OSC_PORT = 39539  # Port VSeeFace Perangkat Kamu
WEBCAM_ID = 0
TARGET_FPS = 30

# ==========================================
# === DATA KALIBRASI FINAL ===
# ==========================================

# 1. LENGAN (1, 1, 1)
ARM_INVERT_X = 1.0
ARM_INVERT_Y = 1.0
ARM_INVERT_Z = 1.0
ARM_GAIN_XY = 1.2
ARM_GAIN_Z = 0.5

# 2. JARI (Finger: L=Z, R=Z | Sign: L=1, R=-1)
# Axis Index: 0=X, 1=Y, 2=Z
FINGER_AXIS_L = 2
FINGER_AXIS_R = 2
FINGER_SIGN_L = 1.0
FINGER_SIGN_R = -1.0
FINGER_SENSITIVITY = 1.3

# 3. JEMPOL (Thumb: L=Y, R=Y | Sign: L=-1, R=-1)
THUMB_AXIS_L = 1
THUMB_AXIS_R = 1
THUMB_SIGN_L = -1.0
THUMB_SIGN_R = 1.0

# ==========================================

# --- TUNING LAINNYA ---
EYE_Y_OFFSET = 0.02
GAZE_SENSITIVITY = 2.0
PITCH_CORRECTION_FACTOR = 0.015
DEADZONE = 0.3
NECK_RATIO = 0.5
EAR_THRESH_CLOSE, EAR_THRESH_OPEN = 0.15, 0.25
MOUTH_OPEN_MIN, MOUTH_OPEN_MAX = 5.0, 40.0


# =========================
# Helper math
# =========================
def euler_to_quaternion(pitch, yaw, roll):
    qx = np.sin(pitch / 2) * np.cos(yaw / 2) * np.cos(roll / 2) - np.cos(pitch / 2) * np.sin(yaw / 2) * np.sin(roll / 2)
    qy = np.cos(pitch / 2) * np.sin(yaw / 2) * np.cos(roll / 2) + np.sin(pitch / 2) * np.cos(yaw / 2) * np.sin(roll / 2)
    qz = np.cos(pitch / 2) * np.cos(yaw / 2) * np.sin(roll / 2) - np.sin(pitch / 2) * np.sin(yaw / 2) * np.cos(roll / 2)
    qw = np.cos(pitch / 2) * np.cos(yaw / 2) * np.cos(roll / 2) + np.sin(pitch / 2) * np.sin(yaw / 2) * np.sin(roll / 2)
    return [qx, qy, qz, qw]


def quat_from_axis_angle(angle, axis_idx):
    s = math.sin(angle / 2.0)
    c = math.cos(angle / 2.0)
    if axis_idx == 0:
        return [s, 0.0, 0.0, c]  # X
    if axis_idx == 1:
        return [0.0, s, 0.0, c]  # Y
    if axis_idx == 2:
        return [0.0, 0.0, s, c]  # Z
    return [0.0, 0.0, 0.0, 1.0]


def get_limb_rotation(start, end, rest_vector):
    v_curr = np.array(end) - np.array(start)
    norm = np.linalg.norm(v_curr)
    if norm < 1e-6:
        return [0, 0, 0, 1]
    v_curr = v_curr / norm

    v_rest = np.array(rest_vector)
    v_rest = v_rest / (np.linalg.norm(v_rest) + 1e-12)

    dot = float(np.dot(v_rest, v_curr))
    dot = max(-1.0, min(1.0, dot))
    angle = math.acos(dot)

    axis = np.cross(v_rest, v_curr)
    axis_len = np.linalg.norm(axis)
    if axis_len < 1e-6:
        return [0, 0, 0, 1]
    axis = axis / axis_len

    sin_half = math.sin(angle / 2)
    qx = axis[0] * sin_half
    qy = axis[1] * sin_half
    qz = axis[2] * sin_half
    qw = math.cos(angle / 2)
    return [qx, qy, qz, qw]


def calculate_ear(face_landmarks, indices, img_w, img_h):
    coords = []
    for idx in indices:
        lm = face_landmarks.landmark[idx]
        coords.append(np.array([lm.x * img_w, lm.y * img_h]))
    v1 = np.linalg.norm(coords[1] - coords[5])
    v2 = np.linalg.norm(coords[2] - coords[4])
    h = np.linalg.norm(coords[0] - coords[3])
    return (v1 + v2) / (2.0 * h + 1e-6)


def get_relative_iris(face_landmarks, iris_idx, inner_idx, outer_idx, img_w, img_h):
    iris = np.array([face_landmarks.landmark[iris_idx].x * img_w, face_landmarks.landmark[iris_idx].y * img_h])
    inner = np.array([face_landmarks.landmark[inner_idx].x * img_w, face_landmarks.landmark[inner_idx].y * img_h])
    outer = np.array([face_landmarks.landmark[outer_idx].x * img_w, face_landmarks.landmark[outer_idx].y * img_h])

    eye_width = np.linalg.norm(outer - inner)
    eye_vec = outer - inner
    eye_vec_norm = eye_vec / (np.linalg.norm(eye_vec) + 1e-6)

    iris_vec = iris - inner
    proj_x = np.dot(iris_vec, eye_vec_norm)
    norm_x = (proj_x / eye_width) * 2.0 - 1.0

    cross_prod = (eye_vec[0] * (iris[1] - inner[1])) - (eye_vec[1] * (iris[0] - inner[0]))
    dist_y = cross_prod / eye_width
    norm_y = dist_y / (eye_width * 0.3)
    return norm_x, norm_y


def get_finger_curl(landmarks, tip_idx, knuckle_idx, wrist_idx):
    tip = np.array([landmarks.landmark[tip_idx].x, landmarks.landmark[tip_idx].y])
    wrist = np.array([landmarks.landmark[wrist_idx].x, landmarks.landmark[wrist_idx].y])
    dist_tip_wrist = np.linalg.norm(tip - wrist)

    knuckle = np.array([landmarks.landmark[knuckle_idx].x, landmarks.landmark[knuckle_idx].y])
    dist_palm = np.linalg.norm(knuckle - wrist)

    ratio = dist_tip_wrist / (dist_palm + 1e-6)
    curl = (ratio - 1.9) / (0.8 - 1.9)
    curl = max(0.0, min(1.0, curl)) * FINGER_SENSITIVITY
    return curl


def apply_deadzone(curr, last, dz):
    if abs(curr - last) < dz:
        return last, last
    return curr, curr


# =========================
# FULLSCREEN COVER (biar abu-abu hilang)
# =========================
def resize_cover(frame, dst_w, dst_h):
    h, w = frame.shape[:2]
    if dst_w <= 0 or dst_h <= 0:
        return frame

    # cover: isi layar full, boleh crop sedikit
    scale = max(dst_w / w, dst_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

    x0 = (nw - dst_w) // 2
    y0 = (nh - dst_h) // 2
    return resized[y0:y0 + dst_h, x0:x0 + dst_w]


# =========================
# Kalman stabilizer
# =========================
class Stabilizer:
    def __init__(self, state_num=2, measure_num=1, cov_process=0.0001, cov_measure=0.1):
        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)
        self.state = np.zeros((state_num, 1), dtype=np.float32)
        self.filter.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.filter.measurementMatrix = np.array([[1, 1]], np.float32)
        self.filter.processNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * cov_process
        self.filter.measurementNoiseCov = np.array([[1]], np.float32) * cov_measure

    def update(self, measurement):
        self.filter.predict()
        self.filter.correct(np.array([[np.float32(measurement)]]))
        self.state = self.filter.statePost
        return self.state[0][0]


# =========================
# Tracker wrapper
# =========================
class VSeeFaceVMC:
    def __init__(self, ip, port):
        self.client = udp_client.SimpleUDPClient(ip, port)

    def bone(self, name, q):
        self.client.send_message("/VMC/Ext/Bone/Pos", [name, 0.0, 0.0, 0.0, float(q[0]), float(q[1]), float(q[2]), float(q[3])])

    def root(self):
        self.client.send_message("/VMC/Ext/Root/Pos", ["Root", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def blend(self, key, val):
        self.client.send_message("/VMC/Ext/Blend/Val", [key, float(val)])


# =========================
# Main app
# =========================
def main():
    # --- INIT PRINTS ---
    print("=" * 60)
    print("=" * 60)
    print(f"Target: {OSC_IP}:{OSC_PORT}")
    print("Starting MediaPipe...")

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    holistic = mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        refine_face_landmarks=True,
        model_complexity=1
    )

    sender = VSeeFaceVMC(OSC_IP, OSC_PORT)

    # Stabilizers (kepala + mata + spine)
    stab_pitch = Stabilizer(cov_process=0.02, cov_measure=0.1)
    stab_yaw = Stabilizer(cov_process=0.02, cov_measure=0.1)
    stab_roll = Stabilizer(cov_process=0.02, cov_measure=0.1)
    stab_eye_x = Stabilizer(cov_process=0.005, cov_measure=0.1)
    stab_eye_y = Stabilizer(cov_process=0.005, cov_measure=0.1)
    stab_spine_roll = Stabilizer(cov_process=0.02, cov_measure=0.1)
    stab_spine_yaw = Stabilizer(cov_process=0.02, cov_measure=0.1)

    # Stabilizer Jari (10 Jari)
    stab_fingers_L = [Stabilizer(cov_process=0.1, cov_measure=0.1) for _ in range(5)]
    stab_fingers_R = [Stabilizer(cov_process=0.1, cov_measure=0.1) for _ in range(5)]

    # Model points (head pose)
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    LEFT_EYE_IDXS = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_IDXS = [362, 385, 387, 263, 373, 380]
    L_IRIS_C, L_IN, L_OUT = 468, 133, 33
    R_IRIS_C, R_IN, R_OUT = 473, 362, 263

    # Config Jari
    FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little"]
    FINGER_INDICES = [(4, 2), (8, 5), (12, 9), (16, 13), (20, 17)]
    BONE_SUFFIXES = ["Proximal", "Intermediate", "Distal"]

    # --- CAMERA ---
    cap = cv2.VideoCapture(WEBCAM_ID)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        print("❌ Cannot open camera!")
        return

    print("✅ Camera opened!")
    print("📹 Tracking started... Press 'Q' to quit\n")
    print("=" * 60)

    # === FULLSCREEN WINDOW SETUP (NO GRAY AREA) ===
    window_name = "VSeeFace Tracking Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # fallback ukuran (kalau getWindowImageRect belum siap)
    win_w, win_h = 1920, 1080

    # Runtime states
    last_raw_pitch, last_raw_yaw, last_raw_roll = 0.0, 0.0, 0.0
    blink_l_state, blink_r_state = 0.0, 0.0
    prev_time = 0.0

    # Local helpers
    def to_unity_vec(mp_vec):
        return np.array([
            mp_vec[0] * ARM_INVERT_X * ARM_GAIN_XY,
            mp_vec[1] * ARM_INVERT_Y * ARM_GAIN_XY,
            mp_vec[2] * ARM_INVERT_Z * ARM_GAIN_Z
        ])

    def draw_face(image, face_landmarks):
        mp_drawing.draw_landmarks(
            image, face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            None, mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
        )

    def head_pose_and_faces(fl, img_w, img_h):
        image_points = np.array([
            (fl.landmark[1].x * img_w, fl.landmark[1].y * img_h),
            (fl.landmark[152].x * img_w, fl.landmark[152].y * img_h),
            (fl.landmark[263].x * img_w, fl.landmark[263].y * img_h),
            (fl.landmark[33].x * img_w, fl.landmark[33].y * img_h),
            (fl.landmark[287].x * img_w, fl.landmark[287].y * img_h),
            (fl.landmark[57].x * img_w, fl.landmark[57].y * img_h)
        ], dtype=np.float64)

        focal_length = img_w
        cam_matrix = np.array(
            [[focal_length, 0, img_w / 2],
             [0, focal_length, img_h / 2],
             [0, 0, 1]], dtype=np.float64
        )
        dist_coeffs = np.zeros((4, 1))

        success_pnp, rot_vec, trans_vec = cv2.solvePnP(
            model_points, image_points, cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        rmat, _ = cv2.Rodrigues(rot_vec)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

        dY = (fl.landmark[263].y * img_h) - (fl.landmark[33].y * img_h)
        dX = (fl.landmark[263].x * img_w) - (fl.landmark[33].x * img_w)
        pitch, yaw, roll = angles[0], angles[1], math.degrees(math.atan2(dY, dX))
        return pitch, yaw, roll

    def send_eyes_and_face(neck_pyr, head_pyr, blinkL, blinkR, mouth_open, eye_xy):
        neck_pitch, neck_yaw, neck_roll = neck_pyr
        head_pitch, head_yaw, head_roll = head_pyr
        smooth_eye_x, smooth_eye_y = eye_xy

        nq = euler_to_quaternion(math.radians(neck_pitch), math.radians(neck_yaw), math.radians(neck_roll))
        sender.bone("Neck", nq)

        hq = euler_to_quaternion(math.radians(head_pitch), math.radians(head_yaw), math.radians(head_roll))
        sender.bone("Head", hq)

        sender.root()
        sender.blend("Blink_L", blinkL)
        sender.blend("Blink_R", blinkR)
        sender.blend("A", mouth_open)

        eq = euler_to_quaternion(math.radians(smooth_eye_y * 70.0), math.radians(smooth_eye_x * 70.0), 0.0)
        sender.bone("LeftEye", eq)
        sender.bone("RightEye", eq)

    def process_hand(hand_landmarks, is_left: bool, finger_stabs):
        mp_drawing.draw_landmarks(image, hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        if is_left:
            side = "Left"
            axis_finger, axis_thumb = FINGER_AXIS_L, THUMB_AXIS_L
            sign_finger, sign_thumb = FINGER_SIGN_L, THUMB_SIGN_L
        else:
            side = "Right"
            axis_finger, axis_thumb = FINGER_AXIS_R, THUMB_AXIS_R
            sign_finger, sign_thumb = FINGER_SIGN_R, THUMB_SIGN_R

        for i, (fname, (tip, knuckle)) in enumerate(zip(FINGER_NAMES, FINGER_INDICES)):
            raw_curl = get_finger_curl(hand_landmarks, tip, knuckle, 0)
            curl = finger_stabs[i].update(raw_curl)

            if fname == "Thumb":
                angle = curl * (math.pi / 2.0) * sign_thumb
                axis = axis_thumb
            else:
                angle = curl * (math.pi / 1.5) * sign_finger
                axis = axis_finger

            fq = quat_from_axis_angle(angle, axis)

            for suf in BONE_SUFFIXES:
                sender.client.send_message(
                    "/VMC/Ext/Bone/Pos",
                    [f"{side}{fname}{suf}", 0.0, 0.0, 0.0, float(fq[0]), float(fq[1]), float(fq[2]), float(fq[3])]
                )

    # --- MAIN LOOP ---
    while cap.isOpened():
        ok, image = cap.read()
        if not ok:
            continue

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        image = cv2.flip(image, 1)
        img_h, img_w, _ = image.shape

        image.flags.writeable = False
        results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        image.flags.writeable = True

        # === 1. FACE TRACKING ===
        if results.face_landmarks:
            draw_face(image, results.face_landmarks)
            fl = results.face_landmarks

            curr_pitch, curr_yaw, curr_roll = head_pose_and_faces(fl, img_w, img_h)

            curr_pitch, last_raw_pitch = apply_deadzone(curr_pitch, last_raw_pitch, DEADZONE)
            curr_yaw, last_raw_yaw = apply_deadzone(curr_yaw, last_raw_yaw, DEADZONE)
            curr_roll, last_raw_roll = apply_deadzone(curr_roll, last_raw_roll, DEADZONE)

            s_pitch = stab_pitch.update(curr_pitch)
            s_yaw = stab_yaw.update(curr_yaw)
            s_roll = stab_roll.update(curr_roll)

            neck_pitch, neck_yaw, neck_roll = s_pitch * NECK_RATIO, s_yaw * NECK_RATIO, s_roll * NECK_RATIO
            head_pitch, head_yaw, head_roll = s_pitch - neck_pitch, s_yaw - neck_yaw, s_roll - neck_roll

            raw_ear_l = calculate_ear(fl, LEFT_EYE_IDXS, img_w, img_h)
            raw_ear_r = calculate_ear(fl, RIGHT_EYE_IDXS, img_w, img_h)

            if raw_ear_l < EAR_THRESH_CLOSE:
                blink_l_state = 1.0
            elif raw_ear_l > EAR_THRESH_OPEN:
                blink_l_state = 0.0

            if raw_ear_r < EAR_THRESH_CLOSE:
                blink_r_state = 1.0
            elif raw_ear_r > EAR_THRESH_OPEN:
                blink_r_state = 0.0

            if s_yaw > 20.0:
                blink_r_state = blink_l_state
            elif s_yaw < -20.0:
                blink_l_state = blink_r_state

            lx, ly = get_relative_iris(fl, L_IRIS_C, L_IN, L_OUT, img_w, img_h)
            rx, ry = get_relative_iris(fl, R_IRIS_C, R_IN, R_OUT, img_w, img_h)

            avg_x = (lx + rx) / 2.0
            avg_y = ((ly + ry) / 2.0) - (s_pitch * PITCH_CORRECTION_FACTOR) + EYE_Y_OFFSET

            if not (blink_l_state > 0.5 or blink_r_state > 0.5):
                smooth_eye_x = stab_eye_x.update(avg_x)
                smooth_eye_y = stab_eye_y.update(avg_y)
            else:
                smooth_eye_x = stab_eye_x.state[0][0]
                smooth_eye_y = stab_eye_y.state[0][0]

            mouth_dist = np.linalg.norm(
                np.array([fl.landmark[13].x * img_w, fl.landmark[13].y * img_h]) -
                np.array([fl.landmark[14].x * img_w, fl.landmark[14].y * img_h])
            )
            mouth_open = max(0.0, min(1.0, (mouth_dist - 5.0) / 35.0))

            send_eyes_and_face(
                (neck_pitch, neck_yaw, neck_roll),
                (head_pitch, head_yaw, head_roll),
                blink_l_state, blink_r_state,
                mouth_open,
                (smooth_eye_x, smooth_eye_y)
            )

        # === 2. BODY & ARM TRACKING ===
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)
            lm = results.pose_landmarks.landmark

            def pvec(idx):
                return [lm[idx].x, lm[idx].y, lm[idx].z]

            l_sh, r_sh = pvec(11), pvec(12)
            spine_roll = stab_spine_roll.update((l_sh[1] - r_sh[1]) * -120.0)
            spine_yaw = stab_spine_yaw.update((l_sh[2] - r_sh[2]) * -80.0)
            sq = euler_to_quaternion(0.0, math.radians(spine_yaw), math.radians(spine_roll))
            sender.bone("Spine", sq)

            if lm[11].visibility > 0.5 and lm[13].visibility > 0.5:
                start = to_unity_vec(pvec(11))
                end = to_unity_vec(pvec(13))
                q_lu = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
                sender.bone("LeftUpperArm", q_lu)

                if lm[15].visibility > 0.5:
                    start = to_unity_vec(pvec(13))
                    end = to_unity_vec(pvec(15))
                    q_ll = get_limb_rotation(start, end, [1.0, 0.0, 0.0])
                    sender.bone("LeftLowerArm", q_ll)

            if lm[12].visibility > 0.5 and lm[14].visibility > 0.5:
                start = to_unity_vec(pvec(12))
                end = to_unity_vec(pvec(14))
                q_ru = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
                sender.bone("RightUpperArm", q_ru)

                if lm[16].visibility > 0.5:
                    start = to_unity_vec(pvec(14))
                    end = to_unity_vec(pvec(16))
                    q_rl = get_limb_rotation(start, end, [-1.0, 0.0, 0.0])
                    sender.bone("RightLowerArm", q_rl)

        # === 3. FINGER TRACKING ===
        if results.left_hand_landmarks:
            process_hand(results.left_hand_landmarks, True, stab_fingers_L)

        if results.right_hand_landmarks:
            process_hand(results.right_hand_landmarks, False, stab_fingers_R)

        # === DISPLAY ===
        cv2.putText(image, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # update ukuran window fullscreen (kalau sudah kebaca)
        try:
            _, _, ww, hh = cv2.getWindowImageRect(window_name)
            if ww > 0 and hh > 0:
                win_w, win_h = ww, hh
        except:
            pass

        display_frame = resize_cover(image, win_w, win_h)
        cv2.imshow(window_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("\n✅ Tracking stopped!")


if __name__ == "__main__":
    main()
