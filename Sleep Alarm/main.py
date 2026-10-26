import os
import cv2
import numpy as np
import mediapipe as mp
import pygame
from collections import deque

AUDIO_PATH = r"C:\Users\Rudra\Downloads\codes\Python\Assets\uth.mp3"

pygame.mixer.init()
if os.path.exists(AUDIO_PATH):
    pygame.mixer.music.load(AUDIO_PATH)
else:
    print(f"File not found: {AUDIO_PATH}")

mp_mesh = mp.solutions.face_mesh
detector = mp_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]

HEAD_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0)
], dtype=np.float64)

calibrated = False
calib_frames = 0
calib_target = 45
base_ears = []
base_pitches = []

ear_limit = 0.20
pitch_limit = -16.0
consec_limit = 24

ear_window = deque(maxlen=5)
pitch_window = deque(maxlen=5)

counter = 0
alarm_on = False

def get_ear(pts, idxs, w, h):
    p = [np.array([pts[i].x * w, pts[i].y * h]) for i in idxs]
    d1 = np.linalg.norm(p[1] - p[5])
    d2 = np.linalg.norm(p[2] - p[4])
    base = np.linalg.norm(p[0] - p[3])
    return (d1 + d2) / (2.0 * base) if base != 0 else 0.0

def get_pitch(pts, w, h):
    targets = [1, 199, 33, 263, 61, 291]
    img_pts = np.array([(pts[i].x * w, pts[i].y * h) for i in targets], dtype=np.float64)

    cam_mat = np.array([
        [w, 0, w / 2],
        [0, w, h / 2],
        [0, 0, 1]
    ], dtype=np.float64)

    ok, rot, _ = cv2.solvePnP(HEAD_3D, img_pts, cam_mat, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0

    rmat, _ = cv2.Rodrigues(rot)
    sy = np.sqrt(rmat[0, 0] ** 2 + rmat[1, 0] ** 2)
    val = np.arctan2(rmat[2, 1], rmat[2, 2]) if sy > 1e-6 else np.arctan2(-rmat[1, 2], rmat[1, 1])
    return float(np.degrees(val))

def draw_hud_box(img, x, y, bw, bh, alpha=0.65):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (20, 20, 20), -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + bw, y + bh), (60, 60, 60), 1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)

    status_flag = False
    ear_val, pitch_val = 0.0, 0.0

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        raw_ear = (get_ear(lm, L_EYE, w, h) + get_ear(lm, R_EYE, w, h)) / 2.0
        raw_pitch = get_pitch(lm, w, h)

        ear_window.append(raw_ear)
        pitch_window.append(raw_pitch)

        ear_val = float(np.median(ear_window))
        pitch_val = float(np.median(pitch_window))

        if not calibrated:
            base_ears.append(ear_val)
            base_pitches.append(pitch_val)
            calib_frames += 1

            draw_hud_box(frame, w // 2 - 180, h // 2 - 35, 360, 70, alpha=0.85)
            cv2.putText(frame, f"Calibrating... {int((calib_frames / calib_target) * 100)}%",
                        (w // 2 - 140, h // 2 + 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0, 215, 255), 1)

            if calib_frames >= calib_target:
                median_base_ear = float(np.median(base_ears))
                median_base_pitch = float(np.median(base_pitches))
                
                ear_limit = round(median_base_ear * 0.72, 2)
                pitch_limit = round(median_base_pitch - 14.0, 1)
                calibrated = True

            cv2.imshow("Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        eyes_shut = ear_val < ear_limit
        head_down = pitch_val < pitch_limit

        if eyes_shut or head_down:
            status_flag = True

    elif calibrated:
        status_flag = True

    if calibrated:
        if status_flag:
            counter += 1
            if counter >= consec_limit:
                if not alarm_on and os.path.exists(AUDIO_PATH):
                    pygame.mixer.music.play(-1)
                    alarm_on = True
        else:
            counter = 0
            if alarm_on:
                pygame.mixer.music.stop()
                alarm_on = False

        draw_hud_box(frame, 20, 20, 280, 105)

        ear_col = (70, 220, 70) if ear_val >= ear_limit else (60, 60, 255)
        pitch_col = (70, 220, 70) if pitch_val >= pitch_limit else (60, 60, 255)

        cv2.putText(frame, f"EAR    : {ear_val:.2f} (Cut: {ear_limit:.2f})", (35, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, ear_col, 1)
        cv2.putText(frame, f"PITCH  : {pitch_val:.1f} deg (Cut: {pitch_limit:.1f})", (35, 78),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, pitch_col, 1)

        dot_col = (0, 0, 255) if alarm_on else (0, 220, 0)
        cv2.circle(frame, (275, 38), 6, dot_col, -1)

        if alarm_on:
            alert_box_w, alert_box_h = 320, 70
            ax = (w - alert_box_w) // 2
            ay = (h - alert_box_h) // 2
            draw_hud_box(frame, ax, ay, alert_box_w, alert_box_h, alpha=0.9)
            cv2.rectangle(frame, (ax, ay), (ax + alert_box_w, ay + alert_box_h), (0, 0, 230), 2)
            cv2.putText(frame, "WAKE UP!", (ax + 55, ay + 48), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)

    cv2.imshow("Monitor", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if pygame.mixer.get_init():
    pygame.mixer.music.stop()
    pygame.mixer.quit()

cap.release()
cv2.destroyAllWindows()