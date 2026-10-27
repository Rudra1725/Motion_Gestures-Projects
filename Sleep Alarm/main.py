import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import pygame
from collections import deque

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_PATH = os.path.join(BASE_DIR, "assets", "uth.mp3")

pygame.mixer.init()
if os.path.exists(AUDIO_PATH):
    pygame.mixer.music.load(AUDIO_PATH)
else:
    print(f"Missing audio at: {AUDIO_PATH}")

mp_mesh = mp.solutions.face_mesh
detector = mp_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

L_EYE = [362, 385, 387, 263, 373, 380]
R_EYE = [33, 160, 158, 133, 153, 144]

NOSE_BRIDGE = 6
CHIN = 152
FOREHEAD = 10

calibrated = False
calib_count = 0
calib_target = 45
sample_ears = []
sample_ratios = []

ear_cutoff = 0.20
down_cutoff = 0.40
up_cutoff = 0.65
trigger_frames = 25

ear_buf = deque(maxlen=5)
ratio_buf = deque(maxlen=5)

drowsy_counter = 0
alarm_ringing = False
show_ui = True

def get_ear(landmarks, indices, w, h):
    p = [np.array([landmarks[i].x * w, landmarks[i].y * h]) for i in indices]
    v1 = np.linalg.norm(p[1] - p[5])
    v2 = np.linalg.norm(p[2] - p[4])
    base = np.linalg.norm(p[0] - p[3])
    return (v1 + v2) / (2.0 * base) if base > 0 else 0.0

def get_head_ratio(landmarks, w, h):
    top = np.array([landmarks[FOREHEAD].x * w, landmarks[FOREHEAD].y * h])
    nose = np.array([landmarks[NOSE_BRIDGE].x * w, landmarks[NOSE_BRIDGE].y * h])
    chin = np.array([landmarks[CHIN].x * w, landmarks[CHIN].y * h])

    face_h = np.linalg.norm(top - chin)
    nose_to_chin = np.linalg.norm(nose - chin)
    return nose_to_chin / face_h if face_h > 0 else 0.5

def draw_pill(img, x, y, w, h, alpha=0.5):
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (15, 15, 15), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 50, 50), 1, cv2.LINE_AA)

def main():
    global calibrated, calib_count, sample_ears, sample_ratios
    global ear_cutoff, down_cutoff, up_cutoff, drowsy_counter, alarm_ringing, show_ui

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera access failed.")
        sys.exit(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.process(rgb)

        is_drowsy_frame = False
        cur_ear, cur_ratio = 0.0, 0.5

        if res.multi_face_landmarks:
            mesh = res.multi_face_landmarks[0].landmark

            raw_ear = (get_ear(mesh, L_EYE, w, h) + get_ear(mesh, R_EYE, w, h)) / 2.0
            raw_ratio = get_head_ratio(mesh, w, h)

            ear_buf.append(raw_ear)
            ratio_buf.append(raw_ratio)

            cur_ear = float(np.median(ear_buf))
            cur_ratio = float(np.median(ratio_buf))

            if not calibrated:
                sample_ears.append(cur_ear)
                sample_ratios.append(cur_ratio)
                calib_count += 1

                if show_ui:
                    draw_pill(frame, w // 2 - 120, h // 2 - 25, 240, 50, alpha=0.8)
                    pct = int((calib_count / calib_target) * 100)
                    cv2.putText(frame, f"Calibrating {pct}%", (w // 2 - 68, h // 2 + 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)

                if calib_count >= calib_target:
                    median_ear = float(np.median(sample_ears))
                    median_ratio = float(np.median(sample_ratios))

                    ear_cutoff = round(median_ear * 0.70, 3)
                    down_cutoff = round(median_ratio * 0.75, 3)
                    up_cutoff = round(median_ratio * 1.30, 3)
                    calibrated = True

                cv2.imshow("VisionGuard", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            eyes_closed = cur_ear < ear_cutoff
            head_down = cur_ratio < down_cutoff
            head_up = cur_ratio > up_cutoff

            if eyes_closed or head_down or head_up:
                is_drowsy_frame = True
        else:
            if calibrated:
                is_drowsy_frame = True

        if calibrated:
            if is_drowsy_frame:
                drowsy_counter += 1
                if drowsy_counter >= trigger_frames:
                    if not alarm_ringing and os.path.exists(AUDIO_PATH):
                        pygame.mixer.music.play(-1)
                        alarm_ringing = True
            else:
                drowsy_counter = 0
                if alarm_ringing:
                    pygame.mixer.music.stop()
                    alarm_ringing = False

            if show_ui:
                # Clean, compact top-left pill
                pill_w, pill_h = 130, 36
                draw_pill(frame, 20, 20, pill_w, pill_h, alpha=0.6)

                dot_col = (0, 70, 255) if alarm_ringing else (80, 220, 100)
                cv2.circle(frame, (38, 38), 5, dot_col, -1, cv2.LINE_AA)

                status_text = "DROWSY" if alarm_ringing else "ACTIVE"
                cv2.putText(frame, status_text, (52, 43),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

                # Modern center alert banner
                if alarm_ringing:
                    box_w, box_h = 240, 56
                    bx = (w - box_w) // 2
                    by = (h - box_h) // 2
                    draw_pill(frame, bx, by, box_w, box_h, alpha=0.9)
                    cv2.rectangle(frame, (bx, by), (bx + box_w, by + box_h), (50, 50, 230), 1, cv2.LINE_AA)
                    cv2.putText(frame, "WAKE UP", (bx + 52, by + 37),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("VisionGuard", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('h'):
            show_ui = not show_ui
        elif key == ord('c'):
            calibrated = False
            calib_count = 0
            sample_ears.clear()
            sample_ratios.clear()

    if pygame.mixer.get_init():
        pygame.mixer.music.stop()
        pygame.mixer.quit()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
