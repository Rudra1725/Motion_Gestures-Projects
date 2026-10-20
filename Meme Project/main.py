import os
import cv2
import mediapipe as mp
import numpy as np
import pygame


pygame.mixer.pre_init(frequency=44100, size=-16, channels=2, buffer=512)
pygame.mixer.init()


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(SCRIPT_DIR, "Assets")

SEARCH_DIRS = [
    ASSETS_DIR,
    SCRIPT_DIR,
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "Assets")),
    os.path.abspath(os.path.join(SCRIPT_DIR, "..")),
]


def load_sound(name_list):
    for folder in SEARCH_DIRS:
        for name in name_list:
            path = os.path.abspath(os.path.join(folder, name))
            if os.path.exists(path):
                try:
                    s = pygame.mixer.Sound(path)
                    s.set_volume(1.0)
                    print(f"[+] Loaded Audio ({name})")
                    return s
                except Exception as e:
                    print(f"[-] Failed loading audio ({name}): {e}")
    return None


aprit_sound = load_sound(["aprit_clean.wav", "aprit.wav", "aprit.mp3"])
arebap_sound = load_sound(["arebap_clean.wav", "arebap.wav", "arebap.mp3"])
dep_sound = load_sound(["dep_clean.wav", "dep.wav", "dep.mp3"])
jaldi_sound = load_sound(["jaldi_audio.wav", "jaldi.wav", "jaldi.mp3"])
kuchu_sound = load_sound(["kuchu_clean.wav", "kuchu.wav", "kuchu.mp3"])
rom_sound = load_sound(["rom_clean.wav", "rom.wav", "rom.mp3"])
faaah_sound = load_sound(["faaah_clean.wav", "faaah.wav", "faaah.mp3"])


video_file = None
for folder in SEARCH_DIRS:
    candidate = os.path.join(folder, "jaldi.mp4")
    if os.path.exists(candidate):
        video_file = candidate
        print("[+] Loaded Video (jaldi.mp4)")
        break

video_cap = cv2.VideoCapture(video_file) if video_file else None


def load_image(names):
    for folder in SEARCH_DIRS:
        for name in names:
            target = os.path.join(folder, name)
            if os.path.exists(target):
                img = cv2.imread(target)
                if img is not None:
                    print(f"[+] Loaded Image ({name})")
                    return img
    return None


meme_img = load_image(["meme.jpg", "download.jpg"])
shock_img = load_image(["shock.jpg", "shocked.jpg", "shock.png"])
up_img = load_image(["up.jpg", "up.png", "look_up.jpg"])

# --- 3. MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.55,
    min_tracking_confidence=0.55,
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def is_mouth_open(mesh_lms):
    upper_lip = mesh_lms[13]
    lower_lip = mesh_lms[14]
    left_corner = mesh_lms[78]
    right_corner = mesh_lms[308]

    vertical_dist = np.hypot(upper_lip.x - lower_lip.x, upper_lip.y - lower_lip.y)
    horizontal_dist = np.hypot(left_corner.x - right_corner.x, left_corner.y - right_corner.y)

    if horizontal_dist == 0:
        return False, 0.0

    mar = vertical_dist / horizontal_dist
    return (mar > 0.32), mar


def is_looking_up(mesh_lms):
    nose_tip = mesh_lms[1]
    glabella = mesh_lms[9]
    mid_eyes = mesh_lms[168]

    z_pitch = glabella.z - nose_tip.z
    vertical_eye_to_nose = nose_tip.y - mid_eyes.y

    is_tilted_up = (z_pitch > 0.045) and (vertical_eye_to_nose < 0.055)
    return is_tilted_up, z_pitch


def are_both_hands_above_head(hand_lms_list, face_top_y):
    if len(hand_lms_list) < 2 or face_top_y is None:
        return False

    h1_wrist = hand_lms_list[0].landmark[0].y
    h1_tip = hand_lms_list[0].landmark[12].y
    h2_wrist = hand_lms_list[1].landmark[0].y
    h2_tip = hand_lms_list[1].landmark[12].y

    h1_highest = min(h1_wrist, h1_tip)
    h2_highest = min(h2_wrist, h2_tip)

    return (h1_highest < face_top_y + 0.10) and (h2_highest < face_top_y + 0.10)


def count_extended_fingers(landmarks):
    extended = 0
    if landmarks[8].y < landmarks[6].y:
        extended += 1
    if landmarks[12].y < landmarks[10].y:
        extended += 1
    if landmarks[16].y < landmarks[14].y:
        extended += 1
    if landmarks[20].y < landmarks[18].y:
        extended += 1
    return extended


def is_hi_wave_gesture(landmarks, face_box):
    wrist = landmarks[0]
    middle_tip = landmarks[12]
    palm_center = landmarks[9]

    extended_count = count_extended_fingers(landmarks)
    hand_upright = middle_tip.y < (wrist.y - 0.10)

    outside_face = True
    if face_box is not None:
        fx, fy, fw, fh = face_box
        is_inside_x = (fx - 20) <= (palm_center.x * 640) <= (fx + fw + 20)
        is_inside_y = (fy - 20) <= (palm_center.y * 480) <= (fy + fh + 40)
        if is_inside_x and is_inside_y:
            outside_face = False

    depth_diff = middle_tip.z - wrist.z
    not_pushed_forward = depth_diff > -0.05 and middle_tip.z > -0.07

    return (extended_count == 4) and hand_upright and outside_face and not_pushed_forward


def is_hand_forward_stop(landmarks):
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]

    fingers_up = (
        landmarks[8].y < landmarks[6].y and
        landmarks[12].y < landmarks[10].y and
        landmarks[16].y < landmarks[14].y and
        landmarks[20].y < landmarks[18].y
    )
    hand_upright = landmarks[12].y < (wrist.y - 0.10)
    knuckle_width = abs(index_mcp.x - pinky_mcp.x)
    is_flat_facing = knuckle_width > 0.07
    depth_diff = landmarks[12].z - wrist.z
    is_forward = depth_diff < -0.07 or landmarks[12].z < -0.10

    return (fingers_up and hand_upright and is_flat_facing and is_forward), depth_diff


def check_phone_to_ear(landmarks, ear_left, ear_right, mouth_pos, face_top_y):
    if mouth_pos is None or face_top_y is None or (ear_left is None and ear_right is None):
        return False

    palm = landmarks[9]
    is_below_forehead = palm.y > (face_top_y + 0.05)
    is_above_mouth = palm.y < (mouth_pos[1] - 0.02)

    d_l = np.hypot(palm.x - ear_left[0], palm.y - ear_left[1]) if ear_left else 999.0
    d_r = np.hypot(palm.x - ear_right[0], palm.y - ear_right[1]) if ear_right else 999.0
    min_ear_dist = min(d_l, d_r)

    return (min_ear_dist < 0.11) and is_below_forehead and is_above_mouth


def is_single_hand_grabbing_head(landmarks, face_top_y, fx_range):
    if face_top_y is None or fx_range is None:
        return False
    palm_center = landmarks[9]
    fx_min, fx_max = fx_range
    is_on_upper_head = palm_center.y < (face_top_y + 0.06)
    is_within_head_width = fx_min <= palm_center.x <= fx_max
    return is_on_upper_head and is_within_head_width


def classify_tight_mouth_gestures(landmarks, mouth_pos):
    if mouth_pos is None:
        return False, False

    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    thumb_tip = landmarks[4]
    palm_center = landmarks[9]

    index_dist = float(np.hypot(index_tip.x - mouth_pos[0], index_tip.y - mouth_pos[1]))
    palm_dist = float(np.hypot(palm_center.x - mouth_pos[0], palm_center.y - mouth_pos[1]))
    extended_count = count_extended_fingers(landmarks)

    thumb_index_dist = float(np.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y))
    thumb_middle_dist = float(np.hypot(thumb_tip.x - middle_tip.x, thumb_tip.y - middle_tip.y))
    is_round_hand = (thumb_index_dist < 0.18) and (thumb_middle_dist < 0.18)

    # 1. APRIT (Shh)
    if (extended_count == 1) and (landmarks[8].y < landmarks[6].y) and (index_dist < 0.11):
        return True, False

    # 2. JALDI (Cupped round hand on lips)
    if is_round_hand and (index_dist < 0.11 or palm_dist < 0.11):
        return False, True

    return False, False


def animated_overlay(bg, overlay, x, y, w, h, alpha, slide_offset):
    if overlay is None or alpha <= 0.01:
        return bg

    bg_h, bg_w, _ = bg.shape
    y_shifted = int(y + slide_offset)

    x1, y1 = max(0, x), max(0, y_shifted)
    x2, y2 = min(bg_w, x + w), min(bg_h, y_shifted + h)

    if x2 <= x1 or y2 <= y1:
        return bg

    resized = cv2.resize(overlay, (x2 - x1, y2 - y1))
    roi = bg[y1:y2, x1:x2].astype(np.float32)
    overlay_crop = resized.astype(np.float32)

    blended = cv2.addWeighted(overlay_crop, alpha, roi, 1.0 - alpha, 0)
    bg[y1:y2, x1:x2] = blended.astype(np.uint8)
    return bg


def draw_minimal_pill_tag(frame, text, accent_bgr):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    x, y = 14, 14
    box_w = text_w + 32
    box_h = text_h + 16

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (18, 18, 22), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), (50, 50, 60), 1)
    dot_y = y + box_h // 2
    cv2.circle(frame, (x + 12, dot_y), 4, accent_bgr, -1)
    cv2.putText(frame, text, (x + 24, y + box_h - baseline - 3), font, scale, (230, 230, 235), thickness, cv2.LINE_AA)


cap = cv2.VideoCapture(0)
last_face_box = None
last_mouth_pos = None
left_ear_pos = None
right_ear_pos = None
face_top_y = None
fx_range = None

active_channel = None
current_playing = None

# Animation State Variables
current_overlay_content = None
active_visual_target = None
anim_alpha = 0.0
FADE_SPEED = 0.14
SLIDE_MAX = 24.0

stop_pose_counter = 0
hi_pose_counter = 0
look_up_counter = 0
DEBOUNCE_FRAMES = 5
LOOK_UP_DEBOUNCE = 6

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mesh_results = face_mesh.process(rgb_frame)
    mouth_open_state = False
    look_up_state = False
    current_mar = 0.0
    current_pitch = 0.0

    if mesh_results.multi_face_landmarks:
        mesh_lms = mesh_results.multi_face_landmarks[0].landmark

        xs = [pt.x for pt in mesh_lms]
        ys = [pt.y for pt in mesh_lms]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        face_top_y = min_y
        fx_range = (min_x - 0.08, max_x + 0.08)

        fx = int((min_x - 0.04) * w)
        fy = int((min_y - 0.08) * h)
        fw = int((max_x - min_x + 0.08) * w)
        fh = int((max_y - min_y + 0.14) * h)
        last_face_box = (fx, fy, fw, fh)

        last_mouth_pos = (mesh_lms[13].x, (mesh_lms[13].y + mesh_lms[14].y) / 2)
        right_ear_pos = (mesh_lms[234].x, mesh_lms[234].y)
        left_ear_pos = (mesh_lms[454].x, mesh_lms[454].y)

        # Facial Expression & Head Pose Trackers
        mouth_open_state, current_mar = is_mouth_open(mesh_lms)
        raw_look_up, current_pitch = is_looking_up(mesh_lms)

        if raw_look_up and not mouth_open_state:
            look_up_counter += 1
            if look_up_counter >= LOOK_UP_DEBOUNCE:
                look_up_state = True
        else:
            look_up_counter = max(0, look_up_counter - 2)

    raw_stop = False
    raw_hi = False
    stop_active = False
    hi_active = False
    shh_active = False
    cupped_mouth_active = False
    single_head_active = False
    both_hands_head_active = False
    phone_to_ear_active = False

    # Face-First Gatekeeper: mouth open or looking up disables hand tracking
    if mouth_open_state or look_up_state:
        stop_pose_counter = 0
        hi_pose_counter = 0
    else:
        hand_results = hands.process(rgb_frame)
        if hand_results.multi_hand_landmarks:
            for hand_lm in hand_results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

            # Two-hand check
            if len(hand_results.multi_hand_landmarks) >= 2:
                both_hands_head_active = are_both_hands_above_head(
                    hand_results.multi_hand_landmarks, face_top_y
                )

            # Single-hand checks
            if not both_hands_head_active:
                primary_hand = hand_results.multi_hand_landmarks[0].landmark

                # 1. Hi Wave -> rom.mp3
                raw_hi = is_hi_wave_gesture(primary_hand, last_face_box)

                # 2. Stop Gesture -> meme.jpg
                raw_stop, _ = is_hand_forward_stop(primary_hand)

                # 3. Head Grab -> arebap.mp3
                single_head_active = is_single_hand_grabbing_head(primary_hand, face_top_y, fx_range)

                # 4. Phone to Ear -> kuchu.mp3
                if not single_head_active and not raw_hi:
                    phone_to_ear_active = check_phone_to_ear(
                        primary_hand, left_ear_pos, right_ear_pos, last_mouth_pos, face_top_y
                    )

                # 5. Mouth Gestures -> aprit / jaldi
                if not single_head_active and not phone_to_ear_active and not raw_hi:
                    shh_active, cupped_mouth_active = classify_tight_mouth_gestures(
                        primary_hand, last_mouth_pos
                    )

        # Debounce Hi Wave
        if raw_hi and not single_head_active and not phone_to_ear_active and not cupped_mouth_active:
            hi_pose_counter += 1
            if hi_pose_counter >= 4:
                hi_active = True
        else:
            hi_pose_counter = max(0, hi_pose_counter - 2)

        # Debounce Stop Gesture
        no_other_active = not (
            both_hands_head_active or hi_active or cupped_mouth_active or
            phone_to_ear_active or shh_active or single_head_active
        )
        if raw_stop and no_other_active:
            stop_pose_counter += 1
            if stop_pose_counter >= DEBOUNCE_FRAMES:
                stop_active = True
        else:
            stop_pose_counter = max(0, stop_pose_counter - 2)

    # --- Active Visual Selection ---
    target_visual = None
    target_img_or_frame = None

    if mouth_open_state and shock_img is not None:
        target_visual = "shock"
        target_img_or_frame = shock_img
    elif look_up_state and up_img is not None:
        target_visual = "up"
        target_img_or_frame = up_img
    elif cupped_mouth_active and video_cap is not None:
        target_visual = "jaldi"
        v_ret, v_frame = video_cap.read()
        if not v_ret:
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            v_ret, v_frame = video_cap.read()
        if v_ret:
            target_img_or_frame = v_frame
    elif stop_active and meme_img is not None:
        target_visual = "stop"
        target_img_or_frame = meme_img
    else:
        if video_cap is not None and active_visual_target != "jaldi":
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Fade Up / Down Animation State Machine ---
    if target_visual is not None:
        anim_alpha = min(1.0, anim_alpha + FADE_SPEED)
        current_overlay_content = target_img_or_frame
        active_visual_target = target_visual
    else:
        anim_alpha = max(0.0, anim_alpha - FADE_SPEED)
        if anim_alpha == 0.0:
            active_visual_target = None
            current_overlay_content = None

    if anim_alpha > 0.0 and current_overlay_content is not None and last_face_box is not None:
        fx, fy, fw, fh = last_face_box
        slide_offset = (1.0 - anim_alpha) * SLIDE_MAX
        frame = animated_overlay(frame, current_overlay_content, fx, fy, fw, fh, anim_alpha, slide_offset)

    # --- Unified Audio Router ---
    target_sound = None
    target_name = None

    if mouth_open_state:
        target_sound = faaah_sound
        target_name = "faaah"
    elif look_up_state:
        target_sound = None
        target_name = "up"
    elif hi_active:
        target_sound = rom_sound
        target_name = "rom"
    elif both_hands_head_active:
        target_sound = dep_sound
        target_name = "dep"
    elif single_head_active:
        target_sound = arebap_sound
        target_name = "arebap"
    elif phone_to_ear_active:
        target_sound = kuchu_sound
        target_name = "kuchu"
    elif cupped_mouth_active:
        target_sound = jaldi_sound
        target_name = "jaldi"
    elif shh_active:
        target_sound = aprit_sound
        target_name = "aprit"

    if target_name != current_playing:
        if active_channel is not None:
            active_channel.stop()
        if target_sound is not None:
            active_channel = target_sound.play(loops=-1)
            current_playing = target_name
        else:
            current_playing = None

    # --- HUD Display ---
    tag_text = "Idle"
    tag_col = (120, 120, 130)

    if mouth_open_state:
        tag_text = "Shock -> Faaah"
        tag_col = (50, 50, 240)
    elif look_up_state:
        tag_text = "Head Up -> Up.jpg"
        tag_col = (255, 180, 50)
    elif hi_active:
        tag_text = "Hi Palm -> Rom"
        tag_col = (0, 220, 255)
    elif both_hands_head_active:
        tag_text = "Both Hands Up -> Dep"
        tag_col = (230, 50, 230)
    elif single_head_active:
        tag_text = "Head Grab -> Are Bap"
        tag_col = (0, 160, 255)
    elif phone_to_ear_active:
        tag_text = "Phone Ear -> Kuchu"
        tag_col = (180, 105, 255)
    elif cupped_mouth_active:
        tag_text = "Mouth Cup -> Jaldi"
        tag_col = (0, 230, 230)
    elif shh_active:
        tag_text = "Shh -> Aprit"
        tag_col = (50, 220, 50)
    elif stop_active:
        tag_text = "Stop -> Meme"
        tag_col = (80, 80, 255)

    draw_minimal_pill_tag(frame, tag_text, tag_col)
    cv2.imshow("Meme Gesture Controller", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if video_cap is not None:
    video_cap.release()
pygame.mixer.quit()
cv2.destroyAllWindows()
