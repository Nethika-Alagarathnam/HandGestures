import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

FINGER_NEON = {
    "thumb":  ((80,  0,  180), (200,  60, 255)),  
    "index":  ((0,   60, 180), ( 80, 160, 255)),   
    "middle": ((120, 0,  180), (220,  80, 255)),  
    "ring":   ((0,   80, 200), (100, 180, 255)),  
    "pinky":  ((100, 0,  160), (200,  60, 255)),   
    "palm":   ((0,   40, 120), ( 60, 120, 220)), 
}

FINGER_CONNECTIONS = {
    "thumb":  [(0,1),(1,2),(2,3),(3,4)],
    "index":  [(0,5),(5,6),(6,7),(7,8)],
    "middle": [(0,9),(9,10),(10,11),(11,12)],
    "ring":   [(0,13),(13,14),(14,15),(15,16)],
    "pinky":  [(0,17),(17,18),(18,19),(19,20)],
    "palm":   [(5,9),(9,13),(13,17),(0,5),(0,17)],
}

prev_landmarks = {}

def neon_line(frame, p1, p2, halo_color, core_color):
    cv2.line(frame, p1, p2, halo_color, 18)   
    cv2.line(frame, p1, p2, halo_color, 12)   
    h2 = tuple(min(255, int(c * 1.4)) for c in halo_color)
    cv2.line(frame, p1, p2, h2,         6)  
    cv2.line(frame, p1, p2, core_color, 2)    
    cv2.line(frame, p1, p2, (240, 230, 255), 1)

def neon_dot(frame, p, halo_color, core_color):
    cv2.circle(frame, p, 12, halo_color,  -1)
    cv2.circle(frame, p, 7,  core_color,  -1)
    cv2.circle(frame, p, 3,  (255, 245, 255), -1)

def draw_skeleton(frame, points):
    for finger, connections in FINGER_CONNECTIONS.items():
        halo, core = FINGER_NEON[finger]
        for (a, b) in connections:
            neon_line(frame, points[a], points[b], halo, core)

    for i, p in enumerate(points):
        if   i <= 4:  halo, core = FINGER_NEON["thumb"]
        elif i <= 8:  halo, core = FINGER_NEON["index"]
        elif i <= 12: halo, core = FINGER_NEON["middle"]
        elif i <= 16: halo, core = FINGER_NEON["ring"]
        else:         halo, core = FINGER_NEON["pinky"]
        neon_dot(frame, p, halo, core)

def draw_trail(frame, hand_id, points):
    if hand_id in prev_landmarks:
        prev_pts = prev_landmarks[hand_id]
        for i, (curr, prev) in enumerate(zip(points, prev_pts)):
            if   i <= 4:  halo, core = FINGER_NEON["thumb"]
            elif i <= 8:  halo, core = FINGER_NEON["index"]
            elif i <= 12: halo, core = FINGER_NEON["middle"]
            elif i <= 16: halo, core = FINGER_NEON["ring"]
            else:         halo, core = FINGER_NEON["pinky"]
            cv2.line(frame, prev, curr, halo, 10)
            cv2.line(frame, prev, curr, core,  1)
    prev_landmarks[hand_id] = points

def draw_hand_to_hand(frame, p0, p1):
    tip_neon = [
        FINGER_NEON["thumb"],
        FINGER_NEON["index"],
        FINGER_NEON["middle"],
        FINGER_NEON["ring"],
        FINGER_NEON["pinky"],
    ]
    for i, idx in enumerate([4, 8, 12, 16, 20]):
        halo, core = tip_neon[i]
        neon_line(frame, p0[idx], p1[idx], halo, core)

def get_finger_states(lm):
    fingers = []
    thumb_tip = np.array([lm[4].x, lm[4].y])
    thumb_mcp = np.array([lm[2].x, lm[2].y])
    index_mcp = np.array([lm[5].x, lm[5].y])
    fingers.append(
        np.linalg.norm(thumb_tip - index_mcp) > np.linalg.norm(thumb_mcp - index_mcp) * 1.2
    )
    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        fingers.append(lm[tip].y < lm[pip].y - 0.02)
    return fingers

def detect_gesture(fingers):
    thumb, index, middle, ring, pinky = fingers
    count = sum(fingers)
    if all(fingers):                                                    return "Open Hand", (200, 60, 255)
    elif not any(fingers):                                              return "Fist",      (100, 60, 255)
    elif thumb and not index and not middle and not ring and not pinky: return "Thumbs Up", (80, 160, 255)
    elif index and middle and not ring and not pinky and not thumb:     return "Peace",    (200, 60, 255)
    elif index and not middle and not ring and not pinky and not thumb: return "Pointing", (80, 160, 255)
    elif thumb and pinky and not index and not middle and not ring:     return "Shaka",    (180, 80, 255)
    elif not thumb and index and middle and ring and pinky:             return "4 Fingers",   (100, 180, 255)
    else: return f"{count} Finger{'s' if count!=1 else ''}", (160, 100, 255)

def calculate_spread(lm):
    return min(int(abs(lm[8].x - lm[20].x) * 300), 100)

def draw_ui(frame, hands_detected, fps, gesture_list, h):
    cv2.putText(frame, f"Hands: {hands_detected}", (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (160, 100, 255), 2)
    cv2.putText(frame, f"FPS: {fps}", (20, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 160, 255), 2)
    for i, (gesture, color, spread) in enumerate(gesture_list):
        y = h - 20 - i * 38
        cv2.putText(frame, f"Hand {i+1}: {gesture}   Spread: {spread}%",
                    (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

cv2.namedWindow("Hand Tracker", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Tracker", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_time = time.time()


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    dark = np.zeros_like(frame)
    frame = cv2.addWeighted(frame, 1.0, dark, 0.0, 0)

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    hands_detected = 0
    all_points     = []
    gesture_list   = []

    if result.multi_hand_landmarks and result.multi_handedness:
        hands_detected = len(result.multi_hand_landmarks)

        for i, (hand_landmarks, _) in enumerate(
            zip(result.multi_hand_landmarks, result.multi_handedness)
        ):
            lm     = hand_landmarks.landmark
            points = [(int(lm[j].x * w), int(lm[j].y * h)) for j in range(21)]
            all_points.append(points)

            draw_trail(frame, i, points)
            draw_skeleton(frame, points)

            fingers = get_finger_states(lm)
            gesture, color = detect_gesture(fingers)
            spread  = calculate_spread(lm)
            gesture_list.append((gesture, color, spread))

        if len(all_points) == 2:
            draw_hand_to_hand(frame, all_points[0], all_points[1])
    else:
        prev_landmarks.clear()

    curr_time = time.time()
    fps       = int(1 / (curr_time - prev_time + 0.001))
    prev_time = curr_time

    draw_ui(frame, hands_detected, fps, gesture_list, h)
    cv2.imshow("Hand Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()