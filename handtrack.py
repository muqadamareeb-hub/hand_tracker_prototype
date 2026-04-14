import cv2 
import mediapipe as mp
from mediapipe.tasks import python 
from mediapipe.tasks.python import vision
import math
import time

# --- Constants ---
SMOOTHING_ALPHA = 0.3       # lower = smoother but more lag, higher = more responsive
TRAIL_DURATION = 1          # seconds
THUMBS_UP_THRESHOLD = 0.02
FLASH_DURATION = 5          # frames
PINCH_THRESHOLD = 0.08

def distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def draw_bezier(frame, p0, p1, p2, color, thickness, steps=20):
    for t in range(steps):
        t1 = t / steps
        t2 = (t + 1) / steps
        # quadratic bezier formula
        x1 = int((1-t1)**2 * p0[0] + 2*(1-t1)*t1 * p1[0] + t1**2 * p2[0])
        y1 = int((1-t1)**2 * p0[1] + 2*(1-t1)*t1 * p1[1] + t1**2 * p2[1])
        x2 = int((1-t2)**2 * p0[0] + 2*(1-t2)*t2 * p1[0] + t2**2 * p2[0])
        y2 = int((1-t2)**2 * p0[1] + 2*(1-t2)*t2 * p1[1] + t2**2 * p2[1])
        cv2.line(frame, (x1, y1), (x2, y2), color, thickness)

# --- Mediapipe Setup ---
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=2,
    min_hand_detection_confidence=0.2,
    min_hand_presence_confidence=0.2,
    min_tracking_confidence=0.2
)
detector = vision.HandLandmarker.create_from_options(options)

# --- Camera Setup ---
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 60)
print("Width:", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print("Height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("FPS:", cap.get(cv2.CAP_PROP_FPS))

cv2.namedWindow("Hand Tracking", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Hand Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- Hand Skeleton Connections ---
connections = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20)  # pinky
]

# --- State Variables ---
trail = []
h_screen, w_screen = 1080, 1920
prev_time = 0
flash_frames = 0
prev_gesture = ""
gesture = ""
smooth_x, smooth_y, smooth_z = None, None, None     # no values assigned on launch
trail_color = (255, 0, 255)                         # default purple
zoom_level = 1.0                                    #default zoom level
zoom_ref_pinch = None      # pinch distance when zoom gesture started
zoom_ref_level = 1.0       # zoom level when zoom gesture started
zoom_sensitivity = 2

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    results = detector.detect(mp_image)

    if results.hand_landmarks:
        num_hands = len(results.hand_landmarks)

        # --- Draw Landmarks and Skeleton on All Detected Hands ---
        for hand in results.hand_landmarks:
            for landmark in hand:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                cv2.circle(frame, (x, y), 9, (0, 255, 0), -1)
            for start, end in connections:
                x_start = int(hand[start].x * w)
                y_start = int(hand[start].y * h)
                x_end = int(hand[end].x * w)
                y_end = int(hand[end].y * h)
                cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 0, 255), 3)

        # --- Gesture Detection (single hand) ---
        if num_hands == 1:
            hand = results.hand_landmarks[0]
            tip_x = int(hand[8].x * w)
            tip_y = int(hand[8].y * h)
            wrist = hand[0]

            # thumb is intentionally excluded - checked separately via dot product
            # this lets closed be reused for thumbs up, thumbs down, etc.
            fingers = [
                (8, 6),   # index
                (12, 10), # middle
                (16, 14), # ring
                (20, 18)  # pinky
            ]
            closed = all(distance(hand[tip], wrist) < distance(hand[joint], wrist) for tip, joint in fingers)

            # --- Individual Finger States ---
            index_open   = distance(hand[8],  wrist) > distance(hand[6],  wrist)
            middle_open  = distance(hand[12], wrist) > distance(hand[10], wrist)
            ring_closed  = distance(hand[16], wrist) < distance(hand[14], wrist)
            pinky_closed = distance(hand[20], wrist) < distance(hand[18], wrist)
            thumb_tucked = distance(hand[4], hand[20]) < distance(hand[1], hand[17])
            hand_size = distance(hand[0], hand[9])
            pinch        = distance(hand[8], hand[4]) < hand_size * 0.3 and not index_open

            # --- Thumb Direction (dot product) ---
            # hand_vec:  wrist→middle base = stable "up" axis of hand (base not tip, tip moves when fingers curl)
            # thumb_vec: wrist→thumb tip = direction thumb is pointing
            # dot product > 0 means thumb points same way as hand = thumb is up relative to hand orientation
            hand_vector  = (hand[9].x - wrist.x, hand[9].y - wrist.y)
            thumb_vector = (hand[4].x - wrist.x, hand[4].y - wrist.y)
            dot          = hand_vector[0] * thumb_vector[0] + hand_vector[1] * thumb_vector[1]
            thumb_extended = (distance(hand[4], hand[9]) > distance(hand[3], hand[9]) and
                              distance(hand[4], hand[8]) > 0.1 and   # far from index tip
                              distance(hand[4], hand[12]) > 0.1)     # far from middle tip
            thumb_up = dot > THUMBS_UP_THRESHOLD and thumb_extended

            # --- Gesture Classification ---
            # thumbs_up checked before closed since it depends on closed being true
            peace     = index_open and middle_open and ring_closed and pinky_closed and thumb_tucked
            thumbs_up = thumb_up and closed

            # reset trail color on gesture change
            

            if peace:
                gesture = "peace"
            elif thumbs_up:
                gesture = "thumbs up"
            elif closed:
                gesture = "closed"
                trail.clear()
                if prev_gesture != "closed":
                    flash_frames = FLASH_DURATION
            elif pinch:
                gesture = "pinch"
                trail_color = (255, 0, 0)
            else:
                gesture = "open"
            if prev_gesture != gesture and gesture != "pinch":
                trail_color = (255, 0, 255)

            # --- Trail (open and pinch both draw) ---
            if gesture in ("open", "pinch"):
                if smooth_x is None:
                    smooth_x, smooth_y, smooth_z = tip_x, tip_y, hand[8].z
                else:
                    smooth_x = int(SMOOTHING_ALPHA * tip_x + (1 - SMOOTHING_ALPHA) * smooth_x)
                    smooth_y = int(SMOOTHING_ALPHA * tip_y + (1 - SMOOTHING_ALPHA) * smooth_y)
                    smooth_z = SMOOTHING_ALPHA * hand[8].z + (1 - SMOOTHING_ALPHA) * smooth_z
                trail.append((smooth_x, smooth_y, smooth_z, time.time()))

                while len(trail) > 0 and time.time() - trail[0][3] > TRAIL_DURATION:
                    trail.pop(0)

                for i in range(1, len(trail) - 1):
                    x0, y0, z0, t0 = trail[i-1]
                    x1, y1, z1, t1 = trail[i]
                    x2, y2, z2, t2 = trail[i+1]
                    depth = abs(z1)
                    thickness = max(2, int(depth * 150))
                    draw_bezier(frame, (x0, y0), (x1, y1), (x2, y2), trail_color, thickness)

            prev_gesture = gesture
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)

        # --- Two Hand Mode ---
        elif num_hands == 2:
            trail.clear()
            hand1 = results.hand_landmarks[0]
            hand2 = results.hand_landmarks[1]
            tip1 = (int(hand1[8].x * w), int(hand1[8].y * h))
            tip2 = (int(hand2[8].x * w), int(hand2[8].y * h))
            cv2.line(frame, tip1, tip2, (0, 255, 255), 6)


            #Zoom idea



            fingers1 = [(8,6),(12,10),(16,14),(20,18)]
            fingers2 = [(8,6),(12,10),(16,14),(20,18)]
            fist1 = all(distance(hand1[tip], hand1[0]) < distance(hand1[joint], hand1[0]) for tip, joint in fingers1)
            fist2 = all(distance(hand2[tip], hand2[0]) < distance(hand2[joint], hand2[0]) for tip, joint in fingers2)
            if fist1 and not fist2:
                ctrl = hand2
            elif fist2 and not fist1:
                ctrl = hand1
            else:
                ctrl = None

            if ctrl is not None:
                pinch_dist = distance(ctrl[4], ctrl[8])
                hand_size = distance(ctrl[0], ctrl[9])
                norm_pinch = pinch_dist / hand_size  # normalized so hand distance doesn't matter

                if zoom_ref_pinch is None:
                    # First frame of zoom gesture - capture baseline
                    zoom_ref_pinch = norm_pinch
                    zoom_ref_level = zoom_level


                # To counter accumulated drift at clamp boundaries

                delta = norm_pinch - zoom_ref_pinch      # how much pinch changed from baseline
                new_zoom = zoom_ref_level + delta * zoom_sensitivity  # scale delta, not absolute position

                if new_zoom < .5:
                    zoom_level = .5
                    zoom_ref_pinch = norm_pinch
                    zoom_level = .5

                elif new_zoom > 6.0:
                    zoom_level = 6.0
                    zoom_ref_pinch = norm_pinch
                    zoom_level = 6.0

                else:
                    zoom_level = new_zoom
                    

            else:
                zoom_ref_pinch = None  # reset reference when gesture ends (fist locked or both open)

    else:
        trail.clear()

     # --- Apply Zoom ---
    if zoom_level > 1.0:
        cx, cy = w // 2, h // 2
        crop_w = int(w / zoom_level)
        crop_h = int(h / zoom_level)
        x1 = max(0, cx - crop_w // 2)
        y1 = max(0, cy - crop_h // 2)
        x2 = min(w, cx + crop_w // 2)
        y2 = min(h, cy + crop_h // 2)
        frame = frame[y1:y2, x1:x2]
        frame = cv2.resize(frame, (w, h))

    # --- Flash Effect on Fist ---
    if flash_frames > 0:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        flash_frames -= 1

    # --- FPS Counter ---
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    cv2.putText(frame, f"FPS: {int(fps)}", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)

    frame = cv2.resize(frame, (w_screen, h_screen))
    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()