import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path
import traceback

#config
W, H = 960, 540
DETECT_W = 320
TARGET_FPS = 30
SMOOTH = 0.36 
FRAME_ALPHA = 1.10
FRAME_BETA = 12
FINGER_UP_GAP = 20
FINGER_EXT_RATIO = 0.58

palette = [
    ("green", (0, 175, 55)),
    ("blue", (190, 135, 0)),
    ("pink", (190, 0, 120)),
    ("orange", (0, 95, 190)),
    ("white", (205, 205, 205)),
]

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = BASE_DIR / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
WINDOW = "Virtual Whiteboard - Google Tasks API"

def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1024: return
    print("[INFO] Downloading Google model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

def create_detector():
    base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base, running_mode=vision.RunningMode.VIDEO, num_hands=1,
        min_hand_detection_confidence=0.55, min_hand_presence_confidence=0.50, min_tracking_confidence=0.50,
    )
    return vision.HandLandmarker.create_from_options(options)

def open_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Camera is not responding")
        
    real_h, real_w = frame.shape[:2]
    return cap, real_h, real_w

def points_from_landmarks(hand):
    return [(int(np.clip(lm.x * W, 0, W - 1)), int(np.clip(lm.y * H, 0, H - 1))) for lm in hand]

def dist(a, b): return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def smooth(prev, target):
    if prev is None: return np.array(target)
    return prev * (1 - SMOOTH) + np.array(target) * SMOOTH

def classify(points):
    idx_up = points[8][1] < points[6][1] - FINGER_UP_GAP
    mid_up = points[12][1] < points[10][1] - FINGER_UP_GAP
    
    palm = max(dist(points[0], points[9]), dist(points[5], points[17]), 1.0)
    idx_ext = dist(points[8], points[5]) > palm * FINGER_EXT_RATIO

    if idx_up and mid_up: mode = "ERASING"
    elif idx_up or idx_ext: mode = "DRAWING"
    else: mode = "IDLE"

    return mode, {"hand": 1, "idx_up": idx_up, "mid_up": mid_up}

class WhiteboardState:
    def __init__(self, w, h):
        self.canvas = np.zeros((h, w, 3), np.uint8)
        self.brush = 8
        self.eraser = 38
        self.color_idx = 0
        self.mode = "IDLE"
        self.prev = None
        self.debug_data = {}

    def clear(self): self.canvas[:] = 0
    def get_color(self): return palette[self.color_idx][1]

def process_frame(frame, detector):
    flipped = cv2.flip(frame, 1)
    small = cv2.resize(flipped, (DETECT_W, int(H * DETECT_W / W)))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, int(time.monotonic() * 1000))
    return result.hand_landmarks[0] if result.hand_landmarks else None

def update_state(state, hand_landmarks):
    points = points_from_landmarks(hand_landmarks)
    state.mode, state.debug_data = classify(points)
    cursor = smooth(state.prev, points[8])

    if state.mode == "DRAWING":
        if state.prev is not None:
            p1, p2 = tuple(state.prev.astype(int)), tuple(cursor.astype(int))
            cv2.line(state.canvas, p1, p2, state.get_color(), state.brush, lineType=cv2.LINE_AA)
        state.prev = cursor
    elif state.mode == "ERASING":
        p = tuple(cursor.astype(int))
        cv2.circle(state.canvas, p, state.eraser, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        state.prev = None
    else:
        state.prev = None

def draw_hud(frame, state, fps, show_help):
    name, color = palette[state.color_idx]
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (620, 130 if show_help else 85), (6, 8, 6), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    cv2.putText(frame, f"MODE: {state.mode} | FPS: {fps:.1f}", (34, 46), 0, 0.6, (180, 180, 180), 1)
    cv2.circle(frame, (42, 72), 6, color, -1)
    cv2.putText(frame, f"Ink: {name} | Brush: {state.brush}px", (65, 78), 0, 0.5, (180, 180, 180), 1)

def handle_key(key, state, flags):
    if key in (27, ord("q")): 
        return False
    
    if key == 255: 
        return True

    #coming soon ig
    toggles = {
        9: "hud",        # Tab
        ord("h"): "help",
        ord("d"): "debug",
        ord("l"): "landmarks",
    }
    
    if key in toggles:
        flag_name = toggles[key]
        flags[flag_name] = not flags[flag_name]
        return True

    if key == ord("c"):
        state.clear()
    elif key == ord("s"):
        save_canvas(state.canvas)
    elif key == ord("["):
        state.brush = max(2, state.brush - 2)
    elif key == ord("]"):
        state.brush = min(50, state.brush + 2)
    elif ord("1") <= key <= ord("5"):
        state.color_idx = key - ord("1")

    return True

def compose(frame, canvas):
    mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    base = cv2.convertScaleAbs(frame, alpha=0.82, beta=-8)
    out = base.copy()
    out[mask > 0] = cv2.addWeighted(base, 0.28, canvas, 0.82, 0)[mask > 0]
    return out

def save_canvas(canvas):
    out_dir = BASE_DIR / "captures"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"whiteboard_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(path), canvas)
    print(f"[OK] Saved canvas: {path}")

def main():
    ensure_model()
    detector = create_detector()
    cap, real_w, real_h = open_camera()
    state = WhiteboardState(W, H)
    flags = {"hud": True, "help": False, "debug": False, "landmarks": False}
    last_time = time.monotonic()

    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: break
            
            frame = cv2.resize(frame, (W, H))
            hands = process_frame(frame, detector)
            if hands:
                update_state(state, hands)
            else:
                state.prev = None
                state.mode = "IDLE"

            out = cv2.flip(frame, 1)
            out = compose(out, state.canvas)
            
            now = time.monotonic()
            fps = 1.0 / (now - last_time)
            last_time = now
            
            if flags["hud"]: draw_hud(out, state, fps, flags["help"])
            
            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            
            if not handle_key(key, state, flags): break

    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()