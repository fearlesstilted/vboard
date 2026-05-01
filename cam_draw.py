import os
import time
import urllib.request
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".matplotlib-cache"))
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = BASE_DIR / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
WINDOW = "Virtual Whiteboard - Google Tasks API"

# tuned on my laptop/webcam, not universal
W, H = 960, 540
DETECT_W = 320
TARGET_FPS = 30
SMOOTH = 0.36 
FRAME_ALPHA = 1.10
FRAME_BETA = 12

brush_size = 8
eraser_size = 38
color_index = 0
palette = [
    ("green", (0, 175, 55)),
    ("blue", (190, 135, 0)),
    ("pink", (190, 0, 120)),
    ("orange", (0, 95, 190)),
    ("white", (205, 205, 205)),
]


def empty_debug():
    return {"hand": 0, "index_up": 0, "index_ext": 0, "middle_up": 0, "middle_ext": 0}


def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1024:
        return
    print("[INFO] Downloading Google hand landmarker model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def create_detector():
    base = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.HandLandmarkerOptions(
        base_options=base,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )
    return vision.HandLandmarker.create_from_options(options)


def open_camera():
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    cap.set(cv2.CAP_PROP_FOURCC, fourcc)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    if not cap.isOpened():
        raise RuntimeError("Camera 0 is not available")
    print(
        f"[INFO] Camera: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}x"
        f"{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f} @ {cap.get(cv2.CAP_PROP_FPS):.1f} fps"
    )
    return cap


def points_from_landmarks(hand):
    points = []
    for lm in hand:
        x = int(np.clip(lm.x * W, 0, W - 1))
        y = int(np.clip(lm.y * H, 0, H - 1))
        points.append((x, y))
    return points


def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def finger_up(points, tip, pip, mcp):
    gap = 20
    tip_y, pip_y, mcp_y = points[tip][1], points[pip][1], points[mcp][1]
    return tip_y < pip_y - gap or tip_y < mcp_y - gap * 2


def finger_extended(points, tip, mcp):
    palm = max(dist(points[0], points[9]), dist(points[5], points[17]), 1.0)
    return dist(points[tip], points[mcp]) > palm * 0.58


def classify(points):
    index_up = finger_up(points, 8, 6, 5)
    middle_up = finger_up(points, 12, 10, 9)
    index_ext = finger_extended(points, 8, 5)
    middle_ext = finger_extended(points, 12, 9)

    if index_up and middle_up:
        mode = "ERASING"
    elif index_up or index_ext:
        mode = "DRAWING"
    else:
        mode = "IDLE"

    debug = {
        "hand": 1,
        "index_up": index_up,
        "index_ext": index_ext,
        "middle_up": middle_up,
        "middle_ext": middle_ext,
    }
    return mode, debug


def smooth(prev, target):
    if prev is None:
        return target
    return (
        int(prev[0] * (1 - SMOOTH) + target[0] * SMOOTH),
        int(prev[1] * (1 - SMOOTH) + target[1] * SMOOTH),
    )


def detector_frame(frame):
    small_h = int(frame.shape[0] * DETECT_W / frame.shape[1])
    small = cv2.resize(frame, (DETECT_W, small_h), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    luma, a, b = cv2.split(lab)
    luma = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(luma)
    boosted = cv2.cvtColor(cv2.merge((luma, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(boosted, alpha=1.08, beta=6)


def compose(frame, canvas):
    mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    base = cv2.convertScaleAbs(frame, alpha=0.82, beta=-8)
    out = base.copy()
    out[mask > 0] = cv2.addWeighted(base, 0.28, canvas, 0.82, 0)[mask > 0]
    return out


def draw_hud(frame, mode, raw_mode, frames, fps, show_help):
    name, color = palette[color_index]
    panel_h = 132 if show_help else 82
    overlay = frame.copy()
    cv2.rectangle(overlay, (18, 18), (620, panel_h), (6, 8, 6), -1)
    cv2.addWeighted(overlay, 0.76, frame, 0.24, 0, frame)
    cv2.rectangle(frame, (18, 18), (620, panel_h), (70, 110, 70), 1, cv2.LINE_AA)

    if mode == "DRAWING":
        mode_color = (120, 220, 120)
    elif mode == "ERASING":
        mode_color = (90, 150, 230)
    else:
        mode_color = (165, 175, 165)

    cv2.putText(frame, f"$ vboard --mode {mode.lower()}", (34, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.58, mode_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"fps={fps:04.1f}", (390, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (165, 175, 165), 1, cv2.LINE_AA)
    cv2.circle(frame, (42, 72), max(4, min(brush_size, 14)), color, -1, lineType=cv2.LINE_AA)
    cv2.putText(frame, f"ink={name} brush={brush_size}px erase={eraser_size}px", (70, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (175, 185, 175), 1, cv2.LINE_AA)

    if show_help:
        cv2.putText(frame, "tab hud  1-5 color  [/] brush  ,/. eraser  c clear  s save", (34, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (150, 160, 150), 1, cv2.LINE_AA)
        cv2.putText(frame, f"l landmarks  d debug  h help  q/esc quit  raw={raw_mode.lower()} frames={frames}", (34, 126), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (115, 125, 115), 1, cv2.LINE_AA)


def draw_debug(frame, debug, raw_mode, stable_mode, frames):
    lines = [
        f"raw={raw_mode} stable={stable_mode} frames={frames}",
        f"hand={debug['hand']} index up={int(debug['index_up'])} ext={int(debug['index_ext'])}",
        f"middle up={int(debug['middle_up'])} ext={int(debug['middle_ext'])} (erase uses up)",
    ]
    x0, y0 = W - 360, 18
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (W - 18, y0 + 132), (12, 16, 22), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (x0 + 14, y0 + 30 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 230, 235), 1, cv2.LINE_AA)


def save_canvas(canvas):
    out_dir = BASE_DIR / "captures"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"whiteboard_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(path), canvas)
    print(f"[OK] Saved canvas: {path}")


def handle_key(key, canvas, flags):
    global brush_size, eraser_size, color_index

    if key in (27, ord("q")):
        return False
    if key == 255:
        return True

    toggles = {
        9: "hud",
        ord("h"): "help",
        ord("d"): "debug",
        ord("l"): "landmarks",
    }
    if key in toggles:
        name = toggles[key]
        flags[name] = not flags[name]
        return True

    if key == ord("c"):
        canvas[:] = 0
    elif key == ord("s"):
        save_canvas(canvas)
    elif key == ord("["):
        brush_size = max(2, brush_size - 2)
    elif key == ord("]"):
        brush_size = min(40, brush_size + 2)
    elif key in (ord(","), ord("-"), ord("_")):
        eraser_size = max(6, eraser_size - 4)
    elif key in (ord("."), ord("+"), ord("=")):
        eraser_size = min(140, eraser_size + 4)
    elif ord("1") <= key <= ord("5"):
        color_index = key - ord("1")

    return True


def main():
    global brush_size, eraser_size, color_index

    ensure_model()
    detector = create_detector()
    cap = open_camera()
    canvas = np.zeros((H, W, 3), np.uint8)

    prev = None
    raw_mode = stable_mode = candidate_mode = "IDLE"
    candidate_frames = 0
    fps = 0.0
    last_frame_time = start_time = time.monotonic()

    flags = {"hud": True, "help": False, "debug": False, "landmarks": False}
    debug = empty_debug()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.monotonic()
            frame = cv2.resize(cv2.flip(frame, 1), (W, H), interpolation=cv2.INTER_AREA)
            frame = cv2.convertScaleAbs(frame, alpha=FRAME_ALPHA, beta=FRAME_BETA)
            rgb = cv2.cvtColor(detector_frame(frame), cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect_for_video(mp_image, int((now - start_time) * 1000))

            cursor = None
            raw_mode = "IDLE"
            debug = empty_debug()

            if result.hand_landmarks:
                points = points_from_landmarks(result.hand_landmarks[0])
                raw_mode, debug = classify(points)
                cursor = smooth(prev, points[8])
                if flags["landmarks"]:
                    for point in points:
                        cv2.circle(frame, point, 4, (210, 80, 255), -1, lineType=cv2.LINE_AA)

            if raw_mode == candidate_mode:
                candidate_frames += 1
            else:
                candidate_mode, candidate_frames = raw_mode, 1
            if candidate_frames >= 2:
                stable_mode = candidate_mode

            action_mode = "IDLE" if raw_mode == "IDLE" else stable_mode
            name, color = palette[color_index]

            if cursor is None or action_mode == "IDLE":
                prev = None
            elif action_mode == "DRAWING":
                if prev is None:
                    prev = cursor
                cv2.line(canvas, prev, cursor, color, brush_size, lineType=cv2.LINE_AA)
                cv2.circle(frame, cursor, brush_size + 6, color, 2, lineType=cv2.LINE_AA)
                prev = cursor
            elif action_mode == "ERASING":
                cv2.circle(canvas, cursor, eraser_size, (0, 0, 0), -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, cursor, eraser_size, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                prev = None

            frame = compose(frame, canvas)
            dt = max(now - last_frame_time, 1e-6)
            fps = fps * 0.9 + (1 / dt) * 0.1
            last_frame_time = now

            if flags["hud"]:
                draw_hud(frame, stable_mode, raw_mode, candidate_frames, fps, flags["help"])
            if flags["debug"]:
                draw_debug(frame, debug, raw_mode, stable_mode, candidate_frames)

            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1) & 0xFF
            if not handle_key(key, canvas, flags):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
