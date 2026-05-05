import os
import sys
import time
import logging
import urllib.request
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("vboard")

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

ColorTuple = Tuple[int, int, int]
PaletteType = List[Tuple[str, ColorTuple]]
Point2D = Tuple[int, int]

#can be different, test on your wbcam
WINDOW_W: int = 960
WINDOW_H: int = 540
DETECT_W: int = 320
TARGET_FPS: int = 30

SMOOTH_FACTOR: float = 0.36 
CAM_CONTRAST: float = 0.82      
CAM_BRIGHTNESS: int = -8        
BLEND_FRAME_WEIGHT: float = 0.28
BLEND_CANVAS_WEIGHT: float = 0.82

FINGER_UP_GAP: int = 20
FINGER_EXT_RATIO: float = 0.58

PALETTE: PaletteType = [
    ("green", (0, 175, 55)),
    ("blue", (190, 135, 0)),
    ("pink", (190, 0, 120)),
    ("orange", (0, 95, 190)),
    ("white", (205, 205, 205)),
]

BASE_DIR: Path = Path(__file__).resolve().parent
MODEL_PATH: Path = BASE_DIR / "hand_landmarker.task"
MODEL_URL: str = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
WINDOW_NAME: str = "Virtual Whiteboard"

@dataclass
class HandState:
    mode: str
    hand_id: int
    idx_up: bool
    mid_up: bool

def get_empty_state() -> HandState:
    return HandState(mode="IDLE", hand_id=0, idx_up=False, mid_up=False)

class WhiteboardState:
    def __init__(self, w: int, h: int) -> None:
        self.canvas: np.ndarray = np.zeros((h, w, 3), dtype=np.uint8)
        self.brush_size: int = 8
        self.eraser_size: int = 38
        self.color_idx: int = 0
        
        self.prev_point: Optional[np.ndarray] = None
        self.current: HandState = get_empty_state()

    def clear(self) -> None:
        self.canvas[:] = 0
        logger.info("Canvas is clean")

    def get_color(self) -> ColorTuple:
        return PALETTE[self.color_idx][1]

def ensure_model(model_url: str, output_path: Path) -> None:
    if output_path.exists() and output_path.stat().st_size > 1024: 
        return
    logger.info(f"Google's cam model downloaded {output_path.name}...")
    try:
        urllib.request.urlretrieve(model_url, output_path)
    except Exception as e:
        logger.error(f"Error with cam loader: {e}")
        sys.exit(1)

def create_detector(model_path: Path) -> vision.HandLandmarker:
    base = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base, 
        running_mode=vision.RunningMode.VIDEO, 
        num_hands=1,
        min_hand_detection_confidence=0.55, 
        min_hand_presence_confidence=0.50, 
        min_tracking_confidence=0.50,
    )
    return vision.HandLandmarker.create_from_options(options)

def open_camera(cam_index: int, width: int, height: int, fps: int) -> Tuple[cv2.VideoCapture, int, int]:
    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")) 
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError(f"Cam {cam_index} is not responding.")
        
    real_h, real_w = frame.shape[:2]
    logger.info(f"Cam is on. Res: {real_w}x{real_h} @ {fps}FPS")
    return cap, real_h, real_w

def get_distance(a: Point2D, b: Point2D) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def smooth_cursor(prev: Optional[np.ndarray], target: Point2D, smooth_factor: float) -> np.ndarray:
    if prev is None:
        return np.array(target, dtype=np.float32) 
    return prev * (1.0 - smooth_factor) + np.array(target, dtype=np.float32) * smooth_factor

def extract_pixel_coords(hand_landmarks: Any, width: int, height: int) -> List[Point2D]:
    points: List[Point2D] = []
    for lm in hand_landmarks:
        x = int(np.clip(lm.x * width, 0, width - 1))
        y = int(np.clip(lm.y * height, 0, height - 1))
        points.append((x, y))
    return points

def save_canvas(canvas: np.ndarray, base_dir: Path) -> None:
    out_dir = base_dir / "captures"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"whiteboard_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(path), canvas)
    logger.info(f"Screenshot has been saved: {path.name}")

def classify_gesture(points: List[Point2D]) -> HandState:
    idx_up: bool = points[8][1] < points[6][1] - FINGER_UP_GAP
    mid_up: bool = points[12][1] < points[10][1] - FINGER_UP_GAP
    
    palm_size = max(get_distance(points[0], points[9]), get_distance(points[5], points[17]), 1.0)
    idx_ext: bool = get_distance(points[8], points[5]) > palm_size * FINGER_EXT_RATIO

    if idx_up and mid_up: 
        mode = "ERASING"
    elif idx_up or idx_ext: 
        mode = "DRAWING"
    else: 
        mode = "IDLE"

    return HandState(mode=mode, hand_id=1, idx_up=idx_up, mid_up=mid_up)

def process_frame(frame: np.ndarray, detector: vision.HandLandmarker, detect_w: int, win_w: int, win_h: int) -> Any:
    flipped = cv2.flip(frame, 1)
    small = cv2.resize(flipped, (detect_w, int(win_h * detect_w / win_w)))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect_for_video(mp_image, int(time.monotonic() * 1000))
    return result.hand_landmarks[0] if result.hand_landmarks else None

def update_state(state: WhiteboardState, hand_landmarks: Any, win_w: int, win_h: int) -> None:
    points = extract_pixel_coords(hand_landmarks, win_w, win_h)
    state.current = classify_gesture(points)
    
    cursor = smooth_cursor(state.prev_point, points[8], SMOOTH_FACTOR)

    if state.current.mode == "DRAWING":
        if state.prev_point is not None:
            p1, p2 = tuple(state.prev_point.astype(int)), tuple(cursor.astype(int))
            cv2.line(state.canvas, p1, p2, state.get_color(), state.brush_size, lineType=cv2.LINE_AA)
        state.prev_point = cursor
    elif state.current.mode == "ERASING":
        p = tuple(cursor.astype(int))
        cv2.circle(state.canvas, p, state.eraser_size, (0, 0, 0), -1, lineType=cv2.LINE_AA)
        state.prev_point = None
    else:
        state.prev_point = None

def compose_image(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    mask = cv2.threshold(cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
    base = cv2.convertScaleAbs(frame, alpha=CAM_CONTRAST, beta=CAM_BRIGHTNESS)
    out = base.copy()
    out[mask > 0] = cv2.addWeighted(base, BLEND_FRAME_WEIGHT, canvas, BLEND_CANVAS_WEIGHT, 0)[mask > 0]
    return out

def draw_hud(frame: np.ndarray, state: WhiteboardState, fps: float, show_help: bool) -> None:
    name, color = PALETTE[state.color_idx]
    overlay = frame.copy()
    
    cv2.rectangle(overlay, (18, 18), (620, 130 if show_help else 85), (6, 8, 6), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    cv2.putText(frame, f"MODE: {state.current.mode} | FPS: {fps:.1f}", (34, 46), 0, 0.6, (180, 180, 180), 1)
    cv2.circle(frame, (42, 72), 6, color, -1)
    cv2.putText(frame, f"Ink: {name} | Brush: {state.brush_size}px", (65, 78), 0, 0.5, (180, 180, 180), 1)

def handle_key(key: int, state: WhiteboardState, flags: Dict[str, bool]) -> bool:
    if key in (27, ord("q")): 
        return False
    if key == 255: 
        return True

    toggles = {
        9: "hud",        
    }
    
    if key in toggles:
        flag_name = toggles[key]
        flags[flag_name] = not flags[flag_name]
        logger.info(f"Flag {flag_name} = {flags[flag_name]}")
        return True

    if key == ord("c"):
        state.clear()
    elif key == ord("s"):
        save_canvas(state.canvas, BASE_DIR)
    elif key == ord("["):
        state.brush_size = max(2, state.brush_size - 2)
    elif key == ord("]"):
        state.brush_size = min(50, state.brush_size + 2)
    elif ord("1") <= key <= ord("5"):
        state.color_idx = key - ord("1")

    return True

def main() -> None:
    ensure_model(MODEL_URL, MODEL_PATH)
    detector = create_detector(MODEL_PATH)
    
    cap, real_h, real_w = open_camera(0, WINDOW_W, WINDOW_H, TARGET_FPS)
    
    state = WhiteboardState(WINDOW_W, WINDOW_H)
    flags: Dict[str, bool] = {"hud": True, "help": False, "debug": False, "landmarks": False}
    last_time = time.monotonic()

    logger.info("Starting the main cycle")
    
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok: 
                logger.error("Lost the cam frame")
                break
            
            frame = cv2.resize(frame, (WINDOW_W, WINDOW_H))
            
            hands = process_frame(frame, detector, DETECT_W, WINDOW_W, WINDOW_H)
            
            if hands:
                update_state(state, hands, WINDOW_W, WINDOW_H)
            else:
                state.prev_point = None
                state.current.mode = "IDLE"

            out = cv2.flip(frame, 1)
            out = compose_image(out, state.canvas)
            
            now = time.monotonic()
            fps = 1.0 / (now - last_time)
            last_time = now
            
            if flags["hud"]: 
                draw_hud(out, state, fps, flags["help"])
            
            cv2.imshow(WINDOW_NAME, out)
            key = cv2.waitKey(1) & 0xFF
            
            if not handle_key(key, state, flags): 
                break

    finally:
        logger.info("im out")
        cap.release()
        detector.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except cv2.error as e:
        logger.critical(f"engine error OpenCV: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.critical(f"hardware error: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Ctrl + c exit.")
        sys.exit(0)
    except Exception as e:
        logger.critical(f"ERROR {e}")
        traceback.print_exc()
        sys.exit(1)