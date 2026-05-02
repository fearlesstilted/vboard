# VBoard: Hand-Tracked Virtual Whiteboard

A highly optimized, zero-latency virtual whiteboard using OpenCV and MediaPipe. 
Draw in the air with your index finger, erase with two fingers. Built with a focus on clean architecture, stable FPS, and real-time performance.

## Run

Install dependencies:

```bash
pip install opencv-python mediapipe numpy
```

Then run:

```bash
python cam_draw.py
```

If you use Conda:

```bash
conda activate whiteboard
python cam_draw.py
```

The MediaPipe model is downloaded automatically if `hand_landmarker.task` is missing.

On Linux, run it from a normal terminal if the OpenCV window does not open from VS Code.

## Controls
some of this was deleted and cooming soon
- `1`-`5` - change color
- `[` / `]` - brush size
- `,` / `.` - eraser size
- `c` - clear canvas
- `s` - save canvas to `captures/`
- `Tab` - toggle HUD
- `h` - help
- `d` - debug panel
- `l` - landmarks
- `q` or `Esc` - quit
 

## Architecture & Performance
- **State Management:** Canvas, colors, and brush properties are encapsulated in a `WhiteboardState` class, removing all global variables.
- **Hardware Optimization:** Uses `MJPG` fourcc format to bypass USB bandwidth bottlenecks on Linux, ensuring stable 30 FPS even on basic webcams.
- **Vectorized Smoothing:** Uses NumPy array operations instead of naive tuple math for zero-jitter cursor tracking.
- **Configurable:** All magic numbers (thresholds, gaps, alpha/beta) are exposed at the top of the file for easy tuning.

## Notes

Good lighting matters. Weak webcams may drop FPS in dark rooms because of auto exposure. Keep the hand near the center of the frame for best tracking.
