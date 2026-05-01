# Hand Whiteboard

Small webcam whiteboard prototype using OpenCV and MediaPipe Tasks.

Show one finger to draw. Show index + middle finger to erase. The project is tuned for a quick demo, not for perfect hand tracking in every lighting condition.

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

## Notes

Good lighting matters. Weak webcams may drop FPS in dark rooms because of auto exposure. Keep the hand near the center of the frame for best tracking.
