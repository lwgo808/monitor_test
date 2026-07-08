# Ergonomic Monitor Tracker

A real-time computer vision system that uses your webcam and MediaPipe Face Mesh to track your posture and head position relative to your monitor. It measures sitting distance, horizontal/vertical alignment, and head orientation — then gives live on-screen feedback and logs recommendations for future monitor control hardware integration.

---

## Features

- Real-time face landmark detection at webcam framerate
- Estimates sitting distance using monocular depth (known face size / observed face size)
- Detects horizontal and vertical misalignment between your eyes/nose and the screen center
- Extracts pitch, yaw, and roll head angles from the 4×4 facial transformation matrix
- Live HUD overlay with annotated landmarks, alignment arrows, and angle gauges
- Logs ergonomic snapshots to `ergonomics_log.json` every 10 seconds
- Designed to forward data to Arduino/ESP32 or a monitor control daemon

---

## Project Structure

```
monitor_test/
├── run.py                      # Root entry point
├── face_landmarker.task        # MediaPipe face model (binary, not in git)
├── requirements.txt
├── ergonomics_log.json         # Created at runtime — per-snapshot metric log
│
├── src/
│   ├── __init__.py
│   ├── main.py                 # Camera loop, drawing pipeline, orchestration
│   ├── face_mesh.py            # FaceMeshTracker — MediaPipe wrapper
│   ├── posture.py              # Metric computation and ergonomic evaluation
│   ├── utils.py                # Landmark math, distance estimation, angle extraction
│   ├── controller.py           # Monitor Control Software (MCS) stub
│   └── camera.py               # Standalone camera preview utility
│
└── tests/
    └── test_tracker.py
```

---

## Setup

**Requirements:** Python 3.9+, a webcam

```bash
# 1. Clone the repo
git clone https://github.com/your-username/monitor_test.git
cd monitor_test

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install opencv-python mediapipe numpy

# 4. Download the MediaPipe face model into the project root
curl -O https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

---

## Usage

Run from the project root — all three commands are equivalent:

```bash
python run.py           # recommended
python -m src.main      # module mode
python src/main.py      # direct
```

Press **`q`** to quit.

---

## File-by-File Reference

### `src/main.py` — Orchestration and drawing pipeline

The application entry point. Runs the webcam loop, feeds frames to the tracker, calls the posture evaluator, and renders every visual layer to the frame on each tick.

**Drawing layers (rendered every frame):**
| Function | What it draws |
|---|---|
| `draw_landmarks()` | Iris centers (yellow), eye line, nose tip (cyan), mouth corners (magenta) |
| `draw_alignment_guides()` | Dashed center line, horizontal offset arrow (eye midpoint → center column), vertical offset arrow (nose → ideal height) |
| `draw_head_axes()` | Three colored arrows from the nose tip showing yaw (red), pitch (green), roll (blue) — built from the rotation matrix |
| `draw_angle_gauges()` | Three horizontal bar gauges (bottom-right) for pitch, yaw, roll — fills green within threshold, red when exceeded |
| `draw_hud()` | Top-left text: distance in cm + current alert messages or "Posture OK" |

Ergonomic logging (`send_to_monitor`) fires on a 10-second interval, not every frame, to avoid log spam.

---

### `src/face_mesh.py` — `FaceMeshTracker`

Wraps MediaPipe `FaceLandmarker` in `LIVE_STREAM` mode. The landmarker runs asynchronously — results arrive in a callback on a separate thread. `FaceMeshTracker` stores the latest result in a thread-safe lock so the main loop can read it without blocking.

**Key methods:**
- `detect_async(mp_image, timestamp_ms)` — submit a frame for processing
- `get_latest()` — returns the most recent `FaceLandmarkerResult` (or `None` if no result yet)
- `close()` — releases the landmarker

The model file (`face_landmarker.task`) is located at the project root. The class raises `FileNotFoundError` with a download URL if the file is missing.

---

### `src/posture.py` — Metric computation and evaluation

Two-stage pipeline: compute raw numbers, then compare against thresholds.

**`compute_metrics(face_result, frame_width, frame_height) → ErgonomicMetrics`**

Extracts six values from the landmark result:

| Metric | How it's computed |
|---|---|
| `distance_cm` | `(known_face_width_cm × focal_length_px) / face_width_px` — Option B monocular depth |
| `horizontal_offset_px` | Eye midpoint x − frame center x; positive = user is right of center |
| `vertical_offset_pct` | `(nose_y − frame_center_y) / frame_height`; positive = user is above center |
| `pitch` | Head nod angle in degrees, from the 4×4 facial transformation matrix |
| `yaw` | Head turn angle in degrees |
| `roll` | Head tilt angle in degrees |

**`evaluate(metrics) → AdjustmentRecommendation`**

Compares each metric against its threshold and builds a list of plain-English messages:

| Threshold constant | Default value |
|---|---|
| `IDEAL_DISTANCE_CM` | 50–70 cm |
| `MAX_HORIZONTAL_OFFSET_PX` | 40 px |
| `MAX_VERTICAL_OFFSET_PCT` | 8% of frame height |
| `MAX_PITCH_DEG` | ±15° |
| `MAX_YAW_DEG` | ±15° |
| `MAX_ROLL_DEG` | ±10° |

---

### `src/utils.py` — Landmark math

Low-level helpers shared across modules.

| Function | Purpose |
|---|---|
| `landmark_to_px(landmark, w, h)` | Converts normalized (0–1) landmark coords to pixel coordinates |
| `midpoint(p1, p2)` | Returns the float midpoint of two pixel points |
| `face_width_px(landmarks, w, h)` | Pixel distance between left and right iris centers |
| `estimate_distance_cm(face_width_px, known_face_width_cm, focal_length_px)` | Monocular distance estimate. Default `focal_length_px=600` is calibrated for ~60 cm; see Calibration below |
| `extract_head_angles(transformation_matrix)` | Decomposes the 4×4 MediaPipe facial transformation matrix into (pitch, yaw, roll) in degrees |

**Landmark indices used:**

| Constant | Index | Point |
|---|---|---|
| `LEFT_EYE_CENTER` | 468 | Left iris center (MediaPipe refined landmarks) |
| `RIGHT_EYE_CENTER` | 473 | Right iris center |
| `NOSE_TIP` | 1 | Nose tip |

---

### `src/controller.py` — Monitor Control Software stub

`send_to_monitor(metrics, rec)` is the integration point between the tracker and any external hardware or system software. Currently it:

1. Appends a JSON snapshot to `ergonomics_log.json` at the project root
2. Prints a one-line status to the terminal when posture is good, or a bulleted alert list when thresholds are exceeded

**Log format (one JSON object per line):**
```json
{
  "timestamp": 1720000000.0,
  "distance_cm": 62.4,
  "horizontal_offset_px": -12.0,
  "vertical_offset_pct": 0.021,
  "pitch": -3.2,
  "yaw": 1.8,
  "roll": 0.5,
  "recommendations": []
}
```

To integrate with hardware, replace the body of `send_to_monitor()` with a `serial.write()` call to an Arduino or ESP32 over USB.

---

### `src/camera.py` — Standalone camera preview

A minimal script that opens the webcam and displays the raw feed. Useful for verifying camera access independently of the full pipeline.

```bash
python src/camera.py
```

---

## Calibration

The default `focal_length_px = 600` in `src/utils.py` is an approximation. For accurate distance readings:

1. Sit exactly **60 cm** from the camera
2. Run the tracker and note the printed `face_width_px` value from the overlay
3. Open [src/utils.py](src/utils.py) and update line 34:
   ```python
   focal_length_px: float = <face_width_px_at_60cm> * 60 / 14
   ```

---

## Roadmap

- [ ] Arduino/ESP32 serial integration in `controller.py`
- [ ] macOS display brightness control via `pyobjc` or `brightness` CLI
- [ ] Display scaling adjustment based on estimated distance
- [ ] Warm color shift after extended session time
- [ ] Calibration wizard to set `focal_length_px` automatically
- [ ] PyTorch model for improved depth estimation in cluttered backgrounds

---

## Dependencies

| Package | Purpose |
|---|---|
| `opencv-python` | Webcam capture, frame rendering, drawing |
| `mediapipe` | Face landmark detection, facial transformation matrix |
| `numpy` | Matrix math for rotation decomposition |

Install: `pip install opencv-python mediapipe numpy`
