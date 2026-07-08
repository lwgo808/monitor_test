from __future__ import annotations
import math
import time
import cv2
import mediapipe as mp
import numpy as np

if __package__:
    from .face_mesh import FaceMeshTracker
    from .posture import compute_metrics, evaluate, ErgonomicMetrics, AdjustmentRecommendation
    from .controller import send_to_monitor
    from .utils import landmark_to_px, LEFT_EYE_CENTER, RIGHT_EYE_CENTER, NOSE_TIP
else:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from src.face_mesh import FaceMeshTracker
    from src.posture import compute_metrics, evaluate, ErgonomicMetrics, AdjustmentRecommendation
    from src.controller import send_to_monitor
    from src.utils import landmark_to_px, LEFT_EYE_CENTER, RIGHT_EYE_CENTER, NOSE_TIP

SNAPSHOT_INTERVAL_S = 10   # log + print every N seconds
AXIS_LENGTH = 70            # pixels for head-orientation arrows


# ── landmark drawing ──────────────────────────────────────────────────────────

def _arrow(frame, p1, p2, color, thickness=2):
    cv2.arrowedLine(frame, p1, p2, color, thickness, tipLength=0.25, line_type=cv2.LINE_AA)


def draw_landmarks(frame: np.ndarray, face_result) -> tuple[int, int] | None:
    """
    Draw key landmarks and return the eye-midpoint pixel, or None if no face.
    Landmarks drawn:
      • Left / right iris centers  (cyan dots)
      • Eye center connecting line (cyan)
      • Eye midpoint               (white dot)
      • Nose tip                   (yellow dot)
      • Mouth corners              (magenta dots)
    """
    if not face_result or not face_result.face_landmarks:
        return None

    lm = face_result.face_landmarks[0]
    h, w = frame.shape[:2]

    l_eye = landmark_to_px(lm[LEFT_EYE_CENTER],  w, h)
    r_eye = landmark_to_px(lm[RIGHT_EYE_CENTER], w, h)
    nose  = landmark_to_px(lm[NOSE_TIP],         w, h)
    l_mouth = landmark_to_px(lm[61],  w, h)
    r_mouth = landmark_to_px(lm[291], w, h)

    eye_mid = (int((l_eye[0] + r_eye[0]) / 2), int((l_eye[1] + r_eye[1]) / 2))

    # Eye line
    cv2.line(frame, l_eye, r_eye, (255, 230, 0), 1, cv2.LINE_AA)
    cv2.circle(frame, l_eye, 5, (255, 220, 0), -1)
    cv2.circle(frame, r_eye, 5, (255, 220, 0), -1)
    cv2.circle(frame, eye_mid, 4, (255, 255, 255), -1)

    # Nose
    cv2.circle(frame, nose, 5, (0, 230, 255), -1)
    cv2.putText(frame, "nose", (nose[0] + 6, nose[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 230, 255), 1, cv2.LINE_AA)

    # Mouth corners
    cv2.circle(frame, l_mouth, 4, (200, 0, 200), -1)
    cv2.circle(frame, r_mouth, 4, (200, 0, 200), -1)
    cv2.line(frame, l_mouth, r_mouth, (200, 0, 200), 1, cv2.LINE_AA)

    return eye_mid


def draw_alignment_guides(frame: np.ndarray, eye_mid: tuple[int, int],
                           metrics: ErgonomicMetrics) -> None:
    """
    Draw:
      • Frame center cross-hair (grey)
      • Ideal horizontal centre line (dashed grey)
      • Horizontal offset arrow: eye midpoint → centre column  (green/red)
      • Vertical offset arrow: nose → ideal height             (green/red)
    """
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2

    # Full-width ideal centre line (dashed)
    for x in range(0, w, 16):
        cv2.line(frame, (x, cy), (min(x + 8, w), cy), (80, 80, 80), 1)

    # Centre cross-hair
    cv2.line(frame, (cx - 18, cy), (cx + 18, cy), (200, 200, 200), 1)
    cv2.line(frame, (cx, cy - 18), (cx, cy + 18), (200, 200, 200), 1)

    # Horizontal offset: eye midpoint → directly above/below centre column
    h_color = (0, 210, 0) if metrics.horizontal_offset_px == 0 or \
        abs(metrics.horizontal_offset_px) <= 40 else (0, 80, 255)
    target_h = (cx, eye_mid[1])
    cv2.line(frame, eye_mid, target_h, h_color, 1, cv2.LINE_AA)
    _arrow(frame, eye_mid, target_h, h_color)
    cv2.putText(frame, f"H {metrics.horizontal_offset_px:+.0f}px",
                (min(eye_mid[0], cx) + 4, eye_mid[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, h_color, 1, cv2.LINE_AA)

    # Vertical offset: nose → ideal nose height (frame centre row)
    nose_y_actual = int((0.5 + metrics.vertical_offset_pct) * h)
    v_color = (0, 210, 0) if abs(metrics.vertical_offset_pct) <= 0.08 else (0, 80, 255)
    cv2.line(frame, (cx, nose_y_actual), (cx, cy), v_color, 1, cv2.LINE_AA)
    _arrow(frame, (cx, nose_y_actual), (cx, cy), v_color)
    cv2.putText(frame, f"V {metrics.vertical_offset_pct*100:+.1f}%",
                (cx + 6, (nose_y_actual + cy) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, v_color, 1, cv2.LINE_AA)


def draw_head_axes(frame: np.ndarray, nose_px: tuple[int, int],
                   pitch: float, yaw: float, roll: float) -> None:
    """
    Draw three orientation arrows from nose tip showing head pose:
      Red   = X axis  (yaw  — turning left/right)
      Green = Y axis  (pitch — nodding up/down)
      Blue  = Z axis  (roll  — tilting sideways)
    All arrows point in the direction the face is rotated.
    """
    p = math.radians(pitch)
    y = math.radians(yaw)
    r = math.radians(roll)

    # Build rotation matrix R = Rz(roll) @ Ry(yaw) @ Rx(pitch)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(p), -math.sin(p)],
                   [0, math.sin(p),  math.cos(p)]])
    Ry = np.array([[ math.cos(y), 0, math.sin(y)],
                   [0, 1, 0],
                   [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0],
                   [math.sin(r),  math.cos(r), 0],
                   [0, 0, 1]])
    R = Rz @ Ry @ Rx

    nx, ny = nose_px

    # Project 3D unit vectors onto the image plane (drop z)
    for vec, color, label in [
        (np.array([1, 0, 0]), (0, 0, 220),  "yaw"),
        (np.array([0, 1, 0]), (0, 200, 0),  "pitch"),
        (np.array([0, 0, 1]), (200, 100, 0),"roll"),
    ]:
        rotated = R @ vec
        # image y is inverted
        end = (int(nx + rotated[0] * AXIS_LENGTH),
               int(ny - rotated[1] * AXIS_LENGTH))
        _arrow(frame, (nx, ny), end, color, thickness=2)
        cv2.putText(frame, label, (end[0] + 4, end[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def draw_angle_gauges(frame: np.ndarray, pitch: float, yaw: float, roll: float) -> None:
    """
    Three horizontal bar gauges in the bottom-right corner.
    Bar fill shows magnitude; colour = green(ok) / red(exceeded).
    """
    h, w = frame.shape[:2]
    bx, by = w - 155, h - 90   # top-left of gauge block
    bar_w, bar_h, gap = 120, 12, 18
    thresholds = {"Pitch": (pitch, 15), "Yaw": (yaw, 15), "Roll": (roll, 10)}

    for i, (label, (val, limit)) in enumerate(thresholds.items()):
        y0 = by + i * gap
        # background
        cv2.rectangle(frame, (bx, y0), (bx + bar_w, y0 + bar_h), (40, 40, 40), -1)
        # fill (clamped to bar width, centred)
        fill = int(abs(val) / limit * (bar_w // 2))
        fill = min(fill, bar_w // 2)
        mid = bx + bar_w // 2
        color = (0, 200, 0) if abs(val) <= limit else (0, 80, 255)
        x0 = mid if val >= 0 else mid - fill
        cv2.rectangle(frame, (x0, y0), (mid + fill if val >= 0 else mid, y0 + bar_h), color, -1)
        # centre tick
        cv2.line(frame, (mid, y0), (mid, y0 + bar_h), (180, 180, 180), 1)
        cv2.putText(frame, f"{label}: {val:+.1f}°",
                    (bx, y0 + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, metrics: ErgonomicMetrics,
             rec: AdjustmentRecommendation) -> None:
    """Top-left text HUD: distance + status messages."""
    ok = not rec.messages
    color = (0, 200, 0) if ok else (0, 80, 255)
    lines = [f"Dist: {metrics.distance_cm:.0f} cm"] + \
            (["Posture OK"] if ok else rec.messages)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)


# ── main loop ─────────────────────────────────────────────────────────────────

def run():
    tracker = FaceMeshTracker()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera.")
        tracker.close()
        return

    last_log_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: could not read frame.")
                break

            timestamp_ms = int(time.monotonic() * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tracker.detect_async(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                timestamp_ms
            )

            face_result = tracker.get_latest()
            h, w = frame.shape[:2]
            metrics = compute_metrics(face_result, w, h)

            if metrics is not None:
                rec = evaluate(metrics)

                # Log on interval
                now = time.monotonic()
                if now - last_log_time >= SNAPSHOT_INTERVAL_S:
                    send_to_monitor(metrics, rec)
                    last_log_time = now

                # Draw visuals every frame
                eye_mid = draw_landmarks(frame, face_result)
                if eye_mid:
                    draw_alignment_guides(frame, eye_mid, metrics)
                    draw_head_axes(frame, landmark_to_px(
                        face_result.face_landmarks[0][NOSE_TIP], w, h),
                        metrics.pitch, metrics.yaw, metrics.roll)
                draw_angle_gauges(frame, metrics.pitch, metrics.yaw, metrics.roll)
                draw_hud(frame, metrics, rec)
            else:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 1, cv2.LINE_AA)

            cv2.imshow("Ergonomic Monitor", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    run()
