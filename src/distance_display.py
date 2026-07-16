from __future__ import annotations
import time
import cv2
import mediapipe as mp
import numpy as np

if __package__:
    from .face_mesh import FaceMeshTracker
    from .posture import compute_metrics
    from .utils import landmark_to_px, NOSE_TIP
else:
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
    from src.face_mesh import FaceMeshTracker
    from src.posture import compute_metrics
    from src.utils import landmark_to_px, NOSE_TIP


def distance_to_scale(distance_cm: float,
                      base_scale: float = 0.8,
                      ref_distance_cm: float = 60.0,
                      min_scale: float = 0.3,
                      max_scale: float = 5.0) -> float:
    """
    Map an estimated distance (cm) to an OpenCV `fontScale`.
    Farther distance -> larger scale (user moves back -> text grows).
    The mapping is linear around `ref_distance_cm` and clamped.
    """
    if distance_cm <= 0 or np.isnan(distance_cm):
        return base_scale
    # scale proportional to distance relative to reference
    scale = base_scale * (distance_cm / ref_distance_cm)
    return float(max(min_scale, min(max_scale, scale)))


def run(word: str = "HELLO", camera_index: int = 0):
    tracker = FaceMeshTracker()
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: could not open camera.")
        tracker.close()
        return

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
                scale = distance_to_scale(metrics.distance_cm)
                thickness = max(1, int(scale * 3))

                # draw the scaled word at image center
                (text_w, text_h), baseline = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
                x = (w - text_w) // 2
                y = (h + text_h) // 2
                # drop shadow for legibility
                cv2.putText(frame, word, (x + 2, y + 2), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                cv2.putText(frame, word, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (200, 200, 255), thickness, cv2.LINE_AA)

                # overlay distance readout
                cv2.putText(frame, f"Dist: {metrics.distance_cm:.0f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
            else:
                cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 1, cv2.LINE_AA)

            cv2.imshow("Distance Scaling Test", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()


if __name__ == "__main__":
    run()
