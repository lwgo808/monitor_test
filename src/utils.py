from __future__ import annotations
import numpy as np


# MediaPipe Face Mesh landmark indices used for ergonomic calculations
LEFT_EYE_CENTER = 468    # left iris center (requires refine_landmarks)
RIGHT_EYE_CENTER = 473   # right iris center
NOSE_TIP = 1
LEFT_MOUTH = 61
RIGHT_MOUTH = 291


def landmark_to_px(landmark, width: int, height: int) -> tuple[int, int]:
    return int(landmark.x * width), int(landmark.y * height)


def midpoint(p1: tuple, p2: tuple) -> tuple[float, float]:
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def face_width_px(landmarks, width: int, height: int) -> float:
    """Pixel distance between outer eye corners — proxy for apparent face size."""
    l = landmark_to_px(landmarks[LEFT_EYE_CENTER], width, height)
    r = landmark_to_px(landmarks[RIGHT_EYE_CENTER], width, height)
    return float(np.linalg.norm(np.array(r) - np.array(l)))


def estimate_distance_cm(face_width_px_val: float,
                          known_face_width_cm: float = 14.0,
                          focal_length_px: float = 600.0) -> float:
    """
    Option B from notes: distance ∝ known_face_size / observed_face_size.
    focal_length_px is calibrated at ~60 cm; adjust for your webcam.
    """
    if face_width_px_val <= 0:
        return 0.0
    return (known_face_width_cm * focal_length_px) / face_width_px_val


def extract_head_angles(transformation_matrix) -> tuple[float, float, float]:
    """
    Returns (pitch, yaw, roll) in degrees from the 4x4 facial transformation matrix.
    Pitch: + = looking up, - = looking down
    Yaw:   + = turned right, - = turned left
    Roll:  + = tilted right, - = tilted left
    """
    mat = np.array(transformation_matrix.data).reshape(4, 4)
    R = mat[:3, :3]

    pitch = float(np.degrees(np.arcsin(-R[2, 0])))
    yaw   = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    roll  = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    return pitch, yaw, roll
