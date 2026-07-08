from __future__ import annotations
from dataclasses import dataclass
from .utils import (
    landmark_to_px, midpoint, face_width_px,
    estimate_distance_cm, extract_head_angles,
    LEFT_EYE_CENTER, RIGHT_EYE_CENTER, NOSE_TIP,
)

# Ergonomic thresholds
IDEAL_DISTANCE_CM = (50, 70)     # comfortable monitor viewing range
MAX_HORIZONTAL_OFFSET_PX = 40   # eye midpoint vs frame center
MAX_VERTICAL_OFFSET_PCT = 0.08   # nose y vs frame center, as fraction of height
MAX_PITCH_DEG = 15
MAX_YAW_DEG = 15
MAX_ROLL_DEG = 10


@dataclass
class ErgonomicMetrics:
    distance_cm: float
    horizontal_offset_px: float   # + = user is right of center
    vertical_offset_pct: float    # + = user is above center
    pitch: float
    yaw: float
    roll: float


@dataclass
class AdjustmentRecommendation:
    distance_ok: bool
    horizontal_ok: bool
    vertical_ok: bool
    angles_ok: bool
    messages: list[str]


def compute_metrics(face_result, frame_width: int, frame_height: int) -> ErgonomicMetrics | None:
    if not face_result or not face_result.face_landmarks:
        return None

    landmarks = face_result.face_landmarks[0]

    # Horizontal offset: eye midpoint vs frame center
    l = landmark_to_px(landmarks[LEFT_EYE_CENTER], frame_width, frame_height)
    r = landmark_to_px(landmarks[RIGHT_EYE_CENTER], frame_width, frame_height)
    eye_mid = midpoint(l, r)
    h_offset = eye_mid[0] - frame_width / 2

    # Vertical offset: nose y vs frame center (normalised)
    nose = landmark_to_px(landmarks[NOSE_TIP], frame_width, frame_height)
    v_offset = (nose[1] - frame_height / 2) / frame_height

    # Distance estimate
    fw = face_width_px(landmarks, frame_width, frame_height)
    distance = estimate_distance_cm(fw)

    # Head angles
    pitch, yaw, roll = 0.0, 0.0, 0.0
    if face_result.facial_transformation_matrixes:
        pitch, yaw, roll = extract_head_angles(
            face_result.facial_transformation_matrixes[0]
        )

    return ErgonomicMetrics(
        distance_cm=distance,
        horizontal_offset_px=h_offset,
        vertical_offset_pct=v_offset,
        pitch=pitch,
        yaw=yaw,
        roll=roll,
    )


def evaluate(metrics: ErgonomicMetrics) -> AdjustmentRecommendation:
    messages = []

    dist_lo, dist_hi = IDEAL_DISTANCE_CM
    distance_ok = dist_lo <= metrics.distance_cm <= dist_hi
    if metrics.distance_cm < dist_lo:
        messages.append(f"Move back — you are too close ({metrics.distance_cm:.0f} cm, ideal {dist_lo}-{dist_hi} cm)")
    elif metrics.distance_cm > dist_hi:
        messages.append(f"Move closer — you are too far ({metrics.distance_cm:.0f} cm, ideal {dist_lo}-{dist_hi} cm)")

    horizontal_ok = abs(metrics.horizontal_offset_px) <= MAX_HORIZONTAL_OFFSET_PX
    if not horizontal_ok:
        direction = "right" if metrics.horizontal_offset_px > 0 else "left"
        messages.append(f"Shift monitor {direction} ({abs(metrics.horizontal_offset_px):.0f} px off-center)")

    vertical_ok = abs(metrics.vertical_offset_pct) <= MAX_VERTICAL_OFFSET_PCT
    if not vertical_ok:
        direction = "down" if metrics.vertical_offset_pct < 0 else "up"
        messages.append(f"Raise/lower monitor — eyes are too {direction}")

    pitch_ok = abs(metrics.pitch) <= MAX_PITCH_DEG
    yaw_ok   = abs(metrics.yaw)   <= MAX_YAW_DEG
    roll_ok  = abs(metrics.roll)  <= MAX_ROLL_DEG
    angles_ok = pitch_ok and yaw_ok and roll_ok
    if not pitch_ok:
        messages.append(f"Tilt monitor: head pitch {metrics.pitch:.1f}°")
    if not yaw_ok:
        messages.append(f"Face the screen: head yaw {metrics.yaw:.1f}°")
    if not roll_ok:
        messages.append(f"Straighten head: roll {metrics.roll:.1f}°")

    return AdjustmentRecommendation(
        distance_ok=distance_ok,
        horizontal_ok=horizontal_ok,
        vertical_ok=vertical_ok,
        angles_ok=angles_ok,
        messages=messages,
    )
