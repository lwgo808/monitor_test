import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path


root_dir = Path(__file__).resolve().parent.parent
candidate_models = [
    root_dir / "pose_landmarker_lite.task",
    root_dir / "pose_landmarker_heavy.task",
    root_dir / "pose_landmarker_full.task",
]
model_path = next((p for p in candidate_models if p.exists()), None)

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


latest_frame = None


def draw_pose_landmarks(frame_rgb: np.ndarray, landmarks: list) -> None:
    height, width = frame_rgb.shape[:2]

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame_rgb, (x, y), 3, (0, 255, 0), -1)

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 7), (4, 5), (5, 6), (6, 8),
        (8, 10), (10, 12), (11, 13), (13, 15), (15, 17), (12, 14),
        (14, 16), (16, 18), (0, 4)
    ]

    for start, end in connections:
        if start < len(landmarks) and end < len(landmarks):
            p1 = landmarks[start]
            p2 = landmarks[end]
            x1 = int(p1.x * width)
            y1 = int(p1.y * height)
            x2 = int(p2.x * width)
            y2 = int(p2.y * height)
            cv2.line(frame_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)


def process_result(result: PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_frame

    frame_rgb = output_image.numpy_view()
    annotated_frame = frame_rgb.copy()

    if result.pose_landmarks:
        draw_pose_landmarks(annotated_frame, result.pose_landmarks[0])

    if result.segmentation_masks:
        mask = np.squeeze(result.segmentation_masks[0].numpy_view())
        mask = (mask * 255).astype(np.uint8)
        mask = np.stack([mask] * 3, axis=-1)
        cv2.imshow("Pose Segmentation Mask", mask)

    latest_frame = cv2.cvtColor(np.array(annotated_frame), cv2.COLOR_RGB2BGR)


def run_pose_landmarker():
    global model_path

    if model_path is None:
        print("No pose model found. Place pose_landmarker_lite.task in the project root and run again.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video stream")
        return

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.LIVE_STREAM,
        output_segmentation_masks=True,
        result_callback=process_result,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_timestamp_ms = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(rgb_frame))
            landmarker.detect_async(mp_image, frame_timestamp_ms)
            frame_timestamp_ms += 100

            if latest_frame is not None:
                cv2.imshow("Pose Landmarker", latest_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pose_landmarker()

