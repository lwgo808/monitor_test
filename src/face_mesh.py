from pathlib import Path
import threading
import mediapipe as mp

_BaseOptions = mp.tasks.BaseOptions
_FaceLandmarker = mp.tasks.vision.FaceLandmarker
_FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
_VisionRunningMode = mp.tasks.vision.RunningMode


def _find_model() -> str:
    root = Path(__file__).resolve().parent.parent
    model = root / "face_landmarker.task"
    if not model.exists():
        raise FileNotFoundError(
            f"face_landmarker.task not found at {root}.\n"
            "Download it from:\n"
            "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/latest/face_landmarker.task"
        )
    return str(model)


class FaceMeshTracker:
    """
    Wraps MediaPipe FaceLandmarker in LIVE_STREAM mode.
    Call detect_async() each frame; read latest results via get_latest().
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_result = None
        self._landmarker = _FaceLandmarker.create_from_options(
            _FaceLandmarkerOptions(
                base_options=_BaseOptions(model_asset_path=_find_model()),
                running_mode=_VisionRunningMode.LIVE_STREAM,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=True,
                result_callback=self._on_result,
            )
        )

    def _on_result(self, result, _output_image, _timestamp_ms: int):
        with self._lock:
            self._latest_result = result

    def detect_async(self, mp_image, timestamp_ms: int):
        self._landmarker.detect_async(mp_image, timestamp_ms)

    def get_latest(self):
        with self._lock:
            return self._latest_result

    def close(self):
        self._landmarker.close()
