model_path = "face_landmarker.task"
import mediapipe as mp


import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

# Create a face landmarker instance with the live stream mode:
def print_result(result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('face landmarker result: {}'.format(result))

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

with FaceLandmarker.create_from_options(options) as landmarker:
    running_mode = VisionRunningMode.LIVE_STREAM
    num_faces = 1
    min_face_detection_confidence = 0.5
    min_face_presence_confidence = 0.5
    output_face_blendshapes = True
    output_face_facial_transformation_matrixes = True
    result_callback = print_result
  # The landmarker is initialized. Use it here.
  # ...