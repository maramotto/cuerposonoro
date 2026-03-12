"""
Metal backend for pose estimation on Mac Apple Silicon.

Uses MediaPipe Tasks API with GPU delegate (Metal), which on Apple Silicon
offloads inference to the M-series GPU. Expected latency: ~15ms vs ~60ms on CPU.

Requirements:
    - Mac with Apple Silicon (M1/M2/M3/M4)
    - mediapipe >= 0.10.0
    - Model file: pose_landmarker_full.task (downloaded automatically if missing)
"""

import os
import urllib.request
import cv2
import numpy as np

from vision_processor.pose import BasePoseEstimator

# URL del modelo oficial de MediaPipe
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/"
    "pose_landmarker_full.task"
)

# Ruta local donde se guarda el modelo (junto al repo)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "pose_landmarker_full.task",
)


def _ensure_model():
    """Download the model file if not already present."""
    if not os.path.exists(_MODEL_PATH):
        print(f"[MetalBackend] Descargando modelo en {_MODEL_PATH} ...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[MetalBackend] Modelo descargado.")


class MetalPoseEstimator(BasePoseEstimator):
    """
    MediaPipe PoseLandmarker with Metal GPU delegate.

    Uses the Tasks API (mp.tasks.vision.PoseLandmarker) instead of the
    legacy mp.solutions.pose. The GPU delegate routes inference through
    Metal on Apple Silicon, reducing pose_ms from ~60ms to ~15ms.

    The result object returned by estimate() is a PoseLandmarkerResult
    (Tasks API), different from the legacy mp.solutions result. All
    conversion to the standard landmark format happens inside this class,
    so the rest of the pipeline sees no difference.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        self._mp = mp
        self._vision = vision

        _ensure_model()

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=_MODEL_PATH,
                delegate=BaseOptions.Delegate.GPU,  # Metal en Apple Silicon
            ),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._frame_timestamp_ms = 0

        # Drawing utilities from the legacy MediaPipe solutions API
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        print("[PoseEstimator] Backend: Metal (Mac Apple Silicon GPU)")

    def estimate(self, frame: np.ndarray):
        """
        Run pose estimation using Metal GPU delegate.

        Returns a PoseLandmarkerResult (Tasks API object).

        Nota: el delegate Metal en Mac requiere formato SRGBA (con canal alfa),
        no SRGB. Usar SRGB provoca un crash en gpu_buffer_storage_cv_pixel_buffer
        con "unsupported ImageFrame format: 1".
        """
        # Metal necesita RGBA, no RGB
        rgba_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba_frame = np.ascontiguousarray(rgba_frame, dtype=np.uint8)
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGBA,
            data=rgba_frame,
        )
        self._frame_timestamp_ms += 33  # ~30 FPS
        return self._landmarker.detect_for_video(mp_image, self._frame_timestamp_ms)

    def get_landmarks(self, results) -> list | None:
        """
        Extract landmarks from Tasks API result.

        Converts NormalizedLandmark objects to the standard dict format
        {x, y, z, visibility} used by FeatureExtractor.
        """
        if not results.pose_landmarks:
            return None

        # results.pose_landmarks es una lista de poses; tomamos la primera
        pose = results.pose_landmarks[0]
        return [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility if hasattr(lm, "visibility") else 1.0,
            }
            for lm in pose
        ]

    def draw_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw skeleton using MediaPipe drawing utilities.

        Converts Tasks API landmarks back to the legacy proto format
        that mp_drawing expects.
        """
        if not results.pose_landmarks:
            return frame

        from mediapipe.framework.formats import landmark_pb2

        pose = results.pose_landmarks[0]
        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for lm in pose:
            lm_proto = landmark_list.landmark.add()
            lm_proto.x = lm.x
            lm_proto.y = lm.y
            lm_proto.z = lm.z

        self._mp_drawing.draw_landmarks(
            frame,
            landmark_list,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        return frame

    def release(self):
        """Release Metal GPU resources."""
        self._landmarker.close()
        print("[PoseEstimator] Metal backend released.")
