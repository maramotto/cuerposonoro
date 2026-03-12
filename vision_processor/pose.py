"""
Pose estimation module for Cuerpo Sonoro.

Provides a common interface (BasePoseEstimator) and automatic backend
selection based on available hardware:

  - TensorRT  → NVIDIA Jetson (GPU Ampere)
  - Metal     → Mac Apple Silicon (GPU M1/M2/M3/M4)
  - CPU       → any machine (fallback, uses MediaPipe on CPU)

The rest of the pipeline (features, MIDI, modes) never needs to know
which backend is active. All backends expose the same four methods:

    estimator.estimate(frame)
    estimator.get_landmarks(results)
    estimator.draw_skeleton(frame, results)
    estimator.release()
"""

from abc import ABC, abstractmethod
import numpy as np


class BasePoseEstimator(ABC):
    """
    Abstract base class for all pose estimation backends.

    Defines the interface that main.py and Config rely on.
    Any backend must implement all four methods below.
    """

    @abstractmethod
    def estimate(self, frame: np.ndarray):
        """
        Run pose estimation on a BGR frame.

        Args:
            frame: BGR image from OpenCV (numpy array).

        Returns:
            Backend-specific result object passed to get_landmarks()
            and draw_skeleton(). Treat it as opaque outside the backend.
        """

    @abstractmethod
    def get_landmarks(self, results) -> list | None:
        """
        Extract landmarks from estimation results.

        Args:
            results: Object returned by estimate().

        Returns:
            List of 33 dicts with keys {x, y, z, visibility},
            or None if no pose was detected.
        """

    @abstractmethod
    def draw_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """
        Draw pose skeleton on a frame.

        Args:
            frame:   BGR image to draw on.
            results: Object returned by estimate().

        Returns:
            Frame with skeleton drawn (may be the same object modified
            in place, or a new array — callers must not assume either).
        """

    @abstractmethod
    def release(self):
        """Release all resources held by this backend."""


# ---------------------------------------------------------------------------
# Keep PoseEstimator as a public name for backwards compatibility.
# Config and any external code that imports PoseEstimator directly
# will get the CPU backend, which is identical to the original class.
# ---------------------------------------------------------------------------

def PoseEstimator(
    model_complexity: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
):
    """
    Backwards-compatible factory.

    Returns a CPUPoseEstimator. Existing code that does:
        from vision_processor.pose import PoseEstimator
        pose = PoseEstimator(model_complexity=1)
    continues to work without any changes.
    """
    from vision_processor.backends.cpu import CPUPoseEstimator
    return CPUPoseEstimator(
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )