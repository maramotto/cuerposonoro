"""
CPU backend for pose estimation.

Uses MediaPipe Pose running on CPU.
Works on any machine with Python and MediaPipe installed:
Mac, Linux, Windows, Jetson (as fallback).

This is the original CuerpoSonoro pose estimator, unchanged,
wrapped as a proper backend class.
"""

import mediapipe as mp
import cv2
import numpy as np

from vision_processor.pose import BasePoseEstimator


class CPUPoseEstimator(BasePoseEstimator):
    """MediaPipe Pose running on CPU. Universal fallback backend."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Args:
            model_complexity:          0=lite, 1=full, 2=heavy.
            min_detection_confidence:  Detection threshold [0.0-1.0].
            min_tracking_confidence:   Tracking threshold [0.0-1.0].
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        print("[PoseEstimator] Backend: CPU (MediaPipe)")

    def estimate(self, frame: np.ndarray):
        """Run pose estimation on a BGR frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.pose.process(rgb_frame)

    def draw_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw pose skeleton on frame."""
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style(),
            )
        return frame

    def get_landmarks(self, results) -> list | None:
        """Extract landmarks as list of dicts with x, y, z, visibility."""
        if not results.pose_landmarks:
            return None

        return [
            {
                "x": lm.x,
                "y": lm.y,
                "z": lm.z,
                "visibility": lm.visibility,
            }
            for lm in results.pose_landmarks.landmark
        ]

    def release(self):
        """Release MediaPipe resources."""
        self.pose.close()
        print("[PoseEstimator] CPU backend released.")
