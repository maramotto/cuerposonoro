"""
FRAGMENT TO ADD TO config.py

Replace the existing create_pose_estimator() method with this block.
Add the _detect_backend() static method immediately before it.
"""

# =========================================================================
# Backend detection
# =========================================================================

@staticmethod
def _detect_backend(requested: str | None = None) -> str:
    """
    Detect the best available pose estimation backend.

    Priority (automatic):
        1. TensorRT  — NVIDIA Jetson (/etc/nv_tegra_release present + tensorrt importable)
        2. Metal     — Mac Apple Silicon (Darwin + arm64)
        3. CPU       — universal fallback

    Args:
        requested: If given ("tensorrt", "metal", "cpu"), skip detection
                   and return that value directly. Raises ValueError if
                   the requested backend is not available.

    Returns:
        One of: "tensorrt", "metal", "cpu"
    """
    import os
    import platform

    if requested:
        requested = requested.lower()
        if requested not in ("tensorrt", "metal", "cpu"):
            raise ValueError(
                f"Unknown backend: '{requested}'. "
                "Valid options: tensorrt, metal, cpu"
            )
        if requested == "tensorrt":
            try:
                import tensorrt  # noqa: F401
            except ImportError:
                raise RuntimeError(
                    "Backend 'tensorrt' requested but TensorRT is not available."
                )
        if requested == "metal":
            if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
                raise RuntimeError(
                    "Backend 'metal' requested but not running on Mac Apple Silicon."
                )
        return requested

    # --- Automatic detection ---

    # 1. Jetson: nv_tegra_release exists and TensorRT is importable
    if os.path.exists("/etc/nv_tegra_release"):
        try:
            import tensorrt  # noqa: F401
            return "tensorrt"
        except ImportError:
            pass  # Jetson without TensorRT in environment, fall through to CPU

    # 2. Mac Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions  # noqa: F401
            return "metal"
        except Exception:
            pass  # MediaPipe without Tasks API, fall through to CPU

    # 3. Universal CPU fallback
    return "cpu"

# =========================================================================
# Factory: pose estimator with backend selection
# =========================================================================

def create_pose_estimator(self):
    """
    Create a PoseEstimator using the best available backend.

    Reads 'pose.backend' from config.yaml if present, otherwise
    auto-detects. Can also be overridden at runtime via:
        config = Config(overrides={"pose.backend": "cpu"})

    Returns:
        CPUPoseEstimator, MetalPoseEstimator, or TensorRTPoseEstimator.
        All expose the same interface (BasePoseEstimator).
    """
    requested = self.get("pose.backend", None)
    backend = self._detect_backend(requested)

    kwargs = dict(
        model_complexity=self.pose_model_complexity,
        min_detection_confidence=self.pose_min_detection_confidence,
        min_tracking_confidence=self.pose_min_tracking_confidence,
    )

    if backend == "tensorrt":
        from vision_processor.backends.tensorrt import TensorRTPoseEstimator
        return TensorRTPoseEstimator(**kwargs)

    elif backend == "metal":
        from vision_processor.backends.metal import MetalPoseEstimator
        return MetalPoseEstimator(**kwargs)

    else:
        from vision_processor.backends.cpu import CPUPoseEstimator
        return CPUPoseEstimator(**kwargs)
