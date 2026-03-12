"""
FRAGMENTO PARA AÑADIR A config.py

Sustituir el método create_pose_estimator() existente por este bloque completo.
Añadir también el método estático _detect_backend() justo antes.
"""

    # =========================================================================
    # Backend detection
    # =========================================================================

    @staticmethod
    def _detect_backend(requested: str | None = None) -> str:
        """
        Detect the best available pose estimation backend.

        Priority (automatic):
            1. TensorRT  — NVIDIA Jetson (nv_tegra_release present + tensorrt importable)
            2. Metal     — Mac Apple Silicon (platform darwin + arm64)
            3. CPU       — universal fallback

        Args:
            requested: If given ("tensorrt", "metal", "cpu"), skip detection
                       and return that value directly. Raises ValueError if
                       the requested backend is not available.

        Returns:
            One of: "tensorrt", "metal", "cpu"
        """
        import platform
        import sys

        if requested:
            requested = requested.lower()
            if requested not in ("tensorrt", "metal", "cpu"):
                raise ValueError(
                    f"Backend desconocido: '{requested}'. "
                    "Opciones: tensorrt, metal, cpu"
                )
            # Validar que el backend pedido está disponible
            if requested == "tensorrt":
                try:
                    import tensorrt  # noqa: F401
                except ImportError:
                    raise RuntimeError(
                        "Backend 'tensorrt' solicitado pero TensorRT no está disponible."
                    )
            if requested == "metal":
                if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
                    raise RuntimeError(
                        "Backend 'metal' solicitado pero no es Mac Apple Silicon."
                    )
            return requested

        # --- Detección automática ---

        # 1. Jetson: nv_tegra_release existe y TensorRT importable
        if os.path.exists("/etc/nv_tegra_release"):
            try:
                import tensorrt  # noqa: F401
                return "tensorrt"
            except ImportError:
                pass  # Jetson sin TensorRT en el entorno → caemos a CPU

        # 2. Mac Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            try:
                # Verificar que la Tasks API de MediaPipe soporta GPU delegate
                from mediapipe.tasks.python.core.base_options import BaseOptions
                # Si el import funciona, asumimos Metal disponible
                return "metal"
            except Exception:
                pass  # MediaPipe sin Tasks API → CPU

        # 3. CPU universal
        return "cpu"

    # =========================================================================
    # Factory: pose estimator con selección de backend
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

        else:  # cpu
            from vision_processor.backends.cpu import CPUPoseEstimator
            return CPUPoseEstimator(**kwargs)
