"""
Unit tests for pose estimation backends.

Tests the BasePoseEstimator interface and the CPU and Metal backends,
using mocked dependencies — no camera, no model files, no GPU required.

Coverage:
  - BasePoseEstimator contract (ABC enforcement, todos los abstractmethod)
  - CPUPoseEstimator: init, BGR->RGB, landmarks, release
  - MetalPoseEstimator: init, BGR->RGBA (fix critico), SRGBA, contiguous,
    detect_for_video, landmarks con/sin visibility, release
  - PoseEstimator() factory: compatibilidad hacia atras, devuelve CPUPoseEstimator
  - Config._detect_backend(): plataforma, override explicito, ValueError

Run:
    pytest tests/unit/test_backends.py -v
"""

import sys
import os
import numpy as np
import pytest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


# =============================================================================
# HELPERS
# =============================================================================

def make_bgr_frame(width=640, height=480) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_mock_landmarks(n=33):
    """Landmarks simulados compatibles con ambas APIs de MediaPipe."""
    lms = []
    for i in range(n):
        lm = MagicMock()
        lm.x = 0.5 + i * 0.001
        lm.y = 0.5 + i * 0.001
        lm.z = 0.0
        lm.visibility = 0.9
        lms.append(lm)
    return lms


def make_solutions_results(detected=True, n=33):
    """Mock results de MediaPipe Solutions API (CPU)."""
    r = MagicMock()
    if detected:
        r.pose_landmarks = MagicMock()
        r.pose_landmarks.landmark = make_mock_landmarks(n)
    else:
        r.pose_landmarks = None
    return r


def make_tasks_results(detected=True, n=33):
    """Mock results de MediaPipe Tasks API (Metal)."""
    r = MagicMock()
    r.pose_landmarks = [make_mock_landmarks(n)] if detected else []
    return r


# =============================================================================
# BasePoseEstimator
# =============================================================================

class TestBasePoseEstimatorContract:

    def _make_minimal(self, missing=None):
        from vision_processor.pose import BasePoseEstimator

        methods = {
            "estimate":      lambda self, frame: None,
            "get_landmarks": lambda self, results: None,
            "draw_skeleton": lambda self, frame, results: frame,
            "release":       lambda self: None,
        }
        if missing:
            del methods[missing]

        return type("Concrete", (BasePoseEstimator,), methods)

    def test_cannot_instantiate_base_directly(self):
        from vision_processor.pose import BasePoseEstimator
        with pytest.raises(TypeError):
            BasePoseEstimator()

    @pytest.mark.parametrize("missing", ["estimate", "get_landmarks", "draw_skeleton", "release"])
    def test_missing_abstract_method_raises(self, missing):
        cls = self._make_minimal(missing=missing)
        with pytest.raises(TypeError):
            cls()

    def test_valid_concrete_instantiates(self):
        cls = self._make_minimal()
        assert cls() is not None


# =============================================================================
# CPUPoseEstimator
# =============================================================================

class TestCPUPoseEstimator:

    @pytest.fixture
    def est(self):
        with patch("vision_processor.backends.cpu.mp") as mock_mp:
            mock_pose = MagicMock()
            mock_mp.solutions.pose.Pose.return_value = mock_pose
            from vision_processor.backends.cpu import CPUPoseEstimator
            estimator = CPUPoseEstimator(model_complexity=1)
            # cpu.py usa self.pose (publico)
            estimator.pose = mock_pose
            yield estimator, mock_pose

    def test_init_ok(self, est):
        estimator, _ = est
        assert estimator is not None

    def test_estimate_converts_bgr_to_rgb(self, est):
        """CPU: COLOR_BGR2RGB, no RGBA."""
        estimator, mock_pose = est
        mock_pose.process.return_value = make_solutions_results()
        with patch("vision_processor.backends.cpu.cv2") as mock_cv2:
            mock_cv2.COLOR_BGR2RGB = 4
            mock_cv2.cvtColor.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
            estimator.estimate(make_bgr_frame())
            assert mock_cv2.cvtColor.call_args[0][1] == mock_cv2.COLOR_BGR2RGB

    def test_estimate_returns_results(self, est):
        estimator, mock_pose = est
        expected = make_solutions_results()
        mock_pose.process.return_value = expected
        with patch("vision_processor.backends.cpu.cv2.cvtColor",
                   return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            assert estimator.estimate(make_bgr_frame()) is expected

    def test_get_landmarks_33_items(self, est):
        estimator, _ = est
        lms = estimator.get_landmarks(make_solutions_results(detected=True))
        assert lms is not None and len(lms) == 33

    def test_get_landmarks_none_when_no_pose(self, est):
        estimator, _ = est
        assert estimator.get_landmarks(make_solutions_results(detected=False)) is None

    def test_get_landmarks_required_keys(self, est):
        estimator, _ = est
        lms = estimator.get_landmarks(make_solutions_results(detected=True))
        for lm in lms:
            assert {"x", "y", "z", "visibility"} <= set(lm.keys())

    def test_release_closes_pose(self, est):
        estimator, mock_pose = est
        estimator.release()
        mock_pose.close.assert_called_once()


# =============================================================================
# MetalPoseEstimator
# =============================================================================

class TestMetalPoseEstimator:
    """
    El bug critico que encontramos al intentar GPU con Metal:
    MediaPipe Tasks GPU delegate requiere SRGBA (4 canales), no SRGB.
    Crash original: 'unsupported ImageFrame format: 1' en
    gpu_buffer_storage_cv_pixel_buffer.cc:154

    Estos tests garantizan que la correccion no se revierta.
    """

    @pytest.fixture
    def est(self):
        """
        metal.py importa mediapipe dentro de __init__, no a nivel de modulo.
        Hay que parchear mediapipe en sys.modules antes de la importacion.
        """
        mock_mp = MagicMock()
        mock_vision = MagicMock()
        mock_landmarker = MagicMock()
        mock_vision.PoseLandmarker.create_from_options.return_value = mock_landmarker

        # Parchear en sys.modules para que los imports dentro de __init__ los cojan
        modules_patch = {
            "mediapipe": mock_mp,
            "mediapipe.tasks": MagicMock(),
            "mediapipe.tasks.python": MagicMock(),
            "mediapipe.tasks.python.vision": mock_vision,
            "mediapipe.tasks.python.core": MagicMock(),
            "mediapipe.tasks.python.core.base_options": MagicMock(),
        }

        with patch.dict("sys.modules", modules_patch), \
             patch("vision_processor.backends.metal._ensure_model"):
            # Importar el modulo limpio en cada test
            if "vision_processor.backends.metal" in sys.modules:
                del sys.modules["vision_processor.backends.metal"]
            from vision_processor.backends.metal import MetalPoseEstimator
            estimator = MetalPoseEstimator()
            estimator._mp = mock_mp
            estimator._landmarker = mock_landmarker
            yield estimator, mock_landmarker, mock_mp

    def test_init_ok(self, est):
        estimator, _, _ = est
        assert estimator is not None

    def test_estimate_converts_bgr_to_rgba_not_rgb(self, est):
        """
        CRITICO: debe usar COLOR_BGR2RGBA (4ch), no COLOR_BGR2RGB (3ch).
        """
        estimator, mock_landmarker, _ = est
        mock_landmarker.detect_for_video.return_value = make_tasks_results(detected=False)
        with patch("vision_processor.backends.metal.cv2") as mock_cv2:
            mock_cv2.COLOR_BGR2RGBA = 3
            mock_cv2.COLOR_BGR2RGB  = 4  # no debe usarse este
            mock_cv2.cvtColor.return_value = np.zeros((480, 640, 4), dtype=np.uint8)
            estimator.estimate(make_bgr_frame())
            assert mock_cv2.cvtColor.call_args[0][1] == mock_cv2.COLOR_BGR2RGBA

    def test_estimate_uses_srgba_image_format(self, est):
        """
        CRITICO: mp.Image debe recibir image_format=mp.ImageFormat.SRGBA.
        """
        estimator, mock_landmarker, mock_mp = est
        mock_landmarker.detect_for_video.return_value = make_tasks_results(detected=False)
        rgba = np.zeros((480, 640, 4), dtype=np.uint8)
        with patch("vision_processor.backends.metal.cv2") as mock_cv2, \
             patch("vision_processor.backends.metal.np") as mock_np:
            mock_cv2.COLOR_BGR2RGBA = 3
            mock_cv2.cvtColor.return_value = rgba
            mock_np.ascontiguousarray.return_value = rgba
            mock_np.uint8 = np.uint8
            estimator.estimate(make_bgr_frame())
            fmt = mock_mp.Image.call_args.kwargs.get("image_format") or \
                  mock_mp.Image.call_args[1].get("image_format")
            assert fmt == mock_mp.ImageFormat.SRGBA

    def test_estimate_calls_ascontiguousarray(self, est):
        """
        Metal necesita array C-contiguo para evitar fallos silenciosos.
        """
        estimator, mock_landmarker, _ = est
        mock_landmarker.detect_for_video.return_value = make_tasks_results(detected=False)
        rgba = np.zeros((480, 640, 4), dtype=np.uint8)
        with patch("vision_processor.backends.metal.cv2") as mock_cv2, \
             patch("vision_processor.backends.metal.np") as mock_np:
            mock_cv2.COLOR_BGR2RGBA = 3
            mock_cv2.cvtColor.return_value = rgba
            mock_np.ascontiguousarray.return_value = rgba
            mock_np.uint8 = np.uint8
            estimator.estimate(make_bgr_frame())
            mock_np.ascontiguousarray.assert_called_once()

    def test_estimate_calls_detect_for_video_not_detect(self, est):
        """Metal usa RunningMode.VIDEO -> detect_for_video, no detect."""
        estimator, mock_landmarker, _ = est
        mock_landmarker.detect_for_video.return_value = make_tasks_results(detected=False)
        rgba = np.zeros((480, 640, 4), dtype=np.uint8)
        with patch("vision_processor.backends.metal.cv2") as mock_cv2, \
             patch("vision_processor.backends.metal.np") as mock_np:
            mock_cv2.COLOR_BGR2RGBA = 3
            mock_cv2.cvtColor.return_value = rgba
            mock_np.ascontiguousarray.return_value = rgba
            mock_np.uint8 = np.uint8
            estimator.estimate(make_bgr_frame())
        mock_landmarker.detect_for_video.assert_called_once()
        mock_landmarker.detect.assert_not_called()

    def test_get_landmarks_33_items(self, est):
        estimator, _, _ = est
        lms = estimator.get_landmarks(make_tasks_results(detected=True, n=33))
        assert lms is not None and len(lms) == 33

    def test_get_landmarks_none_when_empty(self, est):
        estimator, _, _ = est
        assert estimator.get_landmarks(make_tasks_results(detected=False)) is None

    def test_get_landmarks_required_keys(self, est):
        estimator, _, _ = est
        lms = estimator.get_landmarks(make_tasks_results(detected=True))
        for lm in lms:
            assert {"x", "y", "z", "visibility"} <= set(lm.keys())

    def test_get_landmarks_handles_missing_visibility(self, est):
        """Tasks API: si el landmark no tiene .visibility, no debe lanzar."""
        estimator, _, _ = est
        results = MagicMock()
        lms = []
        for i in range(33):
            lm = MagicMock(spec=["x", "y", "z"])  # sin visibility
            lm.x = 0.5
            lm.y = 0.5
            lm.z = 0.0
            lms.append(lm)
        results.pose_landmarks = [lms]
        landmarks = estimator.get_landmarks(results)
        assert landmarks is not None
        for lm in landmarks:
            assert "visibility" in lm

    def test_release_closes_landmarker(self, est):
        estimator, mock_landmarker, _ = est
        estimator.release()
        mock_landmarker.close.assert_called_once()


# =============================================================================
# PoseEstimator() factory (backwards-compatible)
# =============================================================================

class TestPoseEstimatorFactory:

    def test_returns_cpu_estimator(self):
        with patch("vision_processor.backends.cpu.mp") as mock_mp:
            mock_mp.solutions.pose.Pose.return_value = MagicMock()
            from vision_processor.pose import PoseEstimator
            from vision_processor.backends.cpu import CPUPoseEstimator
            assert isinstance(PoseEstimator(), CPUPoseEstimator)

    def test_returns_base_pose_estimator(self):
        from vision_processor.pose import BasePoseEstimator
        with patch("vision_processor.backends.cpu.mp") as mock_mp:
            mock_mp.solutions.pose.Pose.return_value = MagicMock()
            from vision_processor.pose import PoseEstimator
            assert isinstance(PoseEstimator(), BasePoseEstimator)

    def test_accepts_model_complexity(self):
        with patch("vision_processor.backends.cpu.mp") as mock_mp:
            mock_mp.solutions.pose.Pose.return_value = MagicMock()
            from vision_processor.pose import PoseEstimator
            assert PoseEstimator(model_complexity=0) is not None


# =============================================================================
# Config._detect_backend()
# =============================================================================

class TestDetectBackend:

    @pytest.fixture
    def detect(self):
        """Obtiene _detect_backend de Config si ya esta integrado."""
        try:
            from vision_processor.config import Config
            return Config._detect_backend
        except AttributeError:
            pytest.skip("_detect_backend not yet integrated into Config")

    def test_metal_on_darwin_arm64(self, detect):
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="arm64"), \
             patch("os.path.exists", return_value=False):
            assert detect() == "metal"

    def test_no_metal_on_darwin_x86(self, detect):
        with patch("platform.system", return_value="Darwin"), \
             patch("platform.machine", return_value="x86_64"), \
             patch("os.path.exists", return_value=False):
            assert detect() != "metal"

    def test_cpu_on_linux_without_jetson(self, detect):
        with patch("platform.system", return_value="Linux"), \
             patch("os.path.exists", return_value=False):
            assert detect() == "cpu"

    def test_cpu_on_windows(self, detect):
        with patch("platform.system", return_value="Windows"), \
             patch("os.path.exists", return_value=False):
            assert detect() == "cpu"

    def test_explicit_cpu_override(self, detect):
        assert detect(requested="cpu") == "cpu"

    def test_unknown_backend_raises(self, detect):
        with pytest.raises(ValueError, match="Unknown backend"):
            detect(requested="nonexistent")


# =============================================================================
# Consistencia de interfaz entre backends
# =============================================================================

class TestBackendInterfaceConsistency:

    REQUIRED_METHODS = ("estimate", "get_landmarks", "draw_skeleton", "release")

    def test_cpu_has_all_required_methods(self):
        with patch("vision_processor.backends.cpu.mp") as mock_mp:
            mock_mp.solutions.pose.Pose.return_value = MagicMock()
            from vision_processor.backends.cpu import CPUPoseEstimator
            est = CPUPoseEstimator()
            for m in self.REQUIRED_METHODS:
                assert callable(getattr(est, m, None)), f"CPU missing: {m}"

    def test_metal_has_all_required_methods(self):
        mock_mp = MagicMock()
        mock_vision = MagicMock()
        mock_vision.PoseLandmarker.create_from_options.return_value = MagicMock()
        modules_patch = {
            "mediapipe": mock_mp,
            "mediapipe.tasks": MagicMock(),
            "mediapipe.tasks.python": MagicMock(),
            "mediapipe.tasks.python.vision": mock_vision,
            "mediapipe.tasks.python.core": MagicMock(),
            "mediapipe.tasks.python.core.base_options": MagicMock(),
        }
        with patch.dict("sys.modules", modules_patch), \
             patch("vision_processor.backends.metal._ensure_model"):
            if "vision_processor.backends.metal" in sys.modules:
                del sys.modules["vision_processor.backends.metal"]
            from vision_processor.backends.metal import MetalPoseEstimator
            est = MetalPoseEstimator()
            for m in self.REQUIRED_METHODS:
                assert callable(getattr(est, m, None)), f"Metal missing: {m}"
