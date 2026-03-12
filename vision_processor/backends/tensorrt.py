"""
TensorRT backend for pose estimation on NVIDIA Jetson.

Uses MediaPipe Tasks API with the model converted to TensorRT format,
running inference directly on the Jetson's Ampere GPU via TensorRT.
Expected latency: ~15ms vs ~60ms on CPU.

This backend follows the same approach as MetalPoseEstimator but targets
the NVIDIA GPU path. Since MediaPipe's pip wheel does not support GPU on
Jetson (build flags issue), we use TensorRT directly through the ONNX
Runtime TensorRT Execution Provider, using the same pose model converted
to ONNX format.

Requirements:
    - NVIDIA Jetson with JetPack 6+ (CUDA 12.6, TensorRT 10.3)
    - tensorrt Python bindings (already present in the Jetson system)
    - onnx, onnxruntime (CPU version, for preprocessing only)
    - Model: pose_landmarker_full.task (converted to ONNX offline)

NOTE: The first run builds the TensorRT engine from the ONNX model.
      This takes ~2-3 minutes but only happens once. The engine is
      cached in pose_engine.trt next to the model file.
"""

import os
import cv2
import numpy as np

from vision_processor.pose import BasePoseEstimator

_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
)
_ONNX_PATH = os.path.join(_MODEL_DIR, "pose_model.onnx")
_ENGINE_PATH = os.path.join(_MODEL_DIR, "pose_engine.trt")

# Dimensiones de entrada que espera el modelo de pose de MediaPipe
_INPUT_HEIGHT = 256
_INPUT_WIDTH = 256

# Índices de landmarks en la salida del modelo (33 landmarks de BlazePose)
_NUM_LANDMARKS = 33


class TensorRTPoseEstimator(BasePoseEstimator):
    """
    Pose estimator using TensorRT on NVIDIA Jetson GPU.

    Loads a TensorRT engine built from the BlazePose ONNX model.
    Inference runs entirely on the Jetson Ampere GPU.

    The result object returned by estimate() is a numpy array of shape
    (33, 4) with columns [x, y, z, visibility], normalized to [0, 1].
    All conversion to the standard landmark dict format happens here.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401 — inicializa CUDA automáticamente

        self._trt = trt
        self._cuda = cuda

        self._min_confidence = min_detection_confidence

        if not os.path.exists(_ENGINE_PATH):
            if not os.path.exists(_ONNX_PATH):
                raise FileNotFoundError(
                    f"Modelo ONNX no encontrado en {_ONNX_PATH}.\n"
                    "Ejecuta primero: python tools/convert_model.py"
                )
            print("[TensorRTBackend] Construyendo motor TensorRT (primera vez, ~2-3 min)...")
            self._build_engine()
            print("[TensorRTBackend] Motor construido y guardado.")

        print("[TensorRTBackend] Cargando motor TensorRT...")
        self._engine, self._context = self._load_engine()
        self._allocate_buffers()

        # Para dibujar el esqueleto
        import mediapipe as mp
        self._mp_pose = mp.solutions.pose
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles

        print("[PoseEstimator] Backend: TensorRT (Jetson GPU Ampere)")

    def _build_engine(self):
        """Build TensorRT engine from ONNX model and save to disk."""
        trt = self._trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(_ONNX_PATH, "rb") as f:
            if not parser.parse(f.read()):
                errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
                raise RuntimeError(f"Error al parsear ONNX: {errors}")

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        # FP16 si la GPU lo soporta (Ampere sí)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[TensorRTBackend] FP16 activado.")

        serialized = builder.build_serialized_network(network, config)
        with open(_ENGINE_PATH, "wb") as f:
            f.write(serialized)

    def _load_engine(self):
        """Load TensorRT engine from disk."""
        trt = self._trt
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(_ENGINE_PATH, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        context = engine.create_execution_context()
        return engine, context

    def _allocate_buffers(self):
        """Allocate GPU and CPU memory buffers for inference."""
        import pycuda.driver as cuda

        self._inputs = []
        self._outputs = []
        self._bindings = []
        self._stream = cuda.Stream()

        for binding in self._engine:
            size = (
                self._trt.volume(self._engine.get_binding_shape(binding))
            )
            dtype = self._trt.nptype(self._engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))

            if self._engine.binding_is_input(binding):
                self._inputs.append({"host": host_mem, "device": device_mem})
            else:
                self._outputs.append({"host": host_mem, "device": device_mem})

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare frame for TensorRT inference.

        Resizes to 256x256, converts BGR→RGB, normalizes to [0,1],
        and reshapes to (1, 3, 256, 256) NCHW format.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (_INPUT_WIDTH, _INPUT_HEIGHT))
        normalized = resized.astype(np.float32) / 255.0
        # HWC → NCHW
        nchw = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
        return np.ascontiguousarray(nchw)

    def estimate(self, frame: np.ndarray):
        """
        Run pose estimation on GPU via TensorRT.

        Returns numpy array of shape (33, 4): [x, y, z, visibility].
        Returns None if no pose is detected (low confidence).
        """
        import pycuda.driver as cuda

        input_data = self._preprocess(frame)

        # Copiar input a GPU
        np.copyto(self._inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(
            self._inputs[0]["device"],
            self._inputs[0]["host"],
            self._stream,
        )

        # Inferencia
        self._context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle,
        )

        # Copiar output de GPU a CPU
        cuda.memcpy_dtoh_async(
            self._outputs[0]["host"],
            self._outputs[0]["device"],
            self._stream,
        )
        self._stream.synchronize()

        raw = self._outputs[0]["host"].reshape(_NUM_LANDMARKS, -1)

        # Filtrar por confianza (columna 3 = visibility/score)
        if raw.shape[1] >= 4:
            mean_visibility = raw[:, 3].mean()
            if mean_visibility < self._min_confidence:
                return None

        return raw  # shape (33, 4): x, y, z, visibility

    def get_landmarks(self, results) -> list | None:
        """Convert TensorRT output array to standard landmark dict list."""
        if results is None:
            return None

        return [
            {
                "x": float(results[i, 0]),
                "y": float(results[i, 1]),
                "z": float(results[i, 2]),
                "visibility": float(results[i, 3]) if results.shape[1] > 3 else 1.0,
            }
            for i in range(_NUM_LANDMARKS)
        ]

    def draw_skeleton(self, frame: np.ndarray, results) -> np.ndarray:
        """Draw skeleton from TensorRT landmark array."""
        if results is None:
            return frame

        from mediapipe.framework.formats import landmark_pb2

        landmark_list = landmark_pb2.NormalizedLandmarkList()
        for i in range(_NUM_LANDMARKS):
            lm = landmark_list.landmark.add()
            lm.x = float(results[i, 0])
            lm.y = float(results[i, 1])
            lm.z = float(results[i, 2])

        self._mp_drawing.draw_landmarks(
            frame,
            landmark_list,
            self._mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self._mp_drawing_styles.get_default_pose_landmarks_style(),
        )
        return frame

    def release(self):
        """Release TensorRT and CUDA resources."""
        del self._context
        del self._engine
        print("[PoseEstimator] TensorRT backend released.")
