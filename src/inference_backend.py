from __future__ import annotations

from pathlib import Path

from inference_torch import DetectorEngine
from inference_trt import TensorRTDetectorEngine
import gc
import torch


class BaseInferenceBackend:
    """Common inference interface used by the UI and stream paths."""

    def __init__(self, detector: DetectorEngine):
        self._detector = detector

    def predict(self, image, conf_thres: float = 0.35):
        raise NotImplementedError

    def __getattr__(self, name: str):
        # Preserve the existing image, video, preview, and configuration API.
        return getattr(self._detector, name)


class TorchInferenceBackend(BaseInferenceBackend):
    """Common backend adapter for the existing PyTorch detector."""

    def __init__(
        self,
        model_path: str,
        config_path: str,
        device: str | None = None,
        output_root: str = "outputFile",
    ):
        super().__init__(
            DetectorEngine(
                model_path=model_path,
                config_path=config_path,
                device=device,
                output_root=output_root,
            )
        )

    def predict(self, image, conf_thres: float = 0.35):
        return self._detector.infer_frame(image, conf_thres)


class TensorRTInferenceBackend(BaseInferenceBackend):
    """Common backend adapter for the TensorRT detector runtime."""

    def __init__(
        self,
        engine_path: str,
        device: str = "cuda:0",
        output_root: str = "outputFile",
    ):
        resolved_path = _resolve_engine_path(engine_path)
        if not resolved_path.is_file():
            raise FileNotFoundError(
                f"TensorRT engine not found at {engine_path}"
            )

        detector = TensorRTDetectorEngine(
            engine_path=resolved_path,
            engine_path_display=engine_path,
            device=device,
            output_root=output_root,
        )
        detector.ensure_initialized()
        super().__init__(detector)

    def predict(self, image, conf_thres: float = 0.35):
        return self._detector.infer_frame(image, conf_thres)


def _resolve_engine_path(engine_path: str) -> Path:
    path = Path(engine_path).expanduser()
    if path.is_absolute():
        return path

    project_root = Path(__file__).resolve().parent.parent
    candidates = (
        Path.cwd() / path,
        project_root / path,
        Path(__file__).resolve().parent / path,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return path

class DetectorManager:
    """Manages the lifecycle and memory of the active inference backend."""
    
    def __init__(self, model_path, config_path, engine_path, device):
        self.model_path = model_path
        self.config_path = config_path
        self.engine_path = engine_path
        self.device = device
        
        self.active_backend = None
        self.current_backend_type = None

    def set_backend(self, new_backend_type: str):
        if self.current_backend_type == new_backend_type:
            return self.active_backend # Already active

        # 1. Demolish the old engine and free the VRAM
        if self.active_backend is not None:
            print(f"Tearing down {self.current_backend_type} engine...")
            del self.active_backend
            self.active_backend = None
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 2. Use your existing factory to build the new one
        print(f"Spinning up {new_backend_type} engine...")
        self.active_backend = create_inference_backend(
            backend=new_backend_type,
            model_path=self.model_path,
            config_path=self.config_path,
            engine_path=self.engine_path,
            device=self.device
        )
        self.current_backend_type = new_backend_type
        
        return self.active_backend

def create_inference_backend(
    backend: str,
    *,
    model_path: str,
    config_path: str,
    engine_path: str,
    device: str | None = None,
    output_root: str = "outputFile",
) -> BaseInferenceBackend:
    backend_name = backend.strip().lower()
    if backend_name == "torch":
        print("Inference backend: torch")
        return TorchInferenceBackend(
            model_path=model_path,
            config_path=config_path,
            device=device,
            output_root=output_root,
        )
    if backend_name == "trt":
        print(f"Inference backend: TensorRT, engine: {engine_path}")
        return TensorRTInferenceBackend(
            engine_path=engine_path,
            device=device or "cuda:0",
            output_root=output_root,
        )
    raise ValueError(
        f"Unsupported inference backend: {backend}. Expected 'torch' or 'trt'."
    )
