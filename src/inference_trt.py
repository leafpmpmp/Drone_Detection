from __future__ import annotations

import collections
import importlib.util
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference_torch import DetectorEngine


class TRTInference:
    """Low-level TensorRT runner using PyTorch CUDA tensors as device buffers."""

    def __init__(
        self,
        engine_path: str,
        device: str = "cuda:0",
        backend: str = "torch",
        max_batch_size: int = 32,
        verbose: bool = False,
    ):
        try:
            import tensorrt as trt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "TensorRT Python package is required for the TensorRT backend."
            ) from exc

        self.trt = trt
        self.engine_path = engine_path
        self.device = device
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.logger = (
            trt.Logger(trt.Logger.VERBOSE)
            if verbose
            else trt.Logger(trt.Logger.INFO)
        )

        self.engine = self.load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.bindings = self.get_bindings(
            self.engine,
            self.context,
            self.max_batch_size,
            self.device,
        )
        self.bindings_addr = OrderedDict(
            (name, binding.ptr) for name, binding in self.bindings.items()
        )
        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

    def load_engine(self, path: str):
        self.trt.init_libnvinfer_plugins(self.logger, "")
        with open(path, "rb") as engine_file, self.trt.Runtime(
            self.logger
        ) as runtime:
            return runtime.deserialize_cuda_engine(engine_file.read())

    def get_input_names(self) -> list[str]:
        return [
            name
            for name in self.engine
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT
        ]

    def get_output_names(self) -> list[str]:
        return [
            name
            for name in self.engine
            if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.OUTPUT
        ]

    def get_bindings(
        self,
        engine,
        context,
        max_batch_size: int = 32,
        device: str | None = None,
    ) -> OrderedDict:
        binding_type = collections.namedtuple(
            "Binding",
            ("name", "dtype", "shape", "data", "ptr"),
        )
        bindings = OrderedDict()

        for name in engine:
            shape = list(engine.get_tensor_shape(name))
            dtype = self.trt.nptype(engine.get_tensor_dtype(name))

            if shape and shape[0] == -1:
                shape[0] = max_batch_size
                if (
                    engine.get_tensor_mode(name)
                    == self.trt.TensorIOMode.INPUT
                ):
                    context.set_input_shape(name, shape)

            shape = tuple(shape)
            data = torch.from_numpy(np.empty(shape, dtype=dtype)).to(device)
            bindings[name] = binding_type(
                name,
                dtype,
                shape,
                data,
                data.data_ptr(),
            )

        return bindings

    def run_torch(self, blob: dict[str, torch.Tensor]):
        for name in self.input_names:
            if self.bindings[name].shape != blob[name].shape:
                self.context.set_input_shape(name, blob[name].shape)
                self.bindings[name] = self.bindings[name]._replace(
                    shape=blob[name].shape
                )

            if self.bindings[name].data.dtype != blob[name].dtype:
                raise TypeError(
                    f"{name} dtype mismatch: expected "
                    f"{self.bindings[name].data.dtype}, got {blob[name].dtype}"
                )

        self.bindings_addr.update(
            {name: blob[name].data_ptr() for name in self.input_names}
        )
        self.context.execute_v2(list(self.bindings_addr.values()))
        return {
            name: self.bindings[name].data
            for name in self.output_names
        }

    def __call__(self, blob: dict[str, torch.Tensor]):
        if self.backend == "torch":
            return self.run_torch(blob)
        raise NotImplementedError("Only the 'torch' buffer backend is implemented.")


class TensorRTDetectorEngine(DetectorEngine):
    """Existing DetectorEngine media pipeline with TensorRT forward execution."""

    def __init__(
        self,
        engine_path: Path,
        engine_path_display: str,
        device: str = "cuda:0",
        output_root: str = "outputFile",
    ):
        self.engine_path = engine_path
        self.engine_path_display = engine_path_display
        super().__init__(
            model_path=str(engine_path),
            config_path="",
            device=device,
            output_root=output_root,
        )

    def _ensure_model(self, sample_input_path: str, is_video: bool, output_dir: str):
        if self._model is not None:
            return

        if importlib.util.find_spec("tensorrt") is None:
            raise RuntimeError(
                "TensorRT backend dependency missing: tensorrt. "
                "Install TensorRT and its Python package in this environment."
            )

        if not torch.cuda.is_available():
            raise RuntimeError(
                "TensorRT backend requires CUDA and a CUDA-enabled PyTorch installation."
            )

        print(f"Loading TensorRT engine: {self.engine_path_display}")
        try:
            self._model = TRTInference(
                str(self.engine_path),
                device=str(self.device),
                backend="torch",
                max_batch_size=1,
            )
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "TensorRT backend dependency missing: tensorrt. "
                "Install TensorRT and its Python package in this environment."
            ) from exc
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to import TensorRT backend dependencies: {exc}"
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TensorRT engine at {self.engine_path_display}: {exc}"
            ) from exc

    @staticmethod
    def _output_shapes(outputs: Any) -> dict[str, Any]:
        if not isinstance(outputs, dict):
            return {"output": type(outputs).__name__}
        return {
            name: (
                tuple(value.shape)
                if hasattr(value, "shape")
                else type(value).__name__
            )
            for name, value in outputs.items()
        }

    def _run_forward(self, im_data, orig_size):
        outputs = self._model(
            {
                "images": im_data,
                "orig_target_sizes": orig_size,
            }
        )

        shapes = self._output_shapes(outputs)
        required = ("labels", "boxes", "scores")
        if not isinstance(outputs, dict) or any(
            name not in outputs for name in required
        ):
            raise RuntimeError(
                "Unexpected TensorRT output shape: "
                f"{shapes}. Expected outputs named labels, boxes, and scores."
            )

        labels = outputs["labels"]
        boxes = outputs["boxes"]
        scores = outputs["scores"]
        batch_size = im_data.shape[0]

        valid = (
            all(hasattr(value, "ndim") for value in (labels, boxes, scores))
            and labels.ndim == 2
            and scores.ndim == 2
            and boxes.ndim == 3
            and boxes.shape[-1] == 4
            and labels.shape == scores.shape
            and boxes.shape[:2] == labels.shape
            and labels.shape[0] >= batch_size
        )
        if not valid:
            raise RuntimeError(
                "Unexpected TensorRT output shape: "
                f"{shapes}. Expected labels/scores [batch, detections] and "
                "boxes [batch, detections, 4]."
            )

        return (
            labels[:batch_size],
            boxes[:batch_size],
            scores[:batch_size],
        )
