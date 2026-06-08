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
        max_batch_size: int = 1,
        verbose: bool = False,
        debug: bool = False,
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
        self.debug = debug
        self.logger = (
            trt.Logger(trt.Logger.VERBOSE)
            if verbose
            else trt.Logger(trt.Logger.INFO)
        )

        self.engine = self.load_engine(engine_path)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()

        self.input_names = self.get_input_names()
        self.output_names = self.get_output_names()

        self.bindings = self.get_bindings(
            self.engine,
            self.context,
            self.max_batch_size,
            self.device,
        )
        self.bindings_addr = OrderedDict(
            (name, binding.ptr) for name, binding in self.bindings.items()
        )

        if self.debug:
            self._print_engine_io_summary()

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
        max_batch_size: int = 1,
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

            # Dynamic batch support, although current engine is expected to be fixed [1, ...].
            if shape and shape[0] == -1:
                shape[0] = max_batch_size
                if engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT:
                    context.set_input_shape(name, tuple(shape))

            shape = tuple(shape)

            # Allocate output buffers as contiguous CUDA tensors.
            # Input buffers will be replaced by actual input tensor pointers in run_torch().
            data = torch.empty(shape, dtype=self._numpy_dtype_to_torch(dtype), device=device)
            data = data.contiguous()

            bindings[name] = binding_type(
                name=name,
                dtype=dtype,
                shape=shape,
                data=data,
                ptr=data.data_ptr(),
            )

        return bindings

    @staticmethod
    def _numpy_dtype_to_torch(dtype: np.dtype | type) -> torch.dtype:
        dtype = np.dtype(dtype)
        if dtype == np.float32:
            return torch.float32
        if dtype == np.float16:
            return torch.float16
        if dtype == np.int64:
            return torch.int64
        if dtype == np.int32:
            return torch.int32
        if dtype == np.int8:
            return torch.int8
        if dtype == np.uint8:
            return torch.uint8
        if dtype == np.bool_:
            return torch.bool
        raise TypeError(f"Unsupported TensorRT binding dtype: {dtype}")

    def _print_engine_io_summary(self):
        print("=== TensorRT Engine IO Summary ===")
        print(f"Engine path: {self.engine_path}")
        print(f"Input names: {self.input_names}")
        print(f"Output names: {self.output_names}")

        for name in self.engine:
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = self.engine.get_tensor_dtype(name)
            print(f"  {name}: mode={mode}, shape={shape}, dtype={dtype}")

    def _debug_tensor(self, name: str, tensor: torch.Tensor):
        if not self.debug:
            return

        try:
            if tensor.numel() == 0:
                print(f"[TRT DEBUG] {name}: empty tensor")
                return

            if tensor.is_floating_point():
                t_min = float(tensor.min().detach().cpu())
                t_max = float(tensor.max().detach().cpu())
                t_mean = float(tensor.mean().detach().cpu())
                print(
                    f"[TRT DEBUG] {name}: "
                    f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                    f"device={tensor.device}, contiguous={tensor.is_contiguous()}, "
                    f"stride={tensor.stride()}, min={t_min:.6f}, "
                    f"max={t_max:.6f}, mean={t_mean:.6f}"
                )
            else:
                t_min = tensor.min().detach().cpu().item()
                t_max = tensor.max().detach().cpu().item()
                print(
                    f"[TRT DEBUG] {name}: "
                    f"shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                    f"device={tensor.device}, contiguous={tensor.is_contiguous()}, "
                    f"stride={tensor.stride()}, min={t_min}, max={t_max}"
                )
        except Exception as exc:
            print(f"[TRT DEBUG] Failed to inspect tensor {name}: {exc}")

    def run_torch(self, blob: dict[str, torch.Tensor]):
        # Validate required inputs.
        for name in self.input_names:
            if name not in blob:
                raise KeyError(
                    f"Missing TensorRT input '{name}'. "
                    f"Available inputs: {list(blob.keys())}"
                )

        # Prepare input pointers.
        for name in self.input_names:
            tensor = blob[name]

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"TensorRT input '{name}' must be a torch.Tensor, got {type(tensor)}"
                )

            if not tensor.is_cuda:
                raise TypeError(
                    f"TensorRT input '{name}' must be on CUDA device, got {tensor.device}"
                )

            # Critical fix:
            # TensorRT receives only data_ptr() and assumes contiguous memory.
            # PyTorch non-contiguous tensors, especially from permute(), will be read incorrectly.
            tensor = tensor.contiguous()

            expected_dtype = self.bindings[name].data.dtype
            if tensor.dtype != expected_dtype:
                raise TypeError(
                    f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
                )

            current_shape = tuple(tensor.shape)
            binding_shape = tuple(self.bindings[name].shape)

            if binding_shape != current_shape:
                self.context.set_input_shape(name, current_shape)
                self.bindings[name] = self.bindings[name]._replace(
                    shape=current_shape,
                    data=tensor,
                    ptr=tensor.data_ptr(),
                )
            else:
                self.bindings[name] = self.bindings[name]._replace(
                    data=tensor,
                    ptr=tensor.data_ptr(),
                )

            self.bindings_addr[name] = tensor.data_ptr()
            self._debug_tensor(f"input/{name}", tensor)

        # Make sure output binding addresses are current.
        for name in self.output_names:
            output_tensor = self.bindings[name].data.contiguous()
            self.bindings[name] = self.bindings[name]._replace(
                data=output_tensor,
                ptr=output_tensor.data_ptr(),
            )
            self.bindings_addr[name] = output_tensor.data_ptr()

        # execute_v2 uses binding pointers in engine iteration order.
        binding_ptrs = [self.bindings_addr[name] for name in self.engine]
        ok = self.context.execute_v2(binding_ptrs)
        if not ok:
            raise RuntimeError("TensorRT execute_v2 failed.")

        outputs = {}
        for name in self.output_names:
            out = self.bindings[name].data

            # Optional clone:
            # This prevents returned tensors from being overwritten by the next inference.
            # For single-thread immediate drawing, clone is not strictly required.
            # For UI/stream/multithread safety, clone is safer.
            out = out.clone()

            self._debug_tensor(f"output/{name}", out)
            outputs[name] = out

        return outputs

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
        debug: bool = False,
    ):
        self.engine_path = engine_path
        self.engine_path_display = engine_path_display
        self.debug = debug

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
                debug=self.debug,
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

    def _run_forward(self, im_data: torch.Tensor, orig_size: torch.Tensor):
        # Critical safety checks/fixes for TensorRT:
        # - images must be fp32 contiguous NCHW CUDA tensor
        # - orig_target_sizes must be int64 contiguous CUDA tensor
        # Your engine log shows:
        # images: fp32 [1,3,640,640]
        # orig_target_sizes: int64 [1,2]
        im_data = im_data.to(device=self.device, dtype=torch.float32).contiguous()
        orig_size = orig_size.to(device=self.device, dtype=torch.int64).contiguous()

        if im_data.ndim != 4:
            raise RuntimeError(
                f"TensorRT input 'images' must be 4D [N,C,H,W], got {tuple(im_data.shape)}"
            )

        if im_data.shape[1:] != (3, 640, 640):
            raise RuntimeError(
                "TensorRT input 'images' must have shape [N,3,640,640], "
                f"got {tuple(im_data.shape)}"
            )

        if orig_size.ndim != 2 or orig_size.shape[1] != 2:
            raise RuntimeError(
                "TensorRT input 'orig_target_sizes' must have shape [N,2], "
                f"got {tuple(orig_size.shape)}"
            )

        if orig_size.shape[0] != im_data.shape[0]:
            raise RuntimeError(
                "Batch mismatch between images and orig_target_sizes: "
                f"images batch={im_data.shape[0]}, "
                f"orig_target_sizes batch={orig_size.shape[0]}"
            )

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

        labels = labels[:batch_size].contiguous()
        boxes = boxes[:batch_size].contiguous()
        scores = scores[:batch_size].contiguous()

        if self.debug:
            self._debug_outputs(labels, boxes, scores)

        return labels, boxes, scores

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

    def _debug_outputs(
        self,
        labels: torch.Tensor,
        boxes: torch.Tensor,
        scores: torch.Tensor,
    ):
        try:
            print("=== TensorRT Output Debug ===")
            print(
                f"labels: shape={tuple(labels.shape)}, dtype={labels.dtype}, "
                f"min={labels.min().item()}, max={labels.max().item()}"
            )
            print(
                f"boxes: shape={tuple(boxes.shape)}, dtype={boxes.dtype}, "
                f"min={boxes.min().item():.3f}, max={boxes.max().item():.3f}"
            )
            print(
                f"scores: shape={tuple(scores.shape)}, dtype={scores.dtype}, "
                f"min={scores.min().item():.6f}, max={scores.max().item():.6f}"
            )

            k = min(10, scores.shape[1])
            print("First detections:")
            for i in range(k):
                label = labels[0, i].detach().cpu().item()
                score = scores[0, i].detach().cpu().item()
                box = boxes[0, i].detach().cpu().tolist()
                print(f"  {i}: label={label}, score={score:.4f}, box={box}")

        except Exception as exc:
            print(f"[TRT DEBUG] Failed to print output debug info: {exc}")