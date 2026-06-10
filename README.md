# RescueLens: A Dual-Mode Edge Media Inspection System for UAV-Based Search and Rescue

This is the working demonstration system for the ACM MM 2026 DEMO Paper **RescueLens**.
We present RescueLens, a **lightweight**, **fully offline dual-mode** visual recognition system designed for real-time victim detection in the field.

## Run the app

### uv

```
uv run flet run
```

### others

Install requirements:

```
pip install -r requirement.txt
```

Run:

```
python -m flet run
```

or

```
python ./src/main.py
```

### Inference backend

- Model weights should be placed in `src/weights/ ` .

PyTorch remains the default:

```powershell
python src/main.py --backend torch
```

Use the TensorRT engine:

```powershell
python src/main.py --backend trt --engine-path src/weights/model.engine
```

The backend can also be selected with an environment variable:

```powershell
$env:DETECTOR_BACKEND = "trt"
$env:TENSORRT_ENGINE_PATH = "src/weights/model.engine"
python src/main.py
```

Or select backend and model files in the settings page from the app.

- TensorRT is optional. The `trt` backend requires a CUDA-enabled PyTorch
  installation and the TensorRT Python package.

## Tools

### 1. Setup

```powershell
uv add onnx onnxsim onnxruntime
```

### 2. Export onnx

```powershell
python src/tools/export_onnx.py --check -c src/configs/rtv4/rtv4_hgnetv2_${model}_coco.yml -r model.pth
```

### 3. Export TensorRT

```powershell
trtexec --onnx="src/weights/model.onnx" --saveEngine="src/weights/model.engine"
```
