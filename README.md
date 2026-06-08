# DroneDetection app

This is the working demonstration system for the SIGIR 2026 DEMO Paper VRSAR.
We present VRSAR, a lightweight, fully offline visual recognition system designed for real-time victim detection in the field. 

## Run the app

### uv

Default Run:
```
uv run flet run
```

Run with CUDA:
```
uv run --extra cu128 flet run
```

Run with XPU(iGPU):
```
uv run --extra xpu flet run
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

TensorRT is optional. The `trt` backend requires a CUDA-enabled PyTorch
installation and the TensorRT Python package.

## Build the app

### uv

<pre>
uv run (--extra cu128) flet build <i>target_system</i> --module-name main
</pre>
