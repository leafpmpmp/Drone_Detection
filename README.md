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