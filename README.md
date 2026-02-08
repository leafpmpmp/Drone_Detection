# DroneDetection app

This is the working demonstration system for the paper VRSAR.

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