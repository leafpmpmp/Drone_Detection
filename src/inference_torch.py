import os
import sys
import base64
import time
import uuid
import threading
import queue
import cv2
import torch
import torch.nn as nn
import numpy as np

# Import D-FINE core configuration
from engine.core import YAMLConfig

def draw_boxes(images, labels, boxes, scores, conf_thres=0.35):
    """
    OpenCV-based drawing function to replace the rtdetrv2 dependency.
    """
    detect_frame = images[0].copy()
    scr = scores[0]
    
    # Filter by confidence threshold
    valid_mask = scr > conf_thres
    lab = labels[0][valid_mask]
    box = boxes[0][valid_mask]
    scrs = scr[valid_mask]
    
    box_count = len(box)
    
    for j, b in enumerate(box):
        b_np = b.detach().cpu().numpy().astype(int)
        l_id = lab[j].item()
        s_val = scrs[j].item()
        
        # Draw bounding box (Red)
        cv2.rectangle(detect_frame, (b_np[0], b_np[1]), (b_np[2], b_np[3]), (0, 0, 255), 2)
        # Draw label (Blue)
        label_text = f"Class {l_id} {s_val:.2f}"
        cv2.putText(detect_frame, label_text, (b_np[0], max(0, b_np[1] - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
    return detect_frame, box_count


class DFineWrapper(nn.Module):
    """Wrapper to handle D-FINE deploy mode operations natively and fix device mismatches."""
    def __init__(self, cfg, device):
        super().__init__()
        
        # 1. Move to device BEFORE deploy to ensure any auto-generated tensors default to GPU
        cfg.model.to(device)
        cfg.postprocessor.to(device)
        
        self.model = cfg.model.deploy()
        self.postprocessor = cfg.postprocessor.deploy()
        
        # 2. Force sweep hidden python attributes (lists/tuples) that .deploy() stranded on CPU
        self._force_tensors_to_device(self.model, device)
        self._force_tensors_to_device(self.postprocessor, device)

    def _force_tensors_to_device(self, module, device):
        """Recursively forces hidden tensor attributes to the target device."""
        # Check standard attributes
        for attr_name in dir(module):
            # Skip dunder methods to avoid messing with Python internals
            if attr_name.startswith('__'): 
                continue
            
            try:
                attr = getattr(module, attr_name)
                # Catch stray individual tensors
                if isinstance(attr, torch.Tensor):
                    setattr(module, attr_name, attr.to(device))
                # Catch lists or tuples containing tensors (This is where D-FINE hides pos_embeds)
                elif isinstance(attr, (list, tuple)):
                    new_attr = []
                    changed = False
                    for item in attr:
                        if isinstance(item, torch.Tensor):
                            new_attr.append(item.to(device))
                            changed = True
                        else:
                            new_attr.append(item)
                    
                    if changed:
                        # Reassign the updated list/tuple back to the module
                        setattr(module, attr_name, type(attr)(new_attr))
            except Exception:
                pass
                
        # Traverse submodules
        for child in module.children():
            self._force_tensors_to_device(child, device)

    def forward(self, images, orig_target_sizes):
        outputs = self.model(images)
        outputs = self.postprocessor(outputs, orig_target_sizes)
        return outputs


class DetectorEngine:
    def __init__(
        self,
        model_path: str,
        config_path: str,          # Added config path requirement
        device: str | None = None,
        output_root: str = "outputFile",
    ):
        self.model_path = model_path
        self.config_path = config_path 
        
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            self.device = torch.device("xpu")
        else:
            self.device = torch.device("cpu")
        
        print(f"Detector initialized on device: {self.device}")

        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)
        
        self.lang_data = {}

        self._model = None

        self._preview_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._preview_q: queue.Queue = queue.Queue(maxsize=1)
        self._preview_meta_lock = threading.Lock()
        self._preview_meta = {
            "status": "idle",
            "frames": 0,
            "detected_frames": 0,
            "fps_eff": 0.0,
            "last_msg": "",
            "out_video_path": "",
        }

    def set_output_root(self, new_root: str):
        self.output_root = new_root
        os.makedirs(self.output_root, exist_ok=True)

    def set_language(self, lang_data: dict):
        self.lang_data = lang_data

    def _get_text(self, key: str, default: str) -> str:
        return self.lang_data.get(key, default)

    def _ensure_model(self, sample_input_path: str, is_video: bool, output_dir: str):
        if self._model is not None:
            return
            
        print("Loading D-FINE Model Configuration and Weights...")
        
        # Keep your num_classes override if you are passing it here dynamically
        cfg = YAMLConfig(self.config_path, resume=self.model_path, num_classes=2)

        if 'HGNetv2' in cfg.yaml_cfg:
            cfg.yaml_cfg['HGNetv2']['pretrained'] = False

        checkpoint = torch.load(self.model_path, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # Load state and wrap into deployable structure (passing self.device)
        cfg.model.load_state_dict(state)
        self._model = DFineWrapper(cfg, self.device).to(self.device)
        self._model.eval()

    def _prepare_input(self, frame_bgr):
        # D-FINE expects RGB (original used PIL.Image which is RGB)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize on CPU (OpenCV is fast enough and avoids upload overhead for large frames)
        frame_resized = cv2.resize(frame_rgb, (640, 640), interpolation=cv2.INTER_LINEAR)
        
        # Upload as uint8 (4x smaller transfer than float32)
        tensor = torch.from_numpy(frame_resized).permute(2, 0, 1).to(self.device, non_blocking=True)
        
        # Normalize on GPU to Float32
        im_data = tensor.float() / 255.0
        
        return im_data.unsqueeze(0)

    def _run_forward(self, im_data, orig_size):
        # Use autocast to handle mixed precision (Float/Half) automatically
        if self.device.type != "cpu":
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                return self._model(im_data, orig_size)
        else:
            return self._model(im_data, orig_size)

    def _encode_b64_jpg(self, bgr_img):
        ok, buffer = cv2.imencode(".jpg", bgr_img)
        if not ok:
            raise RuntimeError("圖片編碼成 JPG 失敗")
        return base64.b64encode(buffer).decode("utf-8")

    def run_inference(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到檔案: {file_path}")

        ext = os.path.splitext(file_path)[1].lower()
        is_video = ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        is_image = ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

        if not is_video and not is_image:
            raise ValueError(f"不支援的檔案格式: {ext}")

        uid = str(uuid.uuid4())
        out_dir = os.path.join(self.output_root, uid)
        os.makedirs(out_dir, exist_ok=True)

        self._ensure_model(file_path, is_video, out_dir)

        if is_image:
            return self._infer_image(file_path, out_dir)

        return self._infer_video_blocking(file_path, out_dir)

    def _infer_image(self, image_path: str, out_dir: str):
        t0 = time.time()

        img = cv2.imread(image_path)
        if img is None:
            raise RuntimeError(self._get_text("error_read_image", "cv2.imread 讀不到圖片（路徑或格式可能有問題）"))

        h, w = img.shape[:2]
        orig_size = torch.tensor([[w, h]]).to(self.device)

        im_data = self._prepare_input(img)

        output = self._run_forward(im_data, orig_size)
        labels, boxes, scores = output
        detect_frame, box_count = draw_boxes([img], labels, boxes, scores, 0.35)

        out_img_path = os.path.join(out_dir, "result.jpg")
        cv2.imwrite(out_img_path, detect_frame)

        b64_img = self._encode_b64_jpg(detect_frame)
        dt = time.time() - t0

        summary = []
        summary.append(f"{self._get_text('summary_input', '輸入')}: {os.path.abspath(image_path)}")
        summary.append(self._get_text('summary_detected_count', '偵測到 {count} 個目標').format(count=int(box_count)))
        summary.append(f"{self._get_text('summary_time', '耗時')}: {dt:.2f} s")
        summary.append(f"{self._get_text('summary_output_file', '輸出檔案')}: {os.path.abspath(out_img_path)}")

        return b64_img, "\n".join(summary), out_img_path

    def ensure_initialized(self):
        if self._model is not None:
            return
        self._ensure_model("dummy.jpg", False, self.output_root)

    def infer_frame(self, frame_bgr, conf_thres=0.35):
        self.ensure_initialized()
        
        h, w = frame_bgr.shape[:2]
        orig_size = torch.tensor([[w, h]]).to(self.device)

        im_data = self._prepare_input(frame_bgr)

        output = self._run_forward(im_data, orig_size)
        labels, boxes, scores = output
        
        detect_frame, box_count = draw_boxes([frame_bgr], labels, boxes, scores, conf_thres)
        return detect_frame, box_count

    def _infer_video_blocking(self, video_path: str, out_dir: str):
        t0 = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(self._get_text("error_open_video", "無法開啟影片（編碼或路徑可能有問題）"))

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 114514.0  # Default to a high FPS if it cannot be determined

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_video_path = os.path.join(out_dir, "result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(self._get_text("error_create_video", "無法建立輸出影片（mp4v/路徑/權限問題）"))

        orig_size = torch.tensor([[w, h]]).to(self.device)

        frames = 0
        detected_frames = 0
        preview_b64 = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            im_data = self._prepare_input(frame)

            output = self._run_forward(im_data, orig_size)
            labels, boxes, scores = output
            detect_frame, box_count = draw_boxes([frame], labels, boxes, scores, 0.35)

            if box_count > 0:
                detected_frames += 1

            if preview_b64 is None:
                preview_b64 = self._encode_b64_jpg(detect_frame)

            writer.write(detect_frame)
            frames += 1

        cap.release()
        writer.release()

        dt = time.time() - t0
        fps_eff = (frames / dt) if dt > 0 else 0.0

        if preview_b64 is None:
            blank = (torch.zeros((h, w, 3)) * 255).byte().cpu().numpy()
            preview_b64 = self._encode_b64_jpg(blank)

        summary = []
        summary.append(f"{self._get_text('summary_input', '輸入')}: {os.path.abspath(video_path)}")
        summary.append(f"{self._get_text('summary_total_frames', '總幀數')}: {frames}")
        summary.append(f"{self._get_text('summary_detected_frames_count', '偵測到目標的幀數')}: {detected_frames}")
        summary.append(f"{self._get_text('summary_time', '總耗時')}: {dt:.2f} s")
        summary.append(f"{self._get_text('summary_fps', '有效 FPS')}: {fps_eff:.2f}")
        summary.append(f"{self._get_text('summary_output_file', '輸出檔案')}: {os.path.abspath(out_video_path)}")

        return preview_b64, "\n".join(summary), out_video_path

    def start_video_preview(
        self,
        video_path: str,
        out_dir: str | None = None,
        conf: float = 0.35,
        ui_stride: int = 3,
        write_video: bool = True,
    ):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"找不到檔案: {video_path}")

        if self._preview_thread is not None and self._preview_thread.is_alive():
            raise RuntimeError("目前已有預覽推理在進行中，請先 stop_preview()")

        uid = str(uuid.uuid4())
        out_dir = out_dir or os.path.join(self.output_root, uid)
        os.makedirs(out_dir, exist_ok=True)

        self._ensure_model(video_path, True, out_dir)

        self._stop_event.clear()

        with self._preview_meta_lock:
            self._preview_meta = {
                "status": "running",
                "frames": 0,
                "total_frames": 0,
                "detected_frames": 0,
                "fps_eff": 0.0,
                "last_msg": "",
                "out_video_path": "",
                "width": 0,
                "height": 0,
            }

        def worker():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                msg = self._get_text("error_open_video", "無法開啟影片")
                with self._preview_meta_lock:
                    self._preview_meta["status"] = "error"
                    self._preview_meta["last_msg"] = msg
                self._push_preview(("__error__", msg))
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_size = torch.tensor([[w, h]]).to(self.device)
            
            with self._preview_meta_lock:
                self._preview_meta["width"] = w
                self._preview_meta["height"] = h

            writer = None
            out_video_path = ""
            log_f = None
            if write_video:
                out_video_path = os.path.join(out_dir, "result.mp4")
                
                fourcc = cv2.VideoWriter_fourcc(*"avc1")
                writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
                
                if not writer.isOpened():
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))

                if not writer.isOpened():
                     fourcc = cv2.VideoWriter_fourcc(*"hevc")
                     writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
                
                if not writer.isOpened():
                    cap.release()
                    msg = self._get_text("error_create_video", "無法建立輸出影片(Codecs failed)")
                    with self._preview_meta_lock:
                        self._preview_meta["status"] = "error"
                        self._preview_meta["last_msg"] = msg
                    self._push_preview(("__error__", msg))
                    return
                
                log_path = os.path.join(out_dir, "detections.log")
                try:
                    log_f = open(log_path, "w", encoding="utf-8")
                    log_f.write("Frame,Time(s),Detected_Count\n")
                except Exception as e:
                    print(f"Failed to create log file: {e}")

            t0 = time.time()
            frames = 0
            detected_frames = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                im_data = self._prepare_input(frame)

                output = self._run_forward(im_data, orig_size)
                labels, boxes, scores = output
                detect_frame, box_count = draw_boxes([frame], labels, boxes, scores, conf)

                if box_count > 0:
                    detected_frames += 1

                if writer is not None:
                    writer.write(detect_frame)
                    if log_f is not None and box_count > 0:
                        ts = frames / fps if fps > 0 else 0
                        log_f.write(f"{frames},{ts:.2f},{box_count}\n")

                frames += 1

                if frames % ui_stride == 0:
                    now = time.time()
                    dt = now - t0
                    fps_eff = (frames / dt) if dt > 0 else 0.0

                    b64 = self._encode_b64_jpg(detect_frame)
                    msg = self._get_text(
                        "real_time_status",
                        "frames={frames}/{total_frames}, detected_frames={detected_frames}, fps≈{fps}",
                    ).format(
                        frames=frames,
                        total_frames=total_frames,
                        detected_frames=detected_frames,
                        fps=f"{fps_eff:.1f}",
                    )
                    self._push_preview((b64, msg))

                    with self._preview_meta_lock:
                        self._preview_meta["frames"] = frames
                        self._preview_meta["total_frames"] = total_frames
                        self._preview_meta["detected_frames"] = detected_frames
                        self._preview_meta["fps_eff"] = fps_eff
                        self._preview_meta["last_msg"] = msg
                        self._preview_meta["out_video_path"] = out_video_path

            cap.release()
            if writer is not None:
                writer.release()
            if log_f is not None:
                log_f.close()

            dt = time.time() - t0
            fps_eff = (frames / dt) if dt > 0 else 0.0

            status = "stopped" if self._stop_event.is_set() else "done"
            
            summary = []
            summary.append(f"{self._get_text('summary_input', '輸入')}: {os.path.abspath(video_path)}")
            summary.append(f"{self._get_text('summary_total_frames', '總幀數')}: {frames}")
            summary.append(f"{self._get_text('summary_detected_frames_count', '偵測到目標的幀數')}: {detected_frames}")
            summary.append(f"{self._get_text('summary_time', '總耗時')}: {dt:.2f} s")
            summary.append(f"{self._get_text('summary_fps', '有效 FPS')}: {fps_eff:.2f}")
            if write_video:
                summary.append(f"{self._get_text('summary_output_file', '輸出檔案')}: {os.path.abspath(out_video_path)}")

            full_summary_text = "\n".join(summary)

            with self._preview_meta_lock:
                self._preview_meta["status"] = status
                self._preview_meta["last_msg"] = full_summary_text 
                self._preview_meta["out_video_path"] = out_video_path

            self._push_preview(("__done__", full_summary_text))


        self._preview_thread = threading.Thread(target=worker, daemon=True)
        self._preview_thread.start()

    def _push_preview(self, item):
        try:
            while True:
                self._preview_q.get_nowait()
        except queue.Empty:
            pass
        try:
            self._preview_q.put_nowait(item)
        except queue.Full:
            pass

    def get_latest_preview(self):
        try:
            return self._preview_q.get_nowait()
        except queue.Empty:
            return None

    def stop_preview(self):
        self._stop_event.set()

    def get_preview_meta(self):
        with self._preview_meta_lock:
            return dict(self._preview_meta)