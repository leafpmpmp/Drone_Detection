import os
import sys
import base64
import time
import uuid
import threading
import queue
import cv2
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rtdetrv2.tools.infer import InitArgs, draw, initModel


class DetectorEngine:
    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        output_root: str = "outputFile",
    ):
        self.model_path = model_path
        self.device = (
            torch.device(device)
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.output_root = output_root
        os.makedirs(self.output_root, exist_ok=True)

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

    def _ensure_model(self, sample_input_path: str, is_video: bool, output_dir: str):
        if self._model is not None:
            return
        args = InitArgs(sample_input_path, is_video, output_dir, self.device)
        self._model = initModel(args)

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
            raise RuntimeError("cv2.imread 讀不到圖片（路徑或格式可能有問題）")

        h, w = img.shape[:2]
        orig_size = torch.tensor([w, h])[None].to(self.device)

        frame_resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_LINEAR)
        im_data = (
            (torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0)
            .unsqueeze(0)
            .to(self.device)
        )

        output = self._model(im_data, orig_size)
        labels, boxes, scores = output
        detect_frame, box_count = draw([img], labels, boxes, scores, 0.35)

        out_img_path = os.path.join(out_dir, "result.jpg")
        cv2.imwrite(out_img_path, detect_frame)

        b64_img = self._encode_b64_jpg(detect_frame)
        dt = time.time() - t0

        summary = []
        summary.append(f"輸入: {os.path.basename(image_path)}")
        summary.append(f"偵測到 {int(box_count)} 個目標")
        summary.append(f"耗時: {dt:.2f} 秒")
        summary.append(f"輸出圖片: {out_img_path}")

        return b64_img, "\n".join(summary)

    def _infer_video_blocking(self, video_path: str, out_dir: str):
        t0 = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("無法開啟影片（編碼或路徑可能有問題）")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps is None or fps <= 0:
            fps = 30.0

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_video_path = os.path.join(out_dir, "result.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("無法建立輸出影片（mp4v/路徑/權限問題）")

        orig_size = torch.tensor([w, h])[None].to(self.device)

        frames = 0
        detected_frames = 0
        preview_b64 = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(
                frame, (640, 640), interpolation=cv2.INTER_LINEAR
            )
            im_data = (
                (torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0)
                .unsqueeze(0)
                .to(self.device)
            )

            output = self._model(im_data, orig_size)
            labels, boxes, scores = output
            detect_frame, box_count = draw([frame], labels, boxes, scores, 0.35)

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
        summary.append(f"輸入: {os.path.basename(video_path)}")
        summary.append(f"總幀數: {frames}")
        summary.append(f"偵測到目標的幀數: {detected_frames}")
        summary.append(f"總耗時: {dt:.2f} 秒")
        summary.append(f"有效 FPS: {fps_eff:.2f}")
        summary.append(f"輸出影片: {out_video_path}")

        return preview_b64, "\n".join(summary)

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
                "detected_frames": 0,
                "fps_eff": 0.0,
                "last_msg": "",
                "out_video_path": "",
            }

        def worker():
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                with self._preview_meta_lock:
                    self._preview_meta["status"] = "error"
                    self._preview_meta["last_msg"] = "無法開啟影片"
                self._push_preview(("__error__", "無法開啟影片"))
                return

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps is None or fps <= 0:
                fps = 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            orig_size = torch.tensor([w, h])[None].to(self.device)

            writer = None
            out_video_path = ""
            if write_video:
                out_video_path = os.path.join(out_dir, "result.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h))
                if not writer.isOpened():
                    cap.release()
                    with self._preview_meta_lock:
                        self._preview_meta["status"] = "error"
                        self._preview_meta["last_msg"] = "無法建立輸出影片"
                    self._push_preview(("__error__", "無法建立輸出影片"))
                    return

            t0 = time.time()
            last_ui_t = time.time()
            frames = 0
            detected_frames = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_resized = cv2.resize(
                    frame, (640, 640), interpolation=cv2.INTER_LINEAR
                )
                im_data = (
                    (torch.from_numpy(frame_resized).permute(2, 0, 1).float() / 255.0)
                    .unsqueeze(0)
                    .to(self.device)
                )

                output = self._model(im_data, orig_size)
                labels, boxes, scores = output
                detect_frame, box_count = draw([frame], labels, boxes, scores, conf)

                if box_count > 0:
                    detected_frames += 1

                if writer is not None:
                    writer.write(detect_frame)

                frames += 1

                if frames % ui_stride == 0:
                    now = time.time()
                    dt = now - t0
                    fps_eff = (frames / dt) if dt > 0 else 0.0

                    b64 = self._encode_b64_jpg(detect_frame)
                    msg = f"frames={frames}, detected_frames={detected_frames}, fps≈{fps_eff:.1f}"
                    self._push_preview((b64, msg))

                    with self._preview_meta_lock:
                        self._preview_meta["frames"] = frames
                        self._preview_meta["detected_frames"] = detected_frames
                        self._preview_meta["fps_eff"] = fps_eff
                        self._preview_meta["last_msg"] = msg
                        self._preview_meta["out_video_path"] = out_video_path

                    last_ui_t = now

            cap.release()
            if writer is not None:
                writer.release()

            status = "stopped" if self._stop_event.is_set() else "done"
            with self._preview_meta_lock:
                self._preview_meta["status"] = status
                self._preview_meta["last_msg"] = (
                    "已中斷" if status == "stopped" else "完成"
                )
                self._preview_meta["out_video_path"] = out_video_path

            self._push_preview(("__done__", self._preview_meta["last_msg"]))

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
