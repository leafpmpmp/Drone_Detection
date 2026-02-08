import cv2
import threading
import time
import base64
import os
from inference import DetectorEngine

# Custom Threaded Reader to avoid "Reader too slow" and buffer issues
class ThreadedCamera:
    def __init__(self, src):
        self.capture = cv2.VideoCapture(src)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        
        # Check success
        if not self.capture.isOpened():
             self.status = False
        else:
             self.status = True
             
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.update, args=(), daemon=True)
        self.thread.start()

    def update(self):
        while self.running:
            if self.capture.isOpened():
                self.capture.grab() # grab latest
                success, frame = self.capture.retrieve()
                if success:
                    with self.lock:
                        self.frame = frame
                else:
                    # Reconnect logic could be added here
                    time.sleep(0.1)
            else:
                time.sleep(0.1)

    def get_frame(self):
        with self.lock:
             return self.frame

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        self.capture.release()
    
    def isOpened(self):
        return self.status

class StreamManager:
    def __init__(self, detector: DetectorEngine, update_callback):
        self.detector = detector
        self.update_callback = update_callback
        self.running = False
        self.thread = None
        self.cap = None
        self.url = ""
        self.confidence = 0.35 

    def set_confidence(self, conf):
        self.confidence = conf
    
    def start(self, url):
        self.stop()
        self.url = url
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
    
    def stop(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None

    def _stream_loop(self):
        # Force UDP for RTSP if requested or by default to fix latency/dropouts
        # This environment variable helps opencv use UDP
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

        # Use our ThreadedCamera which drops frames automatically
        self.cap = ThreadedCamera(self.url)
        
        if not self.cap.isOpened():
            print(f"Failed to open stream: {self.url}")
            self.running = False
            return

        while self.running:
            frame = self.cap.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            
            try:
                processed_frame, count = self.detector.infer_frame(frame, self.confidence)
                
                # Check if frame is valid
                if processed_frame is None or processed_frame.size == 0:
                    continue

                ok, buffer = cv2.imencode(".jpg", processed_frame)
                if ok:
                    b64_img = base64.b64encode(buffer).decode("utf-8")
                    self.update_callback(b64_img)
            except Exception as e:
                print(f"Stream Error: {e}")
                time.sleep(0.1)
        
        if self.cap:
            self.cap.release()

