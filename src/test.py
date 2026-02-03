import argparse
import base64
import os
import time

import cv2
from ultralytics import RTDETR


class DetectorEngine:
    def __init__(self, model_path: str):
        print(f"正在載入模型: {model_path} ...")
        self.model = RTDETR(model_path)
        print("模型載入完成")

    def run_image(
        self,
        file_path: str,
        imgsz: int | None = None,
        device: str | None = None,
        half: bool = False,
    ):
        results = self.model.predict(
            source=file_path,
            save=False,
            imgsz=imgsz,
            device=device,
            half=half,
            verbose=False,
        )
        result = results[0]
        annotated_frame = result.plot()
        ok, buffer = cv2.imencode(".jpg", annotated_frame)
        if not ok:
            raise RuntimeError("圖片編碼失敗")
        b64_img = base64.b64encode(buffer).decode("utf-8")

        summary = []
        box_count = len(result.boxes)
        if box_count == 0:
            summary.append("未偵測到任何目標。")
        else:
            summary.append(f"偵測到 {box_count} 個目標:")
            class_counts = {}
            for cls in result.boxes.cls:
                name = self.model.names[int(cls)]
                class_counts[name] = class_counts.get(name, 0) + 1
            for name, count in class_counts.items():
                summary.append(f" - {name}: {count} 個")

        return b64_img, "\n".join(summary)

    def run_video(
        self,
        video_path: str,
        output_path: str | None = None,
        imgsz: int | None = None,
        device: str | None = None,
        half: bool = False,
        show: bool = False,
    ):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"無法開啟影片: {video_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_in = cap.get(cv2.CAP_PROP_FPS)
        if fps_in is None or fps_in <= 0:
            fps_in = 30.0

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps_in, (w, h))
            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"無法建立輸出影片: {output_path}")

        frames = 0
        t0 = time.perf_counter()

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = self.model.predict(
                source=frame,
                save=False,
                imgsz=imgsz,
                device=device,
                half=half,
                verbose=False,
            )
            result = results[0]
            annotated = result.plot()

            frames += 1
            dt = time.perf_counter() - t0
            fps_now = frames / dt if dt > 0 else 0.0

            cv2.putText(
                annotated,
                f"FPS: {fps_now:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            if writer:
                writer.write(annotated)

            if show:
                cv2.imshow("RTDETR Inference", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        if writer:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        return frames


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=str, default=r"weights\R50_att_C4_best.pth")
    p.add_argument("--image", type=str, default=None)
    p.add_argument("--video", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--half", action="store_true")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    if not args.image and not args.video:
        raise SystemExit("請提供 --image 或 --video")

    if not os.path.exists(args.weights):
        raise SystemExit(f"找不到權重檔: {args.weights}")

    if args.image and not os.path.exists(args.image):
        raise SystemExit(f"找不到圖片檔: {args.image}")

    if args.video and not os.path.exists(args.video):
        raise SystemExit(f"找不到影片檔: {args.video}")

    engine = DetectorEngine(args.weights)

    if args.image:
        b64_img, summary = engine.run_image(
            args.image, imgsz=args.imgsz, device=args.device, half=args.half
        )
        print("\n--- 📝 辨識結果報告 ---")
        print(summary)
        print("-----------------------")

        output_filename = "test_result.jpg"
        with open(output_filename, "wb") as f:
            f.write(base64.b64decode(b64_img))
        print(f"✅ 測試圖片已儲存至: {output_filename}")
        return

    frames = engine.run_video(
        args.video,
        output_path=args.out,
        imgsz=args.imgsz,
        device=args.device,
        half=args.half,
        show=args.show,
    )
    print(f"✅ 影片推理完成，共處理 {frames} 幀。")
    if args.out:
        print(f"✅ 輸出影片: {args.out}")


if __name__ == "__main__":
    main()
