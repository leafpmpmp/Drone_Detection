# inference.py
import cv2
import base64
import numpy as np
from ultralytics import RTDETR


class DetectorEngine:
    def __init__(self, model_path: str):
        print(f"正在載入模型: {model_path} ...")
        self.model = RTDETR(model_path)
        print("模型載入完成")

    def run_inference(self, file_path: str):
        """
        執行推理並回傳: (base64_image, detection_summary_text)
        """
        # 1. 使用 Ultralytics 進行預測
        results = self.model.predict(source=file_path, save=False)
        result = results[0]  # 取第一張結果

        # 2. 繪製結果圖 (Plot)
        # return numpy array (BGR format)
        annotated_frame = result.plot()

        # 3. 轉換圖片為 Base64 (供 Flet 顯示)
        # Flet 的 Image src_base64 需要這種格式
        _, buffer = cv2.imencode(".jpg", annotated_frame)
        b64_img = base64.b64encode(buffer).decode("utf-8")

        # 4. 整理辨識文字資訊
        summary = []
        box_count = len(result.boxes)
        if box_count == 0:
            summary.append("未偵測到任何目標。")
        else:
            summary.append(f"偵測到 {box_count} 個目標:")
            # 統計類別 (例如: drone: 2, person: 1)
            class_counts = {}
            for cls in result.boxes.cls:
                name = self.model.names[int(cls)]
                class_counts[name] = class_counts.get(name, 0) + 1

            for name, count in class_counts.items():
                summary.append(f" - {name}: {count} 個")

        return b64_img, "\n".join(summary)


if __name__ == "__main__":
    import sys
    import os

    # 1. 檢查是否有輸入圖片路徑
    if len(sys.argv) < 2:
        print("❌ 使用方式錯誤")
        print("請輸入: uv run python src/inference.py <圖片路徑>")
        # 範例: uv run python src/inference.py test.jpg
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"❌ 找不到檔案: {image_path}")
        sys.exit(1)

    # 2. 初始化引擎 (預設使用 rtdetr-l.pt)
    print("🚀 初始化引擎中...")
    engine = DetectorEngine("rtdetr-l.pt")

    # 3. 執行推理
    print(f"🔍 正在辨識: {image_path}")
    try:
        b64_img, summary = engine.run_inference(image_path)

        print("\n--- 📝 辨識結果報告 ---")
        print(summary)
        print("-----------------------")

        # 4. (選用) 將 Base64 轉回圖片存檔，確認繪圖功能是否正常
        output_filename = "test_result.jpg"
        with open(output_filename, "wb") as f:
            f.write(base64.b64decode(b64_img))
        print(f"✅ 測試圖片已儲存至: {output_filename}")

    except Exception as e:
        print(f"❌ 發生錯誤: {e}")
