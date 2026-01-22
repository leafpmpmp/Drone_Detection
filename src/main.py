import os
import flet as ft
from dataclasses import dataclass, field
from inference import DetectorEngine


@dataclass
class State:
    file_picker: ft.FilePicker | None = None
    picked_files: list[ft.FilePickerFile] = field(default_factory=list)


DEBUG = True
state = State()
detector = DetectorEngine(r"weights\rtdetr-l.pt")

# --- Main App ---


async def main(page: ft.Page):
    page.title = "無人機人員/異物偵測系統"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.scroll = ft.ScrollMode.AUTO
    page.window_width = 1000
    page.window_height = 800

    prog_bars: dict[str, ft.ProgressRing] = {}

    def on_upload_progress(e: ft.FilePickerUploadEvent):
        prog_bars[e.file_name].value = e.progress

    async def handle_files_pick(e: ft.Event[ft.Button]):
        state.file_picker = ft.FilePicker(on_upload=on_upload_progress)
        files = await state.file_picker.pick_files(allow_multiple=True)
        print("Picked files:", files)
        state.picked_files = files

        preview_container.controls.clear()

        # update progress bars
        prog_bars.clear()
        upload_progress.controls.clear()
        for f in files:
            prog = ft.ProgressRing(value=0, bgcolor="#eeeeee", width=20, height=20)
            prog_bars[f.name] = prog
            upload_progress.controls.append(ft.Row([prog, ft.Text(f.name)]))
            file_ext = os.path.splitext(f.name)[1].lower()
            is_image = file_ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
            if is_image:
                # 如果是圖片，直接讀取本地路徑顯示縮圖
                thumbnail = ft.Image(
                    src=f.path,
                    width=100,
                    height=100,
                    fit=ft.BoxFit.COVER,
                    border_radius=8,
                )
            else:
                # 如果是影片或其他檔案，顯示一個預設圖示
                thumbnail = ft.Container(
                    content=ft.Icon(
                        ft.Icons.VIDEO_FILE, color=ft.Colors.GREY_400, size=40
                    ),
                    width=100,
                    height=100,
                    bgcolor=ft.Colors.GREY_200,
                    border_radius=8,
                    alignment=ft.alignment.center,
                )
            file_info = ft.Column(
                [
                    ft.Text(
                        f.name,
                        weight=ft.FontWeight.BOLD,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                    ft.Text(
                        f.path,
                        size=12,
                        color=ft.Colors.GREY_500,
                        overflow=ft.TextOverflow.ELLIPSIS,
                    ),
                ],
                expand=True,
            )
            preview_row = ft.Container(
                content=ft.Row([thumbnail, file_info], spacing=15),
                padding=10,
                border=ft.border.all(1, ft.Colors.GREY_300),
                border_radius=10,
                bgcolor=ft.Colors.WHITE,
            )
            preview_container.controls.append(preview_row)

    def dummy_click(e):
        print(
            f"Button clicked: {e.control.text if hasattr(e.control, 'text') else 'Unknown'}"
        )

    def dummy_change(e):
        print(f"Value changed: {e.control.value}")

    async def click_start_inference(e):
        if not state.picked_files:
            detect_status_text.value = "❌ 錯誤: 請先選擇圖片或影片檔案！"
            detect_status_text.color = "red"
            page.update()
            return

        # 1. 介面進入讀取狀態
        detect_status_text.value = "狀態: 正在進行 AI 辨識..."
        detect_status_text.color = "blue"
        detect_progress_bar.visible = True
        detect_progress_bar.value = None  # 設定為 None 會顯示無限加載動畫

        # 清空舊結果
        detect_result_container.controls.clear()
        detect_image_control.visible = False
        page.update()

        try:
            # 2. 遍歷所有選擇的檔案進行辨識
            for file in state.picked_files:
                # 取得檔案路徑 (Flet Desktop 本地路徑)
                file_path = file.path

                # 呼叫 inference.py 的功能
                b64_img, text_result = detector.run_inference(file_path)

                # 3. 更新 UI 顯示結果
                # 設定圖片 src (必須加上 data URI 前綴)
                detect_image_control.src = f"data:image/jpeg;base64,{b64_img}"
                detect_image_control.visible = True

                # 加入文字結果
                detect_result_container.controls.append(
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text(
                                    f"【{file.name}】辨識結果:",
                                    weight=ft.FontWeight.BOLD,
                                    size=16,
                                ),
                                ft.Text(text_result, selectable=True),
                                ft.Divider(),
                            ]
                        ),
                        bgcolor=ft.Colors.GREY_50,
                        padding=10,
                        border_radius=5,
                    )
                )

            detect_status_text.value = "✅ 狀態: 辨識完成"
            detect_status_text.color = "green"

        except Exception as err:
            detect_status_text.value = f"❌ 發生錯誤: {str(err)}"
            detect_status_text.color = "red"
            print(f"Error details: {err}")

        finally:
            # 4. 恢復介面狀態
            detect_progress_bar.visible = False
            page.update()

    def click_clear(e):
        state.picked_files = []
        upload_progress.controls.clear()
        detect_result_container.controls.clear()
        detect_image_control.visible = False
        detect_status_text.value = "狀態: 已清除"
        page.update()

    # --- UI Components ---

    # File Pickers
    detect_file_picker = ft.FilePicker()
    anomaly_file_picker = ft.FilePicker()

    # --- Tab 1: Model Detection ---

    upload_list = ft.Column()
    preview_container = ft.Column(spacing=10, height=100, scroll=ft.ScrollMode.AUTO)

    detect_status_text = ft.Text("狀態: 等待操作")
    # Placeholder 1x1 transparent gif
    pixel = (
        "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    )
    detect_image_control = ft.Image(
        src=pixel, width=640, fit=ft.BoxFit.CONTAIN, visible=False
    )
    detect_progress_bar = ft.ProgressBar(width=400, value=0)
    detect_result_container = ft.Column()

    detect_tab = ft.Container(
        content=ft.Column(
            [
                ft.Text("檔案選擇", size=20, weight=ft.FontWeight.BOLD),
                ft.FilledButton(
                    "選擇圖片/影片",
                    icon=ft.Icons.UPLOAD_FILE,
                    on_click=handle_files_pick,
                ),
                upload_list,
                preview_container,
                upload_progress := ft.Column(),
                ft.Row(
                    [
                        ft.FilledButton(
                            "開始推理",
                            on_click=click_start_inference,
                            style=ft.ButtonStyle(
                                bgcolor=ft.Colors.BLUE, color=ft.Colors.WHITE
                            ),
                        ),
                        ft.FilledButton(
                            "中斷",
                            style=ft.ButtonStyle(
                                bgcolor=ft.Colors.RED, color=ft.Colors.WHITE
                            ),
                        ),
                        ft.FilledButton("清理暫存", on_click=click_clear),
                    ]
                ),
                detect_status_text,
                detect_progress_bar,
                detect_image_control,
                detect_result_container,
            ],
            scroll=ft.ScrollMode.AUTO,
        ),
        padding=20,
    )

    # # --- Tab 2: Anomaly Detection ---

    # anomaly_file_text = ft.Text("No file selected")

    # color_picker_target = ft.TextField(label="增強顏色 (HEX)", value="#FFCC99", width=150, on_change=dummy_change)
    # color_picker_downgrade = ft.TextField(label="削弱顏色 (HEX)", value="#808080", width=150, on_change=dummy_change)

    # rx_slider = ft.Slider(min=1, max=100, divisions=99, value=40, label="RX Threshold: {value}", on_change=dummy_change)
    # tol_slider = ft.Slider(min=1, max=100, divisions=99, value=50, label="Tolerance: {value}", on_change=dummy_change)
    # down_slider = ft.Slider(min=0.0, max=1.0, divisions=20, value=0.5, label="Downgrade Factor: {value}", on_change=dummy_change)

    # anomaly_status_text = ft.Text("狀態: 等待操作")
    # anomaly_image_control = ft.Image(src=pixel, width=640, fit=ft.BoxFit.CONTAIN, visible=False)
    # anomaly_progress_bar = ft.ProgressBar(width=400, value=0)

    # anomaly_tab = ft.Container(
    #     content=ft.Column([
    #         ft.Text("異常檢測參數設定", size=20, weight=ft.FontWeight.BOLD),
    #         ft.FilledButton("選擇影片", icon=ft.Icons.VIDEO_FILE, on_click=dummy_pick_files),
    #         anomaly_file_text,
    #         ft.Row([color_picker_target, color_picker_downgrade]),
    #         ft.Text("RX Detection 強度"), rx_slider,
    #         ft.Text("顏色差異容許度"), tol_slider,
    #         ft.Text("削弱顏色強度"), down_slider,
    #         ft.Row([
    #             ft.FilledButton("開始偵測", on_click=dummy_click, style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE, color=ft.Colors.WHITE)),
    #             ft.FilledButton("中斷", on_click=dummy_click, style=ft.ButtonStyle(bgcolor=ft.Colors.RED, color=ft.Colors.WHITE)),
    #         ]),
    #         anomaly_status_text,
    #         anomaly_progress_bar,
    #         anomaly_image_control
    #     ], scroll=ft.ScrollMode.AUTO),
    #     padding=20
    # )

    # --- Navigation Logic ---

    content_area = ft.Container(content=detect_tab, expand=True)

    def on_tab_change(e):
        idx = e.control.selected_index
        if idx == 0:
            content_area.content = detect_tab
        elif idx == 1:
            content_area.content = detect_tab  # anomaly_tab
        content_area.update()

    rail = ft.NavigationBar(
        on_change=on_tab_change,
        selected_index=0,
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.RADAR, label="模型偵測系統"),
            ft.NavigationBarDestination(icon=ft.Icons.WARNING, label="異常偵測系統"),
        ],
    )

    # Page layout
    page.add(content_area)
    page.navigation_bar = rail
    page.update()


if __name__ == "__main__":
    print("Starting Flet app (Visual Only)...")
    ft.run(main=main, upload_dir="uploads")
