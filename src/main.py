import os
import time
import asyncio
import flet as ft
from dataclasses import dataclass, field
from inference import DetectorEngine


@dataclass
class State:
    picked_files: list[ft.FilePickerFile] = field(default_factory=list)
    preview_task: asyncio.Task | None = None


state = State()
detector = DetectorEngine(r"weights\R50_att_C4_best.pth")


async def main(page: ft.Page):
    page.title = "無人機人員/異物偵測系統"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.scroll = ft.ScrollMode.AUTO
    page.window.width = 1000
    page.window.height = 800

    detect_status_text = ft.Text("狀態: 等待操作")

    pixel = (
        "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    )
    detect_image_control = ft.Image(
        src=pixel, width=800, fit=ft.BoxFit.CONTAIN, visible=False
    )

    preview_container = ft.Column(spacing=10, height=160, scroll=ft.ScrollMode.AUTO)
    upload_progress = ft.Column()
    detect_progress_bar = ft.ProgressBar(width=500, value=0, visible=False)
    detect_result_container = ft.Column()

    file_picker = ft.FilePicker()

    def is_image_file(name: str):
        ext = os.path.splitext(name)[1].lower()
        return ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    def is_video_file(name: str):
        ext = os.path.splitext(name)[1].lower()
        return ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    def b64_to_data_url(b64_str: str) -> str:
        return "data:image/jpeg;base64," + b64_str

    async def handle_files_pick(e):
        result = await file_picker.pick_files(allow_multiple=True)
        if not result:
            return
        state.picked_files = result
        preview_container.controls.clear()
        upload_progress.controls.clear()

        for f in result:
            file_ext = os.path.splitext(f.name)[1].lower()
            if is_image_file(f.name):
                thumbnail = ft.Image(
                    src=f.path,
                    width=100,
                    height=100,
                    fit=ft.BoxFit.COVER,
                    border_radius=8,
                )
            else:
                thumbnail = ft.Container(
                    content=ft.Icon(
                        ft.Icons.VIDEO_FILE, color=ft.Colors.GREY_400, size=40
                    ),
                    width=100,
                    height=100,
                    bgcolor=ft.Colors.GREY_200,
                    border_radius=8,
                    alignment=ft.MainAxisAlignment.CENTER,
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
                    ft.Text(file_ext, size=12, color=ft.Colors.GREY_500),
                ],
                expand=True,
            )

            preview_row = ft.Container(
                content=ft.Row([thumbnail, file_info], spacing=15),
                padding=10,
                border=ft.Border.all(1, ft.Colors.GREY_300),
                border_radius=10,
                bgcolor=ft.Colors.WHITE,
            )
            preview_container.controls.append(preview_row)

        page.update()

    async def inference_stream_loop():
        ui_interval = 0.10
        last_ui_ts = 0.0
        last_b64 = None
        last_msg = ""

        try:
            while True:
                await asyncio.sleep(0.01)

                item = detector.get_latest_preview()
                if item is not None:
                    b64, msg = item

                    if b64 == "__error__":
                        detect_status_text.value = f"❌ {msg}"
                        detect_status_text.color = "red"
                        detect_progress_bar.visible = False
                        detect_status_text.update()
                        detect_progress_bar.update()
                        return

                    if b64 == "__done__":
                        detect_status_text.value = f"✅ {msg}"
                        detect_status_text.color = "green"
                        detect_progress_bar.visible = False
                        detect_status_text.update()
                        detect_progress_bar.update()
                        return

                    last_b64 = b64
                    last_msg = msg

                meta = detector.get_preview_meta()
                if meta.get("status") in ("done", "stopped", "error"):
                    detect_status_text.value = f"狀態: {meta.get('last_msg', '')}"
                    if meta.get("status") == "done":
                        detect_status_text.color = "green"
                    elif meta.get("status") == "stopped":
                        detect_status_text.color = "orange"
                    else:
                        detect_status_text.color = "red"
                    detect_progress_bar.visible = False
                    detect_status_text.update()
                    detect_progress_bar.update()
                    return

                if last_b64 is None:
                    continue

                now = time.time()
                if now - last_ui_ts < ui_interval:
                    continue
                last_ui_ts = now

                detect_image_control.src = b64_to_data_url(last_b64)
                detect_image_control.visible = True
                detect_status_text.value = last_msg
                detect_status_text.color = "blue"

                detect_image_control.update()
                detect_status_text.update()

        except asyncio.CancelledError:
            return

    async def click_start_inference(e):
        if not state.picked_files:
            detect_status_text.value = "❌ 請先選擇圖片或影片"
            detect_status_text.color = "red"
            page.update()
            return

        file = state.picked_files[0]
        file_path = file.path

        detect_result_container.controls.clear()
        detect_image_control.visible = False
        detect_progress_bar.visible = True
        detect_progress_bar.value = None
        detect_status_text.value = "狀態: 推理中（即時顯示）"
        detect_status_text.color = "blue"
        page.update()

        if state.preview_task is not None and not state.preview_task.done():
            detector.stop_preview()
            state.preview_task.cancel()
            state.preview_task = None

        if is_image_file(file.name):
            try:
                b64_img, summary = detector.run_inference(file_path)
                detect_image_control.src = b64_to_data_url(b64_img)
                detect_image_control.visible = True
                detect_status_text.value = "✅ 狀態: 圖片推理完成"
                detect_status_text.color = "green"
                detect_progress_bar.visible = False
                detect_result_container.controls.append(
                    ft.Text(summary, selectable=True)
                )
                page.update()
            except Exception as err:
                detect_status_text.value = f"❌ 發生錯誤: {str(err)}"
                detect_status_text.color = "red"
                detect_progress_bar.visible = False
                page.update()
            return

        if is_video_file(file.name):
            try:
                detector.start_video_preview(
                    video_path=file_path,
                    conf=0.35,
                    ui_stride=3,
                    write_video=True,
                )
                detect_image_control.visible = True
                state.preview_task = asyncio.create_task(inference_stream_loop())
                page.update()
            except Exception as err:
                detect_status_text.value = f"❌ 發生錯誤: {str(err)}"
                detect_status_text.color = "red"
                detect_progress_bar.visible = False
                page.update()
            return

        detect_status_text.value = "❌ 不支援的檔案格式"
        detect_status_text.color = "red"
        detect_progress_bar.visible = False
        page.update()

    def click_stop(e):
        detector.stop_preview()
        if state.preview_task is not None and not state.preview_task.done():
            state.preview_task.cancel()
            state.preview_task = None
        detect_status_text.value = "⏹️ 已送出中斷"
        detect_status_text.color = "orange"
        detect_progress_bar.visible = False
        page.update()

    def click_clear(e):
        detector.stop_preview()
        if state.preview_task is not None and not state.preview_task.done():
            state.preview_task.cancel()
            state.preview_task = None

        state.picked_files = []
        preview_container.controls.clear()
        upload_progress.controls.clear()
        detect_result_container.controls.clear()
        detect_image_control.visible = False
        detect_progress_bar.visible = False
        detect_status_text.value = "狀態: 已清除"
        detect_status_text.color = None
        page.update()

    detect_tab = ft.Container(
        content=ft.Column(
            [
                ft.Text("檔案選擇", size=20, weight=ft.FontWeight.BOLD),
                ft.FilledButton(
                    "選擇圖片/影片",
                    icon=ft.Icons.UPLOAD_FILE,
                    on_click=handle_files_pick,
                ),
                preview_container,
                upload_progress,
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
                            on_click=click_stop,
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

    content_area = ft.Container(content=detect_tab, expand=True)

    def on_tab_change(e):
        content_area.content = detect_tab
        content_area.update()

    page.navigation_bar = ft.NavigationBar(
        on_change=on_tab_change,
        selected_index=0,
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.RADAR, label="模型偵測系統"),
            ft.NavigationBarDestination(icon=ft.Icons.WARNING, label="異常偵測系統"),
        ],
    )

    page.add(content_area)
    page.update()


if __name__ == "__main__":
    ft.run(main=main, upload_dir="uploads")
