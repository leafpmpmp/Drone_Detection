import os
import time
import json
import asyncio
import cv2
import base64
import flet as ft
from dataclasses import dataclass, field
from inference import DetectorEngine
from stream import StreamManager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@dataclass
class State:
    picked_files: list[ft.FilePickerFile] = field(default_factory=list)
    preview_task: asyncio.Task | None = None
    language: str = "zh"
    real_time_render: bool = True
    confidence: float = 0.35
    lang_data: dict = field(default_factory=dict)


state = State()
detector = DetectorEngine(r"weights\R50_att_C4_best.pth")


def load_language(lang_code="zh"):
    # Since we are in src/main.py, the lang folder is at src/lang/
    path = os.path.join(BASE_DIR, "lang", f"{lang_code}.json")
    
    path = os.path.abspath(path)
    if not os.path.exists(path):
        print("LANG PATH not found:", path)
        return {}

    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print("LANG PATH =", path)
        print(f"Error loading language ({lang_code}): {e}")
        return {}


# Initial load
state.lang_data = load_language(state.language)


async def main(page: ft.Page):
    detector.set_language(state.lang_data)
    page.title = state.lang_data.get("title", "無人機人員/異物偵測系統")
    page.theme_mode = ft.ThemeMode.LIGHT
    page.scroll = ft.ScrollMode.AUTO
    page.window.width = 1000
    page.window.height = 800

    detect_status_text = ft.Text(
        state.lang_data.get("status_waiting", "狀態: 等待操作")
    )

    pixel = (
        "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    )
    detect_image_control = ft.Image(
        src=pixel,
        width=800,
        height=600,
        gapless_playback=True,
        fit=ft.BoxFit.CONTAIN,
        visible=False,
    )

    preview_container = ft.Column(spacing=10, height=160, scroll=ft.ScrollMode.AUTO)
    upload_progress = ft.Column()
    detect_progress_bar = ft.ProgressBar(width=500, value=0)
    detect_progress_text = ft.Text("0%")
    detect_progress_row = ft.Row(
        [detect_progress_bar, detect_progress_text],
        visible=False,
        vertical_alignment=ft.CrossAxisAlignment.CENTER,
    )
    detect_result_container = ft.Column()

    file_picker = ft.FilePicker()

    # --- Stream Setup ---
    stream_queue = asyncio.Queue()
    stream_running = [False]
    main_loop = asyncio.get_running_loop()
    
    def stream_callback(b64):
        def _safe_update():
            # This runs inside the event loop, so it's safe to touch asyncio.Queue
            if stream_queue.qsize() > 1:
                try:
                    # Clear old frames to always show the newest
                    while not stream_queue.empty():
                        stream_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            stream_queue.put_nowait(b64)

        try:
            if main_loop and main_loop.is_running():
                main_loop.call_soon_threadsafe(_safe_update)
        except RuntimeError:
            pass
        
    stream_manager = StreamManager(detector, stream_callback)

    async def stream_display_loop():
        print("Starting stream display loop")
        # Throttle UI updates to 15 FPS to prevent freezing
        ui_interval = 1.0 / 15.0
        last_ui_ts = 0.0
        
        try:
            while stream_running[0]:
                try:
                    # Wait for next frame
                    b64 = await stream_queue.get()
                except asyncio.CancelledError:
                    break
                
                if not stream_manager.running:
                    continue
                
                now = time.time()
                if now - last_ui_ts < ui_interval:
                    continue
                last_ui_ts = now
                
                detect_image_control.src = b64_to_data_url(b64)
                detect_image_control.src_base64 = None
                detect_image_control.visible = True
                detect_image_control.update()
                
                # print(f"Updated frame at {now}")
                
        except asyncio.CancelledError:
            print("Stream display loop cancelled")
        except Exception as e:
            print(f"Error in display loop: {e}")

    stream_url_input = ft.TextField(
        label="RTSP/RTMP URL", 
        hint_text="e.g. rtsp://192.168.1.100:8554/cam",
        width=300
    )
    
    async def click_read_stream(e):
        if not stream_running[0]:
            url = stream_url_input.value
            if not url:
                detect_status_text.value = state.lang_data.get("error_no_url", "Please input URL")
                detect_status_text.color = "red"
                detect_status_text.update()
                return

            detector.stop_preview()
            detect_image_control.visible = True
            
            stream_manager.set_confidence(state.confidence)
            stream_manager.start(url)
            stream_running[0] = True
            
            btn_stream_read.content.value = state.lang_data.get("stop_stream", "Stop Stream")
            btn_stream_read.style = ft.ButtonStyle(bgcolor=ft.Colors.RED, color=ft.Colors.WHITE)
            
            if state.preview_task:
                state.preview_task.cancel()
            state.preview_task = asyncio.create_task(stream_display_loop())
            
            detect_status_text.value = f"Streaming: {url}"
            detect_status_text.color = "blue"
            detect_status_text.update()
        else:
            stream_manager.stop()
            stream_running[0] = False
            btn_stream_read.content.value = state.lang_data.get("read_stream", "Read Stream")
            btn_stream_read.style = None
            detect_status_text.value = "Stream Stopped"
            detect_status_text.update()
            
            if state.preview_task:
                state.preview_task.cancel()
                state.preview_task = None
        
        btn_stream_read.update()

    btn_stream_read = ft.FilledButton(
        content=ft.Text(state.lang_data.get("read_stream", "Read Stream")),
        on_click=click_read_stream
    )
    # --- End Stream Setup ---

    def is_image_file(name: str):
        ext = os.path.splitext(name)[1].lower()
        return ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

    def is_video_file(name: str):
        ext = os.path.splitext(name)[1].lower()
        return ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]

    def get_video_thumbnail(video_path: str):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None

            # Resize for thumbnail
            frame = cv2.resize(frame, (100, 100))
            _, buffer = cv2.imencode(".jpg", frame)
            b64 = base64.b64encode(buffer).decode("utf-8")
            return "data:image/jpeg;base64," + b64
        except Exception:
            return None

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
                # Using src=f.path for local files on desktop usually works if allowed.
                thumbnail = ft.Image(
                    src=f.path,
                    width=100,
                    height=100,
                    fit=ft.BoxFit.COVER,
                    border_radius=8,
                )
            elif is_video_file(f.name):
                # Try to get video thumbnail
                thumb_src = get_video_thumbnail(f.path)
                if thumb_src:
                    thumbnail = ft.Image(
                        src=thumb_src,
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
                        detect_progress_row.visible = False
                        detect_status_text.update()
                        detect_progress_row.update()
                        return

                    if b64 == "__done__":
                        detect_status_text.value = f"✅ {msg}"
                        detect_status_text.color = "green"
                        detect_progress_row.visible = False
                        detect_status_text.update()
                        detect_progress_row.update()
                        return

                    last_b64 = b64
                    last_msg = msg

                meta = detector.get_preview_meta()

                # Update progress bar
                current_frames = meta.get("frames", 0)
                total_frames = meta.get("total_frames", 0)
                if total_frames > 0:
                    val = current_frames / total_frames
                    detect_progress_bar.value = val
                    detect_progress_text.value = f"{int(val * 100)}%"
                    detect_progress_row.visible = True
                    detect_progress_row.update()

                if meta.get("status") in ("done", "stopped", "error"):
                    prefix = state.lang_data.get("status_prefix", "狀態: ")
                    detect_status_text.value = f"{prefix}{meta.get('last_msg', '')}"
                    if meta.get("status") == "done":
                        detect_status_text.color = "green"
                    elif meta.get("status") == "stopped":
                        detect_status_text.color = "orange"
                    else:
                        detect_status_text.color = "red"
                    detect_progress_row.visible = False
                    detect_status_text.update()
                    detect_progress_row.update()
                    return

                if last_b64 is None:
                    continue

                now = time.time()
                if now - last_ui_ts < ui_interval:
                    continue
                last_ui_ts = now

                # Use src_base64 for better performance and support in gapless_playback
                if state.real_time_render:
                    detect_image_control.src = b64_to_data_url(last_b64)
                    detect_image_control.src_base64 = None
                    detect_image_control.visible = True
                    detect_image_control.update()

                detect_status_text.value = last_msg
                detect_status_text.color = "blue"
                detect_status_text.update()

        except asyncio.CancelledError:
            return

    async def click_start_inference(e):
        if not state.picked_files:
            detect_status_text.value = state.lang_data.get(
                "error_no_file", "❌ 請先選擇圖片或影片"
            )
            detect_status_text.color = "red"
            page.update()
            return

        file = state.picked_files[0]
        file_path = file.path

        detect_result_container.controls.clear()
        detect_image_control.visible = False
        detect_progress_row.visible = True
        detect_progress_bar.value = None
        detect_progress_text.value = "0%"
        detect_status_text.value = state.lang_data.get(
            "status_inferring", "狀態: 推理中（即時顯示）"
        )
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
                detect_image_control.src_base64 = None
                detect_image_control.visible = True
                detect_status_text.value = state.lang_data.get(
                    "status_image_completed", "✅ 狀態: 圖片推理完成"
                )
                detect_status_text.color = "green"
                detect_progress_row.visible = False
                detect_result_container.controls.append(
                    ft.Text(summary, selectable=True)
                )
                page.update()
            except Exception as err:
                detect_status_text.value = state.lang_data.get(
                    "error_generic", "❌ 發生錯誤: {err}"
                ).format(err=str(err))
                detect_status_text.color = "red"
                detect_progress_row.visible = False
                page.update()
            return

        if is_video_file(file.name):
            try:
                detector.start_video_preview(
                    video_path=file_path,
                    conf=state.confidence,
                    ui_stride=3,
                    write_video=True,
                )
                if state.real_time_render:
                    detect_image_control.visible = True
                state.preview_task = asyncio.create_task(inference_stream_loop())
                page.update()
            except Exception as err:
                detect_status_text.value = state.lang_data.get(
                    "error_generic", "❌ 發生錯誤: {err}"
                ).format(err=str(err))
                detect_status_text.color = "red"
                detect_progress_row.visible = False
                page.update()
            return

        detect_status_text.value = state.lang_data.get(
            "error_unsupported_format", "❌ 不支援的檔案格式"
        )
        detect_status_text.color = "red"
        detect_progress_row.visible = False
        page.update()

    def click_stop(e):
        detector.stop_preview()
        if state.preview_task is not None and not state.preview_task.done():
            state.preview_task.cancel()
            state.preview_task = None
        detect_status_text.value = state.lang_data.get(
            "interrupted_msg", "⏹️ 已送出中斷"
        )
        detect_status_text.color = "orange"
        detect_progress_row.visible = False
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
        detect_progress_row.visible = False
        detect_status_text.value = state.lang_data.get("status_cleared", "狀態: 已清除")
        detect_status_text.color = None
        page.update()

    # --- Controls references for language update ---
    nav_model_dest = ft.NavigationBarDestination(
        icon=ft.Icons.RADAR, label=state.lang_data.get("model_system", "模型偵測系統")
    )
    nav_settings_dest = ft.NavigationBarDestination(
        icon=ft.Icons.SETTINGS, label=state.lang_data.get("settings_title", "設定")
    )

    file_selection_title = ft.Text(
        state.lang_data.get("file_selection", "檔案選擇"),
        size=20,
        weight=ft.FontWeight.BOLD,
    )
    btn_choose_file = ft.FilledButton(
        content=ft.Text(state.lang_data.get("choose_file", "選擇圖片/影片")),
        icon=ft.Icons.UPLOAD_FILE,
        on_click=handle_files_pick,
    )
    btn_start_infer = ft.FilledButton(
        content=ft.Text(state.lang_data.get("infer_button", "開始推理")),
        on_click=click_start_inference,
        style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE, color=ft.Colors.WHITE),
    )
    btn_stop_infer = ft.FilledButton(
        content=ft.Text(state.lang_data.get("interrupt_inference", "中斷")),
        on_click=click_stop,
        style=ft.ButtonStyle(bgcolor=ft.Colors.RED, color=ft.Colors.WHITE),
    )
    btn_cleanup = ft.FilledButton(
        content=ft.Text(state.lang_data.get("cleanup_button", "清理暫存")), on_click=click_clear
    )

    # Settings controls
    settings_title = ft.Text(
        state.lang_data.get("settings_title", "設定"),
        size=20,
        weight=ft.FontWeight.BOLD,
    )
    lang_select_label = ft.Text(state.lang_data.get("language_select", "語言選擇"))

    def on_lang_select(e):
        state.language = e.control.value
        print(f"Language changed to: {state.language}")
        state.lang_data = load_language(state.language)
        detector.set_language(state.lang_data)
        print("Loaded lang data keys:", list(state.lang_data.keys())[:10], "...")
        update_ui_text()
        settings_tab.update()
        detect_tab.update()
        page.update()

    lang_dropdown = ft.Dropdown(
        value=state.language,
        options=[
            ft.DropdownOption(key="zh", text="繁體中文"),
            ft.DropdownOption(key="en", text="English"),
        ],
        width=200,
    )
    lang_dropdown.on_select = on_lang_select

    def on_real_time_change(e):
        state.real_time_render = e.control.value
        page.update()

    real_time_switch = ft.Switch(
        label=state.lang_data.get("real_time_render", "即時影像渲染"),
        value=state.real_time_render,
    )
    real_time_switch.on_change = on_real_time_change

    conf_slider_label = ft.Text(
        f"{state.lang_data.get('conf_threshold', '信心度閥值')}: {state.confidence:.2f}"
    )

    def on_conf_change(e):
        state.confidence = e.control.value
        conf_slider_label.value = f"{state.lang_data.get('conf_threshold', '信心度閥值')}: {state.confidence:.2f}"
        if stream_running[0]:
            stream_manager.set_confidence(state.confidence)
        page.update()

    conf_slider = ft.Slider(
        min=0.1, max=1.0, divisions=18, value=state.confidence, label="{value}"
    )
    conf_slider.on_change = on_conf_change

    def update_ui_text():
        page.title = state.lang_data.get("title", "無人機人員/異物偵測系統")

        nav_model_dest.label = state.lang_data.get("model_system", "模型偵測系統")
        nav_settings_dest.label = state.lang_data.get("settings_title", "設定")

        file_selection_title.value = state.lang_data.get("file_selection", "檔案選擇")
        btn_choose_file.content.value = state.lang_data.get("choose_file", "選擇圖片/影片")
        btn_start_infer.content.value = state.lang_data.get("infer_button", "開始推理")
        btn_stop_infer.content.value = state.lang_data.get("interrupt_inference", "中斷")
        btn_cleanup.content.value = state.lang_data.get("cleanup_button", "清理暫存")
        btn_stream_read.content.value = state.lang_data.get("read_stream", "Read Stream") if not stream_running[0] else state.lang_data.get("stop_stream", "Stop Stream")

        settings_title.value = state.lang_data.get("settings_title", "設定")
        lang_select_label.value = state.lang_data.get("language_select", "語言選擇")
        real_time_switch.label = state.lang_data.get("real_time_render", "即時影像渲染")
        conf_slider_label.value = f"{state.lang_data.get('conf_threshold', '信心度閥值')}: {state.confidence:.2f}"
        detect_status_text.value = state.lang_data.get(
            "status_waiting", "狀態: 等待操作"
        )

        detect_tab.update()
        settings_tab.update()
        page.navigation_bar.update()

    settings_tab = ft.Container(
        content=ft.Column(
            [
                settings_title,
                ft.Divider(),
                lang_select_label,
                lang_dropdown,
                ft.Divider(),
                real_time_switch,
                ft.Divider(),
                conf_slider_label,
                conf_slider,
            ],
            scroll=ft.ScrollMode.AUTO,
        ),
        padding=20,
    )

    detect_tab = ft.Container(
        content=ft.Column(
            [
                file_selection_title,
                ft.Row(
                    [
                        btn_choose_file,
                        stream_url_input,
                        btn_stream_read,
                    ],
                    vertical_alignment=ft.CrossAxisAlignment.START,
                ),
                preview_container,
                upload_progress,
                ft.Row(
                    [
                        btn_start_infer,
                        btn_stop_infer,
                        btn_cleanup,
                    ]
                ),
                detect_status_text,
                detect_progress_row,
                detect_image_control,
                detect_result_container,
            ],
            scroll=ft.ScrollMode.AUTO,
        ),
        padding=20,
    )

    content_area = ft.Container(content=detect_tab, expand=True)

    def on_tab_change(e):
        idx = e.control.selected_index
        if idx == 0:
            content_area.content = detect_tab
        elif idx == 1:
            content_area.content = settings_tab
        content_area.update()

    page.navigation_bar = ft.NavigationBar(
        on_change=on_tab_change,
        selected_index=0,
        destinations=[
            nav_model_dest,
            nav_settings_dest,
        ],
    )

    page.add(content_area)
    page.update()


if __name__ == "__main__":
    ft.run(main=main, upload_dir="uploads")
