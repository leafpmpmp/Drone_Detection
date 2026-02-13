import os
import time
import json
import asyncio
import cv2
import base64
import shutil
import zipfile
from pathlib import Path
import flet as ft
import flet_video as ftv
from dataclasses import dataclass, field
from inference import DetectorEngine
from stream import StreamManager
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add the directory containing main.py (src/) to the DLL search path and system PATH.
# This ensures that external DLLs like openh264 placed in this folder are found by OpenCV.
if os.name == "nt":
    os.environ["PATH"] = BASE_DIR + os.pathsep + os.environ["PATH"]
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(BASE_DIR)
        except Exception:
            pass

@dataclass
class State:
    picked_files: list[ft.FilePickerFile] = field(default_factory=list)
    processed_results: list[dict] = field(default_factory=list)
    preview_task: asyncio.Task | None = None
    language: str = "zh"
    real_time_render: bool = True
    confidence: float = 0.35
    use_custom_path: bool = False
    custom_output_path: str = os.path.abspath("outputFile")
    theme_mode: str = "system"
    lang_data: dict = field(default_factory=dict)

state = State()
detector = DetectorEngine(os.path.join("weights", "R50_att_C4_best.pth"))


def load_language(lang_code="zh"):
    path = os.path.abspath(os.path.join(BASE_DIR, "lang", f"{lang_code}.json"))

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

state.lang_data = load_language(state.language)


async def main(page: ft.Page):
    detector.set_language(state.lang_data)
    page.title = state.lang_data.get("title", "無人機人員/異物偵測系統")
    
    # Set initial theme
    if state.theme_mode == "system":
        page.theme_mode = ft.ThemeMode.SYSTEM
    elif state.theme_mode == "dark":
        page.theme_mode = ft.ThemeMode.DARK
    else:
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

    preview_list = ft.Column(spacing=10, scroll=ft.ScrollMode.AUTO)
    preview_placeholder = ft.Container(
        content=ft.Text(state.lang_data.get("no_file_selected", "尚未選擇檔案"), color=ft.Colors.GREY_400),
        alignment=ft.Alignment(0, 0),
    )

    preview_container = ft.Container(
        content=preview_placeholder,
        height=160,
        border=ft.Border.all(1, ft.Colors.GREY_300),
        border_radius=8,
        padding=10,
    )

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
    folder_picker = ft.FilePicker()
    # ---------------------------------------

    # --- Export UI Setup ---
    export_ui_row = ft.Row(visible=False, alignment=ft.MainAxisAlignment.CENTER)

    def click_export(e):
        # Placeholder for export action
        detect_status_text.value = state.lang_data.get("export_not_implemented", "Export logic to be implemented")
        detect_status_text.update()

    btn_export = ft.FilledButton(
        content=ft.Text(state.lang_data.get("btn_export", "Export Results")),
        on_click=click_export,
        icon=ft.Icons.DOWNLOAD,
        style=ft.ButtonStyle(bgcolor=ft.Colors.GREEN_700, color=ft.Colors.WHITE)
    )
    export_ui_row.controls.append(btn_export)
    # --- End Export UI Setup ---

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
        label=state.lang_data.get("stream_url_label", "RTSP/RTMP URL"), 
        hint_text=state.lang_data.get("stream_url_hint", "e.g. rtsp://192.168.1.100:8554/cam"),
        width=300,
        border_color=ft.Colors.OUTLINE,
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
            
            detect_status_text.value = f"{state.lang_data.get('status_streaming', 'Streaming')}: {url}"
            detect_status_text.color = "blue"
            detect_status_text.update()
        else:
            stream_manager.stop()
            stream_running[0] = False
            btn_stream_read.content.value = state.lang_data.get("read_stream", "Read Stream")
            btn_stream_read.style = None
            detect_status_text.value = state.lang_data.get("status_stream_stopped", "Stream Stopped")
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
        preview_list.controls.clear()
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
                border=ft.Border.all(1, ft.Colors.OUTLINE),
                border_radius=10,
                # Remove hardcoded white background so it adapts to dark mode
                # bgcolor=ft.Colors.WHITE, 
            )
            preview_list.controls.append(preview_row)

        preview_container.content = preview_list
        page.update()

    async def inference_stream_loop(progress_prefix=""):
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
                        detect_status_text.value = f"✗ {msg}"
                        detect_status_text.color = "red"
                        detect_progress_row.visible = False
                        detect_status_text.update()
                        detect_progress_row.update()
                        return

                    if b64 == "__done__":
                        # Now msg contains full summary
                        detect_status_text.value = state.lang_data.get("status_finished", "✓ Finished")
                        detect_status_text.color = "green"
                        detect_progress_row.visible = False
                        detect_status_text.update()
                        detect_progress_row.update()
                        return msg

                    last_b64 = b64
                    last_msg = msg

                meta = detector.get_preview_meta()

                # Update progress bar
                current_frames = meta.get("frames", 0)
                total_frames = meta.get("total_frames", 0)
                if total_frames > 0:
                    val = current_frames / total_frames
                    detect_progress_bar.value = val
                    
                    # Update progress text but keep status text clean with our prefix
                    detect_progress_text.value = f"{int(val * 100)}% ({last_msg})" 
                    detect_progress_row.visible = True
                    detect_progress_row.update()
                    
                    # Ensure status text isn't overwritten by raw frame stats
                    if progress_prefix:
                        detect_status_text.value = progress_prefix
                        detect_status_text.update()

                if meta.get("status") in ("done", "stopped", "error"):
                     # If we missed the __done__ message in the queue but status is done/stopped,
                     # we should still return the summary from meta to ensure result is shown.
                     detect_progress_row.visible = False
                     detect_progress_row.update()
                     
                     if meta.get("status") == "done":
                         detect_status_text.value = state.lang_data.get("status_finished", "✓ Finished")
                         detect_status_text.color = "green"
                         detect_status_text.update()
                         return meta.get("last_msg", "")
                     elif meta.get("status") == "stopped":
                         return meta.get("last_msg", "")
                     else:
                         return None

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
                
                # We moved status text update to progress section to avoid flashing

        except asyncio.CancelledError:
            return

    async def click_start_inference(e):
        if not state.picked_files:
            detect_status_text.value = state.lang_data.get(
                "error_no_file", "✗ 請先選擇圖片或影片"
            )
            detect_status_text.color = "red"
            page.update()
            return        
        # --- Handle Custom Output Path ---
        target_root = "outputFile"
        if state.use_custom_path:
            path_val = state.custom_output_path
            if not path_val or not os.path.exists(path_val):
                 # 2. Show warning if path invalid
                 detect_status_text.value = state.lang_data.get(
                     "error_invalid_path", "✗ Invalid custom output path"
                 )
                 detect_status_text.color = "red"
                 detect_status_text.update()
                 return
            target_root = os.path.abspath(path_val)
        
        try:
             detector.set_output_root(target_root)
        except Exception as err:
             detect_status_text.value = f"✗ Error setting output path: {err}"
             detect_status_text.color = "red"
             detect_status_text.update()
             return
        # --------------------------------
        detect_result_container.controls.clear()
        detect_image_control.visible = False
        detect_progress_row.visible = True
        detect_progress_bar.value = None
        detect_progress_text.value = "0%"

        # Reset Results and Hide Export
        state.processed_results.clear()
        export_ui_row.visible = False
        export_ui_row.update()
        
        status_template = state.lang_data.get("status_inferring", "狀態: 推理中（即時顯示）")
        detect_status_text.value = status_template
        detect_status_text.color = "blue"
        page.update()

        # Stop existing preview/task
        if state.preview_task is not None and not state.preview_task.done():
            if state.preview_task != asyncio.current_task():
                detector.stop_preview()
                state.preview_task.cancel()
                try:
                    await state.preview_task
                except asyncio.CancelledError:
                    pass
                state.preview_task = None

        # Set current task as preview task for cancellation support
        state.preview_task = asyncio.current_task()

        total_files = len(state.picked_files)
        try:
            for i, file in enumerate(state.picked_files):
                file_path = file.path
                
                # Update status
                msg = f"{status_template} [{i+1}/{total_files}] {file.name}"
                detect_status_text.value = msg
                detect_status_text.update()
                
                detect_result_container.controls.append(ft.Text(f"--- File: {file.name} ---", weight=ft.FontWeight.BOLD))
                
                if is_image_file(file.name):
                    try:
                        # Assuming detector.run_inference returns (b64, summary, out_path)
                        b64_img, summary, out_img_path = detector.run_inference(file_path)
                        
                        # Update the main preview image
                        data_url = b64_to_data_url(b64_img)
                        detect_image_control.src = data_url
                        detect_image_control.src_base64 = None
                        detect_image_control.visible = True
                        detect_image_control.update()
                        
                        # Append image result to the results list
                        detect_result_container.controls.append(
                            ft.Image(
                                src=data_url,
                                width=600,
                                fit=ft.BoxFit.CONTAIN,
                                border_radius=8
                            )
                        )

                        detect_result_container.controls.append(
                            ft.Text(summary, selectable=True)
                        )
                        detect_result_container.update()

                        state.processed_results.append({
                            "original": file_path,
                            "output": out_img_path,
                            "type": "image"
                        })
                    except Exception as err:
                        import traceback
                        traceback.print_exc()
                        detect_status_text.value = state.lang_data.get("error_generic", "✗ 發生錯誤: {err}").format(err=str(err))
                        detect_status_text.color = "red"
                        detect_status_text.update()
                        continue

                elif is_video_file(file.name):
                    try:
                        detect_progress_bar.value = 0
                        detect_progress_text.value = "0%"
                        detect_progress_row.visible = True
                        detect_progress_row.update()

                        detector.start_video_preview(
                            video_path=file_path,
                            conf=state.confidence,
                            ui_stride=3,
                            write_video=True,
                        )
                        if state.real_time_render:
                            detect_image_control.visible = True
                        
                        summary_text = await inference_stream_loop(progress_prefix=msg)
                        
                        meta = detector.get_preview_meta()
                        out_video_path = meta.get("out_video_path", "")
                        
                        if out_video_path and os.path.exists(out_video_path):
                             detect_folder = os.path.dirname(out_video_path)
                             log_path = os.path.join(detect_folder, "detections.log")
                             
                             # Convert to absolute path first, then to file URI for reliable playback
                             try:
                                abs_video_path = os.path.abspath(out_video_path)
                                video_uri = Path(abs_video_path).as_uri()
                             except Exception as e:
                                print(f"URI conversion failed: {e}")
                                video_uri = out_video_path

                             # Add Video Player
                             video_w = meta.get("width", 640)
                             video_h = meta.get("height", 480)
                             aspect = video_w / video_h if video_h > 0 else 16/9

                             detect_result_container.controls.append(
                                 ft.Text(f"--- Video Result: {file.name} ---", weight=ft.FontWeight.BOLD)
                             )
                             detect_result_container.controls.append(
                                 ftv.Video(
                                     playlist=[ftv.VideoMedia(video_uri)],
                                     playlist_mode=ftv.PlaylistMode.LOOP,
                                     aspect_ratio=aspect,
                                     autoplay=False, 
                                     filter_quality=ft.FilterQuality.HIGH,
                                     muted=True,
                                 )
                             )

                             state.processed_results.append({
                                 "original": file_path,
                                 "output": out_video_path,
                                 "log": log_path, 
                                 "type": "video"
                             })

                        if summary_text:
                            detect_result_container.controls.append(
                                ft.Text(summary_text, selectable=True)
                            )
                        detect_result_container.update()

                    except Exception as err:
                        import traceback
                        traceback.print_exc()
                        detect_status_text.value = state.lang_data.get("error_generic", "✗ 發生錯誤: {err}").format(err=str(err))
                        detect_status_text.color = "red"
                        detect_status_text.update()
                else:
                    msg = state.lang_data.get(
                        "error_unsupported_format", "✗ Unsupported format: {file}"
                    ).format(file=file.name)
                    detect_result_container.controls.append(ft.Text(msg, color="orange"))
                    detect_result_container.update()

            detect_status_text.value = state.lang_data.get("status_batch_completed", "✓ 所有檔案處理完成")
            detect_status_text.color = "green"
            detect_progress_row.visible = False
            
            if state.processed_results:
                export_ui_row.visible = True
                export_ui_row.update()

            page.update()

        except asyncio.CancelledError:
             detect_status_text.value = state.lang_data.get("interrupted_msg", "⚠ 已送出中斷")
             detect_status_text.color = "orange"
             detect_progress_row.visible = False
             
             if state.processed_results:
                export_ui_row.visible = True
                export_ui_row.update()
                
             page.update()

    async def handle_folder_pick(e):
        result = await folder_picker.get_directory_path()
        if not result:
            return
        state.custom_output_path = result
        txt_custom_path.value = result
    
    def click_stop(e):
        detector.stop_preview()
        if state.preview_task is not None and not state.preview_task.done():
            state.preview_task.cancel()
            state.preview_task = None
        detect_status_text.value = state.lang_data.get(
            "interrupted_msg", "⚠ 已送出中斷"
        )
        detect_status_text.color = "orange"
        detect_progress_row.visible = False
        page.update()

    def click_clear(e):
        detector.stop_preview()
        if state.preview_task is not None and not state.preview_task.done():
            state.preview_task.cancel()
            state.preview_task = None
        
        # Clean up output directory
        if os.path.exists(detector.output_root):
            try:
                shutil.rmtree(detector.output_root)
                os.makedirs(detector.output_root, exist_ok=True)
            except Exception as err:
                print(f"Error cleaning output directory: {err}")

        state.processed_results.clear()
        state.picked_files = []
        
        # Reset preview
        preview_list.controls.clear()
        preview_container.content = preview_placeholder
        
        upload_progress.controls.clear()
        detect_result_container.controls.clear()
        detect_image_control.visible = False
        detect_progress_row.visible = False
        
        export_ui_row.visible = False
        export_ui_row.update()
        
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
        border_color=ft.Colors.OUTLINE,
    )
    lang_dropdown.on_select = on_lang_select

    # --- Theme selection ---
    theme_select_label = ft.Text(state.lang_data.get("theme_select", "Theme Mode"))
    
    def on_theme_change(e):
        val = e.control.value
        state.theme_mode = val
        if val == "system":
            page.theme_mode = ft.ThemeMode.SYSTEM
        elif val == "dark":
            page.theme_mode = ft.ThemeMode.DARK
        else:
            page.theme_mode = ft.ThemeMode.LIGHT
        
        page.update()

    theme_dropdown = ft.Dropdown(
        value=state.theme_mode,
        options=[
            ft.DropdownOption(key="light", text=state.lang_data.get("theme_light", "Light")),
            ft.DropdownOption(key="dark", text=state.lang_data.get("theme_dark", "Dark")),
            ft.DropdownOption(key="system", text=state.lang_data.get("theme_system", "System")),
        ],
        width=200,
        border_color=ft.Colors.OUTLINE,
    )
    theme_dropdown.on_select = on_theme_change

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

    # --- Custom Output Path Controls ---
    def on_path_toggle(e):
        state.use_custom_path = e.control.value
        txt_custom_path.disabled = not state.use_custom_path
        btn_select_folder.disabled = not state.use_custom_path
        settings_tab.update()

    def on_path_change(e):
        # 3. Only update state, don't set detector root immediately
        state.custom_output_path = e.control.value

    switch_custom_path = ft.Switch(
        label=state.lang_data.get("use_custom_path", "自訂輸出路徑"),
        value=state.use_custom_path,
        on_change=on_path_toggle
    )
    
    txt_custom_path = ft.TextField(
        value=state.custom_output_path,
        disabled=not state.use_custom_path,
        expand=True,
        on_change=on_path_change,
        hint_text="Output Path",
        border_color=ft.Colors.OUTLINE,
    )
    
    btn_select_folder = ft.IconButton(
        icon=ft.Icons.FOLDER_OPEN,
        disabled=not state.use_custom_path,
        on_click=handle_folder_pick
    )
    
    row_custom_path = ft.Row([txt_custom_path, btn_select_folder])
    # -----------------------------------

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
        btn_export.content.value = state.lang_data.get("btn_export", "Export Results")
        
        stream_url_input.label = state.lang_data.get("stream_url_label", "RTSP/RTMP URL")
        stream_url_input.hint_text = state.lang_data.get("stream_url_hint", "e.g. rtsp://192.168.1.100:8554/cam")

        preview_placeholder.content.value = state.lang_data.get("no_file_selected", "尚未選擇檔案")

        settings_title.value = state.lang_data.get("settings_title", "設定")
        lang_select_label.value = state.lang_data.get("language_select", "語言選擇")
        theme_select_label.value = state.lang_data.get("theme_select", "Theme Mode")
        
        # Update theme dropdown options
        theme_dropdown.options = [
            ft.DropdownOption(key="light", text=state.lang_data.get("theme_light", "Light")),
            ft.DropdownOption(key="dark", text=state.lang_data.get("theme_dark", "Dark")),
            ft.DropdownOption(key="system", text=state.lang_data.get("theme_system", "System")),
        ]
        
        real_time_switch.label = state.lang_data.get("real_time_render", "即時影像渲染")
        switch_custom_path.label = state.lang_data.get("use_custom_path", "自訂輸出路徑")
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
                theme_select_label,
                theme_dropdown,
                ft.Divider(),
                real_time_switch,
                ft.Divider(),
                conf_slider_label,
                conf_slider,
                ft.Divider(),
                switch_custom_path,
                row_custom_path,
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
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
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
                export_ui_row, # Added export UI at the end
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
    ft.run(main=main, assets_dir="assets", upload_dir="uploads")
