import flet as ft
from dataclasses import dataclass, field

@dataclass
class State:
    file_picker: ft.FilePicker | None = None
    picked_files: list[ft.FilePickerFile] = field(default_factory=list)
state = State()

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

        # update progress bars
        prog_bars.clear()
        upload_progress.controls.clear()
        for f in files:
            prog = ft.ProgressRing(value=0, bgcolor="#eeeeee", width=20, height=20)
            prog_bars[f.name] = prog
            upload_progress.controls.append(ft.Row([prog, ft.Text(f.name)]))


    def dummy_click(e):
        print(f"Button clicked: {e.control.text if hasattr(e.control, 'text') else 'Unknown'}")

    def dummy_change(e):
        print(f"Value changed: {e.control.value}")

    # --- UI Components ---

    # File Pickers
    detect_file_picker = ft.FilePicker()
    anomaly_file_picker = ft.FilePicker()

    # --- Tab 1: Model Detection ---
    
    upload_list = ft.Column()
    
    detect_status_text = ft.Text("狀態: 等待操作")
    # Placeholder 1x1 transparent gif
    pixel = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
    detect_image_control = ft.Image(src=pixel, width=640, fit=ft.BoxFit.CONTAIN, visible=False) 
    detect_progress_bar = ft.ProgressBar(width=400, value=0)
    detect_result_container = ft.Column()

    detect_tab = ft.Container(
        content=ft.Column([
            ft.Text("檔案選擇", size=20, weight=ft.FontWeight.BOLD),
            ft.FilledButton("選擇圖片/影片", icon=ft.Icons.UPLOAD_FILE, on_click=handle_files_pick),
            upload_list,
            upload_progress := ft.Column(),
            ft.Row([
                ft.FilledButton("開始推理", on_click=dummy_click, style=ft.ButtonStyle(bgcolor=ft.Colors.BLUE, color=ft.Colors.WHITE)),
                ft.FilledButton("中斷", on_click=dummy_click, style=ft.ButtonStyle(bgcolor=ft.Colors.RED, color=ft.Colors.WHITE)),
                ft.FilledButton("清理暫存", on_click=dummy_click),
            ]),
            detect_status_text,
            detect_progress_bar,
            detect_image_control,
            detect_result_container
        ], scroll=ft.ScrollMode.AUTO),
        padding=20
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
            content_area.content = detect_tab #anomaly_tab
        content_area.update()
    
    rail = ft.NavigationBar(
        on_change=on_tab_change,
        selected_index=0,
        destinations=[
            ft.NavigationBarDestination(icon=ft.Icons.RADAR, label="模型偵測系統"),
            ft.NavigationBarDestination(icon=ft.Icons.WARNING, label="異常偵測系統"),
        ]
    )
    
    # Page layout
    page.add(content_area)
    page.navigation_bar = rail
    page.update()

if __name__ == "__main__":
    print("Starting Flet app (Visual Only)...")
    ft.run(main=main, upload_dir="uploads")

