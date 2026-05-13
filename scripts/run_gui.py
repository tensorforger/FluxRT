import argparse
import json
import os
import platform
import threading
import time

import cv2
import numpy as np
from PIL import Image

from PySide6.QtCore import Qt, QTimer, Signal, QObject, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QTextEdit,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QVBoxLayout,
    QGridLayout,
    QSizePolicy,
)

from fluxrt import StreamProcessor
from fluxrt.utils import crop_maximal_rectangle

POLL_MS = 40
MAX_CAM_INDEX = 8
if platform.system() == "Windows":
    CAM_BACKEND = cv2.CAP_DSHOW
    CAM_BACKEND_FALLBACK = cv2.CAP_MSMF
else:
    CAM_BACKEND = cv2.CAP_V4L2
    CAM_BACKEND_FALLBACK = None
DEFAULT_CONFIG = "configs/config_with_reference.json"

# ── colour tokens ────────────────────────────────────────────────────────────
BG = "#1e1e1e"
CTRL_BG = "#252526"
ENTRY_BG = "#3c3c3c"
FG = "#cccccc"
DIM_FG = "#666666"
BTN_BG = "#3a3a3a"
BTN_HOVER = "#4e4e4e"
ERR_FG = "#f44747"
VIDEO_BG = "#111111"
STATUS_BG = "#007acc"
STATUS_FG = "#ffffff"
ACCENT = "#007acc"
BORDER = "#4a4a4a"
SEP = "#2d2d2d"

_sp_lock = threading.Lock()


def log(msg: str) -> None:
    print(f"[FluxRT] {msg}", flush=True)


def enumerate_cameras() -> list[tuple[int, str]]:
    found = []
    for i in range(MAX_CAM_INDEX):
        cap = cv2.VideoCapture(i, CAM_BACKEND)
        if not cap.isOpened() and CAM_BACKEND_FALLBACK is not None:
            cap = cv2.VideoCapture(i, CAM_BACKEND_FALLBACK)
        if cap.isOpened():
            found.append((i, f"Camera {i}"))
            cap.release()
    return found


# ── stylesheet ────────────────────────────────────────────────────────────────
STYLESHEET = f"""
* {{
    font-family: "Segoe UI", "Noto Sans", "SF Pro Text", sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background-color: {BG};
}}

/* generic widget base */
QWidget {{
    background-color: {BG};
    color: {FG};
}}

/* video area */
QWidget#video_root,
QWidget#video_panel {{
    background-color: {VIDEO_BG};
}}

/* control panel and its row containers */
QWidget#ctrl_area,
QWidget#ctrl_row {{
    background-color: {CTRL_BG};
}}

QLabel {{
    background-color: transparent;
    color: {FG};
}}

QLabel#video_title {{
    background-color: {VIDEO_BG};
    color: {DIM_FG};
    font-size: 11px;
    padding: 4px 0px;
}}

QLabel#video_lbl {{
    background-color: {VIDEO_BG};
    color: #3a3a3a;
}}

QLabel#dim {{
    color: {DIM_FG};
}}

QLabel#err {{
    color: {ERR_FG};
}}

QLineEdit, QTextEdit {{
    background-color: {ENTRY_BG};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 7px;
    selection-background-color: #264f78;
    selection-color: {FG};
}}

QLineEdit:focus, QTextEdit:focus {{
    border-color: {ACCENT};
}}

QPushButton {{
    background-color: {BTN_BG};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px 14px;
    min-height: 26px;
    min-width: 60px;
}}

QPushButton:hover {{
    background-color: {BTN_HOVER};
    border-color: #666666;
}}

QPushButton:pressed {{
    background-color: #5a5a5a;
}}

QPushButton:disabled {{
    background-color: #262626;
    color: #555555;
    border-color: #333333;
}}

QPushButton#accent {{
    background-color: {ACCENT};
    color: {STATUS_FG};
    border: none;
    font-weight: 600;
}}

QPushButton#accent:hover {{
    background-color: #1a8cd8;
}}

QPushButton#accent:pressed {{
    background-color: #005fa3;
}}

QPushButton#accent:disabled {{
    background-color: #1a3d55;
    color: #6a9ab5;
    border: none;
}}

QComboBox {{
    background-color: {ENTRY_BG};
    color: {FG};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 7px;
    min-height: 26px;
}}

QComboBox:focus {{
    border-color: {ACCENT};
}}

QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: center right;
    width: 22px;
    border-left: 1px solid {BORDER};
    border-top-right-radius: 4px;
    border-bottom-right-radius: 4px;
}}

QComboBox::down-arrow {{
    border-left:  4px solid transparent;
    border-right: 4px solid transparent;
    border-top:   5px solid {DIM_FG};
    width: 0;
    height: 0;
}}

QComboBox QAbstractItemView {{
    background-color: {ENTRY_BG};
    color: {FG};
    border: 1px solid {BORDER};
    selection-background-color: #264f78;
    selection-color: {FG};
    outline: none;
    padding: 2px;
}}

QStatusBar {{
    background-color: {STATUS_BG};
    color: {STATUS_FG};
    font-size: 12px;
    padding: 0 8px;
}}

QStatusBar QLabel {{
    color: {STATUS_FG};
    background-color: transparent;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: #555555;
    border-radius: 4px;
    min-height: 20px;
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
}}
QScrollBar::handle:horizontal {{
    background: #555555;
    border-radius: 4px;
    min-width: 20px;
}}
QScrollBar::add-line:horizontal,
QScrollBar::sub-line:horizontal {{
    width: 0;
}}
"""


# ── cross-thread signals ───────────────────────────────────────────────────────
class _Signals(QObject):
    launch_capture = Signal(object, int)  # (cv2.VideoCapture, cam_idx)
    sp_error = Signal(str)
    camera_error = Signal()
    vcam_error = Signal(str)


# ── main window ────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self, config_path: str, use_int8: bool = False) -> None:
        super().__init__()
        self.setWindowTitle("FluxRT")
        self.resize(1200, 780)

        self.config_path = config_path
        self._use_int8 = use_int8

        self._sp = None
        self._input_tensor = None
        self._output_tensor = None
        self._resolution: dict | None = None
        self._use_ref_image = False
        self._lip_transfer_in_config = False
        self._lip_active = False
        self._sp_loading = False
        self._cfg_w = 576
        self._cfg_h = 320

        self._latest_input: np.ndarray | None = None
        self._latest_output: np.ndarray | None = None
        self._latest_output_bgr: np.ndarray | None = None
        self._frame_lock = threading.Lock()

        self._capture_thread: threading.Thread | None = None
        self._capture_stop = threading.Event()
        self._running = False

        self._vcam_thread: threading.Thread | None = None
        self._vcam_stop = threading.Event()
        self._vcam_cam = None

        self._ref_full_path: str | None = None

        self._sig = _Signals()
        self._build_ui()

        # Connect cross-thread signals after UI exists
        self._sig.launch_capture.connect(self._on_launch_capture)
        self._sig.sp_error.connect(self._on_sp_error)
        self._sig.camera_error.connect(self._on_camera_error)
        self._sig.vcam_error.connect(self._on_vcam_error)

        self._load_config_meta(config_path)

        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_frames)
        self._poll_timer.start(POLL_MS)

        QTimer.singleShot(0, self._begin_start)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── video panels ──────────────────────────────────────────────────────
        video_root = QWidget()
        video_root.setObjectName("video_root")
        video_layout = QHBoxLayout(video_root)
        video_layout.setContentsMargins(6, 6, 6, 6)
        video_layout.setSpacing(6)

        self._input_lbl = self._make_video_panel(video_layout, "Input")
        self._output_lbl = self._make_video_panel(video_layout, "Output")

        # ── control panel ─────────────────────────────────────────────────────
        ctrl_area = QWidget()
        ctrl_area.setObjectName("ctrl_area")
        ctrl_layout = QGridLayout(ctrl_area)
        ctrl_layout.setContentsMargins(14, 10, 14, 12)
        ctrl_layout.setHorizontalSpacing(8)
        ctrl_layout.setVerticalSpacing(6)
        ctrl_layout.setColumnStretch(1, 1)

        row = 0

        # Camera row
        ctrl_layout.addWidget(
            QLabel("Camera:"),
            row,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        cam_row = self._ctrl_row()
        cam_row_l = cam_row.layout()
        self._cam_combo = QComboBox()
        self._cam_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        cam_row_l.addWidget(self._cam_combo)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self._refresh_cameras)
        cam_row_l.addWidget(refresh_btn)
        self._cam_err_lbl = QLabel()
        self._cam_err_lbl.setObjectName("err")
        cam_row_l.addWidget(self._cam_err_lbl)
        cam_row_l.addStretch()
        ctrl_layout.addWidget(cam_row, row, 1, 1, 2)
        row += 1

        # Prompt row
        ctrl_layout.addWidget(
            QLabel("Prompt:"),
            row,
            0,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
        )
        self._prompt_edit = QTextEdit()
        self._prompt_edit.setFixedHeight(72)
        self._prompt_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._prompt_edit.textChanged.connect(self._on_prompt_changed)
        ctrl_layout.addWidget(self._prompt_edit, row, 1, 1, 2)
        row += 1

        # Reference image row (conditionally visible)
        self._ref_widget = self._ctrl_row()
        ref_l = self._ref_widget.layout()
        ref_l.addWidget(QLabel("Reference image:"))
        self._ref_path_lbl = QLabel("(none)")
        self._ref_path_lbl.setObjectName("dim")
        self._ref_path_lbl.setMinimumWidth(260)
        ref_l.addWidget(self._ref_path_lbl)
        browse_ref_btn = QPushButton("Browse")
        browse_ref_btn.clicked.connect(self._browse_reference)
        ref_l.addWidget(browse_ref_btn)
        clear_ref_btn = QPushButton("Clear")
        clear_ref_btn.clicked.connect(self._clear_reference)
        ref_l.addWidget(clear_ref_btn)
        ref_l.addStretch()
        ctrl_layout.addWidget(self._ref_widget, row, 0, 1, 3)
        self._ref_widget.setVisible(False)
        row += 1

        # Lip transfer toggle row
        lip_row = self._ctrl_row()
        lip_l = lip_row.layout()
        self._lip_btn = QPushButton("Enable Lip Transfer")
        self._lip_btn.setEnabled(False)
        self._lip_btn.clicked.connect(self._toggle_lip)
        lip_l.addWidget(self._lip_btn)
        lip_l.addStretch()
        ctrl_layout.addWidget(lip_row, row, 0, 1, 3)
        row += 1

        # Action buttons row
        btn_row = self._ctrl_row()
        btn_row.layout().setContentsMargins(0, 4, 0, 0)
        btn_l = btn_row.layout()
        self._start_btn = QPushButton("Start")
        self._start_btn.setObjectName("accent")
        self._start_btn.setMinimumWidth(90)
        self._start_btn.clicked.connect(self._toggle_start)
        btn_l.addWidget(self._start_btn)
        self._vcam_btn = QPushButton("Enable Virtual Cam")
        self._vcam_btn.setEnabled(False)
        self._vcam_btn.clicked.connect(self._toggle_vcam)
        btn_l.addWidget(self._vcam_btn)
        self._vcam_err_lbl = QLabel()
        self._vcam_err_lbl.setObjectName("err")
        btn_l.addWidget(self._vcam_err_lbl)
        btn_l.addStretch()
        ctrl_layout.addWidget(btn_row, row, 0, 1, 3)

        self.statusBar().showMessage("Ready.")

        root_layout.addWidget(video_root, stretch=1)
        root_layout.addWidget(ctrl_area, stretch=0)

        self._refresh_cameras()

    @staticmethod
    def _ctrl_row() -> QWidget:
        """A transparent-background row container for the control panel."""
        w = QWidget()
        w.setObjectName("ctrl_row")
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        return w

    def _make_video_panel(self, parent_layout: QHBoxLayout, title: str) -> QLabel:
        panel = QWidget()
        panel.setObjectName("video_panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        title_lbl = QLabel(title)
        title_lbl.setObjectName("video_title")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setFixedHeight(22)
        layout.addWidget(title_lbl)

        img_lbl = QLabel("No signal")
        img_lbl.setObjectName("video_lbl")
        img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        img_lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        img_lbl.setMinimumSize(240, 135)
        layout.addWidget(img_lbl)

        parent_layout.addWidget(panel)
        return img_lbl

    # ── config ─────────────────────────────────────────────────────────────────

    def _load_config_meta(self, path: str) -> None:  # noqa: C901
        try:
            with open(path) as f:
                cfg = json.load(f)
            self._use_ref_image = cfg.get("use_reference_image", False)
            self._lip_transfer_in_config = cfg.get("lip_transfer", {}).get("enable", False)
            res = cfg.get("resolution", {})
            self._cfg_w = res.get("width", 576)
            self._cfg_h = res.get("height", 320)
            log(
                f"Config loaded: {path}  res={self._cfg_w}x{self._cfg_h}  use_ref={self._use_ref_image}  lip_transfer={self._lip_transfer_in_config}"
            )
            self._ref_widget.setVisible(self._use_ref_image)
            self._lip_btn.setEnabled(self._lip_transfer_in_config)
            default_prompt = cfg.get("default_prompt", "")
            if default_prompt and not self._prompt_edit.toPlainText().strip():
                self._prompt_edit.blockSignals(True)
                self._prompt_edit.setPlainText(default_prompt)
                self._prompt_edit.blockSignals(False)
        except Exception as exc:
            log(f"Config read error: {exc}")
            self._cfg_w, self._cfg_h = 576, 320

    # ── cameras ────────────────────────────────────────────────────────────────

    def _refresh_cameras(self) -> None:
        log("Scanning for cameras…")
        cams = enumerate_cameras()
        self._cam_combo.clear()
        if cams:
            for _, lbl in cams:
                self._cam_combo.addItem(lbl)
            self._cam_err_lbl.setText("")
            log(f"Cameras found: {[lbl for _, lbl in cams]}")
        else:
            self._cam_err_lbl.setText("No cameras found")
            log("No cameras found")

    def _selected_cam_index(self) -> int | None:
        val = self._cam_combo.currentText()
        if not val:
            return None
        try:
            return int(val.split()[-1])
        except ValueError:
            return None

    # ── prompt / reference ─────────────────────────────────────────────────────

    def _on_prompt_changed(self) -> None:
        prompt = self._prompt_edit.toPlainText()
        if self._sp is not None:
            self._sp.set_prompt(prompt)
        preview = prompt[:70] + ("…" if len(prompt) > 70 else "")
        log(f"Prompt: {preview!r}")

    def _browse_reference(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select reference image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp);;All files (*.*)",
        )
        if not path:
            return
        self._ref_full_path = path
        self._ref_path_lbl.setText(os.path.basename(path))
        log(f"Reference image selected: {path}")
        if self._sp is not None:
            self._apply_reference(path)

    def _clear_reference(self) -> None:
        self._ref_full_path = None
        self._ref_path_lbl.setText("(none)")
        if self._sp is not None and self._use_ref_image:
            try:
                self._sp.set_reference_image(None)
                log("Reference image cleared")
            except Exception as exc:
                log(f"Error clearing reference image: {exc}")

    def _apply_reference(self, path: str) -> None:
        try:
            arr = np.array(Image.open(path).convert("RGB"))
            self._sp.set_reference_image(arr)
            log(f"Reference image applied: {path}")
        except Exception as exc:
            log(f"Reference image load error: {exc}")

    def _toggle_lip(self) -> None:
        self._lip_active = not self._lip_active
        self._lip_btn.setText(
            "Disable Lip Transfer" if self._lip_active else "Enable Lip Transfer"
        )
        if self._sp is not None:
            self._sp.set_lip_transfer(self._lip_active)
        log(f"Lip transfer: {'on' if self._lip_active else 'off'}")

    # ── start / stop ───────────────────────────────────────────────────────────

    def _toggle_start(self) -> None:
        if self._running or self._sp_loading:
            self._stop_capture()
        else:
            self._begin_start()

    def _begin_start(self) -> None:
        cam_idx = self._selected_cam_index()
        if cam_idx is None:
            self._cam_err_lbl.setText("No camera selected")
            log("Start aborted: no camera selected")
            return

        cap = cv2.VideoCapture(cam_idx, CAM_BACKEND)
        if not cap.isOpened():
            cap.release()
            self._cam_err_lbl.setText("Cannot open camera")
            log(f"Start aborted: cannot open camera {cam_idx}")
            return
        self._cam_err_lbl.setText("")

        if self._sp is None:
            self._sp_loading = True
            self._start_btn.setText("Loading…")
            self._start_btn.setEnabled(False)
            self.statusBar().showMessage("Loading model…")
            log("Loading StreamProcessor (this may take a while)")
            prompt = self._prompt_edit.toPlainText()
            threading.Thread(
                target=self._init_sp_thread,
                args=(cap, cam_idx, prompt),
                daemon=True,
            ).start()
        else:
            self._on_launch_capture(cap, cam_idx)

    def _init_sp_thread(self, cap, cam_idx: int, prompt: str) -> None:
        try:
            sp = StreamProcessor(self.config_path)
            if self._use_int8:
                sp.enable_quantization()
            sp.start()
            sp.set_prompt(prompt)
            self._sp = sp
            self._input_tensor = sp.get_input_tensor()
            self._output_tensor = sp.get_output_tensor()
            self._resolution = sp.get_resolution()
            log(f"StreamProcessor ready — resolution={self._resolution}")
            if self._ref_full_path and self._use_ref_image:
                self._apply_reference(self._ref_full_path)
            if self._lip_active:
                sp.set_lip_transfer(True)
            self._sig.launch_capture.emit(cap, cam_idx)
        except Exception as exc:
            log(f"StreamProcessor init error: {exc}")
            cap.release()
            self._sig.sp_error.emit(str(exc))

    @Slot(object, int)
    def _on_launch_capture(self, cap, cam_idx: int) -> None:
        self._sp_loading = False
        self._capture_stop.clear()
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop, args=(cap,), daemon=True
        )
        self._capture_thread.start()
        self._start_btn.setText("Stop")
        self._start_btn.setEnabled(True)
        self._vcam_btn.setEnabled(True)
        self.statusBar().showMessage(f"Running — camera {cam_idx}")
        log(f"Capture started on camera {cam_idx}")
        self._start_vcam()

    @Slot(str)
    def _on_sp_error(self, _err: str) -> None:
        self._sp_loading = False
        self._start_btn.setText("Start")
        self._start_btn.setEnabled(True)
        self.statusBar().showMessage("Model error — see terminal")

    def _stop_capture(self) -> None:
        self._capture_stop.set()
        if self._vcam_cam is not None:
            self._stop_vcam()
        self._running = False
        self._sp_loading = False
        self._start_btn.setText("Start")
        self._start_btn.setEnabled(True)
        self._vcam_btn.setEnabled(False)
        self.statusBar().showMessage("Stopped.")
        with self._frame_lock:
            self._latest_input = self._latest_output = self._latest_output_bgr = None
        for lbl in (self._input_lbl, self._output_lbl):
            lbl.clear()
            lbl.setText("No signal")
        log("Capture stopped")

    # ── capture loop ───────────────────────────────────────────────────────────

    def _capture_loop(self, cap) -> None:
        h = self._resolution["height"]
        w = self._resolution["width"]
        try:
            while not self._capture_stop.is_set():
                ok, frame = cap.read()
                if not ok:
                    log("Camera read error — stopping capture")
                    self._sig.camera_error.emit()
                    break
                cropped = crop_maximal_rectangle(frame, h, w)
                with _sp_lock:
                    self._input_tensor.copy_from(cropped)
                    output_bgr = self._output_tensor.to_numpy()
                input_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                output_rgb = cv2.cvtColor(output_bgr, cv2.COLOR_BGR2RGB)
                with self._frame_lock:
                    self._latest_input = input_rgb
                    self._latest_output = output_rgb
                    self._latest_output_bgr = output_bgr
        finally:
            cap.release()

    @Slot()
    def _on_camera_error(self) -> None:
        self._cam_err_lbl.setText("Camera disconnected")
        self._running = False
        self._start_btn.setText("Start")
        self._start_btn.setEnabled(True)
        self._vcam_btn.setEnabled(False)
        self.statusBar().showMessage("Camera error — see terminal")

    # ── virtual camera ─────────────────────────────────────────────────────────

    def _toggle_vcam(self) -> None:
        if self._vcam_cam is not None:
            self._stop_vcam()
        else:
            self._start_vcam()

    def _start_vcam(self) -> None:
        if self._resolution is None:
            return
        try:
            import pyvirtualcam
            from pyvirtualcam import PixelFormat

            vcam = pyvirtualcam.Camera(
                width=self._resolution["width"],
                height=self._resolution["height"],
                fps=30,
                fmt=PixelFormat.BGR,
            )
            self._vcam_cam = vcam
            self._vcam_err_lbl.setText("")
            self._vcam_btn.setText("Disable Virtual Cam")
            self._vcam_stop.clear()
            self._vcam_thread = threading.Thread(target=self._vcam_loop, daemon=True)
            self._vcam_thread.start()
            log(f"Virtual camera started: {vcam.device}")
            self.statusBar().showMessage(f"Virtual cam active: {vcam.device}")
        except Exception as exc:
            self._vcam_err_lbl.setText(f"VCam error: {exc}")
            log(f"Virtual camera error: {exc}")

    def _stop_vcam(self) -> None:
        self._vcam_stop.set()
        if self._vcam_thread is not None:
            self._vcam_thread.join(timeout=2)
            self._vcam_thread = None
        if self._vcam_cam is not None:
            try:
                self._vcam_cam.__exit__(None, None, None)
            except Exception:
                pass
            self._vcam_cam = None
        self._vcam_btn.setText("Enable Virtual Cam")
        self._vcam_err_lbl.setText("")
        self.statusBar().showMessage("Running.")
        log("Virtual camera stopped")

    def _vcam_loop(self) -> None:
        vcam = self._vcam_cam
        try:
            while not self._vcam_stop.is_set():
                with self._frame_lock:
                    frame = self._latest_output_bgr
                if frame is not None:
                    try:
                        vcam.send(frame)
                        vcam.sleep_until_next_frame()
                    except Exception as exc:
                        log(f"VCam send error: {exc}")
                        self._sig.vcam_error.emit("VCam send error — see terminal")
                        break
                else:
                    time.sleep(0.01)
        finally:
            pass

    @Slot(str)
    def _on_vcam_error(self, msg: str) -> None:
        self._vcam_err_lbl.setText(msg)

    # ── frame rendering ────────────────────────────────────────────────────────

    def _poll_frames(self) -> None:
        with self._frame_lock:
            inp = self._latest_input
            out = self._latest_output
        self._render_frame(self._input_lbl, inp)
        self._render_frame(self._output_lbl, out)

    def _render_frame(self, label: QLabel, frame: np.ndarray | None) -> None:
        if frame is None:
            return
        lw = label.width()
        lh = label.height()
        if lw < 10:
            lw = self._cfg_w
        if lh < 10:
            lh = self._cfg_h
        h, w = frame.shape[:2]
        scale = min(lw / w, lh / h)
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        arr = np.ascontiguousarray(frame)
        if (nw, nh) != (w, h):
            arr = cv2.resize(arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # .copy() detaches QImage from the numpy buffer before it's GC'd
        qimg = QImage(arr.data, nw, nh, nw * 3, QImage.Format.Format_RGB888).copy()
        label.setPixmap(QPixmap.fromImage(qimg))

    # ── close ─────────────────────────────────────────────────────────────────

    def closeEvent(self, event) -> None:
        log("Shutting down")
        self._poll_timer.stop()
        self._capture_stop.set()
        self._vcam_stop.set()
        if self._vcam_cam is not None:
            try:
                self._vcam_cam.__exit__(None, None, None)
            except Exception:
                pass
        if self._sp is not None:
            try:
                self._sp.stop()
            except Exception:
                pass
        event.accept()


# ── entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="FluxRT GUI")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to StreamProcessor config JSON (default: %(default)s)",
    )
    parser.add_argument("--int8", action="store_true", help="Enable int8 quantization")
    args, _ = parser.parse_known_args()

    app = QApplication([])
    app.setStyleSheet(STYLESHEET)
    win = MainWindow(config_path=args.config, use_int8=args.int8)
    win.show()
    app.exec()


if __name__ == "__main__":
    main()
