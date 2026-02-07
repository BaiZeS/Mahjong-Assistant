from __future__ import annotations

import logging
import os
import sys
import subprocess
import cv2
import numpy as np
import win32gui
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from PyQt6 import QtCore, QtGui, QtWidgets

from .constants import TILE_NAME_TO_ID, tile_to_name
from .capture import WindowDetector, ScreenCapture, WindowInfo
from .recognition import TileRecognizer
from .core import LayoutEstimator, GameStateExtractor
from .client import BackendClient

class AnnotateLabel(QtWidgets.QLabel):
    def __init__(self, orig_w: int, orig_h: int, pix: QtGui.QPixmap) -> None:
        super().__init__()
        self.setPixmap(pix)
        self.resize(pix.size())
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.scale_x = orig_w / pix.width()
        self.scale_y = orig_h / pix.height()
        self.rubber = QtWidgets.QRubberBand(QtWidgets.QRubberBand.Shape.Rectangle, self)
        self.origin = QtCore.QPoint()
        self.selection: Optional[QtCore.QRect] = None

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        self.origin = event.pos()
        self.rubber.setGeometry(QtCore.QRect(self.origin, QtCore.QSize()))
        self.rubber.show()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        self.rubber.setGeometry(QtCore.QRect(self.origin, event.pos()).normalized())

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        self.selection = QtCore.QRect(self.origin, event.pos()).normalized()
        self.rubber.hide()

    def selected_rect_original(self) -> Optional[Tuple[int, int, int, int]]:
        if self.selection is None:
            return None
        x = int(self.selection.left() * self.scale_x)
        y = int(self.selection.top() * self.scale_y)
        w = int(self.selection.width() * self.scale_x)
        h = int(self.selection.height() * self.scale_y)
        if w <= 0 or h <= 0:
            return None
        return (x, y, w, h)


class AnnotateDialog(QtWidgets.QDialog):
    def __init__(self, image: np.ndarray, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("手动标注")
        preview = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        preview = np.ascontiguousarray(preview)
        height, width = preview.shape[:2]
        bytes_per_line = preview.strides[0]
        qimg = QtGui.QImage(preview.data, width, height, bytes_per_line, QtGui.QImage.Format.Format_RGB888).copy()
        pix = QtGui.QPixmap.fromImage(qimg)
        max_w, max_h = 900, 600
        pix = pix.scaled(QtCore.QSize(max_w, max_h), QtCore.Qt.AspectRatioMode.KeepAspectRatio, QtCore.Qt.TransformationMode.SmoothTransformation)
        self.label = AnnotateLabel(width, height, pix)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.StandardButton.Ok | QtWidgets.QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.label, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(buttons)

    def selected_rect_original(self) -> Optional[Tuple[int, int, int, int]]:
        return self.label.selected_rect_original()


class BackendManager:
    def __init__(self, port: int = 8765) -> None:
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.logger = logging.getLogger("mahjong.backend_manager")

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
            
        self.process = subprocess.Popen(
            [sys.executable, "-m", "src.mahjong.backend", "--port", str(self.port)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=os.getcwd()
        )
        self.logger.info("Backend started with PID %s", self.process.pid)

    def stop(self) -> None:
        if self.process:
            pid = self.process.pid
            self.logger.info("Stopping backend PID %s", pid)
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                
                if os.name == 'nt':
                    subprocess.call(['taskkill', '/F', '/T', '/PID', str(pid)], 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.process = None
            self.logger.info("Backend stopped")

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None


class MainWindow(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Mahjong Assistant")
        self.setWindowFlags(
            QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.WindowCloseButtonHint
        )
        self.logger = logging.getLogger("mahjong.app")
        self.detector = WindowDetector()
        self.capture = ScreenCapture()
        
        # Determine paths relative to the project root or package
        # Assuming run from project root, templates are in assets/templates
        # But user might have changed it. Let's try to find it.
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        default_dir = os.path.join(project_root, "assets", "templates")
        legacy_dir = os.path.join(project_root, "templates")
        
        templates_dir = default_dir
        
        # Check if default dir is empty but legacy dir has files
        if (not os.path.isdir(default_dir) or not os.listdir(default_dir)) and \
           (os.path.isdir(legacy_dir) and os.listdir(legacy_dir)):
            templates_dir = legacy_dir
        else:
            os.makedirs(default_dir, exist_ok=True)
        
        self.recognizer = TileRecognizer(templates_dir)
        self.layout_estimator = LayoutEstimator()
        self.extractor = GameStateExtractor(self.recognizer, self.layout_estimator)
        
        # Backend management
        self.backend_port = 8765
        self.backend_manager = BackendManager(self.backend_port)
        self.backend_manager.start()
        
        # Ensure backend is killed on app exit
        app = QtWidgets.QApplication.instance()
        if app:
            app.aboutToQuit.connect(self.backend_manager.stop)

        self.backend_client = BackendClient(f"http://127.0.0.1:{self.backend_port}")
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.process_frame)
        self.current_window: Optional[WindowInfo] = None
        self._build_ui()
        self.refresh_windows()

    def _build_ui(self) -> None:
        # Main layout is vertical: Preview (Top) -> Controls/Info (Bottom)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # --- Top Section: Preview ---
        preview_container = QtWidgets.QVBoxLayout()
        preview_container.setSpacing(5)
        self.status_label = QtWidgets.QLabel("状态：未开始")
        self.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label = QtWidgets.QLabel()
        # Set fixed width 600, allow height to expand but keep aspect ratio via scaling
        self.preview_label.setFixedWidth(600)
        self.preview_label.setMinimumHeight(338) # Approx 16:9 for 600 width
        self.preview_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setScaledContents(False)
        self.preview_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.MinimumExpanding)
        preview_container.addWidget(self.status_label)
        preview_container.addWidget(self.preview_label)
        main_layout.addLayout(preview_container)
        
        # --- Bottom Section: Controls & Info ---
        bottom_panel = QtWidgets.QVBoxLayout()
        bottom_panel.setSpacing(10)
        
        # 1. Window & Play Controls (Single Column)
        controls_group = QtWidgets.QGroupBox("控制")
        controls_layout = QtWidgets.QVBoxLayout(controls_group)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setSpacing(5)
        
        self.window_combo = QtWidgets.QComboBox()
        self.refresh_button = QtWidgets.QPushButton("刷新窗口")
        self.start_button = QtWidgets.QPushButton("开始识别")
        self.stop_button = QtWidgets.QPushButton("停止识别")
        
        controls_layout.addWidget(self.window_combo)
        controls_layout.addWidget(self.refresh_button)
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.stop_button)
        
        bottom_panel.addWidget(controls_group)
        
        # 2. Tools (Single Column)
        tools_group = QtWidgets.QGroupBox("工具")
        tools_layout = QtWidgets.QVBoxLayout(tools_group)
        tools_layout.setContentsMargins(10, 10, 10, 10)
        tools_layout.setSpacing(5)
        
        self.templates_button = QtWidgets.QPushButton("选择模板目录")
        self.template_capture_button = QtWidgets.QPushButton("截取当前画面为模板")
        self.annotate_button = QtWidgets.QPushButton("手动标注模板")
        self.calibrate_button = QtWidgets.QPushButton("校准手牌区域")
        self.templates_label = QtWidgets.QLabel(os.path.basename(self.recognizer.templates_dir))
        self.templates_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        
        tools_layout.addWidget(self.templates_button)
        tools_layout.addWidget(self.template_capture_button)
        tools_layout.addWidget(self.annotate_button)
        tools_layout.addWidget(self.calibrate_button)
        tools_layout.addWidget(self.templates_label)
        
        bottom_panel.addWidget(tools_group)
        
        # 3. Settings (Vertical Form)
        settings_group = QtWidgets.QGroupBox("设置")
        settings_layout = QtWidgets.QFormLayout(settings_group)
        settings_layout.setContentsMargins(10, 10, 10, 10)
        settings_layout.setSpacing(5)
        settings_layout.setLabelAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        
        self.red_count_spin = QtWidgets.QSpinBox()
        self.red_count_spin.setRange(0, 8)
        self.red_count_spin.setValue(3)
        self.min_multiplier_spin = QtWidgets.QSpinBox()
        self.min_multiplier_spin.setRange(1, 8)
        self.min_multiplier_spin.setValue(1)
        self.fan_cap_spin = QtWidgets.QSpinBox()
        self.fan_cap_spin.setRange(1, 100)
        self.fan_cap_spin.setValue(40)
        self.play_mode_combo = QtWidgets.QComboBox()
        self.play_mode_combo.addItem("快速模式", "fast")
        self.play_mode_combo.addItem("最大番数", "max")
        self.threshold_spin = QtWidgets.QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 0.95)
        self.threshold_spin.setSingleStep(0.02)
        self.threshold_spin.setValue(0.55)
        self.debug_check = QtWidgets.QCheckBox("调试模式")
        self.debug_check.setChecked(True)
        
        settings_layout.addRow("红中数量:", self.red_count_spin)
        settings_layout.addRow("起胡番数:", self.min_multiplier_spin)
        settings_layout.addRow("封顶番数:", self.fan_cap_spin)
        settings_layout.addRow("识别阈值:", self.threshold_spin)
        settings_layout.addRow("推荐模式:", self.play_mode_combo)
        
        self.que_men_check = QtWidgets.QCheckBox("缺门")
        self.self_draw_check = QtWidgets.QCheckBox("自摸")
        self.self_draw_check.setChecked(True)
        self.gang_hua_check = QtWidgets.QCheckBox("杠上开花")
        self.gang_pao_check = QtWidgets.QCheckBox("杠上炮")
        self.tian_hu_check = QtWidgets.QCheckBox("天胡")
        self.di_hu_check = QtWidgets.QCheckBox("地胡")
        
        checks_layout = QtWidgets.QVBoxLayout()
        checks_layout.addWidget(self.que_men_check)
        checks_layout.addWidget(self.self_draw_check)
        checks_layout.addWidget(self.gang_hua_check)
        checks_layout.addWidget(self.gang_pao_check)
        checks_layout.addWidget(self.tian_hu_check)
        checks_layout.addWidget(self.di_hu_check)
        checks_layout.addWidget(self.debug_check)
        settings_layout.addRow(checks_layout)
        
        bottom_panel.addWidget(settings_group)
        
        # 4. Results & Info (Stacked Vertically)
        info_layout = QtWidgets.QVBoxLayout()
        info_layout.setSpacing(10)
        
        # Hand Group
        hand_group = QtWidgets.QGroupBox("识别手牌")
        hand_layout = QtWidgets.QVBoxLayout(hand_group)
        hand_layout.setContentsMargins(10, 10, 10, 10)
        self.hand_label = QtWidgets.QLabel()
        self.hand_label.setWordWrap(True)
        hand_layout.addWidget(self.hand_label)
        info_layout.addWidget(hand_group)
        
        # Action Group
        action_group = QtWidgets.QGroupBox("操作建议")
        action_layout = QtWidgets.QVBoxLayout(action_group)
        action_layout.setContentsMargins(10, 10, 10, 10)
        self.action_label = QtWidgets.QLabel()
        self.action_label.setWordWrap(True)
        action_layout.addWidget(self.action_label)
        info_layout.addWidget(action_group)
        
        # Recommend Group
        recommend_group = QtWidgets.QGroupBox("出牌推荐")
        recommend_layout = QtWidgets.QVBoxLayout(recommend_group)
        recommend_layout.setContentsMargins(10, 10, 10, 10)
        self.result_list = QtWidgets.QListWidget()
        self.result_list.setMinimumHeight(200)
        recommend_layout.addWidget(self.result_list)
        info_layout.addWidget(recommend_group)
        
        bottom_panel.addLayout(info_layout)
        
        self.discard_label = QtWidgets.QLabel() # Hidden
        self.discard_label.setVisible(False)
        
        main_layout.addLayout(bottom_panel)
        
        # Connect signals
        self.refresh_button.clicked.connect(self.refresh_windows)
        self.start_button.clicked.connect(self.start)
        self.stop_button.clicked.connect(self.stop)
        self.templates_button.clicked.connect(self._choose_templates_dir)
        self.template_capture_button.clicked.connect(self._capture_template)
        self.annotate_button.clicked.connect(self._annotate_template)
        self.calibrate_button.clicked.connect(self._calibrate_hand_bbox)
        self.red_count_spin.valueChanged.connect(self._update_settings)
        self.min_multiplier_spin.valueChanged.connect(self._update_settings)
        self.que_men_check.toggled.connect(self._update_settings)
        self.self_draw_check.toggled.connect(self._update_settings)
        self.gang_hua_check.toggled.connect(self._update_settings)
        self.gang_pao_check.toggled.connect(self._update_settings)
        self.tian_hu_check.toggled.connect(self._update_settings)
        self.di_hu_check.toggled.connect(self._update_settings)
        self.fan_cap_spin.valueChanged.connect(self._update_settings)
        self.play_mode_combo.currentIndexChanged.connect(self._update_settings)
        self.threshold_spin.valueChanged.connect(self._update_settings)
        self.debug_check.toggled.connect(self._update_settings)

    def refresh_windows(self) -> None:
        windows = self.detector.list_windows()
        self.window_combo.clear()
        self_handle = int(self.winId())
        added = 0
        for window in windows:
            if window.handle == self_handle:
                continue
            self.window_combo.addItem(window.title, window)
            added += 1
        if windows:
            self.window_combo.setCurrentIndex(0)
        self.status_label.setText(f"状态：检测到 {added} 个窗口")
        self.logger.info("Refresh windows: %s", added)

    def start(self) -> None:
        data = self.window_combo.currentData()
        if not isinstance(data, WindowInfo):
            self.status_label.setText("状态：未选择窗口")
            return
        if self.recognizer.template_count == 0:
            self.status_label.setText("状态：模板未加载")
            self.logger.warning("Templates not loaded")
            return
        try:
            if not win32gui.IsWindow(data.handle):
                self.status_label.setText("状态：窗口无效")
                return
                
            # Restart backend if it was killed
            if not self.backend_manager.is_running():
                self.backend_manager.start()
                self.logger.info("Restarted backend process")
                
            self.current_window = data
            test_image = self.capture.capture(self.current_window)
            if test_image is None:
                self.status_label.setText("状态：无法捕获窗口")
                return
            self._update_settings()
            self.timer.start(800)
            self.status_label.setText("状态：识别中")
            self.logger.info("Start capture: %s (%s)", data.title, data.handle)
        except Exception as exc:
            self.status_label.setText(f"状态：启动失败 {exc}")
            self.logger.exception("Start failed: %s", exc)

    def stop(self) -> None:
        self.timer.stop()
        self.backend_manager.stop()
        self.status_label.setText("状态：已停止")
        self.logger.info("Stop capture")
        self.close()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.logger.info("Window close event triggered")
        self.timer.stop()
        self.backend_manager.stop()
        event.accept()

    def _capture_template(self) -> None:
        data = self.window_combo.currentData()
        if not isinstance(data, WindowInfo):
            self.status_label.setText("状态：未选择窗口")
            return
        if not win32gui.IsWindow(data.handle):
            self.status_label.setText("状态：窗口无效")
            return
        image = self.capture.capture(data)
        if image is None:
            self.status_label.setText("状态：无法捕获窗口")
            return
        target_dir = self.recognizer.templates_dir
        if not target_dir or not os.path.isdir(target_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            target_dir = os.path.join(project_root, "assets", "templates")
        os.makedirs(target_dir, exist_ok=True)
        filename = f"template_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        path = os.path.join(target_dir, filename)
        saved = cv2.imwrite(path, image)
        if not saved:
            self.status_label.setText("状态：保存失败")
            return
        self.recognizer.templates_dir = target_dir
        self.recognizer.load_templates()
        self.templates_label.setText(os.path.basename(target_dir) or target_dir)
        if self.recognizer.template_count == 0:
            self.status_label.setText(f"状态：已保存 {filename}，但模板未加载")
        else:
            self.status_label.setText(f"状态：已保存 {filename}，模板 {self.recognizer.template_count}/{self.recognizer.template_total}")
        self.logger.info("Template captured: %s", path)

    def _choose_templates_dir(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(self, "选择模板目录", self.recognizer.templates_dir)
        if not directory:
            return
        self.recognizer.templates_dir = directory
        self.recognizer.load_templates()
        self.templates_label.setText(os.path.basename(directory) or directory)
        if self.recognizer.template_total == 0:
            self.status_label.setText("状态：模板未加载")
        else:
            self.status_label.setText(f"状态：模板 {self.recognizer.template_count}/{self.recognizer.template_total}")

    def _annotate_template(self) -> None:
        data = self.window_combo.currentData()
        if not isinstance(data, WindowInfo):
            self.status_label.setText("状态：未选择窗口")
            return
        if not win32gui.IsWindow(data.handle):
            self.status_label.setText("状态：窗口无效")
            return
        image = self.capture.capture(data)
        if image is None:
            self.status_label.setText("状态：无法捕获窗口")
            return
        dialog = AnnotateDialog(image, self)
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        rect = dialog.selected_rect_original()
        if rect is None:
            self.status_label.setText("状态：未选择区域")
            return
        name, ok = QtWidgets.QInputDialog.getText(self, "标注命名", "请输入牌名或编号（如 1w 或 0）")
        if not ok or not name:
            self.status_label.setText("状态：未命名")
            return
        name = name.strip()
        if not (name.isdigit() or name in TILE_NAME_TO_ID):
            self.status_label.setText("状态：命名无效")
            return
        x, y, w, h = rect
        roi = image[y : y + h, x : x + w]
        target_dir = self.recognizer.templates_dir
        if not target_dir or not os.path.isdir(target_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            target_dir = os.path.join(project_root, "assets", "templates")
        os.makedirs(target_dir, exist_ok=True)
        filename = f"{name}.png"
        path = os.path.join(target_dir, filename)
        saved = cv2.imwrite(path, roi)
        if not saved:
            self.status_label.setText("状态：保存失败")
            return
        self.recognizer.templates_dir = target_dir
        self.recognizer.load_templates()
        self.templates_label.setText(os.path.basename(target_dir) or target_dir)
        self.status_label.setText(f"状态：已标注 {filename}")
        self.logger.info("Annotated template saved: %s", path)

    def _calibrate_hand_bbox(self) -> None:
        data = self.window_combo.currentData()
        if not isinstance(data, WindowInfo):
            self.status_label.setText("状态：未选择窗口")
            return
        if not win32gui.IsWindow(data.handle):
            self.status_label.setText("状态：窗口无效")
            return
        image = self.capture.capture(data)
        if image is None:
            self.status_label.setText("状态：无法捕获窗口")
            return
        dialog = AnnotateDialog(image, self)
        dialog.setWindowTitle("校准手牌区域")
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        rect = dialog.selected_rect_original()
        if rect is None:
            self.status_label.setText("状态：未选择区域")
            return
        self.layout_estimator.hand_bbox = rect
        self.status_label.setText("状态：已校准手牌区域")

    def process_frame(self) -> None:
        if self.current_window is None:
            return
        try:
            image = self.capture.capture(self.current_window)
            if image is None:
                return
            state = self.extractor.extract(image)
            
            # Debug Drawing
            debug_img = image.copy()
            layout_dict = state.get("layout")
            if layout_dict:
                # Draw hand regions
                hand_regions = layout_dict.get("hand_regions", [])
                for i, (x, y, w, h) in enumerate(hand_regions):
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    # If we have recognition result for this tile, draw it
                    hand_tiles = state.get("hand", [])
                    if i < len(hand_tiles):
                        name = tile_to_name(hand_tiles[i])
                        cv2.putText(debug_img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                
                # Draw discard regions (optional, can clutter)
                # for player, regions in layout.discard_regions.items():
                #    for x, y, w, h in regions:
                #        cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 1)

            preview = cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB)
            preview = np.ascontiguousarray(preview)
            height, width = preview.shape[:2]
            bytes_per_line = preview.strides[0]
            qimg = QtGui.QImage(
                preview.data,
                width,
                height,
                bytes_per_line,
                QtGui.QImage.Format.Format_RGB888,
            ).copy()
            pix = QtGui.QPixmap.fromImage(qimg)
            target_size = self.preview_label.size()
            if target_size.width() > 0 and target_size.height() > 0:
                pix = pix.scaled(
                    target_size,
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            self.preview_label.setPixmap(pix)
            
            result = self.backend_client.predict(state)
            recommendations = result.get("recommendations", [])
            warning = result.get("warning", "")
            action = result.get("action_advice", {})
            discards = result.get("discard_suggestions", [])
            if warning:
                self.status_label.setText(f"状态：{warning}")
            else:
                self.status_label.setText("状态：识别中")
            self._render_state(state)
            self._render_action(action, discards)
            self._render_recommendations(recommendations, warning)
        except Exception as exc:
            self.timer.stop()
            self.status_label.setText(f"状态：识别异常 {exc}")
            self.logger.exception("Process frame failed: %s", exc)

    def _render_state(self, state: Dict) -> None:
        hand = state.get("hand", [])
        hand_text = " ".join([tile_to_name(t) for t in hand])
        if not hand_text:
            debug = f" 阈值{self.recognizer.last_threshold:.2f} max{self.recognizer.last_max_score:.2f} mean{self.recognizer.last_mean_score:.2f}"
            self.hand_label.setText("未识别" + (debug if self.debug_check.isChecked() else ""))
        else:
            self.hand_label.setText(hand_text)
        discards = state.get("discarded_tiles", [])
        grouped = {i: [] for i in range(4)}
        for d in discards:
            grouped[d["player"]].append(tile_to_name(d["tile"]))
        lines = []
        for player in range(4):
            tiles = " ".join(grouped[player]) if grouped[player] else "-"
            lines.append(f"P{player}: {tiles}")
        current_discard = state.get("current_discard")
        if current_discard is not None:
            lines.append(f"当前出牌: {tile_to_name(current_discard)}")
        self.discard_label.setText(" | ".join(lines))

    def _render_action(self, action: Dict, discards: List[Dict]) -> None:
        action_text = action.get("action", "未知")
        if action_text == "胡":
            action_text = f"胡 {action.get('fan', 0)}颗 {action.get('label', '')}"
        discard_text = ""
        if discards:
            discard_text = " 出牌建议: " + " ".join(
                [f"{tile_to_name(d['tile'])}(向听{d['shanten']})" for d in discards]
            )
        self.action_label.setText(f"{action_text}{discard_text}")

    def _render_recommendations(self, recommendations: List[Dict], warning: str) -> None:
        self.result_list.clear()
        if not recommendations:
            self.result_list.addItem("无满足条件的推荐")
            return
        for idx, item in enumerate(recommendations, start=1):
            tile = item.get("tile", -1)
            score = float(item.get("score", 0))
            fan = int(item.get("fan", 0))
            label = item.get("label", "")
            melds = item.get("melds", [])
            melds_text = " ".join(["".join(m) for m in melds])
            self.result_list.addItem(f"{idx}. {tile_to_name(tile)} {score:.3f} {fan}颗 {label} {melds_text}")

    def _update_settings(self) -> None:
        self.extractor.red_zhong_count = int(self.red_count_spin.value())
        self.extractor.min_hu_multiplier = int(self.min_multiplier_spin.value())
        self.extractor.require_que_men = bool(self.que_men_check.isChecked())
        self.extractor.self_draw = bool(self.self_draw_check.isChecked())
        self.extractor.gang_shang_hua = bool(self.gang_hua_check.isChecked())
        self.extractor.gang_shang_pao = bool(self.gang_pao_check.isChecked())
        self.extractor.tian_hu = bool(self.tian_hu_check.isChecked())
        self.extractor.di_hu = bool(self.di_hu_check.isChecked())
        self.extractor.fan_cap = int(self.fan_cap_spin.value())
        self.extractor.play_mode = str(self.play_mode_combo.currentData())
        self.recognizer.last_threshold = float(self.threshold_spin.value())
        self.recognizer.enable_debug = bool(self.debug_check.isChecked())

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.backend_manager.stop()
        super().closeEvent(event)
