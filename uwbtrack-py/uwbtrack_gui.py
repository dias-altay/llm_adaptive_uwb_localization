"""
UWB Tracker GUI - PySide6 based graphical interface
"""

import sys
import os
import glob
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

try:
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QComboBox, QFileDialog, QMessageBox,
        QFormLayout, QDialog, QLineEdit, QSpinBox, QDoubleSpinBox,
        QCheckBox, QStatusBar, QToolBar, QTextEdit, QTabWidget,
        QSplitter, QGroupBox, QGridLayout, QProgressDialog,
        QSizePolicy
    )
    from PySide6.QtCore import Qt, QThread, QTimer, Signal, QObject
    from PySide6.QtGui import QFont, QIcon, QAction
except Exception as e:
    print("PySide6 import failed.")
    print(f"Exception: {e}")
    print(f"Python executable: {sys.executable}")
    print("sys.path:")
    for p in sys.path:
        print(f"  {p}")
    print(f"  {sys.executable} -m pip install --upgrade PySide6")
    sys.exit(1)

import numpy as np
import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar

from uwbtrack.config import load_config_file
from uwbtrack.geom import parse_anchor_map
from uwbtrack.gui_calibration import CalibrationDialog
from uwbtrack.__main__ import start_tracking_embedded
from uwbtrack.plotting import process_pose_packet

DEFAULT_CONFIG_FILE = "uwbtrack.yaml"

GUI_DEFAULTS = {
    "baud": 115200,
    "anchors": "1:0,0,1.65;2:7.48,0,1.65;3:4.60,3.94,2.665;4:3.50,3.94,2.665;5:1.78,3.94,2.665",
    "min_anchors": 3,
    "solve_2d": True,
    "floor_z": 0.175,
    "min_weight": 0.0,
    "strong_weight": 0.20,
    "certainty_dist": 0.2,
    "outlier_threshold": 6.0,
    "innovation_max": 1.5,
    "weak_avgw": 0.25,
    "stats_every": 20,
    "origin_shift": True,
    "origin_shift_x": 4.20,
    "origin_shift_y": 1.90,
    "max_range_m": 50.0,
    "bias_file": "biases.json",
    "no_calib": True,
    "print_qual": True,
    "debug_geom": False,
    "llm_hybrid_tuning_enabled": False,
    "llm_window_s": 5.0,
    "slow_loop_s": 30.0,
    "mocap_enabled": False,
    "websocket_ip": "ws://192.168.53.225",
    "websocket_port": 8765,
    "mocap_object": "Goal",
    "mocap_id": 2,
    "runs_dir": "runs",
    "log_frames_csv": "",
    "log_windows_csv": "",
    "uwb_ws_url": "ws://10.131.128.4:8766/",
    "max_mocap_age_s": 0.5,
    "logging_enabled": False,
    "show_raw_position": False,
}

class SettingsDialog(QDialog):
    """Settings dialog for configuration parameters"""

    def __init__(self, parent, config: Dict[str, Any]):
        super().__init__(parent)
        self.setWindowTitle("UWB Tracker Settings")
        self.setModal(True)
        self.resize(600, 700)

        self.config = dict(config)
        self.widgets = {}

        layout = QVBoxLayout(self)

        tabs = QTabWidget()
        layout.addWidget(tabs)

        basic_tab = self._create_basic_tab()
        tabs.addTab(basic_tab, "Basic")

        advanced_tab = self._create_advanced_tab()
        tabs.addTab(advanced_tab, "Advanced")

        llm_tab = self._create_llm_tab()
        tabs.addTab(llm_tab, "LLM/AI")

        mocap_tab = self._create_mocap_tab()
        tabs.addTab(mocap_tab, "Motion Capture")

        button_layout = QHBoxLayout()

        load_btn = QPushButton("Load Config File...")
        load_btn.clicked.connect(self._load_config_file)
        button_layout.addWidget(load_btn)

        button_layout.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)

    def _create_basic_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)

        serial_group = QGroupBox("Serial Communication")
        serial_layout = QFormLayout(serial_group)

        self.widgets['baud'] = QSpinBox()
        self.widgets['baud'].setRange(9600, 921600)
        self.widgets['baud'].setValue(self.config.get('baud', GUI_DEFAULTS['baud']))
        serial_layout.addRow("Baud Rate:", self.widgets['baud'])

        from PySide6.QtWidgets import QLineEdit
        self.widgets['uwb_ws_url'] = QLineEdit()
        self.widgets['uwb_ws_url'].setText(str(self.config.get('uwb_ws_url', GUI_DEFAULTS['uwb_ws_url'])))
        serial_layout.addRow("UWB WebSocket URL:", self.widgets['uwb_ws_url'])

        layout.addRow(serial_group)

        anchor_group = QGroupBox("Anchor Configuration")
        anchor_layout = QFormLayout(anchor_group)

        self.widgets['anchors'] = QTextEdit()
        self.widgets['anchors'].setPlainText(str(self.config.get('anchors', GUI_DEFAULTS['anchors'])))
        self.widgets['anchors'].setMaximumHeight(80)
        anchor_layout.addRow("Anchors (format: id:x,y,z;...):", self.widgets['anchors'])

        self.widgets['min_anchors'] = QSpinBox()
        self.widgets['min_anchors'].setRange(3, 10)
        self.widgets['min_anchors'].setValue(self.config.get('min_anchors', GUI_DEFAULTS['min_anchors']))
        anchor_layout.addRow("Minimum Anchors:", self.widgets['min_anchors'])

        layout.addRow(anchor_group)

        pos_group = QGroupBox("Positioning")
        pos_layout = QFormLayout(pos_group)

        self.widgets['solve_2d'] = QCheckBox()
        self.widgets['solve_2d'].setChecked(self.config.get('solve_2d', GUI_DEFAULTS['solve_2d']))
        pos_layout.addRow("2D Floor Solving:", self.widgets['solve_2d'])

        self.widgets['floor_z'] = QDoubleSpinBox()
        self.widgets['floor_z'].setRange(0.0, 10.0)
        self.widgets['floor_z'].setDecimals(3)
        self.widgets['floor_z'].setValue(self.config.get('floor_z', GUI_DEFAULTS['floor_z']))
        pos_layout.addRow("Floor Z (m):", self.widgets['floor_z'])

        layout.addRow(pos_group)

        origin_group = QGroupBox("Origin Shift")
        origin_layout = QFormLayout(origin_group)

        self.widgets['origin_shift'] = QCheckBox()
        self.widgets['origin_shift'].setChecked(self.config.get('origin_shift', GUI_DEFAULTS['origin_shift']))
        origin_layout.addRow("Enable Origin Shift:", self.widgets['origin_shift'])

        self.widgets['origin_shift_x'] = QDoubleSpinBox()
        self.widgets['origin_shift_x'].setRange(-100.0, 100.0)
        self.widgets['origin_shift_x'].setDecimals(3)
        self.widgets['origin_shift_x'].setValue(self.config.get('origin_shift_x', GUI_DEFAULTS['origin_shift_x']))
        origin_layout.addRow("Origin X (m):", self.widgets['origin_shift_x'])

        self.widgets['origin_shift_y'] = QDoubleSpinBox()
        self.widgets['origin_shift_y'].setRange(-100.0, 100.0)
        self.widgets['origin_shift_y'].setDecimals(3)
        self.widgets['origin_shift_y'].setValue(self.config.get('origin_shift_y', GUI_DEFAULTS['origin_shift_y']))
        origin_layout.addRow("Origin Y (m):", self.widgets['origin_shift_y'])

        layout.addRow(origin_group)

        return widget

    def _create_advanced_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)

        filter_group = QGroupBox("Filtering & Robustness")
        filter_layout = QFormLayout(filter_group)

        self.widgets['max_range_m'] = QDoubleSpinBox()
        self.widgets['max_range_m'].setRange(1.0, 1000.0)
        self.widgets['max_range_m'].setValue(self.config.get('max_range_m', GUI_DEFAULTS['max_range_m']))
        filter_layout.addRow("Max Range (m):", self.widgets['max_range_m'])

        self.widgets['certainty_dist'] = QDoubleSpinBox()
        self.widgets['certainty_dist'].setRange(0.0, 10.0)
        self.widgets['certainty_dist'].setDecimals(3)
        self.widgets['certainty_dist'].setValue(self.config.get('certainty_dist', GUI_DEFAULTS['certainty_dist']))
        filter_layout.addRow("Certainty Distance (m):", self.widgets['certainty_dist'])

        self.widgets['outlier_threshold'] = QDoubleSpinBox()
        self.widgets['outlier_threshold'].setRange(1.0, 20.0)
        self.widgets['outlier_threshold'].setValue(self.config.get('outlier_threshold', GUI_DEFAULTS['outlier_threshold']))
        filter_layout.addRow("Outlier Threshold:", self.widgets['outlier_threshold'])

        self.widgets['min_weight'] = QDoubleSpinBox()
        self.widgets['min_weight'].setRange(0.0, 1.0)
        self.widgets['min_weight'].setDecimals(3)
        self.widgets['min_weight'].setValue(self.config.get('min_weight', GUI_DEFAULTS['min_weight']))
        filter_layout.addRow("Min Weight:", self.widgets['min_weight'])

        self.widgets['strong_weight'] = QDoubleSpinBox()
        self.widgets['strong_weight'].setRange(0.0, 1.0)
        self.widgets['strong_weight'].setDecimals(3)
        self.widgets['strong_weight'].setValue(self.config.get('strong_weight', GUI_DEFAULTS['strong_weight']))
        filter_layout.addRow("Strong Weight:", self.widgets['strong_weight'])

        layout.addRow(filter_group)

        kf_group = QGroupBox("Kalman Filter")
        kf_layout = QFormLayout(kf_group)

        self.widgets['innovation_max'] = QDoubleSpinBox()
        self.widgets['innovation_max'].setRange(0.1, 10.0)
        self.widgets['innovation_max'].setValue(self.config.get('innovation_max', GUI_DEFAULTS['innovation_max']))
        kf_layout.addRow("Innovation Max:", self.widgets['innovation_max'])

        self.widgets['weak_avgw'] = QDoubleSpinBox()
        self.widgets['weak_avgw'].setRange(0.0, 1.0)
        self.widgets['weak_avgw'].setDecimals(3)
        self.widgets['weak_avgw'].setValue(self.config.get('weak_avgw', GUI_DEFAULTS['weak_avgw']))
        kf_layout.addRow("Weak Average Weight:", self.widgets['weak_avgw'])

        layout.addRow(kf_group)

        plot_group = QGroupBox("Plotting")
        plot_layout = QFormLayout(plot_group)

        self.widgets['show_raw_position'] = QCheckBox()
        self.widgets['show_raw_position'].setChecked(self.config.get('show_raw_position', GUI_DEFAULTS['show_raw_position']))
        plot_layout.addRow("Show Raw Position:", self.widgets['show_raw_position'])

        layout.addRow(plot_group)

        debug_group = QGroupBox("Debug & Logging")
        debug_layout = QFormLayout(debug_group)

        self.widgets['stats_every'] = QSpinBox()
        self.widgets['stats_every'].setRange(1, 1000)
        self.widgets['stats_every'].setValue(self.config.get('stats_every', GUI_DEFAULTS['stats_every']))
        debug_layout.addRow("Stats Every N frames:", self.widgets['stats_every'])

        self.widgets['print_qual'] = QCheckBox()
        self.widgets['print_qual'].setChecked(self.config.get('print_qual', GUI_DEFAULTS['print_qual']))
        debug_layout.addRow("Print Quality Info:", self.widgets['print_qual'])

        self.widgets['debug_geom'] = QCheckBox()
        self.widgets['debug_geom'].setChecked(self.config.get('debug_geom', GUI_DEFAULTS['debug_geom']))
        debug_layout.addRow("Debug Geometry:", self.widgets['debug_geom'])

        self.widgets['bias_file'] = QLineEdit()
        self.widgets['bias_file'].setText(str(self.config.get('bias_file', GUI_DEFAULTS['bias_file'])))
        debug_layout.addRow("Bias File:", self.widgets['bias_file'])

        layout.addRow(debug_group)

        return widget

    def _create_llm_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)

        llm_group = QGroupBox("LLM Hybrid Tuning")
        llm_layout = QFormLayout(llm_group)

        self.widgets['llm_hybrid_tuning_enabled'] = QCheckBox()
        self.widgets['llm_hybrid_tuning_enabled'].setChecked(self.config.get('llm_hybrid_tuning_enabled', GUI_DEFAULTS['llm_hybrid_tuning_enabled']))
        llm_layout.addRow("Enable LLM Hybrid Tuning:", self.widgets['llm_hybrid_tuning_enabled'])

        self.widgets['llm_window_s'] = QDoubleSpinBox()
        self.widgets['llm_window_s'].setRange(1.0, 60.0)
        self.widgets['llm_window_s'].setValue(self.config.get('llm_window_s', GUI_DEFAULTS['llm_window_s']))
        llm_layout.addRow("Fast Loop Window (s):", self.widgets['llm_window_s'])

        self.widgets['slow_loop_s'] = QDoubleSpinBox()
        self.widgets['slow_loop_s'].setRange(10.0, 120.0)
        self.widgets['slow_loop_s'].setValue(self.config.get('slow_loop_s', GUI_DEFAULTS['slow_loop_s']))
        llm_layout.addRow("Slow Loop Interval (s):", self.widgets['slow_loop_s'])

        layout.addRow(llm_group)

        return widget

    def _create_mocap_tab(self):
        widget = QWidget()
        layout = QFormLayout(widget)

        mocap_group = QGroupBox("Motion Capture")
        mocap_layout = QFormLayout(mocap_group)

        self.widgets['mocap_enabled'] = QCheckBox()
        self.widgets['mocap_enabled'].setChecked(self.config.get('mocap_enabled', GUI_DEFAULTS['mocap_enabled']))
        mocap_layout.addRow("Enable Mocap:", self.widgets['mocap_enabled'])

        self.widgets['websocket_ip'] = QLineEdit()
        self.widgets['websocket_ip'].setText(str(self.config.get('websocket_ip', GUI_DEFAULTS['websocket_ip'])))
        mocap_layout.addRow("WebSocket IP:", self.widgets['websocket_ip'])

        self.widgets['websocket_port'] = QSpinBox()
        self.widgets['websocket_port'].setRange(1000, 65535)
        self.widgets['websocket_port'].setValue(self.config.get('websocket_port', GUI_DEFAULTS['websocket_port']))
        mocap_layout.addRow("WebSocket Port:", self.widgets['websocket_port'])

        self.widgets['mocap_object'] = QLineEdit()
        self.widgets['mocap_object'].setText(str(self.config.get('mocap_object', GUI_DEFAULTS['mocap_object'])))
        mocap_layout.addRow("Mocap Object:", self.widgets['mocap_object'])

        self.widgets['mocap_id'] = QSpinBox()
        self.widgets['mocap_id'].setRange(0, 100)
        self.widgets['mocap_id'].setValue(self.config.get('mocap_id', GUI_DEFAULTS['mocap_id']))
        mocap_layout.addRow("Mocap Object ID:", self.widgets['mocap_id'])

        layout.addRow(mocap_group)

        return widget

    def _load_config_file(self):
        """Load configuration from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration File",
            str(Path.cwd()),
            "YAML files (*.yaml *.yml);;JSON files (*.json);;All files (*)"
        )

        if file_path:
            try:
                config = load_config_file(file_path)
                if config:
                    self._update_widgets_from_config(config)
                    QMessageBox.information(self, "Success", f"Configuration loaded from {file_path}")
                else:
                    QMessageBox.warning(self, "Warning", "Configuration file is empty or invalid")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")

    def _update_widgets_from_config(self, config: Dict[str, Any]):
        """Update widget values from configuration"""
        for key, widget in self.widgets.items():
            if key in config:
                value = config[key]
                if isinstance(widget, QCheckBox):
                    widget.setChecked(bool(value))
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(float(value) if isinstance(widget, QDoubleSpinBox) else int(value))
                elif isinstance(widget, QLineEdit):
                    widget.setText(str(value))
                elif isinstance(widget, QTextEdit):
                    widget.setPlainText(str(value))

    def get_config(self) -> Dict[str, Any]:
        """Get configuration from widgets"""
        config = {}
        for key, widget in self.widgets.items():
            if isinstance(widget, QCheckBox):
                config[key] = widget.isChecked()
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                config[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                config[key] = widget.text().strip()
            elif isinstance(widget, QTextEdit):
                config[key] = widget.toPlainText().strip()
        return config


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()

        self.config = dict(GUI_DEFAULTS)
        self.worker = None
        self.thread = None
        self.tracking = False

        self.raw_positions = []
        self.kf_positions = []
        self.timestamps = []

        self.err_sum_kf = 0.0
        self.err_cnt_kf = 0
        self.err_sum_llm = 0.0
        self.err_cnt_llm = 0

        self.anchor_badges = {}
        self.anchor_badges_layout = None

        self._last_thresholds: Dict[str, float] = {}
        self._last_threshold_update: Optional[float] = None
        self._last_threshold_changes: Dict[str, str] = {}

        self._setup_ui()
        self.session = None
        self.pose_timer = None
        self._artists = {}
        self._with_truth = False
        self._load_initial_config()
        self._refresh_ports()
        self._update_plot()

        self.timer = QTimer()
        self.timer.timeout.connect(self._periodic_update)
        self.timer.start(100)

    def _setup_ui(self):
        """Setup the user interface"""
        self.setWindowTitle("UWB Tracker")
        self.setMinimumSize(1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        control_panel = self._create_control_panel()
        main_layout.addWidget(control_panel)

        status_panel = self._create_status_panel()
        main_layout.addWidget(status_panel)

        try:
            ch = control_panel.sizeHint().height()
            sh = status_panel.sizeHint().height()
            control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            status_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            control_panel.setMaximumHeight(ch)
            status_panel.setMaximumHeight(sh)
        except Exception:
            control_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            status_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        content_splitter = QSplitter(Qt.Horizontal)

        plot_widget = self._create_plot_widget()
        content_splitter.addWidget(plot_widget)

        info_widget = self._create_info_widget()
        content_splitter.addWidget(info_widget)

        content_splitter.setSizes([800, 400])
        main_layout.addWidget(content_splitter)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        self._create_menu_and_toolbar()

    def _create_control_panel(self):
        """Create the top control panel"""
        panel = QGroupBox("Control Panel")
        layout = QHBoxLayout(panel)

        layout.addWidget(QLabel("Port:"))
        self.port_combo = QComboBox()
        self.port_combo.setMinimumWidth(150)
        layout.addWidget(self.port_combo)

        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self._refresh_ports)
        layout.addWidget(refresh_btn)

        layout.addWidget(QLabel("|"))

        self.start_btn = QPushButton("▶️ Start Tracking")
        self.start_btn.clicked.connect(self._start_tracking)
        layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("⏹️ Stop")
        self.stop_btn.clicked.connect(self._stop_tracking)
        self.stop_btn.setEnabled(False)
        layout.addWidget(self.stop_btn)

        layout.addWidget(QLabel("|"))

        calib_btn = QPushButton("🔧 Calibrate")
        calib_btn.clicked.connect(self._open_calibration)
        layout.addWidget(calib_btn)

        settings_btn = QPushButton("⚙️ Settings")
        settings_btn.clicked.connect(self._open_settings)
        layout.addWidget(settings_btn)

        layout.addWidget(QLabel("|"))

        save_config_btn = QPushButton("💾 Save Config")
        save_config_btn.clicked.connect(self._save_config)
        layout.addWidget(save_config_btn)

        load_config_btn = QPushButton("📁 Load Config")
        load_config_btn.clicked.connect(self._load_config)
        layout.addWidget(load_config_btn)

        layout.addStretch()

        self.log_enabled_cb = QCheckBox("Enable Logging")
        self.log_enabled_cb.setChecked(self.config.get('logging_enabled', GUI_DEFAULTS['logging_enabled']))
        layout.addWidget(self.log_enabled_cb)

        return panel

    def _create_status_panel(self):
        """Create status information panel"""
        panel = QGroupBox("Status")
        layout = QGridLayout(panel)

        layout.addWidget(QLabel("True Position (Mocap):"), 0, 0)
        self.gt_pos_label = QLabel("n/a")
        layout.addWidget(self.gt_pos_label, 0, 1)

        layout.addWidget(QLabel("Used Anchors:"), 1, 0)
        self.anchor_badges_widget = QWidget()
        self.anchor_badges_layout = QHBoxLayout(self.anchor_badges_widget)
        self.anchor_badges_layout.setContentsMargins(0, 0, 0, 0)
        self.anchor_badges_layout.setSpacing(6)
        layout.addWidget(self.anchor_badges_widget, 1, 1)
        self._rebuild_anchor_badges()

        layout.addWidget(QLabel("Kalman Filter Position:"), 0, 2)
        self.kf_pos_label = QLabel("(-, -)")
        layout.addWidget(self.kf_pos_label, 0, 3)

        layout.addWidget(QLabel("Average Error (KF):"), 1, 2)
        self.avg_err_kf_label = QLabel("n/a")
        layout.addWidget(self.avg_err_kf_label, 1, 3)

        self.llm_pos_title = QLabel("LLM KF Position:")
        self.llm_pos_label = QLabel("(-, -)")
        layout.addWidget(self.llm_pos_title, 0, 4)
        layout.addWidget(self.llm_pos_label, 0, 5)
        self.llm_pos_title.setVisible(False)
        self.llm_pos_label.setVisible(False)

        self.avg_err_llm_title = QLabel("Average Error (LLM KF):")
        self.avg_err_llm_label = QLabel("n/a")
        layout.addWidget(self.avg_err_llm_title, 1, 4)
        layout.addWidget(self.avg_err_llm_label, 1, 5)
        self.avg_err_llm_title.setVisible(False)
        self.avg_err_llm_label.setVisible(False)

        return panel

    def _create_plot_widget(self):
        """Create matplotlib plot widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        self._plot_layout = layout

        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        self._plot_toolbar = NavigationToolbar(self.canvas, widget)

        layout.addWidget(self._plot_toolbar)
        layout.addWidget(self.canvas)

        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("UWB Tracking Visualization")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        self.raw_line, = self.ax.plot([], [], 'b-', alpha=0.5, linewidth=1, label='Raw Path')
        self.kf_line, = self.ax.plot([], [], 'r-', linewidth=2, label='Filtered Path')
        self.raw_point, = self.ax.plot([], [], 'bo', markersize=8, label='Raw Position')
        self.kf_point, = self.ax.plot([], [], 'ro', markersize=10, label='Filtered Position')
        self.anchor_points = {}
        self.range_circles = {}
        self.ax.legend()
        return widget

    def _create_info_widget(self):
        """Create information widget"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        live_console_splitter = QSplitter(Qt.Vertical)

        data_group = QGroupBox("Live Data")
        data_layout = QVBoxLayout(data_group)

        self.live_data = QTextEdit()
        self.live_data.setReadOnly(True)
        self.live_data.setFont(QFont("Consolas", 10))
        self.live_data.setLineWrapMode(QTextEdit.NoWrap)
        data_layout.addWidget(self.live_data)
        live_console_splitter.addWidget(data_group)

        console_group = QGroupBox("Console")
        console_layout = QVBoxLayout(console_group)

        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setFont(QFont("Consolas", 9))
        console_layout.addWidget(self.console)
        live_console_splitter.addWidget(console_group)

        live_console_splitter.setStretchFactor(0, 2)
        live_console_splitter.setStretchFactor(1, 1)
        live_console_splitter.setSizes([800, 400])

        layout.addWidget(live_console_splitter)
        return widget

    def _rebuild_anchor_badges(self):
        """Build/refresh anchor badges in Status from current config anchors."""
        if not self.anchor_badges_layout:
            return
        while self.anchor_badges_layout.count():
            item = self.anchor_badges_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
        self.anchor_badges.clear()
        try:
            anchors = parse_anchor_map(self.config.get('anchors', ''))
            for aid in sorted(anchors.keys()):
                lbl = QLabel(f"A{aid}")
                lbl.setStyleSheet(self._badge_style(False))
                lbl.setMargin(4)
                self.anchor_badges[aid] = lbl
                self.anchor_badges_layout.addWidget(lbl)
            self.anchor_badges_layout.addStretch()
        except Exception:
            pass

    def _set_anchor_usage(self, used_ids: Optional[List[int]]):
        """Update badge colors: green if used, red otherwise."""
        if not self.anchor_badges:
            return
        used = set(used_ids or [])
        for aid, lbl in self.anchor_badges.items():
            lbl.setStyleSheet(self._badge_style(aid in used))

    @staticmethod
    def _badge_style(is_used: bool) -> str:
        return (
            "QLabel {"
            f"background-color: {'#2ecc71' if is_used else '#e74c3c'};"
            "color: white; border-radius: 6px; padding: 2px 6px;"
            "font-weight: bold;"
            "}"
        )

    def _create_menu_and_toolbar(self):
        """Create menu bar and toolbar"""
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")

        load_action = QAction("Load Config...", self)
        load_action.triggered.connect(self._load_config)
        file_menu.addAction(load_action)

        save_action = QAction("Save Config...", self)
        save_action.triggered.connect(self._save_config)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        tools_menu = menubar.addMenu("Tools")

        settings_action = QAction("Settings...", self)
        settings_action.triggered.connect(self._open_settings)
        tools_menu.addAction(settings_action)

        calib_action = QAction("Calibration...", self)
        calib_action.triggered.connect(self._open_calibration)
        tools_menu.addAction(calib_action)

        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _refresh_ports(self):
        """Refresh available serial ports and add 'uwb_ws' option"""
        current_port = self.port_combo.currentText()
        self.port_combo.clear()

        ports = glob.glob('/dev/ttyACM*')
        ports.sort()

        self.port_combo.setEnabled(True)

        for port in ports:
            self.port_combo.addItem(port)

        self.port_combo.addItem("uwb_ws")

        if current_port in ports or current_port == "uwb_ws":
            self.port_combo.setCurrentText(current_port)

        self._log_console(f"Found {len(ports)} serial ports: {', '.join(ports) if ports else 'none'} (uwb_ws available)")

    def _load_initial_config(self):
        """Load initial configuration from default file"""
        config_path = Path(DEFAULT_CONFIG_FILE)
        if config_path.exists():
            try:
                loaded_config = load_config_file(str(config_path))
                if loaded_config:
                    self.config.update(loaded_config)
                    self._log_console(f"Loaded configuration from {config_path}")
                    self._update_anchor_info()
                else:
                    self._log_console(f"Configuration file {config_path} is empty, using defaults")
            except Exception as e:
                self._log_console(f"Failed to load {config_path}: {e}")
                QMessageBox.warning(self, "Configuration Error",
                                  f"Failed to load {config_path}:\n{str(e)}\n\nUsing default configuration.")
        else:
            self._log_console(f"No configuration file found at {config_path}, using defaults")
        try:
            self.log_enabled_cb.setChecked(self.config.get('logging_enabled', GUI_DEFAULTS['logging_enabled']))
        except Exception:
            pass

    def _update_anchor_info(self):
        """Update anchor information display"""
        try:
            self._rebuild_anchor_badges()
        except Exception as e:
            self._log_console(f"Anchor info update failed: {e}")

    def _update_plot(self):
        """Update the plot with current configuration (placeholder only if not tracking)"""
        if self.tracking or self.session:
            return

        self.ax.clear()

        self.ax.set_title("UWB Tracking Visualization")
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')

        try:
            anchors = parse_anchor_map(self.config.get('anchors', ''))

            xs = [pos[0] for pos in anchors.values()]
            ys = [pos[1] for pos in anchors.values()]

            if xs and ys:
                margin = 1.0
                self.ax.set_xlim(min(xs) - margin, max(xs) + margin)
                self.ax.set_ylim(min(ys) - margin, max(ys) + margin)

                for aid, pos in anchors.items():
                    self.ax.plot(pos[0], pos[1], '^', markersize=12, color='black',
                               markeredgecolor='white', markeredgewidth=2)
                    self.ax.text(pos[0], pos[1] + 0.1, f'A{aid}',
                               ha='center', va='bottom', fontweight='bold')
        except Exception as e:
            self._log_console(f"Error plotting anchors: {e}")

        self.raw_line, = self.ax.plot([], [], 'b-', alpha=0.5, linewidth=1, label='Raw Path')
        self.kf_line, = self.ax.plot([], [], 'r-', linewidth=2, label='Filtered Path')
        self.raw_point, = self.ax.plot([], [], 'bo', markersize=8, label='Raw Position')
        self.kf_point, = self.ax.plot([], [], 'ro', markersize=10, label='Filtered Position')

        self.ax.legend()
        self.canvas.draw()

    def _start_tracking(self):
        """Start UWB tracking via embedded CLI engine"""
        if self.tracking:
            return
        port = self.port_combo.currentText()
        if not port:
            QMessageBox.warning(self, "No Port", "Please select a valid port or 'uwb_ws'.")
            return

        config = dict(self.config)
        config['port'] = port
        config['no_plot'] = False
        config['logging_enabled'] = bool(self.log_enabled_cb.isChecked())
        self.config['logging_enabled'] = config['logging_enabled']

        if 'llm_hybrid_tuning_enabled' in config:
            config['llm_kf_enabled'] = config['hybrid_tuning_enabled'] = config['llm_hybrid_tuning_enabled']

        try:
            if self.session:
                try:
                    self.session['stop']()
                except Exception:
                    pass
                self.session = None

            self.session = start_tracking_embedded(config)
            if not self.session or not self.session.get('figure'):
                QMessageBox.critical(self, "Error", "Failed to start tracking (no figure/session).")
                return

            if self._plot_layout:
                while self._plot_layout.count():
                    item = self._plot_layout.takeAt(0)
                    w = item.widget()
                    if w:
                        w.setParent(None)

                self.figure = self.session['figure']
                self.canvas = FigureCanvas(self.figure)
                self._plot_toolbar = NavigationToolbar(self.canvas, self)
                self._plot_layout.addWidget(self._plot_toolbar)
                self._plot_layout.addWidget(self.canvas)

            self._artists = self.session.get('artists', {})
            self._with_truth = bool(self.session.get('with_truth', False))

            if self.pose_timer:
                self.pose_timer.stop()
            self.pose_timer = QTimer(self)
            self.pose_timer.timeout.connect(self._drain_pose_queue_and_draw)
            self.pose_timer.start(30)

            self.raw_positions.clear()
            self.kf_positions.clear()
            self.timestamps.clear()

            self.err_sum_kf = 0.0
            self.err_cnt_kf = 0
            self.err_sum_llm = 0.0
            self.err_cnt_llm = 0
            self.avg_err_kf_label.setText("n/a")
            self.avg_err_llm_label.setText("n/a")

            llm_enabled_cfg = bool(self.config.get('llm_hybrid_tuning_enabled', False))
            self.llm_pos_title.setVisible(llm_enabled_cfg)
            self.llm_pos_label.setVisible(llm_enabled_cfg)
            self.avg_err_llm_title.setVisible(llm_enabled_cfg)
            self.avg_err_llm_label.setVisible(llm_enabled_cfg)

            self._rebuild_anchor_badges()

            self.tracking = True
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self._log_console(f"Started tracking on {port} (embedded engine)")
            self.status_bar.showMessage(f"Tracking on {port}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start tracking:\n{str(e)}")

    def _drain_pose_queue_and_draw(self):
        """Drain pose queue from embedded engine and update UI + plot (Qt thread)"""
        if not self.session:
            return
        q = self.session.get('pose_queue')
        if not q:
            return

        pkt = None
        try:
            while True:
                pkt = q.get_nowait()
        except Exception:
            pass
        if not pkt:
            return

        x_raw = pkt.get('x_raw'); y_raw = pkt.get('y_raw')
        x_kf_base = pkt.get('x_kf_base', pkt.get('x_kf')); y_kf_base = pkt.get('y_kf_base', pkt.get('y_kf'))
        x_kf_llm = pkt.get('x_kf_llm'); y_kf_llm = pkt.get('y_kf_llm')
        llm_enabled_pkt = bool(pkt.get('llm_enabled', False))
        used = pkt.get('used_ids')
        seq = pkt.get('seq')
        gt_x = pkt.get('gt_x'); gt_y = pkt.get('gt_y')
        t_sec = pkt.get('t')
        mode = pkt.get('mode', 'baseline')
        pv = pkt.get('pv', None)
        rv = pkt.get('rv', None)
        inno_max = pkt.get('inno_max', None)
        win_id = pkt.get('win_id', None)
        thresholds = pkt.get('thresholds', {})
        hybrid_enabled = pkt.get('hybrid_tuning_enabled', False)
        last_threshold_update = pkt.get('last_threshold_update', None)
        explanation = pkt.get('explanation')

        self.llm_pos_title.setVisible(llm_enabled_pkt)
        self.llm_pos_label.setVisible(llm_enabled_pkt)
        self.avg_err_llm_title.setVisible(llm_enabled_pkt)
        self.avg_err_llm_label.setVisible(llm_enabled_pkt)

        if gt_x is not None and gt_y is not None:
            self.gt_pos_label.setText(f"({gt_x:.3f}, {gt_y:.3f})")
        else:
            self.gt_pos_label.setText("n/a")

        if x_kf_base is not None and y_kf_base is not None:
            self.kf_pos_label.setText(f"({x_kf_base:.3f}, {y_kf_base:.3f})")
        else:
            self.kf_pos_label.setText("(-, -)")
        if llm_enabled_pkt and x_kf_llm is not None and y_kf_llm is not None:
            self.llm_pos_label.setText(f"({x_kf_llm:.3f}, {y_kf_llm:.3f})")
        elif not llm_enabled_pkt:
            self.llm_pos_label.setText("n/a")

        if used is not None:
            self._set_anchor_usage(used)

        try:
            if thresholds:
                if last_threshold_update is not None and last_threshold_update != self._last_threshold_update:
                    new_changes: Dict[str, str] = {}
                    for key, cur_val in thresholds.items():
                        try:
                            cur = float(cur_val)
                        except Exception:
                            new_changes[key] = ""
                            continue
                        prev_val = self._last_thresholds.get(key, None)
                        if prev_val is None:
                            new_changes[key] = ""
                        else:
                            if cur > float(prev_val):
                                new_changes[key] = " (+)"
                            elif cur < float(prev_val):
                                new_changes[key] = " (-)"
                            else:
                                new_changes[key] = ""
                    self._last_threshold_changes = new_changes
                    try:
                        self._last_thresholds = {k: float(v) for k, v in thresholds.items()}
                    except Exception:
                        self._last_thresholds = dict(thresholds)
                    self._last_threshold_update = last_threshold_update
        except Exception:
            pass

        parts = []

        if isinstance(seq, int):
            parts.append("• Sequence")
            parts.append(f"  {seq:06d}")
        if isinstance(t_sec, (int, float)):
            dt = time.localtime(int(t_sec))
            dt_str = time.strftime("%Y-%m-%d %H:%M:%S", dt)
            parts.append("• Timestamp")
            parts.append(f"  {dt_str}")

        parts.append("")

        parts.append("• Positions")
        if x_raw is not None and y_raw is not None:
            parts.append(f"  - Raw: ({x_raw:.3f}, {y_raw:.3f})")
        else:
            parts.append("  - Raw: n/a")
        if x_kf_base is not None and y_kf_base is not None:
            parts.append(f"  - Kalman Filter: ({x_kf_base:.3f}, {y_kf_base:.3f})")
        else:
            parts.append("  - Kalman Filter: n/a")
        if llm_enabled_pkt:
            if x_kf_llm is not None and y_kf_llm is not None:
                parts.append(f"  - Adaptive KF: ({x_kf_llm:.3f}, {y_kf_llm:.3f})")
            else:
                parts.append("  - Adaptive KF: n/a")

        parts.append("")

        parts.append("• Errors (vs Ground Truth)")
        have_gt = (gt_x is not None and gt_y is not None)
        if have_gt:
            try:
                if x_raw is not None and y_raw is not None:
                    dxr = x_raw - gt_x; dyr = y_raw - gt_y
                    er = float(np.hypot(dxr, dyr))
                    parts.append(f"  - Raw: v=({dxr:.3f},{dyr:.3f}), |e|={er:.3f}")
                else:
                    parts.append("  - Raw: n/a")

                if x_kf_base is not None and y_kf_base is not None:
                    dxb = x_kf_base - gt_x; dyb = y_kf_base - gt_y
                    eb = float(np.hypot(dxb, dyb))
                    parts.append(f"  - KF: v=({dxb:.3f},{dyb:.3f}), |e|={eb:.3f}")
                    self.err_sum_kf += eb; self.err_cnt_kf += 1
                    self.avg_err_kf_label.setText(f"{(self.err_sum_kf/self.err_cnt_kf):.3f} m")
                else:
                    parts.append("  - KF: n/a")

                if llm_enabled_pkt:
                    if x_kf_llm is not None and y_kf_llm is not None:
                        dxl = x_kf_llm - gt_x; dyl = y_kf_llm - gt_y
                        el = float(np.hypot(dxl, dyl))
                        parts.append(f"  - Adaptive KF: v=({dxl:.3f},{dyl:.3f}), |e|={el:.3f}")
                        self.err_sum_llm += el; self.err_cnt_llm += 1
                        self.avg_err_llm_label.setText(f"{(self.err_sum_llm/self.err_cnt_llm):.3f} m")
                    else:
                        parts.append("  - Adaptive KF: n/a")
            except Exception:
                parts.append("  - (error computing distances)")
        else:
            parts.append("  - Raw: n/a")
            parts.append("  - KF: n/a")
            if llm_enabled_pkt:
                parts.append("  - Adaptive KF: n/a")

        parts.append("")

        parts.append("• Adaptive KF Settings")
        parts.append(f"  - Mode: {str(mode)}")
        if pv is not None and rv is not None:
            parts.append(f"  - Process Var: {float(pv):.3f}")
            parts.append(f"  - Meas Var: {float(rv):.3f}")
        else:
            parts.append("  - Process Var: n/a")
            parts.append("  - Meas Var: n/a")
        parts.append(f"  - Innovation Max: {float(inno_max):.3f}" if inno_max is not None else "  - Innovation Max: n/a")
        parts.append(f"  - Window ID: {int(win_id)}" if isinstance(win_id, int) else "  - Window ID: n/a")

        parts.append("")

        parts.append("• Adaptive KF with LLM Tuning")
        parts.append(f"  - Status: {'Enabled' if hybrid_enabled else 'Disabled'}")

        if thresholds:
            parts.append("")
            parts.append("• Fast Loop Thresholds (5s)")

            parts.append("  - Movement Detection:")
            get_mark = lambda k: self._last_threshold_changes.get(k, "")
            parts.append(f"    * hard_enter_p90: {thresholds.get('hard_enter_p90', 'n/a')}{get_mark('hard_enter_p90')}")
            parts.append(f"    * soft_mean: {thresholds.get('soft_mean', 'n/a')}{get_mark('soft_mean')}")
            parts.append(f"    * soft_innov: {thresholds.get('soft_innov', 'n/a')}{get_mark('soft_innov')}")
            parts.append(f"    * soft_good: {thresholds.get('soft_good', 'n/a')}{get_mark('soft_good')}")

            parts.append("  - Static Detection:")
            parts.append(f"    * quiet_p90: {thresholds.get('quiet_p90', 'n/a')}{get_mark('quiet_p90')}")
            parts.append(f"    * quiet_innov: {thresholds.get('quiet_innov', 'n/a')}{get_mark('quiet_innov')}")
            parts.append(f"    * static_streak: {thresholds.get('static_streak', 'n/a')}{get_mark('static_streak')}")
            parts.append(f"    * moving_streak: {thresholds.get('moving_streak', 'n/a')}{get_mark('moving_streak')}")

            parts.append("")
            parts.append("• Slow Loop Adjustments (30s)")

            if last_threshold_update:
                time_ago = time.time() - last_threshold_update
                parts.append(f"  - Last Updated: {time_ago:.1f}s ago")
            else:
                parts.append("  - Last Updated: never")

            if explanation:
                parts.append("  - Adjustment Rationale:")
                import textwrap
                wrapped_explanation = textwrap.wrap(explanation, width=65)
                for line in wrapped_explanation:
                    parts.append(f"    * {line}")
        else:
            parts.append("  - No threshold data available")

        parts.append("")

        parts.append("• System Info")
        if used is not None:
            parts.append(f"  - Used Anchors: {used}")
        else:
            parts.append("  - Used Anchors: n/a")

        self._set_text_preserve_scroll(self.live_data, "\n".join(parts))

        fig = self.session.get('figure')
        art = self._artists
        if not fig or not art:
            return
        try:
            process_pose_packet(fig, art, pkt, self._with_truth)
            self.canvas.draw_idle()
        except Exception as e:
            self._log_console(f"Plot update error: {e}")

    def _set_text_preserve_scroll(self, text_edit: QTextEdit, text: str):
        """
        Update text_edit plain text while preserving vertical and horizontal scroll
        positions.
        """
        v_sb = text_edit.verticalScrollBar()
        h_sb = text_edit.horizontalScrollBar()

        at_bottom = v_sb.value() >= (v_sb.maximum() - 3)
        at_right = h_sb.value() >= (h_sb.maximum() - 3)
        prev_v = v_sb.value()
        prev_h = h_sb.value()

        text_edit.blockSignals(True)
        text_edit.setPlainText(text)
        text_edit.blockSignals(False)

        if at_bottom:
            v_sb.setValue(v_sb.maximum())
        else:
            v_sb.setValue(min(prev_v, v_sb.maximum()))
        if at_right:
            h_sb.setValue(h_sb.maximum())
        else:
            h_sb.setValue(min(prev_h, h_sb.maximum()))

    def _stop_tracking(self):
        """Stop UWB tracking (embedded engine)"""
        if not self.tracking:
            return

        if self.pose_timer:
            try:
                self.pose_timer.stop()
            except Exception:
                pass
            self.pose_timer = None

        if self.session:
            try:
                stop_cb = self.session.get('stop')
                if callable(stop_cb):
                    stop_cb()
            except Exception:
                pass
            self.session = None

        self.tracking = False
        try:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
        except Exception:
            pass
        self._log_console("Stopping tracking...")
        try:
            self.status_bar.showMessage("Ready")
        except Exception:
            pass

    def _open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self, self.config)
        if dialog.exec() == QDialog.Accepted:
            self.config.update(dialog.get_config())
            self._update_anchor_info()
            self._update_plot()
            self._rebuild_anchor_badges()
            try:
                self.log_enabled_cb.setChecked(self.config.get('logging_enabled', GUI_DEFAULTS['logging_enabled']))
            except Exception:
                pass
            self._log_console("Settings updated")

    def _open_calibration(self):
        """Open calibration dialog"""
        if self.tracking:
            QMessageBox.warning(self, "Tracking Active",
                              "Please stop tracking before starting calibration.")
            return

        port = self.port_combo.currentText()
        if not port:
            QMessageBox.warning(self, "No Port", "Please select a valid port or 'uwb_ws'.")
            return

        try:
            config = dict(self.config)
            config['port'] = port

            dialog = CalibrationDialog(self, config)
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Calibration Error", f"Failed to start calibration:\n{str(e)}")

    def _save_config(self):
        """Save configuration to file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "uwbtrack_config.yaml",
            "YAML files (*.yaml *.yml);;JSON files (*.json);;All files (*)"
        )

        if file_path:
            try:
                if file_path.lower().endswith('.json'):
                    with open(file_path, 'w') as f:
                        json.dump(self.config, f, indent=2)
                else:
                    try:
                        import yaml
                        with open(file_path, 'w') as f:
                            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                    except ImportError:
                        with open(file_path, 'w') as f:
                            f.write("# UWB Tracker Configuration\n")
                            for key, value in self.config.items():
                                f.write(f"{key}: {value}\n")

                self._log_console(f"Configuration saved to {file_path}")
                QMessageBox.information(self, "Success", f"Configuration saved to {file_path}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save configuration:\n{str(e)}")

    def _load_config(self):
        """Load configuration from file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            str(Path.cwd()),
            "YAML files (*.yaml *.yml);;JSON files (*.json);;All files (*)"
        )

        if file_path:
            try:
                loaded_config = load_config_file(file_path)
                if loaded_config:
                    self.config.update(loaded_config)
                    self._update_anchor_info()
                    self._update_plot()
                    try:
                        self.log_enabled_cb.setChecked(self.config.get('logging_enabled', GUI_DEFAULTS['logging_enabled']))
                    except Exception:
                        pass
                    self._log_console(f"Configuration loaded from {file_path}")
                    QMessageBox.information(self, "Success", f"Configuration loaded from {file_path}")
                else:
                    QMessageBox.warning(self, "Warning", "Configuration file is empty or invalid")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load configuration:\n{str(e)}")

    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(self, "About UWB Tracker",
                         "UWB Tracker GUI\n\n"
                         "A PySide6-based graphical interface for UWB positioning.\n\n"
                         "Features:\n"
                         "• Real-time tracking visualization\n"
                         "• Interactive calibration\n"
                         "• Configuration management\n"
                         "• Live data monitoring")

    def _log_console(self, message):
        """Log message to console"""
        timestamp = time.strftime("%H:%M:%S")
        self.console.append(f"[{timestamp}] {message}")

        scrollbar = self.console.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _periodic_update(self):
        """Periodic update function"""
        pass

    def closeEvent(self, event):
        """Handle application close"""
        if self.tracking:
            reply = QMessageBox.question(self, "Tracking Active",
                                         "Tracking is currently active. Stop tracking and exit?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self._stop_tracking()
                event.accept()
            else:
                event.ignore()
        else:
            if self.session:
                try:
                    self.session['stop']()
                except Exception:
                    pass
            event.accept()


def main():
    """Main entry point"""
    app = QApplication(sys.argv)
    app.setApplicationName("UWB Tracker")
    app.setApplicationVersion("1.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
