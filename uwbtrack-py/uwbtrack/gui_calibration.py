"""
GUI Calibration Dialog for UWB Tracking - Visual wrapper around CLI calibration
"""

import time
import json
import queue
import threading
import sys
import io
from pathlib import Path
from typing import Dict, Optional, Any, List
import numpy as np

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QGroupBox, QWidget, QMessageBox, QSplitter
)
from PySide6.QtCore import QObject, Signal, QThread, QTimer, Qt
from PySide6.QtGui import QFont

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from .serial_io import init_serial_connection
from .weights import WEIGHT_CFG
from .geom import parse_anchor_map
from .calibration import calibrate_quality_near_anchors, calibrate_affine_midpoints


class OutputCapture:
    """Captures stdout/stderr and emits to GUI"""

    def __init__(self, signal):
        self.signal = signal
        # keep partial fragment until newline arrives
        self._partial = ""

    def write(self, text):
        # accumulate fragments and only emit full lines (split on '\n')
        if not text:
            return
        self._partial += text
        if '\n' in self._partial:
            parts = self._partial.split('\n')
            for line in parts[:-1]:
                # emit each complete line (without trailing newline)
                try:
                    self.signal.emit(line)
                except Exception:
                    pass
            # keep the last unfinished fragment
            self._partial = parts[-1]

    def flush(self):
        pass


class CalibrationWorker(QObject):
    """Worker that runs the actual calibration functions with output capture"""

    log_message = Signal(str)
    calibration_finished = Signal(dict, dict)
    error_occurred = Signal(str)
    input_requested = Signal(str)
    step_changed = Signal(str, int, list)  # step_type, current_index, highlight_anchors

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.input_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True
        try:
            self.response_queue.put("")
        except:
            pass

    def provide_input(self, response: str):
        """Called by GUI to provide user input"""
        self.response_queue.put(response)

    def run(self):
        """Run calibration with output capture"""
        quality_results = {}
        range_results = {}

        # Capture stdout/stderr
        stdout_capture = OutputCapture(self.log_message)
        stderr_capture = OutputCapture(self.log_message)
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        try:
            anchors_xyz = parse_anchor_map(self.config.get('anchors', ''))
            port = self.config.get('port')
            baud = self.config.get('baud', 115200)

            if not port:
                self.error_occurred.emit("No serial port configured")
                return

            if len(anchors_xyz) < 3:
                self.error_occurred.emit("Need at least 3 anchors for calibration")
                return

            # Redirect output
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            # Monkey patch input() to use our handler
            import builtins
            original_input = builtins.input
            builtins.input = self._gui_input

            try:
                if str(port).lower() == "uwb_ws":
                    from .uwb_ws_io import init_ws_connection
                    ser = init_ws_connection(str(self.config.get('uwb_ws_url', 'ws://10.131.128.4:8766/')), timeout=0.2)
                else:
                    ser = init_serial_connection(port, baud)

                if not self._stop_requested:
                    self.log_message.emit("=== Starting Quality Calibration ===")
                    use_floor = bool(self.config.get('solve_2d', False))
                    floor_z = float(self.config.get('floor_z', 0.0))

                    quality_results = self._run_quality_calibration(
                        ser, anchors_xyz, floor_z=floor_z, use_floor=use_floor
                    )

                if not self._stop_requested:
                    self.log_message.emit("\n=== Starting Range Calibration ===")
                    use_floor = bool(self.config.get('solve_2d', False))
                    floor_z = float(self.config.get('floor_z', 0.0))

                    range_results = self._run_range_calibration(
                        ser, anchors_xyz, floor_z=floor_z, use_floor=use_floor
                    )

                ser.close()

            finally:
                # Restore original input and output
                builtins.input = original_input
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            if not self._stop_requested:
                self.calibration_finished.emit(quality_results, range_results)

        except Exception as e:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            self.error_occurred.emit(f"Calibration error: {str(e)}")

    def _gui_input(self, prompt: str = "") -> str:
        """Replacement for input() that goes through GUI"""
        if self._stop_requested:
            return ""

        self.input_requested.emit(prompt)
        return self.response_queue.get()

    def _run_quality_calibration(self, ser, anchors_xyz, floor_z=0.0, use_floor=False):
        """Run quality calibration with step visualization"""
        from .serial_io import serial_lines
        from .weights import meas_diag
        from collections import defaultdict

        ids_all = sorted(anchors_xyz)
        ser_iter = serial_lines(ser)

        print("\nQuality calibration (diagnostics near each anchor).")
        print("Interactive controls:")
        print("  Enter = measure / accept")
        print("  r     = redo current anchor after measurement")
        print("  b     = go back to previous anchor (overwrite)")
        print()

        anchor_diags = {aid: None for aid in ids_all}

        i = 0
        while i < len(ids_all) and not self._stop_requested:
            aid = ids_all[i]

            # Signal step change for visualization
            self.step_changed.emit("quality", i, [aid])

            ax, ay = anchors_xyz[aid][0], anchors_xyz[aid][1]
            if use_floor:
                base_prompt = f"[{i+1}/{len(ids_all)}] Place TAG under ANCHOR A{aid} (~{ax:.2f},{ay:.2f}, z={floor_z:.2f}) and press Enter (b=back)... "
            else:
                base_prompt = f"[{i+1}/{len(ids_all)}] Place TAG near ANCHOR A{aid} (≤0.3 m) and press Enter (b=back)... "

            cmd = self._gui_input(base_prompt).strip().lower()
            if cmd == 'b':
                if i > 0:
                    i -= 1
                    print("Going back to previous anchor.")
                else:
                    print("Already at first anchor.")
                continue

            print("Waiting 7s for raw data to settle...")
            time.sleep(7)
            print(f"Collecting samples for A{aid}...")

            try: ser.reset_input_buffer()
            except Exception: pass
            try: ser.write(b"S"); ser.flush()
            except Exception: pass
            time.sleep(0.10)

            got = []
            sample_per_anchor = 50
            timeout_s = 10.0
            start = time.time()

            while len(got) < sample_per_anchor and (time.time() - start) < timeout_s and not self._stop_requested:
                try:
                    line = next(ser_iter)
                except StopIteration:
                    break
                from .serial_io import parse_meas
                m = parse_meas(line)
                if not m or m["aid"] != aid:
                    continue
                d = meas_diag(m)
                got.append(d)

            n = len(got)
            if n == 0:
                print(f"  !! No diagnostics collected for A{aid}")
            else:
                snr_med = np.median([g['snr_db'] for g in got])
                rpc_med = np.median([g['rpc'] for g in got])
                print(f"  A{aid}: collected {n} samples (SNR p50 ~ {snr_med:.1f} dB, RXPACC p50 ~ {rpc_med:.0f})")

            while not self._stop_requested:
                decision = self._gui_input("Accept? (Enter=accept, r=redo, b=back) ").strip().lower()
                if decision in ('', 'enter'):
                    anchor_diags[aid] = got
                    i += 1
                    break
                elif decision == 'r':
                    print("Redoing this anchor.")
                    anchor_diags[aid] = None
                    break
                elif decision == 'b':
                    if i > 0:
                        print("Going back.")
                        i -= 1
                        break
                    else:
                        print("Already at first anchor.")
                else:
                    print("Unrecognized input.")

        # Compute results
        snr_db_all = []
        rpc_all = []
        ci_ppm_all = []
        for arr in anchor_diags.values():
            if not arr:
                continue
            snr_db_all.extend([g["snr_db"] for g in arr])
            rpc_all.extend([g["rpc"] for g in arr])
            ci_ppm_all.extend([abs(g["ci_ppm"]) for g in arr])

        if snr_db_all and rpc_all:
            snr_arr = np.array(snr_db_all, float)
            rpc_arr = np.array(rpc_all, float)
            ci_arr = np.array(ci_ppm_all, float) if ci_ppm_all else np.array([WEIGHT_CFG["ci_halfppm"]], float)
            snr_min = float(np.percentile(snr_arr, 10))
            snr_max = float(np.percentile(snr_arr, 90))
            rpc_norm = float(np.percentile(rpc_arr, 90))
            ci_halfppm = float(max(3.0, np.percentile(ci_arr, 75)))
            print(f"\nQuality mapping suggestion:"
                  f"\n  snr_min≈{snr_min:.1f} dB, snr_max≈{snr_max:.1f} dB,"
                  f"\n  rpc_norm≈{rpc_norm:.0f}, ci_halfppm≈{ci_halfppm:.1f}")
            return {"snr_min": snr_min, "snr_max": snr_max,
                    "rpc_norm": max(64.0, rpc_norm), "ci_halfppm": ci_halfppm}
        else:
            print("\n!! Quality calibration failed or aborted; using defaults.")
            return dict(WEIGHT_CFG)

    def _run_range_calibration(self, ser, anchors_xyz, floor_z=0.0, use_floor=False):
        """Run range calibration with step visualization"""
        from .serial_io import serial_lines, parse_meas
        from collections import defaultdict

        ids_all = sorted(anchors_xyz)
        if len(ids_all) < 3:
            print("Need ≥3 anchors for midpoint calibration.", file=sys.stderr)
            return {aid: (1.0, 0.0) for aid in ids_all}

        a1, a2, a3 = ids_all[:3]

        def _xy(p): return (p[0], p[1])
        p12_xy = tuple((np.array(_xy(anchors_xyz[a1])) + np.array(_xy(anchors_xyz[a2]))) / 2.0)
        p13_xy = tuple((np.array(_xy(anchors_xyz[a1])) + np.array(_xy(anchors_xyz[a3]))) / 2.0)
        p23_xy = tuple((np.array(_xy(anchors_xyz[a2])) + np.array(_xy(anchors_xyz[a3]))) / 2.0)

        if use_floor:
            poses = [("mid(A{},A{})".format(a1, a2), (p12_xy[0], p12_xy[1], floor_z), [a1, a2]),
                     ("mid(A{},A{})".format(a1, a3), (p13_xy[0], p13_xy[1], floor_z), [a1, a3]),
                     ("mid(A{},A{})".format(a2, a3), (p23_xy[0], p23_xy[1], floor_z), [a2, a3])]
        else:
            poses = [("mid(A{},A{})".format(a1, a2), p12_xy, [a1, a2]),
                     ("mid(A{},A{})".format(a1, a3), p13_xy, [a1, a3]),
                     ("mid(A{},A{})".format(a2, a3), p23_xy, [a2, a3])]

        print("\nAffine calibration (range): three poses at anchor midpoints:")
        for name, pos, _ in poses:
            if len(pos) == 3:
                print(f"  {name}: floor at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
            else:
                print(f"  {name}: approx at ({pos[0]:.3f}, {pos[1]:.3f}) m")

        print("\nInteractive controls during this calibration:")
        print("  Enter = continue / accept")
        print("  r     = redo this pose (after it is measured)")
        print("  b     = go back to previous pose")
        print(f"\nTarget: ~50 samples per anchor per pose.\n")

        ser_iter = serial_lines(ser)
        meas_means = {aid: [None]*len(poses) for aid in ids_all}
        true_dists = {aid: [None]*len(poses) for aid in ids_all}

        def _pos_vec(p):
            return np.array((p[0], p[1], (p[2] if len(p) == 3 else 0.0)), float)

        anchors_vec = {aid: np.array((v[0], v[1], (v[2] if len(v) == 3 else 0.0)), float)
                       for aid, v in anchors_xyz.items()}

        i = 0
        while i < len(poses) and not self._stop_requested:
            pose_name, pos, highlight_aids = poses[i]

            # Signal step change for visualization
            self.step_changed.emit("range", i, highlight_aids)

            cmd = self._gui_input(f"[{i+1}/{len(poses)}] Ready for pose '{pose_name}'. Place tag and press Enter (b=back)... ").strip().lower()
            if cmd == 'b':
                if i > 0:
                    i -= 1
                    print("Going back to previous pose.")
                    continue
                else:
                    print("Already at first pose.")
                    continue

            print("Waiting 7s for raw data to settle...")
            time.sleep(7)
            print(f"  Collecting samples at {pose_name}...")

            pos_vec = _pos_vec(pos)
            for aid in ids_all:
                true_dists[aid][i] = float(np.linalg.norm(pos_vec - anchors_vec[aid]))

            try: ser.reset_input_buffer()
            except Exception: pass
            try: ser.write(b"S"); ser.flush()
            except Exception: pass
            time.sleep(0.10)

            got = defaultdict(list)
            sample_per_anchor = 50
            timeout_s = 10.0
            start = time.time()

            while (time.time() - start) < timeout_s and not self._stop_requested:
                if all(len(got[aid]) >= sample_per_anchor for aid in ids_all):
                    break
                try:
                    line = next(ser_iter)
                except StopIteration:
                    break
                m = parse_meas(line)
                if not m:
                    continue
                aid = m["aid"]
                if aid in meas_means:
                    got[aid].append(m["range"])

            min_n = min((len(got[aid]) for aid in ids_all), default=0)
            for aid in ids_all:
                arr = np.asarray(got[aid], float)
                mean_distance = float(np.mean(arr)) if arr.size else 0.0
                meas_means[aid][i] = mean_distance
                print(f"  A{aid} at {pose_name}: measured ≈ {mean_distance:.3f} m (N={arr.size})")

            while not self._stop_requested:
                if min_n >= sample_per_anchor or min_n == 0:
                    decision = self._gui_input("Accept this pose? (Enter=accept, r=redo, b=back) ").strip().lower()
                else:
                    decision = self._gui_input(f"Only {min_n} samples (target {sample_per_anchor}). (Enter=accept, r=redo, b=back) ").strip().lower()

                if decision in ('', 'enter'):
                    i += 1
                    break
                elif decision == 'r':
                    print("Redoing this pose.")
                    for aid in ids_all:
                        meas_means[aid][i] = None
                        true_dists[aid][i] = None
                    break
                elif decision == 'b':
                    if i > 0:
                        print("Going back.")
                        i -= 1
                        break
                    else:
                        print("Already at first pose; cannot go back.")
                else:
                    print("Unrecognized input.")

        # Compute calibration
        valid_indices = [k for k in range(len(poses)) if all(meas_means[aid][k] is not None for aid in ids_all)]
        if len(valid_indices) < 2:
            print("Not enough valid midpoint poses; falling back to identity calibration.")
            return {aid: (1.0, 0.0) for aid in ids_all}

        calib_params = {}
        for aid in ids_all:
            d_meas = np.asarray([meas_means[aid][k] for k in valid_indices], float)
            d_true = np.asarray([true_dists[aid][k] for k in valid_indices], float)
            A = np.c_[d_meas, np.ones_like(d_meas)]
            try:
                alpha, beta = np.linalg.lstsq(A, d_true, rcond=None)[0]
            except Exception:
                alpha, beta = 1.0, 0.0
            alpha = float(np.clip(alpha, 0.9, 1.1))
            beta = float(np.clip(beta, -0.5, 0.5))
            calib_params[aid] = (alpha, beta)
            print(f"  A{aid}: d_true ≈ {alpha:.4f} * d_meas + {beta:.4f}")

        print("\nRange calibration complete.")
        return calib_params


class CalibrationDialog(QDialog):
    """Visual calibration wizard dialog"""

    def __init__(self, parent, config: Dict):
        super().__init__(parent)
        self.setWindowTitle("UWB Calibration Wizard")
        self.setModal(True)
        # make overall window slightly smaller and compact
        self.resize(960, 520)

        self.config = config
        self.anchors = parse_anchor_map(config.get('anchors', ''))

        self.worker = None
        self.worker_thread = None
        self.current_step = ""
        self.current_highlights = []

        self._setup_ui()
        self._setup_plot()

    def _setup_ui(self):
        """Setup the UI with split layout"""
        layout = QHBoxLayout(self)
        # tighten outer margins and spacing (less top padding)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(6)

        # Create splitter
        splitter = QSplitter()
        layout.addWidget(splitter)

        # Left side - matplotlib plot
        self.plot_widget = QWidget()
        # keep anchor layout compact but flexible
        self.plot_widget.setMinimumWidth(340)
        plot_layout = QVBoxLayout(self.plot_widget)
        plot_layout.setContentsMargins(4, 4, 4, 4)
        plot_layout.setSpacing(4)

        # canvas: let it expand but provide a sensible minimum
        self.figure = Figure(figsize=(4, 4), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumSize(320, 320)
        plot_layout.addWidget(self.canvas)

        splitter.addWidget(self.plot_widget)

        # Right side - controls
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        # tighter right-side margins so header is closer to content
        right_layout.setContentsMargins(6, 4, 6, 6)
        right_layout.setSpacing(6)

        # Header
        header = QLabel("UWB Calibration Wizard")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(header)

        # Current step
        step_group = QGroupBox("Current Step")
        step_layout = QVBoxLayout(step_group)
        # bring content closer to the title and remove extra vertical gap
        step_group.setContentsMargins(6, 2, 6, 6)
        step_layout.setSpacing(6)

        self.step_label = QLabel("Ready to start calibration")
        self.step_label.setWordWrap(True)
        step_layout.addWidget(self.step_label)

        # Buttons
        button_widget = QWidget()
        button_layout = QHBoxLayout(button_widget)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        self.continue_btn = QPushButton("Continue / Accept")
        self.continue_btn.clicked.connect(lambda: self._send_response(""))
        self.continue_btn.setEnabled(False)
        # reduced height to ~2/3 of previous (≈40px) and consistent font
        self.continue_btn.setFixedHeight(40)
        self.continue_btn.setStyleSheet("font-size:13px;")
        button_layout.addWidget(self.continue_btn)

        self.redo_btn = QPushButton("Redo (r)")
        self.redo_btn.clicked.connect(lambda: self._send_response("r"))
        self.redo_btn.setEnabled(False)
        self.redo_btn.setFixedHeight(40)
        self.redo_btn.setStyleSheet("font-size:13px;")
        button_layout.addWidget(self.redo_btn)

        self.back_btn = QPushButton("Back (b)")
        self.back_btn.clicked.connect(lambda: self._send_response("b"))
        self.back_btn.setEnabled(False)
        self.back_btn.setFixedHeight(40)
        self.back_btn.setStyleSheet("font-size:13px;")
        button_layout.addWidget(self.back_btn)

        step_layout.addWidget(button_widget)
        right_layout.addWidget(step_group)

        # Status/Log area — show only the latest message as selectable text
        log_group = QGroupBox("Current Message")
        log_layout = QVBoxLayout(log_group)
        log_group.setContentsMargins(6, 4, 6, 6)

        self.status_label = QLabel("")
        self.status_label.setFont(QFont("Consolas", 11))
        self.status_label.setWordWrap(True)
        # Allow selecting/copying the displayed text
        self.status_label.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.status_label.setMinimumHeight(80)
        log_layout.addWidget(self.status_label)

        right_layout.addWidget(log_group)

        # Control buttons
        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(0, 4, 0, 0)
        control_layout.setSpacing(8)

        self.start_btn = QPushButton("Start Calibration")
        self.start_btn.clicked.connect(self._start_calibration)
        # match reduced action button height / font
        self.start_btn.setFixedHeight(40)
        self.start_btn.setStyleSheet("font-size:13px;")
        control_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton("Stop & Close")
        self.stop_btn.clicked.connect(self._stop_and_close)
        self.stop_btn.setEnabled(False)
        # match reduced action button height / font
        self.stop_btn.setFixedHeight(40)
        self.stop_btn.setStyleSheet("font-size:13px;")
        control_layout.addWidget(self.stop_btn)

        control_layout.addStretch()

        right_layout.addLayout(control_layout)

        splitter.addWidget(right_widget)
        # set initial splitter sizes to 50/50 of the dialog width (~960)
        splitter.setSizes([480, 480])
        splitter.setHandleWidth(6)

        # Initial status
        self._log("Calibration wizard ready")
        self._log(f"Found {len(self.anchors)} anchors: {list(self.anchors.keys())}")

    def _setup_plot(self):
        """Setup matplotlib plot showing anchors"""
        # ensure figure is the smaller one created earlier
        self.figure.set_size_inches(4, 4)
        # tighten subplot margins so title and axes don't leave too much whitespace
        try:
            self.figure.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.98)
        except Exception:
            pass

        self.ax = self.figure.add_subplot(111)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        # smaller title padding
        self.ax.set_title("Anchor Layout", pad=6)

        # Plot all anchors
        self.anchor_artists = {}
        self.highlight_artists = {}

        for aid, pos in self.anchors.items():
            x, y = pos[0], pos[1]
            # Normal anchor point (slightly larger, circular)
            scatter = self.ax.scatter(x, y, c='blue', s=100, marker='o', zorder=5, edgecolors='k', linewidths=0.6)
            self.anchor_artists[aid] = scatter

            # Text label (close to marker)
            self.ax.text(x, y+0.16, f'A{aid}', ha='center', va='bottom', fontsize=9, fontweight='bold')

            # Highlight circle (initially invisible) - slightly smaller radius
            circle = plt.Circle((x, y), 0.42, fill=False, color='green', linewidth=2.6, visible=False, zorder=4)
            self.ax.add_patch(circle)
            self.highlight_artists[aid] = circle

        # Midpoint visuals
        self.midpoint_lines = {}
        self.midpoint_points = {}

        if len(self.anchors) >= 3:
            aids = sorted(self.anchors.keys())
            a1, a2, a3 = aids[:3]
            pairs = [(a1, a2), (a1, a3), (a2, a3)]

            for pair in pairs:
                aid1, aid2 = pair
                pos1, pos2 = self.anchors[aid1], self.anchors[aid2]
                mid_x = (pos1[0] + pos2[0]) / 2
                mid_y = (pos1[1] + pos2[1]) / 2

                # Dashed lines to midpoint
                line1, = self.ax.plot([pos1[0], mid_x], [pos1[1], mid_y], 'g--', linewidth=1.6, visible=False, zorder=3)
                line2, = self.ax.plot([pos2[0], mid_x], [pos2[1], mid_y], 'g--', linewidth=1.6, visible=False, zorder=3)
                # Midpoint circle
                point = self.ax.scatter(mid_x, mid_y, c='green', s=120, marker='o', visible=False, zorder=6)

                self.midpoint_lines[pair] = [line1, line2]
                self.midpoint_points[pair] = point

        self._update_plot_limits()
        self.canvas.draw_idle()

    def _update_plot_limits(self):
        """Update plot limits to show all anchors with padding"""
        if not self.anchors:
            return

        xs = [pos[0] for pos in self.anchors.values()]
        ys = [pos[1] for pos in self.anchors.values()]

        x_range = max(xs) - min(xs)
        y_range = max(ys) - min(ys)
        padding = max(1.0, max(x_range, y_range) * 0.2)

        self.ax.set_xlim(min(xs) - padding, max(xs) + padding)
        self.ax.set_ylim(min(ys) - padding, max(ys) + padding)

    def _log(self, message: str):
        """Show only the latest message (replace previous displayed text)."""
        if message is None:
            message = ""
        # strip a single trailing newline (calibration prints end with '\n')
        if message.endswith("\n"):
            message = message[:-1]
        # update the label so it always shows the latest message only
        # (label is selectable)
        try:
            self.status_label.setText(message)
        except Exception:
            # fallback: no-op if label not available yet
            pass

    def _start_calibration(self):
        """Start calibration process"""
        # Update UI state
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # Create and start worker
        self.worker = CalibrationWorker(self.config)
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)

        # Connect signals
        self.worker.log_message.connect(self._log)
        self.worker.input_requested.connect(self._handle_input_request)
        self.worker.step_changed.connect(self._handle_step_change)
        self.worker.calibration_finished.connect(self._on_calibration_finished)
        self.worker.error_occurred.connect(self._on_error)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.calibration_finished.connect(self.worker_thread.quit)
        self.worker.error_occurred.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)

        self.worker_thread.start()
        self._log("Starting calibration...")

    def _handle_input_request(self, prompt: str):
        """Handle input request from calibration worker"""
        self.step_label.setText(prompt)

        # Enable appropriate buttons
        self.continue_btn.setEnabled(True)
        self.redo_btn.setEnabled("r=" in prompt or "redo" in prompt.lower())
        self.back_btn.setEnabled("b=" in prompt or "back" in prompt.lower())

    def _handle_step_change(self, step_type: str, current_index: int, highlight_anchors: List[int]):
        """Handle step change for visualization"""
        # Clear previous highlights
        for artist in self.highlight_artists.values():
            artist.set_visible(False)
        for lines in self.midpoint_lines.values():
            for line in lines:
                line.set_visible(False)
        for point in self.midpoint_points.values():
            point.set_visible(False)

        self.current_step = step_type
        self.current_highlights = highlight_anchors

        if step_type == "quality":
            # Highlight single anchor
            if highlight_anchors:
                aid = highlight_anchors[0]
                if aid in self.highlight_artists:
                    self.highlight_artists[aid].set_visible(True)

        elif step_type == "range":
            # Highlight anchor pair and midpoint
            if len(highlight_anchors) == 2:
                aid1, aid2 = highlight_anchors
                # Highlight both anchors
                if aid1 in self.highlight_artists:
                    self.highlight_artists[aid1].set_visible(True)
                if aid2 in self.highlight_artists:
                    self.highlight_artists[aid2].set_visible(True)

                # Show midpoint lines and point
                pair = tuple(sorted([aid1, aid2]))
                if pair in self.midpoint_lines:
                    for line in self.midpoint_lines[pair]:
                        line.set_visible(True)
                if pair in self.midpoint_points:
                    self.midpoint_points[pair].set_visible(True)

        self.canvas.draw()

    def _send_response(self, response: str):
        """Send response back to calibration worker"""
        if self.worker:
            self.worker.provide_input(response)

        # Disable buttons until next prompt
        self.continue_btn.setEnabled(False)
        self.redo_btn.setEnabled(False)
        self.back_btn.setEnabled(False)

        self.step_label.setText("Processing...")

    def _on_calibration_finished(self, quality_results: Dict, range_results: Dict):
        """Handle calibration completion"""
        self._log("\n=== Calibration Complete ===")

        # Save results
        if quality_results or range_results:
            bias_file = self.config.get('bias_file', 'biases.json')
            self._save_results(bias_file, quality_results, range_results)

        # Clear highlights
        for artist in self.highlight_artists.values():
            artist.set_visible(False)
        for lines in self.midpoint_lines.values():
            for line in lines:
                line.set_visible(False)
        for point in self.midpoint_points.values():
            point.set_visible(False)
        self.canvas.draw()

        QMessageBox.information(self, "Calibration Complete",
                              "Calibration finished successfully!\n\nWindow will close automatically.")
        self.accept()

    def _on_error(self, error_msg: str):
        """Handle calibration error"""
        self._log(f"Error: {error_msg}")
        QMessageBox.critical(self, "Calibration Error", error_msg)
        self.reject()

    def _save_results(self, bias_file: str, quality_results: Dict, range_results: Dict):
        """Save calibration results to file"""
        try:
            # Load existing data
            existing_data = {}
            try:
                with open(bias_file, 'r') as f:
                    existing_data = json.load(f)
            except FileNotFoundError:
                pass

            # Update with new results
            if quality_results:
                existing_data['quality'] = quality_results
                WEIGHT_CFG.update(quality_results)
                self._log("Quality calibration results:")
                for key, value in quality_results.items():
                    self._log(f"  {key}: {value:.2f}")

            if range_results:
                existing_data['range'] = {
                    str(aid): [alpha, beta] for aid, (alpha, beta) in range_results.items()
                }
                self._log("Range calibration results:")
                for aid, (alpha, beta) in range_results.items():
                    self._log(f"  A{aid}: true_range = {alpha:.4f} * measured_range + {beta:.4f}")

            # Save to file
            with open(bias_file, 'w') as f:
                json.dump(existing_data, f, indent=2)

            self._log(f"Results saved to {bias_file}")

        except Exception as e:
            self._log(f"Failed to save calibration: {str(e)}")
            QMessageBox.warning(self, "Save Error", f"Failed to save calibration:\n{str(e)}")

    def _stop_and_close(self):
        """Stop calibration and close immediately"""
        if self.worker:
            self.worker.stop()
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(1000)
        self.reject()

    def closeEvent(self, event):
        """Handle dialog close - always stop immediately"""
        if self.worker:
            self.worker.stop()
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(1000)
        event.accept()
