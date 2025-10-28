import csv, os, time, threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .mocap.websocket_io import start_ws_receiver

UWB_FRAME_HEADER = [
    "t","seq","win_id","x_raw","y_raw","x_kf_base","y_kf_base","x_kf_llm","y_kf_llm",
    "inno_base","inno_llm","avg_weight","used_ids","mode","pv","rv","inno_max","gate_drop_base","gate_drop_llm"
]
WINDOWS_HEADER = ["win_id","t_start","t_end","n_pose_frames","n_total_frames","pct_good","mean_speed",
                  "innov_p95","avgw_mean","avgw_p25","avgw_p75","mode","pv","rv","inno_max","advice_json","summary_json"]
MOCAP_HEADER = ["timestamp","frame","mode","object","id","x","y","z","qx","qy","qz","qw"]
JOIN_HEADER_SUFFIX = ["mocap_ts","mocap_frame","mocap_x","mocap_y","mocap_qx","mocap_qy","mocap_qz","mocap_qw",
                      "err_x_base","err_y_base","err_xy_base","err_x_llm","err_y_llm","err_xy_llm"]

@dataclass
class _MocapSample:
    ts: float
    frame: int
    x_m: float
    y_m: float
    qx: float
    qy: float
    qz: float
    qw: float

class RunLogger:
    def __init__(self, args: Dict[str, Any]):
        self.args = args
        self.max_mocap_age_s = float(args.get("max_mocap_age_s", 0.5))
        self.logging_enabled = bool(args.get("logging_enabled", True))

        runs_dir = Path(args.get("runs_dir") or "runs")
        self.runs_dir = runs_dir if runs_dir.is_absolute() else Path.cwd() / runs_dir
        if self.logging_enabled:
            (self.runs_dir / "uwb").mkdir(parents=True, exist_ok=True)
            (self.runs_dir / "mocap").mkdir(parents=True, exist_ok=True)

        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.frames_path = self.runs_dir / "uwb" / f"frames_{stamp}.csv"
        self.windows_path = self.runs_dir / "uwb" / f"windows_{stamp}.csv"
        self.mocap_path = self.runs_dir / "mocap" / f"run_{stamp}.csv"
        self.join_path = self.runs_dir / f"runs_{stamp}.csv"

        self.frames_f = None; self.frames_w = None
        self.windows_f = None; self.windows_w = None
        self.join_f = None; self.join_w = None
        if self.logging_enabled:
            self.frames_f = open(self.frames_path, "w", newline="")
            self.frames_w = csv.writer(self.frames_f)
            self.frames_w.writerow(UWB_FRAME_HEADER); self.frames_f.flush()

            self.windows_f = open(self.windows_path, "w", newline="")
            self.windows_w = csv.writer(self.windows_f)
            self.windows_w.writerow(WINDOWS_HEADER); self.windows_f.flush()

            self.join_f = open(self.join_path, "w", newline="")
            self.join_w = csv.writer(self.join_f)
            self.join_w.writerow(UWB_FRAME_HEADER + JOIN_HEADER_SUFFIX); self.join_f.flush()

        self.mocap_f = None
        self.mocap_w = None
        self._mocap_buf = deque(maxlen=20000)
        self._mocap_lock = threading.Lock()
        self._mocap_thread = None
        self._mocap_stop = None

        if bool(args.get("mocap_enabled", False)):
            if self.logging_enabled:
                self.mocap_f = open(self.mocap_path, "w", newline="")
                self.mocap_w = csv.writer(self.mocap_f)
                self.mocap_w.writerow(MOCAP_HEADER); self.mocap_f.flush()
            ws_url = self._build_ws_url(args.get("websocket_ip"), args.get("websocket_port"))
            obj = args.get("mocap_object", "Goal")
            oid = args.get("mocap_id", 2)
            self._mocap_thread, self._mocap_stop = start_ws_receiver(ws_url, self._on_mocap_row, object_filter=obj, id_filter=str(oid))

    def _build_ws_url(self, ip_or_url: Optional[str], port: Optional[int]) -> str:
        if not ip_or_url:
            return f"ws://127.0.0.1:{int(port or 8765)}"
        s = str(ip_or_url)
        if s.startswith("ws://") or s.startswith("wss://"):
            return s if (":" in s.split("//", 1)[1]) else f"{s}:{int(port or 8765)}"
        return f"ws://{s}:{int(port or 8765)}"

    def _on_mocap_row(self, row: Dict[str, Any]):
        if self.logging_enabled and (self.mocap_w is not None):
            self.mocap_w.writerow([row.get(k) for k in MOCAP_HEADER])
            self.mocap_f.flush()
        try:
            ts = float(row.get("timestamp"))
            frame = int(row.get("frame") or 0)
            x_m = float(row.get("x")) / 1000.0
            y_m = float(row.get("y")) / 1000.0
            qx = float(row.get("qx") or 0.0)
            qy = float(row.get("qy") or 0.0)
            qz = float(row.get("qz") or 0.0)
            qw = float(row.get("qw") or 1.0)
        except Exception:
            return
        with self._mocap_lock:
            self._mocap_buf.append(_MocapSample(ts, frame, x_m, y_m, qx, qy, qz, qw))

    def log_window(self, win_id: int, t_start: float, t_end: float, *,
                   n_pose_frames: int, n_total_frames: int, pct_good: float, mean_speed: float,
                   innov_p95: float, avgw_mean: float, avgw_p25: float, avgw_p75: float,
                   advice: Dict[str, Any], summary: Dict[str, Any]):
        if not self.logging_enabled or self.windows_w is None:
            return
        mode = advice.get("mode", summary.get("mode", "baseline"))
        pv = float(advice.get("pv", summary.get("pv", 0.0)))
        rv = float(advice.get("rv", summary.get("rv", 0.0)))
        inno_max = float(advice.get("innovation_max", summary.get("innovation_max", 0.0)))
        self.windows_w.writerow([
            win_id, t_start, t_end, n_pose_frames, n_total_frames, pct_good, mean_speed,
            innov_p95, avgw_mean, avgw_p25, avgw_p75, mode, pv, rv, inno_max,
            advice, summary
        ])
        self.windows_f.flush()

    def log_frame(self, *, t: float, seq: int, win_id: int,
                  x_raw: float, y_raw: float,
                  x_kf_base: float, y_kf_base: float,
                  x_kf_llm: float, y_kf_llm: float,
                  inno_base: float, inno_llm: float,
                  avg_weight: float, used_ids: List[int],
                  mode: str, pv: float, rv: float, inno_max: float,
                  gate_drop_base: bool, gate_drop_llm: bool):
        used_str = ";".join(str(u) for u in used_ids)
        uwb_row = [t, seq, win_id, x_raw, y_raw, x_kf_base, y_kf_base, x_kf_llm, y_kf_llm,
                   inno_base, inno_llm, avg_weight, used_str, mode, pv, rv, inno_max,
                   int(bool(gate_drop_base)), int(bool(gate_drop_llm))]
        if not self.logging_enabled:
            return
        if self.frames_w is not None:
            self.frames_w.writerow(uwb_row)
            self.frames_f.flush()
        if self.join_w is not None:
            moc_ts, moc_frame, moc_x, moc_y, qx, qy, qz, qw = self._interp_mocap(t)
            if moc_ts is not None:
                err_x_b = x_kf_base - moc_x
                err_y_b = y_kf_base - moc_y
                err_xy_b = (err_x_b**2 + err_y_b**2) ** 0.5
                err_x_l = x_kf_llm - moc_x
                err_y_l = y_kf_llm - moc_y
                err_xy_l = (err_x_l**2 + err_y_l**2) ** 0.5
                self.join_w.writerow(uwb_row + [moc_ts, moc_frame, moc_x, moc_y, qx, qy, qz, qw,
                                                err_x_b, err_y_b, err_xy_b, err_x_l, err_y_l, err_xy_l])
            else:
                self.join_w.writerow(uwb_row + [None]*14)
            self.join_f.flush()

    def _interp_mocap(self, t: float) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        with self._mocap_lock:
            if len(self._mocap_buf) < 1:
                return (None,)*8
            buf = list(self._mocap_buf)

        max_age = float(getattr(self, "max_mocap_age_s", 0.5))

        if t < buf[0].ts:
            s = buf[0]
            return (s.ts, s.frame, s.x_m, s.y_m, s.qx, s.qy, s.qz, s.qw) if abs(s.ts - t) <= max_age else (None,)*8
        if t > buf[-1].ts:
            s = buf[-1]
            return (s.ts, s.frame, s.x_m, s.y_m, s.qx, s.qy, s.qz, s.qw) if abs(s.ts - t) <= max_age else (None,)*8

        lo, hi = 0, len(buf)-1
        while lo <= hi:
            mid = (lo + hi) // 2
            if buf[mid].ts < t:
                lo = mid + 1
            else:
                hi = mid - 1
        i1 = max(1, lo)
        s0, s1 = buf[i1-1], buf[i1]

        nearest_dt = min(abs(t - s0.ts), abs(s1.ts - t))
        if nearest_dt > max_age:
            return (None,)*8

        if s1.ts == s0.ts:
            s = s0
            return s.ts, s.frame, s.x_m, s.y_m, s.qx, s.qy, s.qz, s.qw

        a = (t - s0.ts) / (s1.ts - s0.ts)
        xi = s0.x_m + a * (s1.x_m - s0.x_m)
        yi = s0.y_m + a * (s1.y_m - s0.y_m)
        qn = s0 if abs(s0.ts - t) <= abs(s1.ts - t) else s1
        return t, qn.frame, xi, yi, qn.qx, qn.qy, qn.qz, qn.qw

    def has_mocap(self) -> bool:
        return self._mocap_thread is not None

    def interp_mocap(self, t: float) -> Tuple[Optional[float], Optional[int], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
        return self._interp_mocap(t)

    def close(self):
        try:
            if self._mocap_stop is not None:
                self._mocap_stop.set()
        except Exception:
            pass
        for f in (getattr(self, "frames_f", None),
                  getattr(self, "windows_f", None),
                  getattr(self, "mocap_f", None),
                  getattr(self, "join_f", None)):
            try:
                if f: f.close()
            except Exception:
                pass
