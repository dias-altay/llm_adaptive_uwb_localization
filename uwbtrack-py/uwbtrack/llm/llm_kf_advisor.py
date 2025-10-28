from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple, Any
import numpy as np
import threading
import time

from .llm_agent import LLMAdvisorAgent, _MODE_DEFAULT_KF
from .hyper_tuner import HyperTuner, DEFAULT_THRESHOLDS


@dataclass
class KFTuningAdvisor:
    window_s: float
    default_innovation_max: float
    agent: Optional[LLMAdvisorAgent] = None
    mode_dwell_w: int = 2
    default_kf_process: float = 0.15
    default_kf_meas: float = 0.30
    enable_hyper_tuner: bool = False
    slow_loop_interval: float = 30.0

    _pose_frames: Deque[Dict[str, Any]] = field(default_factory=lambda: deque())
    _all_frames: Deque[float] = field(default_factory=lambda: deque())
    _last_adv_t: float = 0.0
    _win_id: int = 0
    _current: Dict[str, Any] = field(default_factory=lambda: {
        "mode": "baseline",
        "kf": {"process_var": 0.15, "meas_var": 0.30, "innovation_max": 1.5}
    })
    _win_t0: float = 0.0
    _mode_last_switch_winid: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _static_consec_count: int = 0
    _moving_consec_count: int = 0
    _last_mode_switch_time: float = 0.0
    _thresholds: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    _hyper_tuner: Optional[HyperTuner] = None

    _last_thresholds_snapshot: Dict[str, float] = field(default_factory=dict)
    _last_threshold_update_time: Optional[float] = None
    _last_threshold_explanation: Optional[str] = None

    _cwin_centroids: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=8))
    _cwin_times: Deque[float] = field(default_factory=lambda: deque(maxlen=8))

    def __post_init__(self):
        self._current["kf"]["process_var"] = float(self.default_kf_process)
        self._current["kf"]["meas_var"] = float(self.default_kf_meas)
        self._current["kf"]["innovation_max"] = float(self.default_innovation_max)

        if self.enable_hyper_tuner and self.agent:
            print(f"[KFTuningAdvisor] Initializing HyperTuner with slow_loop_interval={self.slow_loop_interval}s")
            self._hyper_tuner = HyperTuner(agent=self.agent, slow_loop_interval=self.slow_loop_interval)
        else:
            if self.enable_hyper_tuner:
                print(f"[KFTuningAdvisor] WARNING: Hybrid tuning enabled but LLM agent is None!")
            else:
                print(f"[KFTuningAdvisor] Hybrid tuning is disabled")

    def current_knobs(self) -> Tuple[str, float, float, float, int]:
        kf = self._current["kf"]
        mode = self._current.get("mode", "baseline")

        if mode == "baseline":
            return (
                "baseline",
                float(self.default_kf_process),
                float(self.default_kf_meas),
                float(self.default_innovation_max),
                int(self._win_id)
            )

        return (
            mode,
            float(kf["process_var"]),
            float(kf["meas_var"]),
            float(kf["innovation_max"]),
            int(self._win_id)
        )

    def note_any_frame(self, t: float):
        self._all_frames.append(t)
        while self._all_frames and (t - self._all_frames[0]) > self.window_s:
            self._all_frames.popleft()

    def note_pose_frame(
        self,
        t: float,
        v_inst: float,
        innov_b: float,
        avgw: float,
        gate_b: bool = False,
        x: Optional[float] = None,
        y: Optional[float] = None
    ):
        self._pose_frames.append({
            "t": t, "v": v_inst, "innov_b": innov_b, "avgw": avgw, "gate": bool(gate_b),
            "x": (float(x) if x is not None else None),
            "y": (float(y) if y is not None else None),
        })
        while self._pose_frames and (t - self._pose_frames[0]["t"]) > self.window_s:
            self._pose_frames.popleft()

    def _apply_advice_direct(self, mode: str):
        if not isinstance(mode, str): return
        if mode not in _MODE_DEFAULT_KF: return
        kf = _MODE_DEFAULT_KF[mode]
        with self._lock:
            self._current = {
                "mode": mode,
                "kf": {
                    "process_var": float(kf["process_var"]),
                    "meas_var": float(kf["meas_var"]),
                    "innovation_max": float(kf["innovation_max"]),
                }
            }
            self._mode_last_switch_winid = int(self._win_id)
            self._last_mode_switch_time = time.time()

    def maybe_advise(self, now: float):
        if self._last_adv_t == 0.0:
            self._last_adv_t = now
            self._win_t0 = now
            return None

        if (now - self._last_adv_t) < self.window_s:
            return None

        try:
            n_pose = len(self._pose_frames)
            n_all = len(self._all_frames)
            pct_good = (n_pose / max(1, n_all)) if n_all else 0.0

            if n_pose:
                v = np.array([p["v"] for p in self._pose_frames], dtype=float)
                inn = np.array([p["innov_b"] for p in self._pose_frames], dtype=float)
                w = np.array([p["avgw"] for p in self._pose_frames], dtype=float)
                g = np.array([1.0 if p.get("gate", False) else 0.0 for p in self._pose_frames], dtype=float)

                mean_speed = float(np.mean(v)) if v.size else 0.0
                speed_p90  = float(np.percentile(v, 90)) if v.size else 0.0
                innov_p50  = float(np.percentile(inn, 50)) if inn.size else 1.0
                innov_p95  = float(np.percentile(inn, 95)) if inn.size else 1.0
                avgw_mean  = float(np.mean(w)) if w.size else 0.0
                avgw_p25   = float(np.percentile(w, 25)) if w.size else 0.0
                avgw_p75   = float(np.percentile(w, 75)) if w.size else 0.0
                gate_rate  = float(np.mean(g)) if g.size else 0.0

                xs = [p["x"] for p in self._pose_frames if p.get("x") is not None and p.get("y") is not None]
                ys = [p["y"] for p in self._pose_frames if p.get("x") is not None and p.get("y") is not None]
                ts = [p["t"] for p in self._pose_frames if p.get("x") is not None and p.get("y") is not None]

                if len(xs) >= 2:
                    net_disp = float(np.hypot(xs[-1] - xs[0], ys[-1] - ys[0]))
                    start_pos = (float(xs[0]), float(ys[0]))
                    end_pos = (float(xs[-1]), float(ys[-1]))

                    total_path_length = 0.0
                    for i in range(1, len(xs)):
                        total_path_length += float(np.hypot(xs[i] - xs[i-1], ys[i] - ys[i-1]))

                    jitter_ratio = (total_path_length / max(0.01, net_disp)) if net_disp > 0 else float("inf")

                    cx, cy = float(np.mean(xs)), float(np.mean(ys))
                    rms = float(np.sqrt(np.mean((np.array(xs)-cx)**2 + (np.array(ys)-cy)**2)))

                    span_s = max(1.0, min(self.window_s / 3.0, 2.0))
                    t0 = self._win_t0
                    early_idx = [i for i, t in enumerate(ts) if (t - t0) <= span_s]
                    late_idx  = [i for i, t in enumerate(ts) if (now - t) <= span_s]
                    if len(early_idx) < 3:
                        cut = max(1, int(0.1 * len(xs)))
                        early_idx = list(range(0, cut))
                    if len(late_idx) < 3:
                        cut = max(1, int(0.1 * len(xs)))
                        late_idx = list(range(len(xs) - cut, len(xs)))

                    ex = float(np.mean(np.array(xs)[early_idx])); ey = float(np.mean(np.array(ys)[early_idx]))
                    lx = float(np.mean(np.array(xs)[late_idx]));  ly = float(np.mean(np.array(ys)[late_idx]))
                    edge_shift_1s = float(np.hypot(lx - ex, ly - ey))

                else:
                    net_disp = 0.0; start_pos = (None, None); end_pos = (None, None)
                    total_path_length = 0.0; jitter_ratio = 1.0; rms = 0.0; edge_shift_1s = 0.0
                    cx = cy = 0.0

            else:
                mean_speed = 0.0; speed_p90 = 0.0; innov_p50 = 1.0; innov_p95 = 1.0
                avgw_mean = 0.0; avgw_p25 = 0.0; avgw_p75 = 0.0; gate_rate = 0.0
                net_disp = 0.0; total_path_length = 0.0; start_pos = (None, None); end_pos = (None, None)
                jitter_ratio = 1.0; rms = 0.0; edge_shift_1s = 0.0; cx = cy = 0.0

            if self.enable_hyper_tuner and self._hyper_tuner is None and self.agent is not None:
                print("[KFTuningAdvisor] Reinitializing HyperTuner")
                self._hyper_tuner = HyperTuner(agent=self.agent, slow_loop_interval=self.slow_loop_interval)

            if self._hyper_tuner:
                self._thresholds = self._hyper_tuner.get_current_thresholds()

            th = self._thresholds
            hard_enter_p90 = th.get("hard_enter_p90", DEFAULT_THRESHOLDS["hard_enter_p90"])
            soft_mean = th.get("soft_mean", DEFAULT_THRESHOLDS["soft_mean"])
            soft_innov = th.get("soft_innov", DEFAULT_THRESHOLDS["soft_innov"])
            soft_good = th.get("soft_good", DEFAULT_THRESHOLDS["soft_good"])
            quiet_p90 = th.get("quiet_p90", DEFAULT_THRESHOLDS["quiet_p90"])
            quiet_innov = th.get("quiet_innov", DEFAULT_THRESHOLDS["quiet_innov"])
            static_streak_req = int(th.get("static_streak", DEFAULT_THRESHOLDS["static_streak"]))
            moving_streak_req = int(th.get("moving_streak", DEFAULT_THRESHOLDS["moving_streak"]))

            static_edge_max = th.get("static_edge_max", DEFAULT_THRESHOLDS["static_edge_max"])
            static_netd_max = th.get("static_netd_max", DEFAULT_THRESHOLDS["static_netd_max"])
            move_edge_min = th.get("move_edge_min", DEFAULT_THRESHOLDS["move_edge_min"])
            move_netd_min = th.get("move_netd_min", DEFAULT_THRESHOLDS["move_netd_min"])
            move_path_min = th.get("move_path_min", DEFAULT_THRESHOLDS["move_path_min"])
            static_jitter_min = th.get("static_jitter_min", DEFAULT_THRESHOLDS["static_jitter_min"])

            static_rms_max = th.get("static_rms_max", DEFAULT_THRESHOLDS["static_rms_max"])
            static_drift_max = th.get("static_drift_max", DEFAULT_THRESHOLDS["static_drift_max"])

            static_cwin_r_max = th.get("static_cwin_r_max", DEFAULT_THRESHOLDS["static_cwin_r_max"])
            static_cwin_drift_max = th.get("static_cwin_drift_max", DEFAULT_THRESHOLDS["static_cwin_drift_max"])

            min_pct_good = th.get("min_pct_good", DEFAULT_THRESHOLDS["min_pct_good"])
            max_gate_rate = th.get("max_gate_rate", DEFAULT_THRESHOLDS["max_gate_rate"])
            cooldown_s = th.get("cooldown_s", DEFAULT_THRESHOLDS["cooldown_s"])

            print(
                f"[KFTuningAdvisor] Thresholds: hard_enter_p90={hard_enter_p90:.1f}, quiet_p90={quiet_p90:.1f}, "
                f"quiet_innov={quiet_innov:.3f}, move_edge_min={move_edge_min:.2f}, move_netd_min={move_netd_min:.2f}, "
                f"move_path_min={move_path_min:.2f}, static_rms_max={static_rms_max:.2f}, static_drift_max={static_drift_max:.2f}, "
                f"cwin_r_max={static_cwin_r_max:.2f}, cwin_drift_max={static_cwin_drift_max:.2f}"
            )
            print(
                f"[KFTuningAdvisor] Metrics: speed_p90={speed_p90:.1f}, mean_speed={mean_speed:.1f}, innov_p95={innov_p95:.3f}, "
                f"net_disp={net_disp:.3f}m, edge_shift_1s={edge_shift_1s:.3f}m, path={total_path_length:.3f}m, "
                f"jitter_ratio={jitter_ratio:.1f}, pct_good={pct_good:.2f}, gate_rate={gate_rate:.2f}"
            )

            quality_ok = (pct_good >= min_pct_good) and (gate_rate <= max_gate_rate)

            if quality_ok and (np.isfinite(cx) and np.isfinite(cy)):
                current_mode = self._current.get("mode", "baseline")
                if current_mode != "static" or len(self._cwin_centroids) < 2 or (now - self._cwin_times[-1] if self._cwin_times else 0) > 2.0:
                    self._cwin_centroids.append((cx, cy))
                    self._cwin_times.append(now)
            if len(self._cwin_centroids) >= 3:
                arr = np.array(self._cwin_centroids, dtype=float)
                mx, my = np.median(arr[:,0]), np.median(arr[:,1])
                cross_win_r = float(np.sqrt(np.mean((arr[:,0]-mx)**2 + (arr[:,1]-my)**2)))
                cross_win_drift = float(np.hypot(arr[-1,0]-arr[0,0], arr[-1,1]-arr[0,1]))
            else:
                cross_win_r = 1e9
                cross_win_drift = 1e9

            moving_speed_rule = (speed_p90 >= hard_enter_p90) or \
                                (mean_speed >= soft_mean and innov_p95 >= soft_innov and pct_good >= soft_good)
            path_suggests_moving = (total_path_length >= move_path_min) and \
                                   (jitter_ratio <= (static_jitter_min * 0.75)) and \
                                   (net_disp >= 0.06)
            moving_geom_rule = (edge_shift_1s >= move_edge_min) or (net_disp >= move_netd_min) or path_suggests_moving

            static_quiet_rule = (speed_p90 <= quiet_p90) and (innov_p95 <= quiet_innov)
            static_geom_rule = (edge_shift_1s <= static_edge_max) and (net_disp <= static_netd_max)
            static_jitter_rule = (jitter_ratio >= static_jitter_min) or (net_disp <= static_netd_max)
            static_cluster_rule = (rms <= static_rms_max) and (edge_shift_1s <= static_drift_max)
            static_xwin_rule = (cross_win_r <= static_cwin_r_max) and (cross_win_drift <= static_cwin_drift_max)

            if self._current.get("mode") == "static":
                static_xwin_rule = (cross_win_r <= static_cwin_r_max * 0.5) and (cross_win_drift <= static_cwin_drift_max * 0.5)
            else:
                static_xwin_rule = (cross_win_r <= static_cwin_r_max) and (cross_win_drift <= static_cwin_drift_max)

            moving_evidence = quality_ok and (moving_speed_rule or moving_geom_rule) and (not static_cluster_rule) and (not static_xwin_rule)
            static_evidence = quality_ok and (
                (static_quiet_rule and static_geom_rule) or
                (static_geom_rule and static_jitter_rule) or
                static_cluster_rule or
                static_xwin_rule
            )

            if moving_evidence and not static_evidence:
                self._moving_consec_count += 1
                self._static_consec_count = 0
            elif static_evidence and not moving_evidence:
                self._static_consec_count += 1
                self._moving_consec_count = 0
            else:
                self._moving_consec_count = max(0, self._moving_consec_count - 1)
                self._static_consec_count = max(0, self._static_consec_count - 1)

            margins = {
                "hard_enter": float(speed_p90 - hard_enter_p90),
                "soft_mean": float(mean_speed - soft_mean),
                "soft_innov": float(innov_p95 - soft_innov),
                "soft_good": float(pct_good - soft_good),
                "quiet_p90": float(quiet_p90 - speed_p90),
                "quiet_innov": float(quiet_innov - innov_p95),
                "move_edge": float(edge_shift_1s - move_edge_min),
                "move_netd": float(net_disp - move_netd_min),
                "move_path": float(total_path_length - move_path_min),
                "static_edge": float(static_edge_max - edge_shift_1s),
                "static_netd": float(static_netd_max - net_disp),
                "static_jitter": float(jitter_ratio - static_jitter_min),
                "static_rms": float(static_rms_max - rms),
                "static_drift": float(static_drift_max - edge_shift_1s),
                "pct_good": float(pct_good - th.get("min_pct_good", DEFAULT_THRESHOLDS["min_pct_good"])),
                "gate_rate": float(th.get("max_gate_rate", DEFAULT_THRESHOLDS["max_gate_rate"]) - gate_rate),
                "static_xwin_r": float(static_cwin_r_max - cross_win_r),
                "static_xwin_drift": float(static_cwin_drift_max - cross_win_drift),
            }
            rule_hits = {
                "moving_rule1": bool(speed_p90 >= hard_enter_p90),
                "moving_rule2": bool(mean_speed >= soft_mean and innov_p95 >= soft_innov and pct_good >= soft_good),
                "static_rule": bool(static_quiet_rule),
                "static_cluster": bool(static_cluster_rule),
                "static_xwin": bool(static_xwin_rule),
            }

            allow_switch = True
            if self._last_mode_switch_time:
                if (now - self._last_mode_switch_time) < cooldown_s:
                    allow_switch = False

            chosen_mode = self._current.get("mode", "baseline")
            prev_mode = chosen_mode

            if allow_switch and (self._moving_consec_count >= moving_streak_req):
                chosen_mode = "moving"
            elif allow_switch and (self._static_consec_count >= static_streak_req):
                chosen_mode = "static"
            else:
                if chosen_mode == "baseline":
                    if self._moving_consec_count >= max(1, moving_streak_req - 1):
                        chosen_mode = "moving"
                    elif self._static_consec_count >= max(1, static_streak_req - 1):
                        chosen_mode = "static"

            if chosen_mode != prev_mode:
                print(
                    f"[KFTuningAdvisor] MODE CHANGE: {prev_mode} → {chosen_mode} "
                    f"(moving_streak={self._moving_consec_count}/{moving_streak_req}, "
                    f"static_streak={self._static_consec_count}/{static_streak_req}, cooldown_ok={allow_switch})"
                )
                self._apply_advice_direct(chosen_mode)
                self._last_mode_switch_time = now
                summary_transition = f"{prev_mode}_to_{chosen_mode}"
                if chosen_mode == "moving":
                    self._static_consec_count = 0
                elif chosen_mode == "static":
                    self._moving_consec_count = 0
            else:
                summary_transition = None

            summary = {
                "n_frames": n_pose,
                "pct_good": pct_good,
                "mean_speed": mean_speed,
                "speed_p90": speed_p90,
                "innov_p50": innov_p50,
                "innov_p95": innov_p95,
                "avgw_mean": avgw_mean,
                "avgw_p25": avgw_p25,
                "avgw_p75": avgw_p75,
                "gate_rate": gate_rate,
                "last_mode": self._current.get("mode", "baseline"),
                "win_len_s": float(self.window_s),
                "frames_per_s": float(n_pose) / max(1.0, float(self.window_s)),
                "net_disp": net_disp,
                "static_streak": int(self._static_consec_count),
                "start_pos": start_pos,
                "end_pos": end_pos,
                "total_path_length": total_path_length,
                "pos_variance": float((rms * rms) if rms else 0.0),
                "rms": float(rms),
                "edge_shift_1s": edge_shift_1s,
                "jitter_ratio": jitter_ratio,
                "centroid": (cx, cy),
                "cross_win_r": cross_win_r if np.isfinite(cross_win_r) else None,
                "cross_win_drift": cross_win_drift if np.isfinite(cross_win_drift) else None,
                "rule_hits": rule_hits,
                "margins": margins,
                "mode_transition": summary_transition,
            }

            if self._hyper_tuner:
                try:
                    print(f"[KFTuningAdvisor] Window {self._win_id}: send to HyperTuner (mode={chosen_mode})")
                    self._hyper_tuner.note_window_summary(summary, chosen_mode, self._win_id)
                except Exception as e:
                    print(f"[KFTuningAdvisor] Error sending window to HyperTuner: {e}")
                    import traceback; traceback.print_exc()

            win_id = self._win_id + 1
            t_start, t_end = self._win_t0, now
            self._win_id = win_id
            self._last_adv_t = now
            self._win_t0 = now

            try:
                if self._hyper_tuner and hasattr(self._hyper_tuner, 'last_adjustment_time'):
                    self._last_thresholds_snapshot = dict(self._thresholds)
                    self._last_threshold_update_time = float(self._hyper_tuner.last_adjustment_time or 0.0)
                    self._last_threshold_explanation = self._hyper_tuner.last_adjustment_explanation
            except Exception as e:
                print(f"[KFTuningAdvisor] Cache thresholds snapshot error: {e}")

            return {
                "win_id": win_id - 1,
                "t_start": t_start,
                "t_end": t_end,
                "summary": summary,
                "advice": {
                    "mode": self._current.get("mode", "baseline"),
                    "kf": self._current.get("kf", {})
                }
            }

        except Exception as e:
            print(f"[KFTuningAdvisor] Uncaught error in maybe_advise: {e}")
            import traceback; traceback.print_exc()
            self._win_id += 1
            self._last_adv_t = now
            self._win_t0 = now
            return None
