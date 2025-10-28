import json
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass, field

from .llm_agent import LLMAdvisorAgent

DEFAULT_THRESHOLDS = {
    "hard_enter_p90": 60.0,
    "soft_mean": 20.0,
    "soft_innov": 0.25,
    "soft_good": 0.80,
    "quiet_p90": 30.0,
    "quiet_innov": 0.08,
    "static_streak": 2,
    "moving_streak": 2,
    "static_edge_max": 0.03,
    "static_netd_max": 0.05,
    "move_edge_min": 0.15,
    "move_netd_min": 0.25,
    "move_path_min": 0.50,
    "static_jitter_min": 8.0,
    "static_rms_max": 0.04,
    "static_drift_max": 0.03,
    "static_cwin_r_max": 0.06,
    "static_cwin_r_max": 0.06,
    "static_cwin_drift_max": 0.04,
    "min_pct_good": 0.30,
    "max_gate_rate": 0.70,
    "cooldown_s": 5.0
}

THRESHOLD_BOUNDS = {
    "hard_enter_p90": (30.0, 100.0),
    "soft_mean": (10.0, 40.0),
    "soft_innov": (0.08, 0.50),
    "soft_good": (0.60, 0.95),
    "quiet_p90": (15.0, 50.0),
    "quiet_innov": (0.08, 0.20),
    "static_streak": (1, 3),
    "moving_streak": (1, 3),
    "static_edge_max": (0.04, 0.15),
    "static_netd_max": (0.06, 0.25),
    "move_edge_min": (0.08, 0.30),
    "move_netd_min": (0.18, 0.60),
    "move_path_min": (0.30, 1.20),
    "static_jitter_min": (3.0, 20.0),
    "static_rms_max": (0.03, 0.30),
    "static_drift_max": (0.04, 0.20),
    "static_cwin_r_max": (0.05, 0.40),
    "static_cwin_drift_max": (0.05, 0.35),
    "min_pct_good": (0.20, 0.80),
    "max_gate_rate": (0.30, 0.90),
    "cooldown_s": (0.0, 10.0),
}

MAX_STEP_FRACTION = 0.10
_MIN_EVAL_FOR_FREEZE = 4

def _clip_step(old: float, new: float, lo: float, hi: float, frac: float) -> float:
    rng = hi - lo
    max_step = rng * frac
    delta = max(-max_step, min(max_step, new - old))
    v = old + delta
    return max(lo, min(hi, v))

@dataclass
class HyperTuner:
    window_summaries: deque = field(default_factory=lambda: deque(maxlen=12))
    mode_history: deque = field(default_factory=lambda: deque(maxlen=12))
    thresholds: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_THRESHOLDS))
    last_slow_update: float = field(default_factory=lambda: 0.0)
    last_adjustment_time: float = field(default_factory=lambda: 0.0)
    slow_loop_interval: float = field(default_factory=lambda: 30.0)
    agent: Optional[LLMAdvisorAgent] = None

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _metrics_history: List[Dict[str, Any]] = field(default_factory=list)
    _llm_thread: Optional[threading.Thread] = None

    _last_significant_change: float = field(default_factory=lambda: 0.0)
    _min_stability_interval: float = field(default_factory=lambda: 25.0)

    last_adjustment_explanation: Optional[str] = field(default=None)

    _window_count: int = field(default_factory=lambda: 0)
    _llm_call_count: int = field(default_factory=lambda: 0)
    _llm_success_count: int = field(default_factory=lambda: 0)
    _llm_failure_count: int = field(default_factory=lambda: 0)
    _last_llm_response: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        print(f"[HyperTuner] Initialized (slow loop {self.slow_loop_interval:.0f}s)")
        self.last_slow_update = time.time() - (self.slow_loop_interval * 0.8)

    def note_window_summary(self, summary: Dict[str, Any], mode: str, win_id: int) -> None:
        try:
            with self._lock:
                if "timestamp" not in summary:
                    summary["timestamp"] = time.time()
                self.window_summaries.append(summary)
                self.mode_history.append({"mode": mode, "win_id": win_id, "timestamp": summary["timestamp"]})
                self._window_count += 1

                now = time.time()
                time_since = now - self.last_slow_update
                enough_windows = len(self.window_summaries) >= 6
                stable = (now - self._last_significant_change) >= self._min_stability_interval

                print(f"[HyperTuner] Window {win_id} noted: mode={mode}, net_disp={summary.get('net_disp',0):.3f}m, "
                      f"edge_shift_1s={summary.get('edge_shift_1s',0):.3f}m, rms={summary.get('rms',0):.3f}m, jitter_ratio={summary.get('jitter_ratio',0):.1f}")

                if time_since >= self.slow_loop_interval and enough_windows and stable:
                    self._trigger_slow_loop(now)

        except Exception as e:
            print(f"[HyperTuner] ❌ Error in note_window_summary: {e}")
            import traceback; traceback.print_exc()

    def get_current_thresholds(self) -> Dict[str, float]:
        with self._lock:
            return dict(self.thresholds)

    def get_last_llm_response(self) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._last_llm_response

    def _trigger_slow_loop(self, now: float) -> None:
        self.last_slow_update = now
        if self._llm_thread and self._llm_thread.is_alive():
            print("[HyperTuner] Slow loop already running")
            return

        def _run():
            try:
                metrics = self._calculate_performance_metrics()
                conf = metrics.get("confusion", {})
                n_eval = int(metrics.get("n_eval", 0))
                if (n_eval >= _MIN_EVAL_FOR_FREEZE and
                    metrics.get("accuracy", 0.0) >= 0.98 and
                    conf.get("moving_but_fast_static_or_baseline", 0) == 0 and
                    conf.get("static_but_fast_moving_or_baseline", 0) == 0 and
                    not metrics.get("stuck_baseline", False)):
                    print("[HyperTuner] ✅ Perfect agreement — freezing thresholds this cycle")
                    self.last_adjustment_time = time.time()
                    self.last_adjustment_explanation = "No change: perfect agreement."
                    return

                self._metrics_history.append(metrics)
                self._metrics_history[:] = self._metrics_history[-5:]

                if not self.agent:
                    print("[HyperTuner] ⚠️ No LLM agent; applying heuristic nudge if needed")
                    self._maybe_apply_small_fallback(metrics)
                    return

                payload = self._create_llm_payload(metrics)
                self._llm_call_count += 1
                print("[HyperTuner] 🧠 Requesting strategy…")
                adjustments = self._request_threshold_adjustments(payload)
                if adjustments:
                    self._llm_success_count += 1
                    self._apply_threshold_adjustments(adjustments)
                else:
                    self._llm_failure_count += 1
                    self._maybe_apply_small_fallback(metrics)
            except Exception as e:
                self._llm_failure_count += 1
                print(f"[HyperTuner] ❌ Slow loop error: {e}")
                import traceback; traceback.print_exc()

        self._llm_thread = threading.Thread(target=_run, name="HyperTuner_SlowLoop", daemon=True)
        self._llm_thread.start()

    @staticmethod
    def _classify_ref_with_conf(summary: Dict[str, Any], thresholds: Dict[str, float]) -> Tuple[str, float]:
        edge = float(summary.get("edge_shift_1s", 0.0))
        netd = float(summary.get("net_disp", 0.0))
        path_len = float(summary.get("total_path_length", 0.0))
        jitter_ratio = float(summary.get("jitter_ratio", 1.0))
        speed_p90 = float(summary.get("speed_p90", 0.0))
        innov_p95 = float(summary.get("innov_p95", 1.0))
        rms = float(summary.get("rms", 0.0))
        pct_good = float(summary.get("pct_good", 0.0))
        gate_rate = float(summary.get("gate_rate", 0.0))
        cwin_r = float(summary.get("cross_win_r", 1e9))
        cwin_drift = float(summary.get("cross_win_drift", 1e9))

        hard_enter_p90 = thresholds.get("hard_enter_p90", 60.0)
        quiet_p90 = thresholds.get("quiet_p90", 30.0)
        quiet_innov = thresholds.get("quiet_innov", 0.12)

        static_edge_max = thresholds.get("static_edge_max", 0.08)
        static_netd_max = thresholds.get("static_netd_max", 0.12)
        move_edge_min = thresholds.get("move_edge_min", 0.15)
        move_netd_min = thresholds.get("move_netd_min", 0.25)
        move_path_min = thresholds.get("move_path_min", 0.50)
        static_jitter_min = thresholds.get("static_jitter_min", 8.0)

        static_rms_max = thresholds.get("static_rms_max", 0.10)
        static_drift_max = thresholds.get("static_drift_max", 0.08)

        static_cwin_r_max = thresholds.get("static_cwin_r_max", 0.15)
        static_cwin_drift_max = thresholds.get("static_cwin_drift_max", 0.12)

        min_pct_good = thresholds.get("min_pct_good", 0.30)
        max_gate_rate = thresholds.get("max_gate_rate", 0.70)

        if pct_good < min_pct_good or gate_rate > max_gate_rate:
            return "unknown", 0.3

        cwin_static = (cwin_r <= static_cwin_r_max) and (cwin_drift <= static_cwin_drift_max)
        if cwin_static:
            return "static", 0.97

        if (rms <= static_rms_max) and (edge <= static_drift_max):
            return "static", 0.95

        cues_m = 0
        if edge >= move_edge_min: cues_m += 1
        if netd >= move_netd_min: cues_m += 1
        if (path_len >= move_path_min) and (jitter_ratio <= (static_jitter_min * 0.75)) and (netd >= 0.06): cues_m += 1
        if speed_p90 >= hard_enter_p90: cues_m += 1

        static_by_edge_net = (edge <= static_edge_max and netd <= static_netd_max)
        static_by_slow_clean = (speed_p90 <= quiet_p90 and innov_p95 <= quiet_innov)
        static_by_jitter = (jitter_ratio >= static_jitter_min and netd <= max(static_netd_max, 0.15))
        cues_s = int(static_by_edge_net) + int(static_by_slow_clean) + int(static_by_jitter)

        if cues_m >= 1 and cues_m >= cues_s:
            conf = min(1.0, 0.55 + 0.12 * cues_m)
            return "moving", conf

        if cues_s >= 1 and cues_s > cues_m:
            conf = min(1.0, 0.55 + 0.15 * cues_s)
            return "static", conf

        return "unknown", 0.4

    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        with self._lock:
            summaries = list(self.window_summaries)
            mode_hist = list(self.mode_history)
            th = dict(self.thresholds)

        n = min(len(summaries), len(mode_hist))
        if n == 0:
            return {"timestamp": time.time(), "confusion": {}, "accuracy": 1.0, "n_eval": 0, "current_thresholds": th, "stuck_baseline": False}

        ref_moving = ref_static = ref_unknown = 0
        fast_moving = fast_static = fast_baseline = 0
        w_fn_move = 0.0
        w_fp_move = 0.0
        w_correct = 0.0
        w_total = 0.0

        rows = []
        baseline_streak = 0
        max_baseline_streak = 0

        for s, m in zip(summaries[-n:], mode_hist[-n:]):
            ref, conf = self._classify_ref_with_conf(s, th)
            mode = m.get("mode", "baseline")

            if mode == "baseline":
                baseline_streak += 1
            else:
                max_baseline_streak = max(max_baseline_streak, baseline_streak)
                baseline_streak = 0

            if ref == "moving":
                ref_moving += 1
                if mode in ("static", "baseline"):
                    w_fn_move += conf
                else:
                    w_correct += conf
                w_total += conf
            elif ref == "static":
                ref_static += 1
                if mode in ("moving", "baseline"):
                    w_fp_move += conf
                else:
                    w_correct += conf
                w_total += conf
            else:
                ref_unknown += 1

            if mode == "moving": fast_moving += 1
            elif mode == "static": fast_static += 1
            else: fast_baseline += 1

            rows.append({
                "win_id": m.get("win_id"),
                "mode": mode,
                "ref": ref,
                "ref_conf": conf,
                "speed_p90": s.get("speed_p90", 0.0),
                "mean_speed": s.get("mean_speed", 0.0),
                "pct_good": s.get("pct_good", 0.0),
                "innov_p95": s.get("innov_p95", 0.0),
                "edge_shift_1s": s.get("edge_shift_1s", 0.0),
                "net_disp": s.get("net_disp", 0.0),
                "total_path_length": s.get("total_path_length", 0.0),
                "rms": s.get("rms", 0.0),
                "jitter_ratio": s.get("jitter_ratio", 0.0),
                "gate_rate": s.get("gate_rate", 0.0),
                "cross_win_r": s.get("cross_win_r", None),
                "cross_win_drift": s.get("cross_win_drift", None),
                "rule_hits": s.get("rule_hits", {}),
                "margins": s.get("margins", {}),
            })

        max_baseline_streak = max(max_baseline_streak, baseline_streak)
        stuck_baseline = (fast_baseline >= int(0.66 * n)) or (max_baseline_streak >= 3)

        accuracy = (w_correct / max(1e-6, w_total)) if w_total > 0 else 1.0
        n_eval = ref_moving + ref_static

        def _avg_margin(key: str) -> float:
            vals = [float(r.get("margins", {}).get(key, 0.0)) for r in rows if r.get("margins") is not None]
            return float(np.mean(vals)) if vals else 0.0

        avg_margins = {
            "hard_enter": _avg_margin("hard_enter"),
            "soft_mean": _avg_margin("soft_mean"),
            "soft_innov": _avg_margin("soft_innov"),
            "soft_good": _avg_margin("soft_good"),
            "quiet_p90": _avg_margin("quiet_p90"),
            "quiet_innov": _avg_margin("quiet_innov"),
            "move_edge": _avg_margin("move_edge"),
            "move_netd": _avg_margin("move_netd"),
            "move_path": _avg_margin("move_path"),
            "static_edge": _avg_margin("static_edge"),
            "static_netd": _avg_margin("static_netd"),
            "static_jitter": _avg_margin("static_jitter"),
            "static_rms": _avg_margin("static_rms"),
            "static_drift": _avg_margin("static_drift"),
            "pct_good": _avg_margin("pct_good"),
            "gate_rate": _avg_margin("gate_rate"),
            "static_xwin_r": _avg_margin("static_xwin_r"),
            "static_xwin_drift": _avg_margin("static_xwin_drift"),
        }

        def _hit_rate(flag: str) -> float:
            vals = [1.0 if r.get("rule_hits", {}).get(flag, False) else 0.0 for r in rows]
            return float(np.mean(vals)) if vals else 0.0

        rule_hit_rates = {
            "moving_rule1": _hit_rate("moving_rule1"),
            "moving_rule2": _hit_rate("moving_rule2"),
            "static_rule": _hit_rate("static_rule"),
            "static_cluster": _hit_rate("static_cluster"),
            "static_xwin": _hit_rate("static_xwin"),
        }

        jitter_fp_signature = (
            (avg_margins["static_rms"] > 0.0 or avg_margins["static_xwin_r"] > 0.0) and
            (avg_margins["static_drift"] > 0.0 or avg_margins["static_xwin_drift"] > 0.0) and
            (avg_margins["static_jitter"] > 0.0) and
            (avg_margins["move_path"] > 0.0) and
            (avg_margins["move_edge"] < 0.0) and
            (avg_margins["move_netd"] < 0.0)
        )

        metrics = {
            "timestamp": time.time(),
            "n_eval": int(n_eval),
            "accuracy": float(accuracy),
            "confusion": {
                "ref_moving": ref_moving,
                "ref_static": ref_static,
                "ref_unknown": ref_unknown,
                "fast_moving": fast_moving,
                "fast_static": fast_static,
                "fast_baseline": fast_baseline,
                "moving_but_fast_static_or_baseline": float(w_fn_move),
                "static_but_fast_moving_or_baseline": float(w_fp_move)
            },
            "recent_windows": rows[-6:],
            "current_thresholds": th,
            "aggregates": {
                "avg_edge_shift_1s": float(np.mean([r["edge_shift_1s"] for r in rows])) if rows else 0.0,
                "avg_net_disp": float(np.mean([r["net_disp"] for r in rows])) if rows else 0.0,
                "avg_speed_p90": float(np.mean([r["speed_p90"] for r in rows])) if rows else 0.0,
                "avg_ref_conf": float(np.mean([r["ref_conf"] for r in rows])) if rows else 0.0,
            },
            "avg_margins": avg_margins,
            "rule_hit_rates": rule_hit_rates,
            "stuck_baseline": bool(stuck_baseline),
            "patterns": {"jitter_fp_signature": bool(jitter_fp_signature)}
        }
        print(f"[HyperTuner] Confusion(wt): FNm={metrics['confusion']['moving_but_fast_static_or_baseline']:.2f}, "
              f"FPm={metrics['confusion']['static_but_fast_moving_or_baseline']:.2f}, "
              f"acc={metrics['accuracy']:.2f} (n_eval={n_eval}, stuck_baseline={stuck_baseline}, jitterFP={jitter_fp_signature})")
        return metrics

    def _create_llm_payload(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "current_thresholds": metrics.get("current_thresholds", {}),
            "threshold_bounds": THRESHOLD_BOUNDS,
            "max_step_fraction": MAX_STEP_FRACTION,
            "confusion": metrics.get("confusion", {}),
            "accuracy": metrics.get("accuracy", 1.0),
            "n_eval": metrics.get("n_eval", 0),
            "stuck_baseline": metrics.get("stuck_baseline", False),
            "avg_margins": metrics.get("avg_margins", {}),
            "patterns": metrics.get("patterns", {}),
        }

    def _request_threshold_adjustments(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.agent:
            return None
        try:
            result = self.agent.advise_thresholds(payload)
            self._last_llm_response = result
            return result
        except Exception as e:
            print(f"[HyperTuner] LLM error: {e}")
            import traceback; traceback.print_exc()
            return None

    def _apply_threshold_adjustments(self, adjustments: Dict[str, Any]) -> None:
        if not adjustments or "thresholds" not in adjustments:
            print("[HyperTuner] No threshold adjustments")
            return

        new_th = adjustments["thresholds"]
        explanation = adjustments.get("explanation", "Thresholds updated.")
        significant = False
        changed = False

        with self._lock:
            for k, v in new_th.items():
                if k not in THRESHOLD_BOUNDS or k not in self.thresholds:
                    continue
                lo, hi = THRESHOLD_BOUNDS[k]
                old = float(self.thresholds[k])
                stepped = _clip_step(old, float(v), lo, hi, MAX_STEP_FRACTION)
                if abs(stepped - old) > 1e-6:
                    changed = True
                    if abs(stepped - old) / max(1e-6, hi - lo) > 0.05:
                        significant = True
                    self.thresholds[k] = stepped
                    print(f"[HyperTuner] {k}: {old:.4g} -> {stepped:.4g} (target {float(v):.4g})")

            self.last_adjustment_time = time.time()
            self.last_adjustment_explanation = explanation
            if significant:
                self._last_significant_change = time.time()

        if changed:
            print("[HyperTuner] ✅ Threshold adjustments applied")
        else:
            print("[HyperTuner] ℹ️ No effective changes (step-limit/identical)")

    def _maybe_apply_small_fallback(self, metrics: Dict[str, Any]) -> None:
        conf = metrics.get("confusion", {})
        n_eval = int(metrics.get("n_eval", 0))
        if n_eval < 4 and not metrics.get("stuck_baseline", False):
            return
        fn_move = float(conf.get("moving_but_fast_static_or_baseline", 0.0))
        fp_move = float(conf.get("static_but_fast_moving_or_baseline", 0.0))
        jitter_fp = bool(metrics.get("patterns", {}).get("jitter_fp_signature", False))
        avg_m = metrics.get("avg_margins", {})

        with self._lock:
            th = dict(self.thresholds)

        if jitter_fp or (fp_move > fn_move and (avg_m.get("move_path", 0.0) > 0.0) and (avg_m.get("static_rms", 0.0) > 0.0)):
            print("[HyperTuner] Fallback: static-but-jittery detected → widen cluster & raise geom moving gates")
            adj = {
                "quiet_p90": th["quiet_p90"] * 1.10,
                "quiet_innov": th["quiet_innov"] * 1.10,
                "static_rms_max": min(THRESHOLD_BOUNDS["static_rms_max"][1], th["static_rms_max"] * 1.20),
                "static_drift_max": min(THRESHOLD_BOUNDS["static_drift_max"][1], th["static_drift_max"] * 1.20),
                "move_path_min": min(THRESHOLD_BOUNDS["move_path_min"][1], th["move_path_min"] * 1.15),
                "move_edge_min": min(THRESHOLD_BOUNDS["move_edge_min"][1], th["move_edge_min"] * 1.10),
                "move_netd_min": min(THRESHOLD_BOUNDS["move_netd_min"][1], th["move_netd_min"] * 1.05),
                "moving_streak": max(1, min(3, int(th["moving_streak"] + 1))),
                "static_cwin_r_max": min(THRESHOLD_BOUNDS["static_cwin_r_max"][1], th["static_cwin_r_max"] * 1.15),
                "static_cwin_drift_max": min(THRESHOLD_BOUNDS["static_cwin_drift_max"][1], th["static_cwin_drift_max"] * 1.15),
            }
            self._apply_threshold_adjustments({"thresholds": adj, "explanation": "Fallback jitter-static widening."})
            return

        if (fn_move > fp_move) or (metrics.get("stuck_baseline", False) and (avg_m.get("hard_enter", 0.0) < 0.0)):
            print("[HyperTuner] Fallback: make MOVING easier")
            adj = {
                "hard_enter_p90": th["hard_enter_p90"] * 0.90,
                "soft_mean": th["soft_mean"] * 0.90,
                "soft_innov": th["soft_innov"] * 0.95,
                "soft_good": max(0.6, th["soft_good"] * 0.97),
                "move_edge_min": th["move_edge_min"] * 0.90,
                "move_netd_min": th["move_netd_min"] * 0.90,
                "move_path_min": th["move_path_min"] * 0.90,
                "static_streak": max(1, min(3, int(th["static_streak"] + 1))),
                "moving_streak": max(1, min(3, int(th["moving_streak"]))),
            }
        else:
            print("[HyperTuner] Fallback: make STATIC easier / MOVING harder")
            adj = {
                "quiet_p90": th["quiet_p90"] * 1.15,
                "quiet_innov": th["quiet_innov"] * 1.15,
                "static_streak": max(1, int(th["static_streak"] - 1)),
                "hard_enter_p90": th["hard_enter_p90"] * 1.05,
                "soft_mean": th["soft_mean"] * 1.05,
                "soft_innov": th["soft_innov"] * 1.05,
                "move_edge_min": th["move_edge_min"] * 1.10,
                "move_netd_min": th["move_netd_min"] * 1.10,
                "move_path_min": th["move_path_min"] * 1.10,
                "min_pct_good": min(0.8, th["min_pct_good"] * 1.05),
                "max_gate_rate": max(0.3, th["max_gate_rate"] * 0.98),
            }
        self._apply_threshold_adjustments({"thresholds": adj, "explanation": "Fallback auto-nudge by confusion bias."})
