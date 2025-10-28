from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, List
import json, time, os

JsonCaller = Callable[[List[Dict[str, Any]], Dict[str, Any]], str]

SCHEMA_TEXT = """
Output strict JSON only (no prose), matching:
{
  "mode": "static" | "moving" | "erratic",
  "kf": { "process_var": number, "meas_var": number, "innovation_max": number }
}
"""

THRESHOLD_STRATEGY_PROMPT = """
You are deciding how to nudge thresholds for a 5s fast loop (moving vs static).
You will NOT output any numeric thresholds. Pick ONE strategy label only.

You get:
- accuracy (0..1)
- confusion weights (FP_moving = static_but_fast_moving_or_baseline; FN_moving = moving_but_fast_static_or_baseline)
- patterns: jitter_fp_signature (True if path is large but edge/net small and cluster/jitter evidence is strong), stuck_baseline
- avg_margins (sign of these indicates which side of threshold typical windows sit on)

Pick one of:
- "NO_CHANGE": if accuracy >= 0.90 and both FP and FN <= 1.0
- "EASE_STATIC": when FP_moving dominates (fast loop over-calls moving on static)
- "EASE_MOVING": when FN_moving dominates (fast loop under-calls moving)
- "TIGHTEN_QUALITY": when data quality is the issue (stuck_baseline or pct_good/gate_rate margins are often negative)
- "WIDEN_CLUSTER": when jitter_fp_signature is True (accept bigger cluster/rms/drift for static)

Output STRICT JSON:
{ "strategy": "NO_CHANGE" | "EASE_STATIC" | "EASE_MOVING" | "TIGHTEN_QUALITY" | "WIDEN_CLUSTER", "notes": "one short sentence" }
"""

_MODE_DEFAULT_KF = {
    "static":  {"process_var": 0.001, "meas_var": 0.30, "innovation_max": 0.20},
    "moving":  {"process_var": 0.03, "meas_var": 0.10, "innovation_max": 0.80},
    "erratic": {"process_var": 1.20, "meas_var": 0.18, "innovation_max": 1.90},
}

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))

@dataclass
class LLMAdvisorAgent:
    caller: Optional[JsonCaller] = None
    temperature: float = 0.0
    max_tokens: int = 200
    caller_options: Optional[Dict[str, Any]] = None
    log_path: Optional[str] = None

    pv_min: float = 0.03; pv_max: float = 1.50
    rv_min: float = 0.03; rv_max: float = 0.25
    inno_min: float = 0.60; inno_max: float = 2.00

    def _fallback(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        v_p90 = float(summary.get("speed_p90", summary.get("mean_speed", 0.0)))
        i95 = float(summary.get("innov_p95", 1.0))
        last_mode = summary.get("last_mode", "baseline")
        static_streak = int(summary.get("static_streak", 0))
        if v_p90 >= 60.0:
            mode = "moving"
        elif (v_p90 <= 30.0) and (i95 <= 0.12) and (static_streak >= 2):
            mode = "static"
        else:
            mode = last_mode if isinstance(last_mode, str) else "baseline"
        kf = _MODE_DEFAULT_KF.get(mode, _MODE_DEFAULT_KF.get("moving"))
        return {
            "mode": mode,
            "kf": {
                "process_var": float(_clamp(kf["process_var"], self.pv_min, self.pv_max)),
                "meas_var": float(_clamp(kf["meas_var"], self.rv_min, self.rv_max)),
                "innovation_max": float(_clamp(kf["innovation_max"], self.inno_min, self.inno_max)),
            }
        }

    def _threshold_fallback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        curr = payload.get("current_thresholds", {})
        return {"thresholds": dict(curr), "explanation": "No LLM: hold thresholds."}

    def _log_jsonl(self, rec: Dict[str, Any]):
        if not self.log_path: return
        try:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        except Exception:
            pass
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _strategy_to_adjustment(self, strategy: str, bounds: Dict[str, Any], curr: Dict[str, float], patterns: Dict[str, Any]) -> Dict[str, float]:
        th = dict(curr)
        def step(key, frac_delta):
            lo, hi = bounds[key]
            rng = hi - lo
            target = _clamp(th[key] + frac_delta * rng, lo, hi)
            th[key] = float(target)

        if strategy == "NO_CHANGE":
            return th

        if strategy == "EASE_STATIC":
            step("quiet_p90", +0.15)
            step("quiet_innov", +0.15)
            step("static_streak", -0.10)
            step("hard_enter_p90", +0.05)
            step("soft_mean", +0.05)
            step("soft_innov", +0.05)
            step("move_edge_min", +0.10)
            step("move_netd_min", +0.10)
            step("move_path_min", +0.10)
            step("static_rms_max", +0.10)
            step("static_drift_max", +0.10)
            step("min_pct_good", +0.05)
            step("max_gate_rate", -0.05)
            return th

        if strategy == "EASE_MOVING":
            step("hard_enter_p90", -0.10)
            step("soft_mean", -0.10)
            step("soft_innov", -0.10)
            step("soft_good", -0.05)
            step("move_edge_min", -0.10)
            step("move_netd_min", -0.10)
            step("move_path_min", -0.10)
            step("static_streak", +0.10)
            return th

        if strategy == "TIGHTEN_QUALITY":
            step("min_pct_good", +0.10)
            step("max_gate_rate", -0.10)
            step("static_rms_max", +0.05)
            step("static_drift_max", +0.05)
            return th

        if strategy == "WIDEN_CLUSTER":
            step("static_rms_max", +0.20)
            step("static_drift_max", +0.20)
            if "static_cwin_r_max" in th: step("static_cwin_r_max", +0.15)
            if "static_cwin_drift_max" in th: step("static_cwin_drift_max", +0.15)
            step("move_path_min", +0.15)
            step("move_edge_min", +0.10)
            step("move_netd_min", +0.05)
            step("quiet_p90", +0.10)
            step("quiet_innov", +0.10)
            step("moving_streak", +0.10)
            return th

        return th

    def advise_thresholds(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if self.caller is None:
            return self._threshold_fallback(payload)

        confusion = payload.get("confusion", {})
        patterns = payload.get("patterns", {})
        avg_margins = payload.get("avg_margins", {})
        summary = {
            "accuracy": float(payload.get("accuracy", 1.0)),
            "n_eval": int(payload.get("n_eval", 0)),
            "FP_moving": float(confusion.get("static_but_fast_moving_or_baseline", 0.0)),
            "FN_moving": float(confusion.get("moving_but_fast_static_or_baseline", 0.0)),
            "patterns": {
                "jitter_fp_signature": bool(patterns.get("jitter_fp_signature", False)),
                "stuck_baseline": bool(payload.get("stuck_baseline", False)),
            },
            "avg_margins": {
                k: float(v) for k, v in avg_margins.items() if k in (
                    "quiet_p90","quiet_innov","move_path","move_edge","move_netd",
                    "static_rms","static_drift","static_jitter","pct_good","gate_rate"
                )
            }
        }

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": THRESHOLD_STRATEGY_PROMPT},
            {"role": "user", "content": json.dumps(summary)}
        ]
        opts: Dict[str, Any] = {"temperature": self.temperature, "max_tokens": max(48, self.max_tokens)}
        if self.caller_options: opts.update(self.caller_options)

        t0 = time.perf_counter()
        log_rec = {
            "t": time.time(),
            "type": "threshold_tuning",
            "metrics": {"confusion": confusion, "accuracy": summary["accuracy"]},
            "current_thresholds": payload.get("current_thresholds", {}),
            "options": opts,
            "prompt": json.dumps(summary),
            "system_prompt": THRESHOLD_STRATEGY_PROMPT
        }

        try:
            text = self.caller(messages, opts)
            elapsed = time.perf_counter() - t0

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                import re
                m = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
                if not m: raise
                data = json.loads(m.group(1))

            strategy = str(data.get("strategy", "NO_CHANGE"))
            bounds = payload.get("threshold_bounds", {})
            curr = payload.get("current_thresholds", {})
            th = self._strategy_to_adjustment(strategy, bounds, curr, summary.get("patterns", {}))

            for k, (lo, hi) in bounds.items():
                if k not in th and k in curr:
                    th[k] = float(curr[k])
                if k in th:
                    th[k] = _clamp(float(th[k]), lo, hi)

            out = {
                "thresholds": th,
                "explanation": f"Strategy={strategy}. " + (data.get("notes") or "Small bounded tweaks.")
            }
            log_rec.update({"ok": True, "elapsed_ms": int(elapsed*1000), "raw": text, "advice": out})
            self._log_jsonl(log_rec)
            return out

        except Exception as e:
            elapsed = time.perf_counter() - t0
            log_rec.update({"ok": False, "elapsed_ms": int(elapsed*1000), "error": str(e)})
            self._log_jsonl(log_rec)
            return self._threshold_fallback(payload)
