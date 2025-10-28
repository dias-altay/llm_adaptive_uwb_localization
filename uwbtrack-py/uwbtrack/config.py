import argparse, sys, json
from pathlib import Path
from typing import Dict, Any, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

DEFAULT_CONFIG_PATHS = [
    Path("./uwbtrack.yaml"),
    Path.home() / ".config" / "uwbtrack" / "config.yaml",
    Path("./uwbtrack.json"),
]

def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    ext = path.suffix.lower()
    if ext in (".yaml", ".yml"):
        if yaml is None:
            raise RuntimeError("PyYAML is not installed but a YAML config was requested.")
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    elif ext == ".json":
        with open(path, "r") as f:
            return json.load(f) or {}
    else:
        raise ValueError(f"Unsupported config extension for {path}")

def load_config(path: str | None) -> Tuple[Dict[str, Any], Path | None]:
    if path:
        p = Path(path)
        return _load_yaml_or_json(p), p if p.exists() else None
    for p in DEFAULT_CONFIG_PATHS:
        if p.exists():
            return _load_yaml_or_json(p), p
    return {}, None

def load_config_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a YAML/JSON configuration file and return a normalized dict suitable for the GUI.
    - Returns {} when the file does not exist or is invalid.
    - Normalizes top-level keys by replacing '-' with '_'.
    - Resolves common path-like keys relative to the config file location.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        return {}

    raw = _load_yaml_or_json(p)
    if not isinstance(raw, dict):
        return {}

    cfg: Dict[str, Any] = {}
    for k, v in raw.items():
        nk = k.replace("-", "_")
        cfg[nk] = v

    cfg_dir = p.parent

    for key in ("bias_file", "log_frames_csv", "log_windows_csv", "runs_dir"):
        if key in cfg and cfg.get(key):
            cfg[key] = _resolve_relative_to(cfg_dir, cfg.get(key))

    return cfg

def add_cli_args(ap: argparse.ArgumentParser) -> None:
    s = argparse.SUPPRESS
    ap.add_argument("--config", default=s, help="Path to YAML/JSON config (defaults to uwbtrack.yaml)")
    ap.add_argument("--port", default=s)
    ap.add_argument("--baud", type=int, default=s)
    ap.add_argument("--anchors", default=s)
    ap.add_argument("--min-anchors", type=int, default=s)
    ap.add_argument("--no-plot", action="store_true", default=s)
    ap.add_argument("--bias-file", default=s)
    ap.add_argument("--no-calib", action="store_true", default=s)
    ap.add_argument("--print-qual", action="store_true", default=s)
    ap.add_argument("--debug-geom", action="store_true", default=s)
    ap.add_argument("--solve-2d", action="store_true", default=s)
    ap.add_argument("--floor-z", type=float, default=s)
    ap.add_argument("--certainty-dist", type=float, default=s)
    ap.add_argument("--outlier-threshold", type=float, default=s)
    ap.add_argument("--min-weight", type=float, default=s)
    ap.add_argument("--stats-every", type=int, default=s)
    ap.add_argument("--strong-weight", type=float, default=s)
    ap.add_argument("--innovation-max", type=float, default=s)
    ap.add_argument("--weak-avgw", type=float, default=s)
    ap.add_argument("--kf-process", type=float, default=s, help="Kalman process noise (std-dev)")
    ap.add_argument("--kf-meas", type=float, default=s, help="Kalman measurement noise (std-dev)")
    ap.add_argument("--log-frames", dest="log_frames_csv", default=s,
                    help="CSV path for per-frame logs (RAW/KF/LLM, gates, knobs)")
    ap.add_argument("--log-windows", dest="log_windows_csv", default=s,
                    help="CSV path for per-window (advisory) logs")
    ap.add_argument("--origin-shift", dest="origin_shift", action="store_true", default=s,
                    help="Shift coordinate frame so that (origin_shift_x, origin_shift_y) becomes (0,0)")
    ap.add_argument("--origin-shift-x", type=float, default=s, help="X of the new origin in the old frame")
    ap.add_argument("--origin-shift-y", type=float, default=s, help="Y of the new origin in the old frame")
    ap.add_argument("--max-range-m", type=float, default=s,
                    help="Hard cap for accepted UWB range (meters). If unset, auto-derives from anchor span.")
    ap.add_argument("--llm-kf", dest="llm_kf_enabled", action="store_true", default=s,
                    help="Enable LLM-tuned KF advisory (parallel to baseline KF)")
    ap.add_argument("--llm-window-s", type=float, default=s,
                    help="Advisory window length in seconds (default 5.0)")
    ap.add_argument("--llm-log", dest="llm_log_csv", default=s,
                    help="CSV path to log raw, baseline KF, LLM-KF, and advisory outputs")
    ap.add_argument("--hybrid-tuning", dest="hybrid_tuning_enabled", action="store_true", default=s,
                    help="Enable hybrid fast+slow loop with LLM threshold tuning")
    ap.add_argument("--slow-loop-s", type=float, default=s,
                    help="Slow loop interval in seconds (default 30.0)")
    ap.add_argument("--llama-url", default=s, help="llama.cpp server base URL (default http://localhost:8080)")
    ap.add_argument("--llama-timeout", type=float, default=s, help="HTTP timeout seconds for llama.cpp (default 1.5)")
    ap.add_argument("--llama-max-tokens", type=int, default=s, help="Max tokens for llama response (default 80)")
    ap.add_argument("--llama-num-gpu", type=int, default=s)
    ap.add_argument("--mocap-enabled", action="store_true", default=s,
                    help="Enable mocap WebSocket ground truth logging and joining")
    ap.add_argument("--websocket-ip", default=s,
                    help="Mocap WebSocket server IP or URL (ws://host:port or host)")
    ap.add_argument("--websocket-port", type=int, default=s,
                    help="Mocap WebSocket server port (default 8765)")
    ap.add_argument("--mocap-object", default=s, help="Mocap object name to filter (default Goal)")
    ap.add_argument("--mocap-id", type=int, default=s, help="Mocap object id to filter (default 2)")
    ap.add_argument("--runs-dir", default=s, help="Base directory to store runs (default ./runs)")
    ap.add_argument("--uwb-ws-url", default=s, help="UWB WebSocket URL (e.g., ws://10.0.0.5:8766/)")
    ap.add_argument("--logging-enabled", action="store_true", default=s,
                    help="Enable writing CSV logs (default: on unless disabled in config)")

def _resolve_relative_to(base_dir: Path | None, value: str | None) -> str | None:
    if not value:
        return value
    p = Path(value)
    if p.is_absolute() or base_dir is None:
        return str(p)
    return str((base_dir / p).resolve())

def parse_and_merge(argv=None) -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="UWB Tracking (robust trilateration + KF)")
    add_cli_args(ap)
    ns, _unknown = ap.parse_known_args(argv)
    cli = vars(ns)

    cfg_path_arg = cli.get("config", None)
    cfg, cfg_path = load_config(cfg_path_arg if cfg_path_arg is not argparse.SUPPRESS else None)
    cfg_dir = cfg_path.parent if cfg_path is not None else None

    eff: Dict[str, Any] = dict(cfg)
    for k, v in cli.items():
        if v is not argparse.SUPPRESS:
            eff[k] = v

    eff.setdefault("baud", 115200)
    eff.setdefault("anchors", "1:0,0,1.65;2:7.48,0,1.65;3:4.60,3.94,2.60")
    eff.setdefault("min_anchors", 3)
    eff.setdefault("no_plot", False)
    eff.setdefault("bias_file", "biases.json")
    eff.setdefault("no_calib", True)
    eff.setdefault("print_qual", True)
    eff.setdefault("debug_geom", False)
    eff.setdefault("solve_2d", True)
    eff.setdefault("floor_z", 0.0)
    eff.setdefault("certainty_dist", 0.2)
    eff.setdefault("outlier_threshold", 3.0)
    eff.setdefault("min_weight", 0.05)
    eff.setdefault("stats_every", 20)
    eff.setdefault("strong_weight", 0.20)
    eff.setdefault("innovation_max", 1.5)
    eff.setdefault("weak_avgw", 0.25)
    eff.setdefault("kf_process", 0.15)
    eff.setdefault("kf_meas", 0.30)
    eff.setdefault("origin_shift", False)
    eff.setdefault("origin_shift_x", 0.0)
    eff.setdefault("origin_shift_y", 0.0)
    eff.setdefault("max_range_m", None)
    eff.setdefault("llm_kf_enabled", False)
    eff.setdefault("llm_window_s", 5.0)
    eff.setdefault("log_frames_csv", "")
    eff.setdefault("log_windows_csv", "")
    eff.setdefault("llm_log_csv", "")
    eff.setdefault("llama_url", "http://localhost:11434")
    eff.setdefault("llama_timeout", 1.5)
    eff.setdefault("llama_max_tokens", 80)
    eff.setdefault("llama_num_gpu", None)
    eff.setdefault("mocap_enabled", False)
    eff.setdefault("websocket_ip", "ws://192.168.53.225")
    eff.setdefault("websocket_port", 8765)
    eff.setdefault("mocap_object", "Goal")
    eff.setdefault("mocap_id", 2)
    eff.setdefault("runs_dir", "runs")
    eff.setdefault("uwb_ws_url", "ws://10.131.128.4:8766/")
    eff.setdefault("logging_enabled", True)
    eff.setdefault("hybrid_tuning_enabled", False)
    eff.setdefault("slow_loop_s", 30.0)

    if "llm_hybrid_tuning_enabled" in eff:
        eff["llm_kf_enabled"] = eff["hybrid_tuning_enabled"] = eff["llm_hybrid_tuning_enabled"]

    eff.setdefault("show_raw_position", False)

    eff["bias_file"] = _resolve_relative_to(cfg_dir, eff.get("bias_file"))
    if eff.get("log_frames_csv"):
        eff["log_frames_csv"] = _resolve_relative_to(cfg_dir, eff.get("log_frames_csv"))
    if eff.get("log_windows_csv"):
        eff["log_windows_csv"] = _resolve_relative_to(cfg_dir, eff.get("log_windowsCsv"))
    if eff.get("runs_dir"):
        eff["runs_dir"] = _resolve_relative_to(cfg_dir, eff.get("runs_dir"))

    if not eff.get("port"):
        ap.error("Missing --port (you can also set `port:` in uwbtrack.yaml)")
    return eff
