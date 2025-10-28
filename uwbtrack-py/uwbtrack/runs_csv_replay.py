import argparse, csv, time, math, sys, os, re
from pathlib import Path

try:
    from .plotting import init_plot_with_truth, update_plot_with_truth, init_plot, update_plot
    from .geom import parse_anchor_map
except ImportError:
    from plotting import init_plot_with_truth, update_plot_with_truth, init_plot, update_plot  # type: ignore
    from geom import parse_anchor_map  # type: ignore

ANCHOR_SPEC_RE = re.compile(r"(\d+)\s*:\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)(?:\s*,\s*([-+0-9.eE]+))?")

def parse_anchor_inline(spec: str):
    """
    Parse "1:0,0,1.5;2:3.2,0,1.5;3:1.0,2.0,1.5" (z optional) -> {id:(x,y[,z])}
    Ignores empty segments; raises if no valid anchors parsed.
    """
    anchors = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"\s*(\d+)\s*:\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)(?:\s*,\s*([-+0-9.eE]+))?\s*", part)
        if not m:
            continue
        aid = int(m.group(1))
        x = float(m.group(2)); y = float(m.group(3))
        anchors[aid] = (x, y, float(m.group(4))) if m.group(4) else (x, y)
    if not anchors:
        raise ValueError(f"Could not parse any anchors from inline spec: {spec}")
    return anchors

def load_anchors_from_yaml(cfg_path: Path):
    """
    Robust extraction:
      1) Try to capture inline string after 'anchors:' on one line.
      2) Fallback: scan entire file for anchor specs and aggregate.
    """
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    text = cfg_path.read_text()

    m = re.search(r'anchors:\s*(?P<q>["\'])?(?P<body>[0-9:;,\s.+\-eE]+)(?P=q)', text)
    anchors = {}
    if m:
        body = m.group("body").strip()
        try:
            anchors = parse_anchor_inline(body)
        except Exception:
            anchors = {}

    if not anchors or len(anchors) < 4:
        scanned = {}
        for m2 in ANCHOR_SPEC_RE.finditer(text):
            aid = int(m2.group(1))
            x = float(m2.group(2)); y = float(m2.group(3))
            z = float(m2.group(4)) if m2.group(4) else None
            scanned[aid] = (x, y, z) if z is not None else (x, y)
        if scanned:
            anchors = scanned

    if not anchors:
        raise ValueError(f"No anchors found in {cfg_path}")
    return anchors

def parse_origin_shift_from_yaml(cfg_path: Path):
    """
    Extract origin_shift, origin_shift_x, origin_shift_y from YAML text.
    Defaults: enabled=False, x=0.0, y=0.0 if absent.
    Comments (# ...) are ignored line-wise.
    """
    enabled = False
    ox = 0.0
    oy = 0.0
    try:
        for raw in cfg_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("origin_shift:"):
                val = line.split(":", 1)[1].strip().lower()
                enabled = val in ("true", "yes", "1")
            elif line.startswith("origin_shift_x:"):
                try: ox = float(line.split(":", 1)[1].strip())
                except Exception: pass
            elif line.startswith("origin_shift_y:"):
                try: oy = float(line.split(":", 1)[1].strip())
                except Exception: pass
    except Exception:
        pass
    return {"enabled": enabled, "x": ox, "y": oy}

def resolve_anchors(arg_value: str | None, cfg_path: Path):
    """
    Resolution priority:
      1) If arg_value provided and is an existing file -> parse_anchor_map(file).
      2) If arg_value provided and not a file -> parse as inline spec.
      3) Else load anchors from cfg_path (uwbtrack.yaml).
    Returns dict {aid:(x,y)} (drops z if present).
    """
    if arg_value:
        p = Path(arg_value).expanduser()
        if p.exists() and p.is_file():
            anchors_xyz = parse_anchor_map(str(p))
        else:
            anchors_xyz = parse_anchor_inline(arg_value)
    else:
        anchors_xyz = load_anchors_from_yaml(cfg_path)
    anchors_2d = {aid: (pos[0], pos[1]) for aid, pos in anchors_xyz.items()}
    return anchors_2d

def parse_args():
    p = argparse.ArgumentParser(
        description="Replay a joined runs_*.csv file in real time (or accelerated)."
    )
    p.add_argument("csv", help="Path to joined runs_*.csv (with mocap columns).")
    p.add_argument("-a", "--anchors",
                   help="Either: path to anchors file OR inline spec like '1:0,0,1.65;2:7.48,0,1.65;...'. "
                        "If omitted, anchors are read from uwbtrack.yaml.")
    p.add_argument("--config", default="../uwbtrack.yaml",
                   help="Config YAML to read anchors from when --anchors not provided (default: uwbtrack.yaml).")
    p.add_argument("--kf", choices=["base", "llm"], default="llm",
                   help="Which KF track to visualize (default: llm).")
    p.add_argument("--speed", type=float, default=1.0,
                   help="Time scaling (>1 faster, <1 slower).")
    p.add_argument("--start-seq", type=int, default=1,
                   help="Start at (first) sequence number >= this (default 1).")
    p.add_argument("--limit", type=int, default=0,
                   help="Limit number of frames (0 = all).")
    p.add_argument("--no-circles", action="store_true",
                   help="Disable range circles (slightly faster).")
    p.add_argument("--quiet", action="store_true",
                   help="Reduce stdout printing.")
    return p.parse_args()

def load_rows(csv_path: Path, start_seq: int):
    rows = []
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                seq = int(row.get("seq") or 0)
                if seq < start_seq:
                    continue
                t = float(row["t"])
            except Exception:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit("No rows found (check --start-seq or file path).")
    return rows

def has_mocap(rows):
    for row in rows:
        if row.get("mocap_ts") not in ("", None):
            try:
                float(row["mocap_ts"])
                return True
            except Exception:
                pass
    return False

def build_latest(anchors_xyz, used_ids, x_raw, y_raw):
    latest = {}
    for aid in used_ids:
        if aid in anchors_xyz:
            ax, ay = anchors_xyz[aid][0], anchors_xyz[aid][1]
            rng = math.hypot(x_raw - ax, y_raw - ay)
            latest[aid] = (rng, 1.0)
    return latest

def main():
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"File not found: {csv_path}")
    cfg_path = Path(args.config).expanduser().resolve()
    try:
        anchors_xyz = resolve_anchors(args.anchors, cfg_path)
    except Exception as e:
        raise SystemExit(f"Failed to resolve anchors: {e}")

    origin_shift_applied = False
    if args.anchors is None:
        osh = parse_origin_shift_from_yaml(cfg_path)
        if osh["enabled"]:
            dx, dy = float(osh["x"]), float(osh["y"])
            if abs(dx) > 1e-12 or abs(dy) > 1e-12:
                anchors_xyz = {aid: (pos[0] - dx, pos[1] - dy) for aid, pos in anchors_xyz.items()}
                origin_shift_applied = True
                print(f"[origin] Shifted anchors by (-{dx:.3f}, -{dy:.3f}); new origin is old ({dx:.3f}, {dy:.3f}).")

    rows = load_rows(csv_path, args.start_seq)
    mocap_available = has_mocap(rows)

    if mocap_available:
        fig, ax, raw_dot, tag_dot, gt_dot, err_line, circles, info_txt = init_plot_with_truth(anchors_xyz)
    else:
        fig, ax, raw_dot, tag_dot, circles = init_plot(anchors_xyz)

    first_t = float(rows[0]["t"])
    real_start = time.time()

    total = len(rows)
    limit = args.limit if args.limit > 0 else total
    speed = max(1e-6, args.speed)
    use_llm = (args.kf == "llm")

    if not args.quiet:
        print(f"[replay] file={csv_path.name} anchors={len(anchors_xyz)} frames={len(rows)} start_seq={args.start_seq} "
              f"mocap={mocap_available} speed={args.speed} KF={args.kf} src={'arg' if args.anchors else cfg_path.name}"
              f"{' origin_shift' if origin_shift_applied else ''}")

    try:
        for idx, row in enumerate(rows[: (args.limit if args.limit > 0 else len(rows))], 1):
            t_row = float(row["t"])
            target_elapsed = (t_row - first_t) / max(1e-6, args.speed)
            while True:
                now_elapsed = time.time() - real_start
                dt = target_elapsed - now_elapsed
                if dt <= 0:
                    break
                time.sleep(min(dt, 0.05))

            try:
                x_raw = float(row["x_raw"]); y_raw = float(row["y_raw"])
                if (args.kf == "llm") and row.get("x_kf_llm") not in (None, "", "None"):
                    x_kf = float(row["x_kf_llm"]); y_kf = float(row["y_kf_llm"])
                else:
                    x_kf = float(row["x_kf_base"]); y_kf = float(row["y_kf_base"])
            except Exception:
                continue

            used_ids = []
            us = row.get("used_ids") or ""
            if us:
                try:
                    used_ids = [int(s) for s in us.split(";") if s.strip().isdigit()]
                except Exception:
                    used_ids = []

            latest = build_latest(anchors_xyz, used_ids, x_raw, y_raw) if (not args.no_circles) else {}

            gt_x = gt_y = None
            mocap_ts = row.get("mocap_ts")
            if mocap_available and mocap_ts not in (None, "", "None"):
                try:
                    gt_x = float(row.get("mocap_x"))
                    gt_y = float(row.get("mocap_y"))
                except Exception:
                    gt_x = gt_y = None

            if mocap_available:
                update_plot_with_truth(
                    fig, raw_dot, tag_dot, gt_dot, err_line, circles,
                    x_raw, y_raw, x_kf, y_kf, used_ids, latest,
                    gt_x=gt_x, gt_y=gt_y, t=t_row
                )
            else:
                update_plot(fig, raw_dot, tag_dot, circles,
                            x_raw, y_raw, x_kf, y_kf, used_ids, latest)

            if not args.quiet and (idx % 25 == 0 or idx == 1):
                print(f"[replay] {idx}/{limit} t_rel={(t_row-first_t):.2f}s raw=({x_raw:.3f},{y_raw:.3f}) kf=({x_kf:.3f},{y_kf:.3f})")

        if not args.quiet:
            print("[replay] done.")
        print("Press Ctrl+C or close the plot window to exit.")
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\n[replay] interrupted.")
    except Exception as e:
        print(f"[replay] error: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
