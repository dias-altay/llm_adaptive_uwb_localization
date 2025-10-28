import sys, time, json
from pathlib import Path
import numpy as np
import threading
import queue

from .config import parse_and_merge
from .geom import parse_anchor_map, robust_solve_with_residuals, _reflect_across_line, _weighted_cost
from .serial_io import init_serial_connection, serial_lines, parse_meas
from .weights import WEIGHT_CFG, RollingQual, meas_weight, meas_diag
from .kalman import CVKalman2D
from .calibration import calibrate_quality_near_anchors, calibrate_affine_midpoints

from .logging_csv import RunLogger

from .llm.callers.ollama_caller import ollama_caller
from .llm.llm_agent import LLMAdvisorAgent
from .llm.llm_kf_advisor import KFTuningAdvisor
from .llm.hyper_tuner import HyperTuner

def start_tracking_embedded(cfg: dict):
    """
    Launch full tracking (same logic as main()) in a background thread but
    return the matplotlib figure so a Qt GUI can embed it.
    """
    import matplotlib.pyplot as _plt
    stop_event = threading.Event()
    session = {
        'figure': None,
        'ax': None,
        'artists': {},
        'pose_queue': queue.Queue(maxsize=2000),
        'thread': None,
        'stop': stop_event.set,
        'with_truth': False,
        'alive': lambda: (session['thread'].is_alive() if session.get('thread') else False),
    }

    def _run():
        args = dict(cfg)
        try:
            anchors_xyz = parse_anchor_map(args["anchors"])
            if bool(args.get("origin_shift", False)):
                dx = float(args.get("origin_shift_x", 0.0) or 0.0)
                dy = float(args.get("origin_shift_y", 0.0) or 0.0)
                if abs(dx) > 1e-12 or abs(dy) > 1e-12:
                    anchors_xyz = {
                        aid: ((p[0]-dx, p[1]-dy, p[2]) if len(p)==3 else (p[0]-dx, p[1]-dy))
                        for aid, p in anchors_xyz.items()
                    }
                    print(f"[origin] Shifted anchors by (-{dx:.3f}, -{dy:.3f}) (embedded).")
            anchors_have_z = (len(next(iter(anchors_xyz.values()))) == 3)
            xs = [p[0] for p in anchors_xyz.values()]
            ys = [p[1] for p in anchors_xyz.values()]
            span_diag = float(np.hypot(max(xs)-min(xs), max(ys)-min(ys))) if len(xs) >= 2 else 5.0
            auto_max_range = max(5.0, span_diag * 2.0 + 5.0)
            max_range_cfg = args.get("max_range_m", None)
            max_range_m = float(max_range_cfg) if (max_range_cfg not in (None, "", False)) else auto_max_range

            if args.get("solve_2d", False):
                solve_anchors = {aid: (p[0], p[1]) for aid, p in anchors_xyz.items()}
                dz_map = {aid: (p[2] if len(p) >= 3 else 0.0) - args["floor_z"] for aid, p in anchors_xyz.items()}
                solve_dims = 2
            else:
                solve_anchors = anchors_xyz
                dz_map = {aid: 0.0 for aid in anchors_xyz}
                solve_dims = len(next(iter(anchors_xyz.values())))
            required_min = max(int(args["min_anchors"]), solve_dims + 1)

            try:
                if str(args.get("port", "")).lower() == "uwb_ws":
                    from .uwb_ws_io import init_ws_connection
                    ser = init_ws_connection(str(args.get("uwb_ws_url", "")), timeout=0.2)
                else:
                    ser = init_serial_connection(args["port"], args["baud"])
            except Exception as e:
                print(f"[embedded-error] IO open failed: {e}")
                return

            calib_params = {}
            bias_path = Path(args["bias_file"])
            if args.get("no_calib", False):
                if bias_path.exists():
                    with open(bias_path) as f:
                        raw = json.load(f)
                    if "range" in raw:
                        calib_params = {int(k): tuple(v) for k, v in raw["range"].items()}
                    if "quality" in raw:
                        WEIGHT_CFG.update(raw["quality"])
            else:
                use_floor_calib = (anchors_have_z or args.get("solve_2d", False))
                qual_cfg = calibrate_quality_near_anchors(ser, anchors_xyz, floor_z=args["floor_z"], use_floor=use_floor_calib)
                WEIGHT_CFG.update(qual_cfg)
                calib_params = calibrate_affine_midpoints(ser, anchors_xyz, floor_z=args["floor_z"], use_floor=use_floor_calib)
                payload = {
                    "quality": {k: float(v) for k, v in WEIGHT_CFG.items()},
                    "range": {str(k): [float(v[0]), float(v[1])] for k, v in calib_params.items()}
                }
                with open(bias_path, "w") as f:
                    json.dump(payload, f, indent=2)

            do_plot = not bool(args.get("no_plot", False))
            plot_with_truth = False
            show_raw = bool(args.get("show_raw_position", False))
            if do_plot:
                if bool(args.get("mocap_enabled", False)):
                    from .plotting import init_plot_with_truth
                    fig, ax, raw_dot, tag_dot, gt_dot, err_line, circles, info_txt = init_plot_with_truth(anchors_xyz, embedded=True, show_raw=show_raw)
                    plot_with_truth = True
                    session['artists'] = {'raw_dot': raw_dot,'tag_dot': tag_dot,'gt_dot': gt_dot,'err_line': err_line,'circles': circles}
                else:
                    from .plotting import init_plot
                    fig, ax, raw_dot, tag_dot, circles = init_plot(anchors_xyz, embedded=True, show_raw=show_raw)
                    session['artists'] = {'raw_dot': raw_dot,'tag_dot': tag_dot,'circles': circles}
                session['figure'] = fig
                session['ax'] = ax
                session['with_truth'] = plot_with_truth

            logger = RunLogger(args)
            kf_base = CVKalman2D(dt=0.1, process_var=float(args.get("kf_process", 0.15)), meas_var=float(args.get("kf_meas", 0.30)))
            kf_llm  = CVKalman2D(dt=0.1, process_var=float(args.get("kf_process", 0.15)), meas_var=float(args.get("kf_meas", 0.30)))

            if 'llm_hybrid_tuning_enabled' in args:
                args['llm_kf_enabled'] = args['hybrid_tuning_enabled'] = args['llm_hybrid_tuning_enabled']

            advisor_enabled = bool(args.get("llm_kf_enabled", False))
            agent = None
            if advisor_enabled:
                agent = LLMAdvisorAgent(
                    caller=ollama_caller,
                    temperature=0.0,
                    max_tokens=int(args.get("llama_max_tokens", 80)),
                    caller_options={
                        "base_url": args.get("llama_url", "http://localhost:11434"),
                        "model": args.get("llama_model", "llama3.1:8b-instruct-q4_K_M"),
                        "timeout": float(args.get("llama_timeout", 1.5)),
                        **({"num_gpu": int(args["llama_num_gpu"])} if args.get("llama_num_gpu") is not None else {})
                    },
                    log_path=(args.get("llm_log_csv") or None),
                )

            hybrid_tuning_enabled = bool(args.get("hybrid_tuning_enabled", False))
            slow_loop_interval = float(args.get("slow_loop_s", 30.0))
            print(f"[embedded] Hybrid tuning enabled: {hybrid_tuning_enabled}, slow loop: {slow_loop_interval}s")
            if agent:
                print(f"[embedded] LLM agent is available for tuning")
            else:
                print(f"[embedded] No LLM agent available")

            tuner = KFTuningAdvisor(
                window_s=float(args.get("llm_window_s", 5.0)),
                default_innovation_max=float(args["innovation_max"]),
                agent=agent,
                default_kf_process=float(args.get("kf_process", 0.15)),
                default_kf_meas=float(args.get("kf_meas", 0.30)),
                enable_hyper_tuner=hybrid_tuning_enabled,
                slow_loop_interval=slow_loop_interval
            )

            if hybrid_tuning_enabled:
                if tuner._hyper_tuner is not None:
                    print(f"[embedded] HyperTuner successfully initialized")
                else:
                    print(f"[embedded] ERROR: HyperTuner failed to initialize")
                    if agent is None:
                        print(f"[embedded] Cause: LLM agent is None")
                        return

            logger = RunLogger(args)
            last_raw_sample = None
            last_pose_time = time.time()
            stale_notified = False
            latest = {}
            last_seen = {}
            max_meas_age_s = float(args.get("max_meas_age_s", 2.0))
            rq = RollingQual(maxlen=400)
            frames_in = poses_out = seq = 0

            gt_last_xy = (None, None)
            gt_last_ts = None

            for line in serial_lines(ser):
                if stop_event.is_set():
                    break
                now = time.time()

                if max_meas_age_s > 0:
                    for a, ts in list(last_seen.items()):
                        if (now - ts) > max_meas_age_s:
                            last_seen.pop(a, None)
                            latest.pop(a, None)

                if (now - last_pose_time) > 3.0 and not stale_notified:
                    latest.clear()
                    last_seen.clear()
                    stale_notified = True

                m = parse_meas(line)
                if not m:
                    continue
                frames_in += 1
                tuner.note_any_frame(now)
                aid = m["aid"]; rng_meas = m["range"]

                xsane = (np.isfinite(rng_meas) and rng_meas > 0.0 and rng_meas <= (max_range_m * 3.0))
                if not xsane:
                    latest.pop(aid, None); last_seen.pop(aid, None)
                    continue

                alpha, beta = calib_params.get(aid, (1.0, 0.0))
                rng_corr = max(0.0, alpha * rng_meas + beta)
                d = meas_diag(m)
                rq.push(d["snr_db"], d["rpc"])
                rq.update_cfg(WEIGHT_CFG, time.time(), period=1.0)
                dz = dz_map.get(aid, 0.0)
                if args.get("solve_2d", False):
                    if rng_corr < abs(dz):
                        latest.pop(aid, None); last_seen.pop(aid, None)
                        continue
                    v = rng_corr * rng_corr - dz * dz
                    rng_use = float(np.sqrt(v)) if v > 0.0 else 0.0
                else:
                    rng_use = rng_corr
                if (not np.isfinite(rng_use)) or rng_use <= 0.0 or rng_use > (max_range_m * 1.5):
                    latest.pop(aid, None); last_seen.pop(aid, None)
                    continue

                w = meas_weight(m)
                latest[aid] = (rng_use, w)
                last_seen[aid] = now

                current_anchors = sorted(set(solve_anchors) & set(latest))
                if len(current_anchors) < required_min:
                    continue

                candidates = {
                    a: latest[a] for a in current_anchors
                    if (latest[a][1] >= float(args["min_weight"])) and np.isfinite(latest[a][0]) and (0.0 < latest[a][0] <= max_range_m)
                }
                if len(candidates) < required_min:
                    continue
                strong_thresh = max(float(args["min_weight"]), float(args["strong_weight"]))
                if float(args["strong_weight"]) > 0:
                    good = sum(1 for (_d0, ww) in candidates.values() if ww >= strong_thresh)
                    if good < 2:
                        continue
                certain_anchor = None; min_dist = float('inf')
                cd = float(args["certainty_dist"])
                if cd > 0:
                    for a, (dist, _w_) in candidates.items():
                        if 0 < dist < cd and dist < min_dist:
                            min_dist = dist; certain_anchor = a
                try:
                    if certain_anchor is not None:
                        est = solve_anchors[certain_anchor]; used_ids = {certain_anchor}
                    else:
                        est, used_ids = robust_solve_with_residuals(
                            solve_anchors, candidates,
                            required_min=required_min,
                            k_mad=float(args["outlier_threshold"])
                        )
                        if est is not None and len(candidates) >= 2:
                            top2 = sorted(candidates.items(), key=lambda kv: (-kv[1][1], kv[1][0]))[:2]
                            a_pt = solve_anchors[top2[0][0]]; b_pt = solve_anchors[top2[1][0]]
                            p_ref = _reflect_across_line(est, a_pt, b_pt)
                            if _weighted_cost(p_ref, solve_anchors, candidates) + 0.02 < _weighted_cost(est, solve_anchors, candidates):
                                est = p_ref
                except Exception:
                    continue
                if est is None or len(used_ids) < required_min:
                    continue
                x_raw, y_raw = float(est[0]), float(est[1])
                if (not np.isfinite(x_raw)) or (not np.isfinite(y_raw)):
                    continue
                if last_raw_sample is not None:
                    dt = max(1e-3, now - last_raw_sample[0])
                    v_inst = float(np.hypot(x_raw - last_raw_sample[1], y_raw - last_raw_sample[2]) / dt)
                else:
                    v_inst = 0.0
                last_raw_sample = (now, x_raw, y_raw)
                avgw = float(np.mean([w for (_d1, w) in candidates.values()])) if candidates else 0.0
                kf_base.predict()
                xpb, ypb = kf_base.state()
                innov_b = float(np.hypot(x_raw - xpb, y_raw - ypb))
                gate_b = (innov_b > float(args["innovation_max"])) and (avgw < float(args["weak_avgw"]))
                if not gate_b:
                    kf_base.update([x_raw, y_raw])
                x_kf_b, y_kf_b = kf_base.state()
                mode, pv, rv, inno_max, win_id = (None, 1.0, 1.0, float(args["innovation_max"]), None)
                if advisor_enabled:
                    mode, pv, rv, inno_max, win_id = tuner.current_knobs()
                if advisor_enabled:
                    mode, pv, rv, inno_max, win_id = tuner.current_knobs()
                    if mode == "baseline":
                        kf_llm.set_velocity_damping(1.0)
                        kf_llm.set_position_scaling(1.0)
                        kf_llm.set_noise(
                            process_var=float(args.get("kf_process", 0.15)),
                            meas_var=float(args.get("kf_meas", 0.30))
                        )
                        inno_max = float(args["innovation_max"])
                    elif mode == "static":
                        kf_llm.set_velocity_damping(0.0025)
                        kf_llm.set_position_scaling(0.005)
                        kf_llm.set_noise(process_var=pv, meas_var=rv)
                    elif mode == "moving":
                        kf_llm.set_velocity_damping(0.05)
                        kf_llm.set_position_scaling(0.20)
                        kf_llm.set_noise(process_var=pv, meas_var=rv)
                    else:
                        kf_llm.set_velocity_damping(0.8)
                        kf_llm.set_position_scaling(1.0)
                        kf_llm.set_noise(process_var=pv, meas_var=rv)
                else:
                    kf_llm.set_velocity_damping(1.0)
                    kf_llm.set_position_scaling(1.0)
                    kf_llm.set_noise(
                        process_var=float(args.get("kf_process", 0.15)),
                        meas_var=float(args.get("kf_meas", 0.30))
                    )
                kf_llm.predict()
                xpl, ypl = kf_llm.state()
                innov_l = float(np.hypot(x_raw - xpl, y_raw - ypl))
                gate_l = (innov_l > inno_max) and (avgw < float(args["weak_avgw"]))
                if not gate_l:
                    kf_llm.update([x_raw, y_raw])
                x_kf_l, y_kf_l = kf_llm.state()
                tuner.note_pose_frame(now, v_inst=v_inst, innov_b=innov_b, avgw=avgw, gate_b=gate_b, x=x_raw, y=y_raw)
                win_info = tuner.maybe_advise(now) if advisor_enabled else None
                if win_info:
                    logger.log_window(
                        win_info["win_id"], win_info["t_start"], win_info["t_end"],
                        n_pose_frames=len(tuner._pose_frames),
                        n_total_frames=len(tuner._all_frames),
                        pct_good=win_info["summary"]["pct_good"],
                        mean_speed=win_info["summary"]["mean_speed"],
                        innov_p95=win_info["summary"]["innov_p95"],
                        avgw_mean=win_info["summary"]["avgw_mean"],
                        avgw_p25=win_info["summary"]["avgw_p25"],
                        avgw_p75=win_info["summary"]["avgw_p75"],
                        advice=win_info["advice"],
                        summary=win_info["summary"]
                    )
                poses_out += 1
                used = sorted(list(used_ids))
                raw_str = f"RAW=({x_raw:.3f},{y_raw:.3f})"
                base_str = f"KF=({x_kf_b:.3f},{y_kf_b:.3f})"
                llm_str = f" LLM=({x_kf_l:.3f},{y_kf_l:.3f})" if advisor_enabled else ""
                seq_print = seq
                print(f"{seq_print:06d} {raw_str}  {base_str}{llm_str}")

                logger.log_frame(
                    t=now, seq=poses_out, win_id=win_id,
                    x_raw=x_raw, y_raw=y_raw,
                    x_kf_base=x_kf_b, y_kf_base=y_kf_b,
                    x_kf_llm=x_kf_l, y_kf_llm=y_kf_l,
                    inno_base=innov_b, inno_llm=innov_l,
                    avg_weight=avgw,
                    used_ids=used,
                    mode=mode, pv=pv, rv=rv, inno_max=inno_max,
                    gate_drop_base=gate_b, gate_drop_llm=gate_l
                )
                last_pose_time = now; stale_notified = False

                if do_plot:
                    px, py = (x_kf_l, y_kf_l) if advisor_enabled else (x_kf_b, y_kf_b)
                    gt_x = gt_y = None
                    gt_stale = False
                    gt_last_x = gt_last_y = None
                    gt_age_s = None
                    if plot_with_truth and logger.has_mocap():
                        moc_ts, moc_frame, _gt_x, _gt_y, gqx, gqy, gqz, gqw = logger.interp_mocap(now)
                        if moc_ts is not None:
                            gt_x, gt_y = _gt_x, _gt_y
                            gt_last_xy = (gt_x, gt_y)
                            gt_last_ts = now
                        else:
                            if gt_last_ts is not None and gt_last_xy != (None, None):
                                gt_stale = True
                                gt_last_x, gt_last_y = gt_last_xy
                                gt_age_s = float(now - gt_last_ts)

                    thresholds = {}
                    last_threshold_update = None
                    threshold_explanation = None
                    hybrid_tuning_enabled = bool(args.get("hybrid_tuning_enabled", False))
                    if hybrid_tuning_enabled and tuner._hyper_tuner:
                        thresholds = tuner._hyper_tuner.get_current_thresholds()
                        last_threshold_update = tuner._hyper_tuner.last_adjustment_time
                        threshold_explanation = tuner._hyper_tuner.last_adjustment_explanation

                    pkt = {
                        't': now,
                        'x_raw': x_raw, 'y_raw': y_raw,
                        'x_kf': px, 'y_kf': py,
                        'x_kf_base': x_kf_b, 'y_kf_base': y_kf_b,
                        'x_kf_llm': (x_kf_l if advisor_enabled else None),
                        'y_kf_llm': (y_kf_l if advisor_enabled else None),
                        'used_ids': used,
                        'latest': dict(latest),
                        'gt_x': gt_x, 'gt_y': gt_y,
                        'gt_stale': gt_stale,
                        'gt_last_x': gt_last_x, 'gt_last_y': gt_last_y,
                        'gt_age_s': gt_age_s,
                        'mode': (tuner.current_knobs()[0] if advisor_enabled else "baseline"),
                        'pv': (tuner.current_knobs()[1] if advisor_enabled else float(args.get("kf_process", 0.15))),
                        'rv': (tuner.current_knobs()[2] if advisor_enabled else float(args.get("kf_meas", 0.30))),
                        'inno_max': (tuner.current_knobs()[3] if advisor_enabled else float(args["innovation_max"])),
                        'win_id': (tuner.current_knobs()[4] if advisor_enabled else 0),
                        'llm_enabled': advisor_enabled,
                        'thresholds': thresholds,
                        'hybrid_tuning_enabled': hybrid_tuning_enabled,
                        'last_threshold_update': last_threshold_update,
                        'explanation': threshold_explanation
                    }
                    try:
                        session['pose_queue'].put_nowait(pkt)
                    except queue.Full:
                        pass

                seq += 1
        except Exception as e:
            print(f"[embedded-exception] {e}", file=sys.stderr)
        finally:
            try: ser.close()
            except Exception: pass
            try: logger.close()
            except Exception: pass

    t = threading.Thread(target=_run, name="UWBTrackEmbedded", daemon=True)
    t.start()

    for _ in range(200):
        if session['figure'] is not None or not t.is_alive():
            break
        time.sleep(0.01)
    session['thread'] = t
    return session

def main(argv=None):
    args = parse_and_merge(argv)
    anchors_xyz = parse_anchor_map(args["anchors"])

    if bool(args.get("origin_shift", False)):
        dx = float(args.get("origin_shift_x", 0.0) or 0.0)
        dy = float(args.get("origin_shift_y", 0.0) or 0.0)
        if abs(dx) > 1e-12 or abs(dy) > 1e-12:
            anchors_xyz = {
                aid: ((pos[0] - dx, pos[1] - dy, pos[2]) if len(pos) == 3 else (pos[0] - dx, pos[1] - dy))
                for aid, pos in anchors_xyz.items()
            }
            print(f"[origin] Shifted anchors by (-{dx:.3f}, -{dy:.3f}); new origin is old ({dx:.3f}, {dy:.3f}).")

    anchors_have_z = (len(next(iter(anchors_xyz.values()))) == 3)

    xs = [p[0] for p in anchors_xyz.values()]
    ys = [p[1] for p in anchors_xyz.values()]
    span_diag = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys))) if len(xs) >= 2 else 5.0
    auto_max_range = max(5.0, span_diag * 2.0 + 5.0)
    max_range_cfg = args.get("max_range_m", None)
    max_range_m = float(max_range_cfg) if (max_range_cfg not in (None, "", False)) else auto_max_range
    print(f("[limits] max_range_m={max_range_m:.2f} m ({'cfg' if max_range_cfg not in (None, '', False) else 'auto'})"))

    if args.get("solve_2d", False):
        solve_anchors = {aid: (pos[0], pos[1]) for aid, pos in anchors_xyz.items()}
        dz_map = {aid: (pos[2] if len(pos) >= 3 else 0.0) - args["floor_z"] for aid, pos in anchors_xyz.items()}
        solve_dims = 2
        print(f"\n[planar] Solving in XY at z={args['floor_z']:.2f} m (planarized ranges)\n")
    else:
        solve_anchors = anchors_xyz
        dz_map = {aid: 0.0 for aid in anchors_xyz}
        solve_dims = len(next(iter(anchors_xyz.values())))

    required_min = max(int(args["min_anchors"]), solve_dims + 1)
    if len(anchors_xyz) < required_min:
        print(f"Error: Need at least {required_min} anchors for {solve_dims}D localization.", file=sys.stderr)
        sys.exit(1)
        sys.exit(1)

    print("Anchors (meters):")
    for aid, pos in sorted(anchors_xyz.items()):
        if len(pos) == 3:
            print(f"  A{aid}: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")
        else:
            print(f"  A{aid}: ({pos[0]:.3f}, {pos[1]:.3f})")

    try:
        if str(args.get("port", "")).lower() == "uwb_ws":
            from .uwb_ws_io import init_ws_connection
            ser = init_ws_connection(str(args.get("uwb_ws_url", "")), timeout=0.2)
        else:
            ser = init_serial_connection(args["port"], args["baud"])
    except Exception as e:
        print(f"Failed to open IO: {e}", file=sys.stderr)
        sys.exit(2)

    calib_params = {}
    bias_path = Path(args["bias_file"])
    if args.get("no_calib", False):
        if bias_path.exists():
            print(f"Loaded calibration from {bias_path.resolve()}\n")
            with open(bias_path) as f:
                raw = json.load(f)
            if "range" in raw:
                calib_params = {int(k): tuple(v) for k, v in raw["range"].items()}
            if "quality" in raw:
                WEIGHT_CFG.update(raw["quality"])
        else:
            print("No calibration file found, using defaults.\n")
    else:
        use_floor_calib = (anchors_have_z or args.get("solve_2d", False))
        qual_cfg = calibrate_quality_near_anchors(ser, anchors_xyz, floor_z=args["floor_z"], use_floor=use_floor_calib)
        WEIGHT_CFG.update(qual_cfg)
        calib_params = calibrate_affine_midpoints(ser, anchors_xyz, floor_z=args["floor_z"], use_floor=use_floor_calib)
        payload = {
            "quality": {k: float(v) for k, v in WEIGHT_CFG.items()},
            "range": {str(k): [float(v[0]), float(v[1])] for k, v in calib_params.items()}
        }
        with open(bias_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved calibration to {bias_path.resolve()}\n")

    do_plot = not bool(args.get("no_plot", False))
    show_raw = bool(args.get("show_raw_position", False))
    if do_plot:
        if bool(args.get("mocap_enabled", False)):
            from .plotting import init_plot_with_truth
            fig, ax, raw_dot, tag_dot, gt_dot, err_line, circles, info_txt = init_plot_with_truth(anchors_xyz, show_raw=show_raw)
            plot_with_truth = True
            artists = {'raw_dot': raw_dot,'tag_dot': tag_dot,'gt_dot': gt_dot,'err_line': err_line,'circles': circles}
        else:
            from .plotting import init_plot
            fig, ax, raw_dot, tag_dot, circles = init_plot(anchors_xyz, show_raw=show_raw)
            plot_with_truth = False
            artists = {'raw_dot': raw_dot,'tag_dot': tag_dot,'circles': circles}
        pose_queue = queue.Queue(maxsize=2000)
        plot_stop_event = threading.Event()
        from .plotting import start_plot_consumer
        plot_thread = start_plot_consumer(pose_queue, fig, artists, plot_with_truth, plot_stop_event, fps=30.0)
        import matplotlib.pyplot as _plt
    else:
        pose_queue = None
        plot_stop_event = None
        plot_thread = None
        plot_with_truth = False

    logger = RunLogger(args)

    print("Tracking… (Ctrl+C to stop)")
    kf_base = CVKalman2D(dt=0.1, process_var=float(args.get("kf_process", 0.15)), meas_var=float(args.get("kf_meas", 0.30)))
    kf_llm  = CVKalman2D(dt=0.1, process_var=float(args.get("kf_process", 0.15)), meas_var=float(args.get("kf_meas", 0.30)))

    latest = {}
    last_seen = {}
    max_meas_age_s = float(args.get("max_meas_age_s", 2.0))
    rq = RollingQual(maxlen=400)
    seq = 0; frames_in = 0; poses_out = 0

    advisor_enabled = bool(args.get("llm_kf_enabled", False))
    agent = None
    if advisor_enabled:
        agent = LLMAdvisorAgent(
            caller=ollama_caller,
            temperature=0.0,
            max_tokens=int(args.get("llama_max_tokens", 80)),
            caller_options={
                "base_url": args.get("llama_url", "http://localhost:11434"),
                "model": args.get("llama_model", "llama3.1:8b-instruct-q4_K_M"),
                "timeout": float(args.get("llama_timeout", 1.5)),
                **({"num_gpu": int(args["llama_num_gpu"])} if args.get("llama_num_gpu") is not None else {})
            },
            log_path=(args.get("llm_log_csv") or None),
        )

    hybrid_tuning_enabled = bool(args.get("hybrid_tuning_enabled", False))
    slow_loop_interval = float(args.get("slow_loop_s", 30.0))

    if agent:
        print(f"[main] LLM agent is available for tuning")
    else:
        print(f"[main] No LLM agent available")

    tuner = KFTuningAdvisor(
        window_s=float(args.get("llm_window_s", 5.0)),
        default_innovation_max=float(args["innovation_max"]),
        agent=agent,
        default_kf_process=float(args.get("kf_process", 0.15)),
        default_kf_meas=float(args.get("kf_meas", 0.30)),
        enable_hyper_tuner=hybrid_tuning_enabled,
        slow_loop_interval=slow_loop_interval
    )

    if hybrid_tuning_enabled:
        print(f"[main] Hybrid tuning enabled with fast_loop={args.get('llm_window_s', 5.0)}s, slow_loop={slow_loop_interval}s")
        if tuner._hyper_tuner is not None:
            print(f"[main] HyperTuner successfully initialized")
        else:
            print(f"[main] ERROR: HyperTuner failed to initialize")
            if agent is None:
                print(f"[main] Cause: LLM agent is None")

    last_raw_sample = None
    last_pose_time = time.time()
    stale_notified = False

    gt_last_xy = (None, None)
    gt_last_ts = None

    try:
        for line in serial_lines(ser):
            now = time.time()

            if max_meas_age_s > 0:
                for a, ts in list(last_seen.items()):
                    if (now - ts) > max_meas_age_s:
                        last_seen.pop(a, None)
                        latest.pop(a, None)

            if (now - last_pose_time) > 3.0 and not stale_notified:
                latest.clear()
                last_seen.clear()
                stale_notified = True
                print("[watchdog] Cleared stale measurements due to no pose for 3s.")

            m = parse_meas(line)
            if not m:
                continue
            frames_in += 1
            tuner.note_any_frame(now)

            aid = m["aid"]
            rng_meas = m["range"]

            if not np.isfinite(rng_meas) or rng_meas <= 0.0 or rng_meas > (max_range_m * 3.0):
                latest.pop(aid, None); last_seen.pop(aid, None)
                continue

            alpha, beta = calib_params.get(aid, (1.0, 0.0))
            rng_corr = max(0.0, alpha * rng_meas + beta)

            d = meas_diag(m)
            rq.push(d["snr_db"], d["rpc"])
            rq.update_cfg(WEIGHT_CFG, time.time(), period=1.0)

            dz = dz_map.get(aid, 0.0)
            if args.get("solve_2d", False):
                if rng_corr < abs(dz):
                    latest.pop(aid, None); last_seen.pop(aid, None)
                    continue
                v = rng_corr * rng_corr - dz * dz
                rng_use = float(np.sqrt(v)) if v > 0.0 else 0.0
            else:
                rng_use = rng_corr

            if (not np.isfinite(rng_use)) or rng_use <= 0.0 or rng_use > (max_range_m * 1.5):
                latest.pop(aid, None); last_seen.pop(aid, None)
                continue

            w = meas_weight(m)
            latest[aid] = (rng_use, w)
            last_seen[aid] = now

            current_anchors = sorted(set(solve_anchors) & set(latest))
            if len(current_anchors) < required_min:
                continue

            candidates = {
                a: latest[a] for a in current_anchors
                if (latest[a][1] >= float(args["min_weight"])) and np.isfinite(latest[a][0]) and (0.0 < latest[a][0] <= max_range_m)
            }
            if len(candidates) < required_min:
                continue

            strong_thresh = max(float(args["min_weight"]), float(args["strong_weight"]))
            if float(args["strong_weight"]) > 0:
                good = sum(1 for (_d_, ww) in candidates.values() if ww >= strong_thresh)
                if good < 2:
                    continue

            certain_anchor = None; min_dist = float('inf')
            cd = float(args["certainty_dist"])
            if cd > 0:
                for a, (dist, _w_) in candidates.items():
                    if 0 < dist < cd and dist < min_dist:
                        min_dist = dist; certain_anchor = a

            try:
                if certain_anchor is not None:
                    est = solve_anchors[certain_anchor]; used_ids = {certain_anchor}
                else:
                    est, used_ids = robust_solve_with_residuals(
                        solve_anchors, candidates, required_min=required_min, k_mad=float(args["outlier_threshold"])
                    )
                    if est is not None and len(candidates) >= 2:
                        top2 = sorted(candidates.items(), key=lambda kv: (-kv[1][1], kv[1][0]))[:2]
                        ids2 = [top2[0][0], top2[1][0]]
                        a_pt = solve_anchors[ids2[0]]; b_pt = solve_anchors[ids2[1]]
                        p_ref = _reflect_across_line(est, a_pt, b_pt)
                        if _weighted_cost(p_ref, solve_anchors, candidates) + 0.02 < _weighted_cost(est, solve_anchors, candidates):
                            est = p_ref
            except Exception:
                continue

            if est is None or len(used_ids) < required_min:
                continue

            x_raw, y_raw = float(est[0]), float(est[1])

            if (not np.isfinite(x_raw)) or (not np.isfinite(y_raw)) or (abs(x_raw) > (max_range_m * 2.0)) or (abs(y_raw) > (max_range_m * 2.0)):
                continue

            if last_raw_sample is not None:
                dt = max(0.01, now - last_raw_sample[0])
                dx = x_raw - last_raw_sample[1]
                dy = y_raw - last_raw_sample[2]
                dist = float(np.hypot(dx, dy))
                if dist < 0.05:
                    v_inst = 0.0
                else:
                    v_inst = min(dist / dt, 10.0)
            else:
                v_inst = 0.0

            last_raw_sample = (now, x_raw, y_raw)

            avgw = float(np.mean([w for (_d0, w) in candidates.values()])) if candidates else 0.0

            kf_base.predict()
            x_pred_b, y_pred_b = kf_base.state()
            innovation_b = float(np.hypot(x_raw - x_pred_b, y_raw - y_pred_b))
            gate_b = (innovation_b > float(args["innovation_max"])) and (avgw < float(args["weak_avgw"]))
            if not gate_b:
                kf_base.update([x_raw, y_raw])
            x_kf_b, y_kf_b = kf_base.state()

            mode, pv, rv, inno_max, win_id = tuner.current_knobs()

            if mode == "baseline":
                kf_llm.set_velocity_damping(1.0)
                kf_llm.set_position_scaling(1.0)
                kf_llm.set_noise(
                    process_var=float(args.get("kf_process", 0.15)),
                    meas_var=float(args.get("kf_meas", 0.30))
                )
            elif mode == "static":
                kf_llm.set_velocity_damping(0.0025)
                kf_llm.set_position_scaling(0.005)
                kf_llm.set_noise(process_var=pv, meas_var=rv)
            elif mode == "moving":
                kf_llm.set_velocity_damping(0.05)
                kf_llm.set_position_scaling(0.20)
                kf_llm.set_noise(process_var=pv, meas_var=rv)
            else:
                kf_llm.set_velocity_damping(0.8)
                kf_llm.set_position_scaling(1.0)
                kf_llm.set_noise(process_var=pv, meas_var=rv)

            kf_llm.predict()
            x_pred_l, y_pred_l = kf_llm.state()
            innovation_l = float(np.hypot(x_raw - x_pred_l, y_raw - y_pred_l))
            gate_l = (innovation_l > inno_max) and (avgw < float(args["weak_avgw"]))
            if not gate_l:
                kf_llm.update([x_raw, y_raw])
            x_kf_l, y_kf_l = kf_llm.state()

            tuner.note_pose_frame(now, v_inst=v_inst, innov_b=innovation_b, avgw=avgw, gate_b=gate_b, x=x_raw, y=y_raw)

            win_info = tuner.maybe_advise(now)
            if win_info is not None:
                logger.log_window(
                    win_info["win_id"], win_info["t_start"], win_info["t_end"],
                    n_pose_frames=len(tuner._pose_frames),
                    n_total_frames=len(tuner._all_frames),
                    pct_good=win_info["summary"]["pct_good"],
                    mean_speed=win_info["summary"]["mean_speed"],
                    innov_p95=win_info["summary"]["innov_p95"],
                    avgw_mean=win_info["summary"]["avgw_mean"],
                    avgw_p25=win_info["summary"]["avgw_p25"],
                    avgw_p75=win_info["summary"]["avgw_p75"],
                    advice=win_info["advice"],
                    summary=win_info["summary"]
                )

            poses_out += 1
            used = sorted(list(used_ids))
            raw_str = f"RAW=({x_raw:.3f},{y_raw:.3f})"
            kf_str  = f"KF=({x_kf_b:.3f},{y_kf_b:.3f})"
            llm_str = f"LLM=({x_kf_l:.3f},{y_kf_l:.3f})" if advisor_enabled else ""
            gt_str = ""
            err_str = ""
            if bool(args.get("mocap_enabled", False)) and logger.has_mocap():
                moc_ts, moc_frame, gt_x, gt_y, gqx, gqy, gqz, gqw = logger.interp_mocap(now)
                if moc_ts is not None and gt_x is not None and gt_y is not None:
                    px, py = (x_kf_l, y_kf_l) if advisor_enabled else (x_kf_b, y_kf_b)
                    dx, dy = (px - gt_x), (py - gt_y)
                    de = float(np.hypot(dx, dy))
                    gt_str = f"  GT=({gt_x:.3f},{gt_y:.3f})"
                    err_str = f"  E=({dx:.3f},{dy:.3f})|e|={de:.3f}"

            if args.get("print_qual", False):
                txt = " ".join([f"A{a}={latest[a][0]:.3f}m[w={latest[a][1]:.2f}]" for a in used])
                txt += f"  [aw={avgw:.2f}, innoB={innovation_b:.2f}"
                if advisor_enabled:
                    txt += f", innoL={innovation_l:.2f}, mode={mode}, pv={pv:.2f}, rv={rv:.2f}, imax={inno_max:.2f}]"
                else:
                    txt += "]"
            else:
                txt = " ".join([f"A{a}={latest[a][0]:.3f}m" for a in used])
            print(f"{seq:06d} {raw_str}  {kf_str}  {llm_str}{gt_str}{err_str}  | {txt}")
            seq += 1

            logger.log_frame(
                t=now, seq=poses_out, win_id=win_id,
                x_raw=x_raw, y_raw=y_raw,
                x_kf_base=x_kf_b, y_kf_base=y_kf_b,
                x_kf_llm=x_kf_l, y_kf_llm=y_kf_l,
                inno_base=innovation_b, inno_llm=innovation_l,
                avg_weight=avgw,
                used_ids=used,
                mode=mode,
                pv=pv, rv=rv, inno_max=inno_max,
                gate_drop_base=gate_b, gate_drop_llm=gate_l
            )

            if do_plot:
                px, py = (x_kf_l, y_kf_l) if advisor_enabled else (x_kf_b, y_kf_b)
                gt_x = gt_y = None
                gt_stale = False
                gt_last_x = gt_last_y = None
                gt_age_s = None
                if plot_with_truth and logger.has_mocap():
                    moc_ts, moc_frame, _gt_x, _gt_y, gqx, gqy, gqz, gqw = logger.interp_mocap(now)
                    if moc_ts is not None:
                        gt_x, gt_y = _gt_x, _gt_y
                        gt_last_xy = (gt_x, gt_y)
                        gt_last_ts = now
                    else:
                        if gt_last_ts is not None and gt_last_xy != (None, None):
                            gt_stale = True
                            gt_last_x, gt_last_y = gt_last_xy
                            gt_age_s = float(now - gt_last_ts)

                thresholds = {}
                last_threshold_update = None
                threshold_explanation = None
                hybrid_tuning_enabled = bool(args.get("hybrid_tuning_enabled", False))
                if hybrid_tuning_enabled and tuner._hyper_tuner:
                    thresholds = tuner._hyper_tuner.get_current_thresholds()
                    last_threshold_update = tuner._hyper_tuner.last_adjustment_time
                    threshold_explanation = tuner._hyper_tuner.last_adjustment_explanation

                pkt = {
                    't': now,
                    'x_raw': x_raw, 'y_raw': y_raw,
                    'x_kf': px, 'y_kf': py,
                    'x_kf_base': x_kf_b, 'y_kf_base': y_kf_b,
                    'x_kf_llm': (x_kf_l if advisor_enabled else None),
                    'y_kf_llm': (y_kf_l if advisor_enabled else None),
                    'used_ids': used,
                    'latest': dict(latest),
                    'gt_x': gt_x, 'gt_y': gt_y,
                    'gt_stale': gt_stale,
                    'gt_last_x': gt_last_x, 'gt_last_y': gt_last_y,
                    'gt_age_s': gt_age_s,
                    'mode': (tuner.current_knobs()[0] if advisor_enabled else "baseline"),
                    'pv': (tuner.current_knobs()[1] if advisor_enabled else float(args.get("kf_process", 0.15))),
                    'rv': (tuner.current_knobs()[2] if advisor_enabled else float(args.get("kf_meas", 0.30))),
                    'inno_max': (tuner.current_knobs()[3] if advisor_enabled else float(args["innovation_max"])),
                    'win_id': (tuner.current_knobs()[4] if advisor_enabled else 0),
                    'llm_enabled': advisor_enabled,
                    'thresholds': thresholds,
                    'hybrid_tuning_enabled': hybrid_tuning_enabled,
                    'last_threshold_update': last_threshold_update,
                    'explanation': threshold_explanation
                }
                try:
                    pose_queue.put_nowait(pkt)
                except queue.Full:
                    pass

                if frames_in % 25 == 0:
                    try:
                        _plt.pause(0.001)
                    except Exception:
                        pass

    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        try: ser.close()
        except Exception: pass
        try: logger.close()
        except Exception: pass
        if plot_stop_event:
            plot_stop_event.set()
        if plot_thread and plot_thread.is_alive():
            plot_thread.join(timeout=1.0)

if __name__ == "__main__":
    main()
