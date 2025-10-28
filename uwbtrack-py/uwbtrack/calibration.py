import sys, time, json
from collections import defaultdict
from pathlib import Path
import numpy as np

from .serial_io import serial_lines, parse_meas
from .weights import WEIGHT_CFG, meas_diag

def calibrate_affine_midpoints(ser, anchors_xyz, sample_per_anchor=50, timeout_s=10.0,
                               floor_z=0.0, use_floor=False):
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
        poses = [("mid(A{},A{})".format(a1, a2), (p12_xy[0], p12_xy[1], floor_z)),
                 ("mid(A{},A{})".format(a1, a3), (p13_xy[0], p13_xy[1], floor_z)),
                 ("mid(A{},A{})".format(a2, a3), (p23_xy[0], p23_xy[1], floor_z))]
    else:
        poses = [("mid(A{},A{})".format(a1, a2), p12_xy),
                 ("mid(A{},A{})".format(a1, a3), p13_xy),
                 ("mid(A{},A{})".format(a2, a3), p23_xy)]

    print("\nAffine calibration (range): three poses at anchor midpoints:")
    for name, pos in poses:
        if len(pos) == 3:
            print(f"  {name}: floor at ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) m")
        else:
            print(f"  {name}: approx at ({pos[0]:.3f}, {pos[1]:.3f}) m")

    print("\nInteractive controls during this calibration:")
    print("  Enter = continue / accept")
    print("  r     = redo this pose (after it is measured)")
    print("  b     = go back to previous pose")
    print(f"\nTarget: ~{sample_per_anchor} samples per anchor per pose.\n")

    ser_iter = serial_lines(ser)
    meas_means = {aid: [None]*len(poses) for aid in ids_all}
    true_dists = {aid: [None]*len(poses) for aid in ids_all}

    def _pos_vec(p):
        return np.array((p[0], p[1], (p[2] if len(p) == 3 else 0.0)), float)

    anchors_vec = {aid: np.array((v[0], v[1], (v[2] if len(v) == 3 else 0.0)), float)
                   for aid, v in anchors_xyz.items()}

    i = 0
    while i < len(poses):
        pose_name, pos = poses[i]
        cmd = input(f"[{i+1}/{len(poses)}] Ready for pose '{pose_name}'. Place tag and press Enter (b=back)... ").strip().lower()
        if cmd == 'b':
            if i > 0:
                i -= 1
                print("Going back to previous pose.")
                continue
            else:
                print("Already at first pose.")
                continue

        print("Waiting 7s for raw data to settle..."); time.sleep(7)
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
        start = time.time()
        while (time.time() - start) < timeout_s:
            if all(len(got[aid]) >= sample_per_anchor for aid in ids_all):
                break
            try: line = next(ser_iter)
            except StopIteration: break
            m = parse_meas(line)
            if not m: continue
            aid = m["aid"]
            if aid in meas_means:
                got[aid].append(m["range"])

        min_n = min((len(got[aid]) for aid in ids_all), default=0)
        for aid in ids_all:
            arr = np.asarray(got[aid], float)
            mean_distance = float(np.mean(arr)) if arr.size else 0.0
            meas_means[aid][i] = mean_distance
            print(f"  A{aid} at {pose_name}: measured ≈ {mean_distance:.3f} m (N={arr.size})")

        # Post-step decision if low samples
        while True:
            if min_n >= sample_per_anchor or min_n == 0:
                decision = input("Accept this pose? (Enter=accept, r=redo, b=back) ").strip().lower()
            else:
                decision = input(f"Only {min_n} samples (target {sample_per_anchor}). (Enter=accept, r=redo, b=back) ").strip().lower()
            if decision in ('', 'enter'):
                i += 1
                break
            elif decision == 'r':
                print("Redoing this pose.")
                # wipe stored data for this index
                for aid in ids_all:
                    meas_means[aid][i] = None
                    true_dists[aid][i] = None
                break  # redo same i
            elif decision == 'b':
                if i > 0:
                    print("Going back.")
                    i -= 1
                    break
                else:
                    print("Already at first pose; cannot go back.")
            else:
                print("Unrecognized input.")
    # Compute calibration using only completed indices (all anchors must have non-None)
    valid_indices = [k for k in range(len(poses)) if all(meas_means[aid][k] is not None for aid in ids_all)]
    if len(valid_indices) < 2:
        print("Not enough valid midpoint poses; falling back to identity calibration.")
        return {aid: (1.0, 0.0) for aid in ids_all}

    calib_params = {}
    for aid in ids_all:
        d_meas = np.asarray([meas_means[aid][k] for k in valid_indices], float)
        d_true = np.asarray([true_dists[aid][k] for k in valid_indices], float)
        A = np.c_[d_meas, np.ones_like(d_meas)]
        try: alpha, beta = np.linalg.lstsq(A, d_true, rcond=None)[0]
        except Exception: alpha, beta = 1.0, 0.0
        alpha = float(np.clip(alpha, 0.9, 1.1))
        beta = float(np.clip(beta, -0.5, 0.5))
        calib_params[aid] = (alpha, beta)
        print(f"  A{aid}: d_true ≈ {alpha:.4f} * d_meas + {beta:.4f}")
    print("\nRange calibration complete.")
    return calib_params

def calibrate_quality_near_anchors(ser, anchors_xyz, sample_per_anchor=50, timeout_s=10.0,
                                   floor_z=0.0, use_floor=False):
    ids_all = sorted(anchors_xyz)
    if len(ids_all) < 3:
        print("Need ≥3 anchors for quality calibration.", file=sys.stderr)
    ser_iter = serial_lines(ser)

    print("\nQuality calibration (diagnostics near each anchor).")
    print("Interactive controls:")
    print("  Enter = measure / accept")
    print("  r     = redo current anchor after measurement")
    print("  b     = go back to previous anchor (overwrite)")
    print()

    # Store per-anchor diagnostics so we can overwrite
    anchor_diags = {aid: None for aid in ids_all}

    i = 0
    while i < len(ids_all):
        aid = ids_all[i]
        ax, ay = anchors_xyz[aid][0], anchors_xyz[aid][1]
        if use_floor:
            base_prompt = f"[{i+1}/{len(ids_all)}] Place TAG under ANCHOR A{aid} (~{ax:.2f},{ay:.2f}, z={floor_z:.2f}) and press Enter (b=back)... "
        else:
            base_prompt = f"[{i+1}/{len(ids_all)}] Place TAG near ANCHOR A{aid} (≤0.3 m) and press Enter (b=back)... "
        cmd = input(base_prompt).strip().lower()
        if cmd == 'b':
            if i > 0:
                i -= 1
                print("Going back to previous anchor.")
            else:
                print("Already at first anchor.")
            continue

        print("Waiting 7s for raw data to settle..."); time.sleep(7)
        print(f"Collecting samples for A{aid}...")
        try: ser.reset_input_buffer()
        except Exception: pass
        try: ser.write(b"S"); ser.flush()
        except Exception: pass
        time.sleep(0.10)
        got = []
        start = time.time()
        while len(got) < sample_per_anchor and (time.time() - start) < timeout_s:
            try: line = next(ser_iter)
            except StopIteration: break
            m = parse_meas(line)
            if not m or m["aid"] != aid: continue
            d = meas_diag(m); got.append(d)

        n = len(got)
        if n == 0:
            print(f"  !! No diagnostics collected for A{aid}")
        else:
            snr_med = np.median([g['snr_db'] for g in got])
            rpc_med = np.median([g['rpc'] for g in got])
            print(f"  A{aid}: collected {n} samples (SNR p50 ~ {snr_med:.1f} dB, RXPACC p50 ~ {rpc_med:.0f})")

        while True:
            decision = input("Accept? (Enter=accept, r=redo, b=back) ").strip().lower()
            if decision in ('', 'enter'):
                anchor_diags[aid] = got
                i += 1
                break
            elif decision == 'r':
                print("Redoing this anchor.")
                anchor_diags[aid] = None
                break  # redo same i
            elif decision == 'b':
                if i > 0:
                    print("Going back.")
                    i -= 1
                    break
                else:
                    print("Already at first anchor.")
            else:
                print("Unrecognized input.")

    # Aggregate all collected diagnostics
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
