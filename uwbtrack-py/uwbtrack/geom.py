import itertools
from typing import Dict, Tuple, Set
import numpy as np

def parse_anchor_map(s: str):
    """
    Accepts 'id:x,y' or 'id:x,y,z'. Returns {aid: (x,y)} or {aid: (x,y,z)}.
    If any anchor has 3 components, all anchors are promoted to 3D (z=0 by default).
    """
    raw = {}
    dims = 2
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        k, v = part.split(":")
        aid = int(k.strip())
        comps = [float(t) for t in v.split(",")]
        if len(comps) == 2:
            raw[aid] = tuple(comps)
        elif len(comps) == 3:
            raw[aid] = tuple(comps)
            dims = 3
        else:
            raise ValueError(f"Anchor '{part}' must be x,y or x,y,z")
    if dims == 3:
        out = {aid: (xy[0], xy[1], (xy[2] if len(xy) == 3 else 0.0)) for aid, xy in raw.items()}
    else:
        out = {aid: (xy[0], xy[1]) for aid, xy in raw.items()}
    return out

def choose_reference_anchor(ranges_dict, ref_eps=0.2):
    best = None; best_score = -1.0
    for a, (rng, w) in ranges_dict.items():
        score = float(w) / (float(rng) + float(ref_eps))
        if score > best_score:
            best_score, best = score, a
    return best

def trilaterate(anchors_pos, ranges_dict, ref_eps=0.2):
    """Weighted LS trilateration (2D/3D). ranges_dict[aid] = (range_m, weight)."""
    ids = sorted(set(anchors_pos) & set(ranges_dict))
    if not ids:
        return None
    d = len(next(iter(anchors_pos.values())))
    if len(ids) < d + 1:
        return None
    filtered_ranges = {aid: ranges_dict[aid] for aid in ids}
    a0_best = choose_reference_anchor(filtered_ranges, ref_eps=ref_eps)
    x1 = np.asarray(anchors_pos[a0_best], float)
    r1, w1 = filtered_ranges[a0_best]
    A, b, roww = [], [], []
    for aid in ids:
        if aid == a0_best: continue
        xi = np.asarray(anchors_pos[aid], float)
        ri, wi = filtered_ranges[aid]
        A.append(2.0 * (xi - x1))
        b.append(r1**2 - ri**2 + xi.dot(xi) - x1.dot(x1))
        wrow = (w1 * wi) / (w1 + wi + 1e-9)
        roww.append(float(wrow))
    A = np.asarray(A, float); b = np.asarray(b, float)
    Wsqrt = np.sqrt(np.asarray(roww, float)).reshape(-1, 1)
    Aw = A * Wsqrt; bw = b * Wsqrt.ravel()
    try:
        sol, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
    except Exception:
        return None
    d_out = len(next(iter(anchors_pos.values())))
    return tuple(float(v) for v in sol[:d_out])

def _compute_residuals(est, anchors_pos, ranges_dict):
    if est is None: return {}
    est = np.asarray(est, float)
    out = {}
    for aid, (rng_meas, _w) in ranges_dict.items():
        pred = float(np.linalg.norm(est - np.asarray(anchors_pos[aid], float)))
        res = float(rng_meas - pred)
        out[aid] = (res, abs(res))
    return out

def _mad_gating(residuals_abs, k=2.5):
    if not residuals_abs: return set()
    ids = list(residuals_abs.keys())
    arr = np.array([residuals_abs[i] for i in ids], float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad < 1e-6:
        std = float(np.std(arr))
        if std < 1e-3: return set(ids)
        scale = 1.4826 * std
    else:
        scale = 1.4826 * mad
    inliers = [i for i, v in zip(ids, arr) if (abs(v - med) / (scale + 1e-9)) < k]
    return set(inliers)

def robust_solve_with_residuals(anchors_pos, ranges_dict, required_min, k_mad=2.5):
    """Residual MAD gate, then tiny RANSAC fallback."""
    ids_all = list(ranges_dict.keys())
    est0 = trilaterate(anchors_pos, ranges_dict)
    if est0 is not None:
        res_dict = _compute_residuals(est0, anchors_pos, ranges_dict)
        abs_map = {aid: v[1] for aid, v in res_dict.items()}
        keep_ids = _mad_gating(abs_map, k=k_mad)
        if len(keep_ids) >= required_min:
            est1 = trilaterate(anchors_pos, {aid: ranges_dict[aid] for aid in keep_ids})
            if est1 is not None:
                res2 = _compute_residuals(est1, anchors_pos, {aid: ranges_dict[aid] for aid in keep_ids})
                abs2 = {aid: v[1] for aid, v in res2.items()}
                keep2 = _mad_gating(abs2, k=k_mad)
                if len(keep2) >= required_min:
                    est2 = trilaterate(anchors_pos, {aid: ranges_dict[aid] for aid in keep2})
                    if est2 is not None:
                        return est2, set(keep2)
                return est1, set(keep_ids)
    best_cost = float("inf"); best_est = None; best_inliers: Set[int] = set()
    for subset in itertools.combinations(ids_all, required_min):
        sub_dict = {aid: ranges_dict[aid] for aid in subset}
        est_sub = trilaterate(anchors_pos, sub_dict)
        if est_sub is None: continue
        res_all = _compute_residuals(est_sub, anchors_pos, ranges_dict)
        abs_map = {aid: v[1] for aid, v in res_all.items()}
        inl = _mad_gating(abs_map, k=k_mad)
        if len(inl) < required_min: continue
        cost = float(np.median(list(abs_map.values()))) if abs_map else float("inf")
        if (len(inl) > len(best_inliers)) or (len(inl) == len(best_inliers) and cost < best_cost):
            best_cost = cost; best_est = est_sub; best_inliers = inl
    if best_est is not None and len(best_inliers) >= required_min:
        est_final = trilaterate(anchors_pos, {aid: ranges_dict[aid] for aid in best_inliers})
        if est_final is not None:
            return est_final, set(best_inliers)
        return best_est, set(best_inliers)
    return None, set()

def _reflect_across_line(p, a, b):
    p = np.asarray(p, float); a = np.asarray(a, float); b = np.asarray(b, float)
    ab = b - a; nrm = np.linalg.norm(ab)
    if nrm < 1e-9: return tuple(p)
    ab /= nrm
    proj = a + ab * np.dot(p - a, ab)
    pref = proj + (proj - p)
    return (float(pref[0]), float(pref[1]))

def _weighted_cost(q, anchors_pos, ranges_dict):
    q = np.asarray(q, float)
    s = 0.0
    for aid, (rng, w) in ranges_dict.items():
        pred = np.linalg.norm(q - np.asarray(anchors_pos[aid], float))
        s += w * abs(rng - pred)
    return float(s)
