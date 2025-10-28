import numpy as np

WEIGHT_CFG = {
    "snr_min": 6.0,
    "snr_max": 50.0,
    "rpc_norm": 1024.0,
    "ci_halfppm": 6.0,
}

class RollingQual:
    """Online auto-scaling of SNR/RXPACC from rolling percentiles."""
    def __init__(self, maxlen=400):
        from collections import deque
        self.snr = deque(maxlen=maxlen)
        self.rpc = deque(maxlen=maxlen)
        self.last_update = 0.0

    def push(self, snr_db, rpc):
        self.snr.append(float(snr_db))
        self.rpc.append(float(rpc))

    def update_cfg(self, cfg, now, period=1.0):
        if now - self.last_update < period or len(self.snr) < 40:
            return
        snr_arr = np.array(self.snr, float)
        rpc_arr = np.array(self.rpc, float)
        s0 = float(np.percentile(snr_arr, 10))
        s1 = float(np.percentile(snr_arr, 90))
        r90 = float(np.percentile(rpc_arr, 90))
        if s1 - s0 < 4.0:
            s0, s1 = s0 - 2.0, s1 + 2.0
        cfg["snr_min"] = s0
        cfg["snr_max"] = s1
        cfg["rpc_norm"] = max(64.0, r90)
        self.last_update = now

def meas_diag(m):
    fp = float(max(m.get("fp1", 0), m.get("fp2", 0), m.get("fp3", 0)))
    noise = float(max(1, m.get("stdNoise", 50), m.get("maxNoise", 50)))
    snr = max(1.0, fp / noise)
    snr_db = 20.0 * np.log10(snr)
    ci_ppm = abs(m.get("ci", 0.0)) * 1e6
    rpc = int(m.get("rpc", 0))
    return {"snr_db": float(snr_db), "rpc": rpc, "ci_ppm": float(ci_ppm)}

def meas_weight(m):
    fp = float(max(m.get("fp1", 0), m.get("fp2", 0), m.get("fp3", 0)))
    noise = float(max(1, m.get("stdNoise", 50), m.get("maxNoise", 50)))
    snr = max(1.0, fp / noise)
    snr_db = 20.0 * np.log10(snr)
    s0 = WEIGHT_CFG["snr_min"]; s1 = WEIGHT_CFG["snr_max"]
    snr_w = float(np.clip((snr_db - s0) / max(1e-6, (s1 - s0)), 0.0, 1.0))
    rpc = float(m.get("rpc", 0))
    rpc_w = float(np.clip(rpc / max(1.0, WEIGHT_CFG["rpc_norm"]), 0.0, 1.0))
    ci_ppm = abs(m.get("ci", 0.0)) * 1e6
    ci_half = max(1e-6, WEIGHT_CFG["ci_halfppm"])
    ci_w = 1.0 / (1.0 + (ci_ppm / ci_half))
    w = snr_w * rpc_w * ci_w
    return max(0.05, min(1.0, w))
