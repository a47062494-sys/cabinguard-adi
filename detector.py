"""
CabinGuard-ADI - rule-based two-stage detector (the logic under test).

Pipeline
  1. calibrate()      per-driver baselines from the first CALIB_S seconds
  2. features()       per-modality features on a 10 Hz decision grid
  3. decide()         Stage-1 alert / Stage-2 verification per class, labelling,
                      fatigue routing; all thresholds come from PARAMS

"Missing data is never evidence": camera features only vote when a face is tracked.

Sensor health (added 7 Oct 2026, see health()): every modality votes only while its stream is
healthy, and for as long after a fault as its feature windows still contain faulty samples.
A stuck IMU or a lost telemetry/camera stream is reported as DEGRADED and can no longer create
LOC or hypoglycaemia evidence. On fault-free data the detector output is unchanged.

Recalibration (added 8 Oct 2026, see find_recal()): a modality that was faulty during the
initial calibration is no longer lost for the whole drive. Once it is back, the first clean
20-s window in which the system raised NO Stage-1 alert becomes its new calibration window;
the modality votes again only after that window (+ the usual 3-s / 60-s exclusion).
On fault-free data nothing changes (bit-identical output).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal

from generator import FS_IMU, FS_CAM, FS_TEL, CALIB_S

DT = 0.1  # decision step [s]

# --------------------------------------------------------------------------------------
# Thresholds (source or rationale noted next to each value)
# --------------------------------------------------------------------------------------
PARAMS = dict(
    # eyes / gaze / head
    ear_closed_frac=0.60,    # eye "closed" if EAR < 0.60 x driver's open-eye EAR (~0.18-0.20)
    closed_s=2.0,            # continuous closure counted as LOC evidence (>= microsleep upper bound)
    unresp_closed_s=6.0,     # Euro NCAP 2026 unresponsive-driver criterion (overrides fatigue gate)
    stare_s=5.0,             # no blink for >= 5 s with fixed gaze
    stare_gaze_sd=1.5,       # deg, 3-s gaze SD below which gaze is "fixed"
    gaze_away_s=3.0,         # yaw deviation not returning within 3 s (Euro NCAP)
    yaw_dev_deg=20.0,
    pitch_drop_deg=15.0,     # head drop vs baseline
    perclos=0.12,            # fatigue gate (PERCLOS over 60 s)
    fatigue_hold_s=120.0,    # fatigue context persists 120 s (hypoglycaemia routed to fatigue warning)
    # IMU
    tilt_deg=10.0,           # trunk tilt change from calibrated gravity vector
    quiet_ratio=0.5,         # voluntary-motion RMS below 0.5 x baseline
    band_ratio=2.0,          # 1.5-8 Hz residual RMS above 2.0 x baseline
    rhythm=0.50,             # two-period autocorrelation (rhythmicity)
    hf_ratio=3.0,            # 8-20 Hz residual (tonic tremor) above 3 x baseline
    # telemetry
    frozen_frac=0.25,        # median |steering rate| below 0.25 x baseline ...
    frozen_abs=0.8,          # ... and below 0.8 deg/s
    frozen_s=2.0,
    # face (stroke)
    asym_z=4.0,
    asym_abs=0.02,
    energy_z=1.5,            # face-motion energy for rhythmic (seizure) evidence
    talk_energy_z=3.0,       # above this the face is "busy" (talking): asymmetry is not scored
    # hypoglycaemia (60-s window, Lehmann et al. 2024)
    hy_micro=0.6, hy_abrupt=2, hy_lane=1.5, hy_gaze=0.7,
    # stage timing
    loc_verify_s=3.0,        # LOC: >=2 modalities for 3 s
    sz_verify_s=1.5,         # seizure: >=2 rhythmic modalities for 1.5 s (window 3 s)
    st_stage1_s=3.0, st_verify_s=10.0,
    hy_stage1_s=5.0, hy_verify_s=15.0,
    label_refine_s=30.0,     # etiology field may be refined for 30 s after verification (vehicle response unchanged)
    # sensor health
    imu_stuck_s=0.1,         # IMU unhealthy after >= 0.1 s of bit-identical samples (MPU6050 I2C hang)
    health_short_s=3.0,      # a modality stays excluded 3 s after recovery (length of the short feature windows)
    health_long_s=60.0,      # ... and 60 s for the hypoglycaemia criteria (60-s windows)
    tel_timeout_s=0.15,      # telemetry watchdog warning: 3 missed periods at 20 Hz (logged, no exclusion)
    cam_timeout_s=0.10,      # camera watchdog warning: 3 missed frames at 30 Hz (logged, no exclusion)
    tel_gap_tol_s=0.5,       # gap tolerated without excluding telemetry (held value, < 1 steering correction)
    cam_gap_tol_s=0.5,       # gap tolerated without excluding the camera (< 2 blinks' duration)
    calib_face_min=0.8,      # calibration valid only if a face is tracked >= 80% of the time
    # recalibration after a calibration fault (8 Oct 2026; not tuned: same length as the initial calibration)
    recal_enable=True,
    recal_s=20.0,            # a recalibration window is as long as the initial calibration (CALIB_S)
)
RECAL_MODALITIES = ("belt", "tel", "cam")   # the seat-frame reference IMU is not recalibrated (see README)


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _bp(x, fs, lo, hi, axis=0):
    sos = signal.butter(3, [lo / (fs / 2), hi / (fs / 2)], "band", output="sos")
    return signal.sosfiltfilt(sos, x, axis=axis)


def _hp(x, fs, fc, axis=0):
    sos = signal.butter(2, fc / (fs / 2), "high", output="sos")
    return signal.sosfiltfilt(sos, x, axis=axis)


def _lp(x, fs, fc, axis=0):
    sos = signal.butter(2, fc / (fs / 2), "low", output="sos")
    return signal.sosfiltfilt(sos, x, axis=axis)


def _roll(x, n, how="mean"):
    s = pd.Series(x)
    r = s.rolling(max(int(n), 1), min_periods=1)
    return getattr(r, how)().to_numpy()


def _at(t_src, x, t_dec):
    idx = np.clip(np.searchsorted(t_src, t_dec, side="right") - 1, 0, len(t_src) - 1)
    return x[idx]


def _runlen(b, fs):
    """Duration [s] of the current run of True ending at each sample (causal)."""
    b = np.asarray(b, bool)
    c = np.cumsum(b)
    reset = np.maximum.accumulate(np.where(~b, c, 0))
    return (c - reset) / fs


def _rhythm(x, fs, t_src, t_dec, win_s, lag_lo, lag_hi):
    """Peak of the normalised (unbiased) autocorrelation in [lag_lo, lag_hi] over a causal window.
    x may be (N,) or (N, k); axes are pooled."""
    x = np.atleast_2d(x.T).T if x.ndim == 1 else x
    W = int(win_s * fs)
    idx = np.clip(np.searchsorted(t_src, t_dec, side="right"), W, len(t_src))
    win = sliding_window_view(x, W, axis=0)          # (N-W+1, k, W)
    seg = win[idx - W]                                # (M, k, W)
    seg = seg - seg.mean(-1, keepdims=True)
    nfft = 2 * W
    X = np.fft.rfft(seg, nfft, axis=-1)
    ac = np.fft.irfft(np.abs(X) ** 2, nfft, axis=-1)[..., :W].sum(1)  # (M, W)
    ac = ac / (ac[:, :1] + 1e-12)
    lags = np.arange(W)
    ac = ac * W / np.maximum(W - lags, 1)
    lo, hi = int(lag_lo * fs), int(lag_hi * fs)
    # rhythmicity = the pattern must repeat over TWO periods (rejects narrow-band noise,
    # whose autocorrelation decays after one period): min(ac[P], ac[2P])
    P = lo + ac[:, lo:hi + 1].argmax(1)
    r1 = ac[np.arange(len(P)), P]
    P2 = np.minimum(2 * P, W - 1)
    r2 = ac[np.arange(len(P)), P2]
    return np.minimum(r1, r2)


# --------------------------------------------------------------------------------------
# sensor health
# --------------------------------------------------------------------------------------
def _imu_live(x, fs, stuck_s):
    """False once the IMU has repeated a bit-identical 3-axis sample for >= stuck_s."""
    same = np.r_[False, np.all(np.diff(x, axis=0) == 0, axis=1)]
    return _runlen(same, fs) < stuck_s


def _cal_masks(tr, calib=None):
    """Calibration masks on each native time base. Default = the initial calibration [0, CALIB_S);
    calib = {modality: (t_start, t_end)} replaces it for a recalibrated modality."""
    W = {"belt": (0.0, CALIB_S), "frame": (0.0, CALIB_S), "tel": (0.0, CALIB_S), "cam": (0.0, CALIB_S)}
    W.update(calib or {})
    tb = {"belt": tr["t_imu"], "frame": tr["t_imu"], "tel": tr["t_tel"], "cam": tr["t_cam"]}
    return W, {k: (tb[k] >= a) & (tb[k] < b) for k, (a, b) in W.items()}


def health(tr, t_dec, calib=None):
    """Per-modality health on the decision grid.

    Raw health: IMU = not stuck (from the data); telemetry / camera = new messages still arriving
    (arrival flags; absent = always arriving). A modality is excluded while unhealthy and for the
    length of the feature windows after recovery, so that no window containing faulty samples
    can vote. Returns raw and extended masks."""
    P = PARAMS
    H = {}
    live = {k: _imu_live(tr[k], FS_IMU, P["imu_stuck_s"]) for k in ("belt", "frame")}
    # watchdog: age of the newest message held by the receive buffer (rx timestamp; absent = fresh)
    tel_age = tr["t_tel"] - np.asarray(tr.get("tel_rx_t", tr["t_tel"]))
    cam_age = tr["t_cam"] - np.asarray(tr.get("cam_rx_t", tr["t_cam"]))
    # short gaps (jitter, a few dropped frames) are tolerated: the stream is only UNAVAILABLE once its
    # newest message is older than the gap tolerance
    tel_live = tel_age <= P["tel_gap_tol_s"]
    cam_live = cam_age <= P["cam_gap_tol_s"]
    # calibration validity: a modality that was faulty during calibration has no valid baseline
    W, M = _cal_masks(tr, calib)
    face = np.asarray(tr["face_ok"], bool)
    H["calib_valid"] = dict(
        belt=bool(live["belt"][M["belt"]].all()), frame=bool(live["frame"][M["frame"]].all()),
        tel=bool(tel_live[M["tel"]].all()),
        cam=bool(cam_live[M["cam"]].all()) and float((face & cam_live)[M["cam"]].mean()) >= P["calib_face_min"])
    # a recalibrated modality only exists from the end of its calibration window
    # (initial calibration: t_dec starts at CALIB_S, so this is always true)
    after = {k: t_dec >= W[k][1] - 1e-9 for k in W}
    H["belt_raw"] = _at(tr["t_imu"], live["belt"], t_dec) & H["calib_valid"]["belt"] & after["belt"]
    H["frame_raw"] = _at(tr["t_imu"], live["frame"], t_dec) & H["calib_valid"]["frame"] & after["frame"]
    H["tel_raw"] = _at(tr["t_tel"], tel_live, t_dec) & H["calib_valid"]["tel"] & after["tel"]
    H["cam_raw"] = _at(tr["t_cam"], cam_live, t_dec) & H["calib_valid"]["cam"] & after["cam"]
    for k in ("belt", "frame", "tel", "cam"):
        bad = ~H[k + "_raw"].astype(bool)
        H[k + "_ok"] = _roll(bad.astype(float), round(P["health_short_s"] / DT), "max") == 0
        H[k + "_ok60"] = _roll(bad.astype(float), round(P["health_long_s"] / DT), "max") == 0
    return H


# --------------------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------------------
def features(tr, use_reference_imu=True, calib=None, recal=True):
    """calib: {modality: (t_start, t_end)} recalibration windows (None = initial calibration only).
    recal: look for recalibration windows when a modality was faulty during calibration."""
    t_dec = np.arange(CALIB_S, tr["T"] - 1e-9, DT)
    P = PARAMS
    F = dict(t=t_dec)
    F.update({"h_" + k: v for k, v in health(tr, t_dec, calib).items()})
    _, M = _cal_masks(tr, calib)
    cal_i = M["belt"]          # belt-IMU baselines (and the reference-filter fit) use the belt window
    cal_c = M["cam"]
    cal_t = M["tel"]

    # ---------------- IMU ----------------
    belt, frame, ti = tr["belt"], tr["frame"], tr["t_imu"]
    bh, fh = _hp(belt, FS_IMU, 0.3), _hp(frame, FS_IMU, 0.3)
    frame_live = _imu_live(frame, FS_IMU, P["imu_stuck_s"])
    if use_reference_imu and not frame_live[cal_i].all():
        use_reference_imu = False          # no valid reference during calibration: no cancellation filter
    elif use_reference_imu and not frame_live.all():
        use_reference_imu = "partial"
    if use_reference_imu:
        # Wiener FIR (30 taps = 0.3 s) from frame -> belt, fitted on calibration only
        L = 30
        Xf = np.concatenate([sliding_window_view(np.pad(fh[:, a], (L - 1, 0)), L)[:, ::-1] for a in range(3)], 1)
        Xc = Xf[cal_i]
        A = Xc.T @ Xc + 1e-3 * np.eye(Xc.shape[1]) * np.trace(Xc.T @ Xc) / Xc.shape[1]
        W = np.linalg.solve(A, Xc.T @ bh[cal_i])
        res = bh - Xf @ W
        if use_reference_imu == "partial":
            # reference IMU stuck: vibration cancellation is switched off while it is unhealthy
            res = np.where(frame_live[:, None], res, bh)
    else:
        res = bh
    rb = _bp(res, FS_IMU, 1.5, 8)
    rq = _bp(res, FS_IMU, 0.3, 3)
    rh = _bp(res, FS_IMU, 8, 20)
    def rms_roll(x, win):
        return np.sqrt(_roll((x ** 2).sum(1), win * FS_IMU))
    band = rms_roll(rb, 3); quiet = rms_roll(rq, 3); hf = rms_roll(rh, 2)
    band_b = np.median(band[cal_i][300:]); quiet_b = np.median(quiet[cal_i][300:]); hf_b = np.median(hf[cal_i][200:])
    F["band_ratio"] = _at(ti, band, t_dec) / band_b
    F["quiet_ratio"] = _at(ti, quiet, t_dec) / quiet_b
    F["hf_ratio"] = _at(ti, hf, t_dec) / hf_b
    F["imu_rhythm"] = _rhythm(rb, FS_IMU, ti, t_dec, 3.0, 0.125, 0.8)
    g = _lp(belt, FS_IMU, 0.5)
    g0 = g[cal_i].mean(0)
    cosang = (g @ g0) / (np.linalg.norm(g, axis=1) * np.linalg.norm(g0))
    F["tilt"] = _at(ti, np.degrees(np.arccos(np.clip(cosang, -1, 1))), t_dec)
    F["res_norm"] = _at(ti, np.linalg.norm(rb, axis=1), t_dec)

    # ---------------- camera ----------------
    tc = tr["t_cam"]
    ok = tr["face_ok"].astype(bool)
    ear_b = np.median(tr["ear"][cal_c & ok])
    closed = (tr["ear"] < P["ear_closed_frac"] * ear_b) & ok
    F["closed_run"] = _at(tc, _runlen(closed, FS_CAM), t_dec)
    # a camera outage resets the no-blink timer: a stare must be observed on live frames
    cam_live = (tc - np.asarray(tr.get("cam_rx_t", tc))) <= P["cam_gap_tol_s"]
    last_closed = np.maximum.accumulate(np.where(closed | ~cam_live, tc, -1e9))
    since_closed = tc - np.maximum(last_closed, 0)
    gaze_sd3 = _roll(tr["gaze"], 3 * FS_CAM, "std")
    stare = ok & ~closed & (since_closed >= P["stare_s"]) & (gaze_sd3 < P["stare_gaze_sd"])
    F["stare"] = _at(tc, stare, t_dec)
    F["perclos"] = _at(tc, _roll(closed.astype(float), 60 * FS_CAM), t_dec)
    pitch_b = np.median(tr["pitch"][cal_c & ok])
    pitch_m = _roll(np.where(ok, tr["pitch"], np.nan), FS_CAM, "median")
    F["pitch_drop"] = np.nan_to_num(_at(tc, pitch_b - pitch_m, t_dec), nan=0.0)
    yaw_b = np.median(tr["yaw"][cal_c & ok])
    ydev = ok & (np.abs(tr["yaw"] - yaw_b) > P["yaw_dev_deg"])
    F["yaw_dev_run"] = _at(tc, _runlen(ydev, FS_CAM), t_dec)
    F["face_ok"] = _at(tc, _roll(ok.astype(float), FS_CAM), t_dec)
    a3 = _roll(tr["asym"], 3 * FS_CAM, "median")
    mu, sd = np.mean(a3[cal_c][3 * FS_CAM:]), max(np.std(a3[cal_c][3 * FS_CAM:]), 0.003)
    F["asym_z"] = _at(tc, (a3 - mu) / sd, t_dec)
    F["asym_d"] = _at(tc, a3 - mu, t_dec)
    e3 = _roll(tr["motion"], 3 * FS_CAM)
    eb = np.median(e3[cal_c][3 * FS_CAM:]); emad = max(np.median(np.abs(tr["motion"][cal_c] - np.median(tr["motion"][cal_c]))), 0.05)
    F["energy_z"] = _at(tc, (e3 - eb) / emad, t_dec)
    F["face_rhythm"] = _rhythm(_hp(tr["motion"], FS_CAM, 0.5), FS_CAM, tc, t_dec, 3.0, 0.2, 0.8)
    F["gaze"] = _at(tc, tr["gaze"], t_dec)
    gsd1 = _roll(tr["gaze"], FS_CAM, "std")
    g60 = _roll(gsd1, 60 * FS_CAM, "median")
    F["gaze_ratio"] = _at(tc, g60, t_dec) / np.median(gsd1[cal_c][FS_CAM:])

    # ---------------- telemetry ----------------
    tt = tr["t_tel"]
    rate = np.gradient(tr["steer"]) * FS_TEL
    rrms = np.sqrt(_roll(rate ** 2, 2 * FS_TEL))
    rb_ = np.median(rrms[cal_t][2 * FS_TEL:])
    # "no purposeful correction": median |steering rate| over 2 s, so brief involuntary jerks
    # (e.g. syncopal myoclonus, Lempert 1994) do not masquerade as driver input
    rmed = _roll(np.abs(rate), 2 * FS_TEL, "median")
    rmed_b = np.median(rmed[cal_t][2 * FS_TEL:])
    frozen = rmed < min(P["frozen_frac"] * rmed_b, P["frozen_abs"])
    F["frozen_run"] = _at(tt, _runlen(frozen, FS_TEL), t_dec)
    F["rate_ratio"] = _at(tt, rrms, t_dec) / rb_
    F["steer_rhythm"] = _rhythm(_bp(rate, FS_TEL, 1.0, 8.0), FS_TEL, tt, t_dec, 3.0, 0.15, 0.8)
    micro = _bp(tr["steer"], FS_TEL, 0.3, 2.0)
    # robust (median-absolute) amplitude so that rare abrupt corrections do not mask fewer micro-corrections
    F["micro_ratio"] = _at(tt, _roll(np.abs(micro), 60 * FS_TEL, "median"), t_dec) / np.median(np.abs(micro[cal_t]))
    thr = max(3 * 1.4826 * np.median(np.abs(rate[cal_t] - np.median(rate[cal_t]))), 10.0)
    pk, _ = signal.find_peaks(np.abs(rate), height=thr, distance=FS_TEL)
    pk_ind = np.zeros(len(tt)); pk_ind[pk] = 1
    F["abrupt"] = _at(tt, _roll(pk_ind, 60 * FS_TEL, "sum"), t_dec)
    F["lane_ratio"] = _at(tt, _roll(tr["lane"], 60 * FS_TEL, "std"), t_dec) / max(np.std(tr["lane"][cal_t]), 0.15)
    F["pedal"] = _at(tt, tr["pedal"], t_dec)
    F["steer_rate"] = _at(tt, np.abs(rate), t_dec)
    F["recal"] = dict(calib or {})

    if recal and calib is None and P["recal_enable"]:
        windows = find_recal(tr, F)
        if windows:
            return features(tr, use_reference_imu, calib=windows, recal=False)
    return F


def _bad_per_step(t_src, bad, t_dec):
    """True at a decision step if ANY native sample in (t - DT, t] is bad (no fault slips between steps)."""
    c = np.r_[0, np.cumsum(bad)]
    hi = np.searchsorted(t_src, t_dec, side="right")
    lo = np.searchsorted(t_src, t_dec - DT, side="right")
    return (c[hi] - c[lo]) > 0


def find_recal(tr, F):
    """Recalibration windows for modalities that were faulty during the initial calibration.

    Window = the first recal_s seconds in which (1) the modality's stream is healthy on every sample
    (camera: face tracked >= calib_face_min of the time), and (2) the detector, running WITHOUT that
    modality, raised NO Stage-1 alert. Condition (2) is the safety rule: never learn a baseline while
    the other sensors already see something abnormal. It cannot exclude an event that the other
    sensors have not seen yet (e.g. the slow start of hypoglycaemia); run_recal.py measures this."""
    P = PARAMS
    bad_cal = [k for k in RECAL_MODALITIES if not F["h_calib_valid"][k]]
    if not bad_cal:
        return {}
    t = F["t"]
    s1 = decide(F)["any_s1"]
    n = round(P["recal_s"] / DT)
    out = {}
    for k in bad_cal:
        if k == "belt":
            bad = _bad_per_step(tr["t_imu"], ~_imu_live(tr["belt"], FS_IMU, P["imu_stuck_s"]), t)
            extra = np.ones(len(t), bool)
        elif k == "tel":
            age = tr["t_tel"] - np.asarray(tr.get("tel_rx_t", tr["t_tel"]))
            bad = _bad_per_step(tr["t_tel"], age > P["tel_gap_tol_s"], t)
            extra = np.ones(len(t), bool)
        else:
            tc = tr["t_cam"]
            live = (tc - np.asarray(tr.get("cam_rx_t", tc))) <= P["cam_gap_tol_s"]
            bad = _bad_per_step(tc, ~live, t)
            face = _at(tc, (np.asarray(tr["face_ok"], bool) & live).astype(float), t)
            extra = _roll(face, n, "mean") >= P["calib_face_min"]
        good = ~bad & ~s1
        ok = (_runlen(good, 1 / DT) >= P["recal_s"] - 1e-9) & extra
        idx = np.flatnonzero(ok)
        if len(idx):
            b = round(float(t[idx[0]] + DT), 3)
            out[k] = (round(b - P["recal_s"], 3), b)
    return out


# --------------------------------------------------------------------------------------
# decision logic
# --------------------------------------------------------------------------------------
def _cum(b, win_s):
    return _roll(b.astype(float), round(win_s / DT), "sum") * DT


def decide(F, P=None, use=("imu", "cam", "tel")):
    P = {**PARAMS, **(P or {})}
    t = F["t"]
    ones = np.ones(len(t), bool)
    hk = lambda k: F.get("h_" + k, ones).astype(bool)
    # a modality votes only if it is used AND healthy (missing data is never evidence)
    cam = ("cam" in use) & hk("cam_ok")
    imu = ("imu" in use) & hk("belt_ok")
    tel = ("tel" in use) & hk("tel_ok")
    cam60 = ("cam" in use) & hk("cam_ok60")
    tel60 = ("tel" in use) & hk("tel_ok60")
    face = F["face_ok"] > 0.9

    # PERCLOS (60-s window) is valid only if the camera was healthy over the whole window;
    # without it the conservative 6-s closure rule applies and no fatigue warning is claimed
    perclos_ok = cam60
    drowsy_ctx = perclos_ok & (F["perclos"] > P["perclos"])
    closed_ev = F["closed_run"] >= np.where(drowsy_ctx | ~perclos_ok, P["unresp_closed_s"], P["closed_s"])
    cam_loc = cam & face & (closed_ev | F["stare"] | (F["pitch_drop"] > P["pitch_drop_deg"]) |
                            (F["yaw_dev_run"] >= P["gaze_away_s"]))
    imu_loc = imu & ((F["tilt"] > P["tilt_deg"]) | (F["quiet_ratio"] < P["quiet_ratio"]))
    tel_loc = tel & (F["frozen_run"] >= P["frozen_s"])
    n_loc = cam_loc.astype(int) + imu_loc + tel_loc

    imu_rh = imu & (F["imu_rhythm"] > P["rhythm"]) & (F["band_ratio"] > P["band_ratio"])
    cam_rh = cam & face & (F["face_rhythm"] > P["rhythm"]) & (F["energy_z"] > P["energy_z"])
    tel_rh = tel & (F["steer_rhythm"] > P["rhythm"]) & (F["rate_ratio"] > 1.5)
    n_rh = imu_rh.astype(int) + cam_rh + tel_rh

    asym_ev = cam & face & (F["asym_z"] > P["asym_z"]) & (F["asym_d"] > P["asym_abs"]) & \
        (F["energy_z"] < P["talk_energy_z"]) & (F["closed_run"] < 0.5)

    hy_c = np.stack([
        tel60 & (F["micro_ratio"] < P["hy_micro"]),
        tel60 & (F["abrupt"] >= P["hy_abrupt"]),
        tel60 & (F["lane_ratio"] > P["hy_lane"]),
        cam60 & (F["gaze_ratio"] < P["hy_gaze"]),
    ]).sum(0)
    hy_ok_time = t >= CALIB_S + 30  # need a partially filled 60-s window
    fatigue_recent = _roll(drowsy_ctx.astype(float), round(P["fatigue_hold_s"] / DT), "max") > 0

    # ---- Stage 1 (alert / driver prompt) ----
    s1 = dict(
        loc=cam_loc | (n_loc >= 2),
        # camera-only rhythm (e.g. talking) never prompts: >= 2 rhythmic modalities for 0.5 s
        seizure=_cum(n_rh >= 2, 1.0) >= 0.5,
        stroke=_cum(asym_ev, P["st_stage1_s"] + 1.0) >= P["st_stage1_s"] - 1e-6,
        hypoglycemia=hy_ok_time & (_cum(hy_c >= 3, P["hy_stage1_s"]) >= P["hy_stage1_s"] - 1e-6),
    )
    # ---- Stage 2 (verification) ----
    s2 = dict(
        loc=_cum(n_loc >= 2, P["loc_verify_s"] + 0.5) >= P["loc_verify_s"] - 1e-6,
        seizure=_cum(n_rh >= 2, 3.0) >= P["sz_verify_s"] - 1e-6,
        stroke=_cum(asym_ev, P["st_verify_s"] + 1.0) >= P["st_verify_s"] - 1e-6,
        # fatigue context in the last 120 s routes the same driving signature to the fatigue warning
        # the fatigue gate needs the camera (PERCLOS): without it, drowsiness and hypoglycaemia cannot be
        # told apart, so hypoglycaemia is not verified (Stage 1 still alerts)
        hypoglycemia=hy_ok_time & ~fatigue_recent & cam60 &
        (_cum(hy_c >= 3, P["hy_verify_s"]) >= P["hy_verify_s"] - 1e-6),
    )
    any_s1 = np.any(np.stack(list(s1.values())), 0)
    any_s2 = np.any(np.stack(list(s2.values())), 0)

    # fatigue routing (drowsiness is a separate warning, not ADI)
    fatigue = drowsy_ctx

    # ---- DEGRADED state: a stream is unhealthy; vehicle response is never triggered by it ----
    raw_bad = {k: ~hk(k + "_raw") for k in ("belt", "frame", "tel", "cam")}
    # DEGRADED lasts as long as any modality is excluded (incl. the post-fault exclusion windows)
    excluded = {k: ~hk(k + "_ok") for k in ("belt", "frame")}
    excluded.update({k: ~(hk(k + "_ok") & hk(k + "_ok60")) for k in ("tel", "cam")})
    degraded = np.any(np.stack(list(excluded.values())), 0)
    cam_face = cam & face                           # camera only counts if it actually sees the face
    n_healthy = cam_face.astype(int) + imu + tel
    not_assessable = dict(
        loc=n_healthy < 2, seizure=n_healthy < 2,   # corroboration needs >= 2 healthy modalities
        stroke=~cam_face,                           # stroke evidence is camera-only
        hypoglycemia=~tel60 | ~cam60,               # 3 of 4 criteria from telemetry; fatigue gate from camera
    )

    # ---- labelling at first verification ----
    out = dict(t=t, s1=s1, s2=s2, any_s1=any_s1, any_s2=any_s2, fatigue=fatigue,
               degraded=degraded, raw_bad=raw_bad, excluded=excluded, not_assessable=not_assessable,
               n_loc=n_loc, n_rh=n_rh, cam_loc=cam_loc, imu_loc=imu_loc, tel_loc=tel_loc, asym_ev=asym_ev, hy_c=hy_c)

    def label_at(i):
        if s2["seizure"][i]:
            return "seizure"
        if s2["loc"][i]:
            collapse = (F["pitch_drop"][i] > P["pitch_drop_deg"]) or (F["closed_run"][i] >= P["closed_s"])
            tonic = (F["yaw_dev_run"][i] >= P["gaze_away_s"]) or (F["hf_ratio"][i] > P["hf_ratio"] and bool(imu[i]))
            if tonic and not collapse:
                return "ADI-unspecified"
            if s2["stroke"][i] and not collapse:
                return "ADI-unspecified"
            return "syncope"
        if s2["hypoglycemia"][i] and s2["stroke"][i]:
            return "ADI-unspecified"
        if s2["stroke"][i]:
            return "stroke"
        if s2["hypoglycemia"][i]:
            return "hypoglycemia"
        return None

    out["label_at"] = label_at
    return out


def first_true(mask, t, t_from=-np.inf):
    idx = np.flatnonzero(mask & (t >= t_from))
    return (idx[0], t[idx[0]]) if len(idx) else (None, None)


def events_per_run(mask, t, refractory=30.0, t_from=-np.inf, t_to=np.inf):
    """Rising edges of mask with a refractory period (count of distinct alarms)."""
    m = mask & (t >= t_from) & (t < t_to)
    edges = np.flatnonzero(m & ~np.r_[False, m[:-1]])
    out, last = [], -np.inf
    for e in edges:
        if t[e] - last >= refractory:
            out.append(t[e]); last = t[e]
    return out


def run_trial(F, D, onset=None):
    """Summarise decisions for one trial."""
    t = D["t"]
    P = PARAMS
    res = {}
    if onset is not None:
        # only a verification that STARTS at or after onset counts (an ongoing pre-onset false
        # verification is not credited as a detection)
        s2 = D["any_s2"]
        s2_edge = s2 & ~np.r_[False, s2[:-1]]
        i, tv = first_true(s2_edge, t, onset)
        j, ts1 = first_true(D["any_s1"], t, onset)
        res["verified"] = tv is not None
        res["latency"] = (tv - onset) if tv is not None else np.nan
        res["s1_latency"] = (ts1 - onset) if ts1 is not None else np.nan
        res["label"] = D["label_at"](i) if i is not None else None
        final = res["label"]
        res["label_latency"] = res["latency"]
        if i is not None:
            # refine within label_refine_s: a later, more specific verification wins
            for k in range(i, min(len(t), i + int(P["label_refine_s"] / DT))):
                lab = D["label_at"](k)
                if (lab == "seizure" and final != "seizure") or (final == "ADI-unspecified" and lab not in (None, "ADI-unspecified")):
                    final = lab
                    res["label_latency"] = t[k] - onset
                    if lab == "seizure":
                        break
        res["final_label"] = final
        pre_end = onset
    else:
        pre_end = np.inf
    res["false_verifications"] = len(events_per_run(D["any_s2"], t, t_to=pre_end))
    res["prompts"] = len(events_per_run(D["any_s1"], t, refractory=10.0, t_to=pre_end))
    res["fatigue"] = len(events_per_run(D["fatigue"], t, refractory=60.0, t_to=pre_end))
    res["exposure_h"] = (min(pre_end, t[-1] + DT) - t[0]) / 3600
    # prompts falling inside drowsiness contexts
    return res


# --------------------------------------------------------------------------------------
# reference detectors (baselines)
# --------------------------------------------------------------------------------------
def baselines(F, onset=None):
    t = F["t"]
    # B1: Euro NCAP 2026 unresponsive-driver criterion: eyes closed >= 6 s, or gaze away
    #     (>20 deg) not returning: 3-4 s long distraction + 3 s no return => >= 7 s
    gaze_away = np.abs(F["gaze"]) > 20
    ga_run = _runlen(gaze_away, 1 / DT)
    b1 = (F["closed_run"] >= 6.0) | (ga_run >= 7.0)
    # B2: B1 + production-style input override: any accelerator or steering input => attentive
    inp = (F["pedal"] > 0.05) | (F["steer_rate"] > 2.0)
    since_input = _runlen(~inp, 1 / DT)
    b2 = b1 & (since_input >= 6.0)
    out = {}
    for name, m in (("B1", b1), ("B2", b2)):
        if onset is not None:
            _, tv = first_true(m, t, onset)
            out[name] = dict(verified=tv is not None, latency=(tv - onset) if tv is not None else np.nan,
                             false=len(events_per_run(m, t, t_to=onset)))
        else:
            out[name] = dict(false=len(events_per_run(m, t)))
    return out
