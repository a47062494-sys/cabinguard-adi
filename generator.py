"""
CabinGuard-ADI - synthetic multimodal trial generator (Phase 1 validation substitute).

Produces time-aligned streams at their native rates:
  * belt IMU and seat-frame (reference) IMU, MPU6050-like, 100 Hz, 3 axes [m/s^2]
      x = fore-aft, y = lateral, z = vertical
  * webcam face-landmark features, 30 Hz
      ear        eye aspect ratio (Soukupova & Cech 2016)
      pitch      head pitch [deg], negative = head dropping forward
      yaw        head yaw [deg]
      gaze       horizontal gaze angle [deg]
      asym       mouth/face landmark asymmetry index [-]
      motion     landmark motion energy [a.u.]
      face_ok    1 if a face is tracked
  * simulator telemetry, 20 Hz
      steer [deg], pedal [0..1], lane [m]

Every numeric parameter used for an event signature is listed in SIGNATURES with a
provenance tag: LIT (taken from a cited source), LIT-DIR (direction/shape from a source,
magnitude assumed), ASSUMED (team assumption, to be re-estimated in Phase 2).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

FS_IMU, FS_CAM, FS_TEL = 100, 30, 20
G = 9.81
CALIB_S = 20.0          # clean normal driving used for per-driver calibration

EVENT_CLASSES = ["seizure", "syncope", "stroke", "hypoglycemia"]

# --------------------------------------------------------------------------------------
# Signature parameters and their provenance
# --------------------------------------------------------------------------------------
SIGNATURES = {
    # ---- Seizure (generalized tonic-clonic + mild/focal variant) ----
    "sz_tonic_s": ((10, 20), "LIT", "Tonic phase usually 10-20 s (GTCS clinical description; secondary source, replace with primary)"),
    "sz_clonic_s": ((25, 50), "LIT-DIR", "Total ictal period usually <= ~1 min; clonic longer than tonic"),
    "sz_clonic_f_hz": ((4.0, 1.5), "LIT-DIR", "Clonic jerks start fast/low-amplitude, slow down and grow (GTCS description); "
                                            "4-8 Hz dominant ACM power in GTCS (Joo et al. 2017)"),
    "sz_belt_amp": ((1.5, 3.0), "ASSUMED", "Trunk (belt) jerk amplitude m/s^2; wrist values in literature are larger"),
    "sz_version_deg": ((20, 45), "LIT-DIR", "Head/eye version during tonic phase; magnitude assumed"),
    "sz_mild_frac": (0.25, "ASSUMED", "Share of mild/focal seizures (no tonic phase, 0.3x amplitude, driver conscious)"),
    "sz_pedal_press_frac": (0.5, "ASSUMED", "Tonic leg extension presses accelerator"),
    # ---- Syncope / cardiac LOC ----
    "sy_collapse_frac": (0.87, "LIT-DIR", "2/15 described cases upright-unconscious (Nilsson et al. 2026); all others modelled as collapse"),
    "sy_eyes_open_frac": (0.76, "LIT", "Eyes remained open in 76% (Lempert et al. 1994)"),
    "sy_myoclonus_frac": (0.90, "LIT", "Myoclonus in 90% of induced syncope (Lempert et al. 1994)"),
    "sy_myoclonus_start_s": ((2.6, 0.9), "LIT", "Myoclonus starts 2.6 +/- 0.9 s after LOC (Lempert 1994)"),
    "sy_myoclonus_dur_s": ((6.6, 3.5), "LIT", "Myoclonus lasts 6.6 +/- 3.5 s, arrhythmic/multifocal (Lempert 1994)"),
    "sy_transient_dur_s": ((12.1, 4.4), "LIT", "Induced syncope lasted 12.1 +/- 4.4 s, range 4.5-21.7 s (Lempert 1994)"),
    "sy_persistent_frac": (0.5, "ASSUMED", "Share of non-recovering LOC (cardiac arrest dominates fatal crashes, Nilsson 2026)"),
    "sy_pedal_sustained_frac": (0.62, "LIT", "15/24 cases sustained accelerator, no braking (Nilsson et al. 2026)"),
    "sy_steer": ("frozen", "LIT", "Near-zero steering in all 7 clean-signal cases (Nilsson et al. 2026)"),
    "sy_head_drop_deg": ((20, 45), "ASSUMED", "Head pitch drop when collapsing"),
    # ---- Stroke ----
    "st_clear_frac": (0.65, "ASSUMED", "Share of clear (vs mild) facial droop"),
    "st_asym_clear": ((0.06, 0.12), "ASSUMED", "Landmark asymmetry increase, clear droop (CPSS facial droop item: Kothari 1999)"),
    "st_asym_mild": ((0.02, 0.045), "ASSUMED", "Landmark asymmetry increase, mild droop"),
    "st_gaze_dev_frac": (0.25, "ASSUMED", "Sustained gaze deviation (10-20 deg)"),
    "st_drift_frac": (0.5, "ASSUMED", "Lateral drift from arm weakness"),
    # ---- Hypoglycaemia ----
    "hy_ramp_s": ((120, 300), "ASSUMED", "Time from first sign to full impairment"),
    "hy_micro_drop": (0.6, "LIT-DIR", "Fewer micro-corrections (Lehmann et al. 2024, real cars)"),
    "hy_abrupt_per_min": (3.0, "LIT-DIR", "More abrupt corrections / 'hectic' behaviour (Lehmann 2024)"),
    "hy_gaze_drop": (0.5, "LIT-DIR", "Monotonous gaze with abrupt shifts (Lehmann 2024)"),
    "hy_lane_gain": (1.5, "LIT-DIR", "Driving performance impaired at BG 3.4-4.0 mmol/l already (Cox et al. 2000)"),
    # ---- Normal-driving confounders ----
    "nd_resonance_hz": ((4.0, 6.0), "LIT", "Seated-body vertical resonance ~4-6 Hz (Fairley & Griffin 1989)"),
    "nd_blink_dur_s": ((0.10, 0.40), "LIT", "Blink lasts ~100-400 ms (Soukupova & Cech 2016)"),
    "nd_microsleep_s": ((1.0, 2.5), "LIT", "Microsleep = 1-2 s eye closure (Euro NCAP 2026); upper bound stretched to 2.5 s as a stress case"),
    "nd_drowsy_frac": (0.4, "ASSUMED", "Share of normal trials containing a drowsiness episode"),
    "nd_speech_hz": ((3.5, 5.5), "ASSUMED", "Syllabic jaw/face motion while talking"),
    "nd_pose_asym": (0.04, "ASSUMED", "Apparent landmark asymmetry per |sin(head yaw)| after frontalisation"),
}


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _filt_noise(rng, n, fs, lo=None, hi=None, sd=1.0, order=2):
    pad = int(fs * 3)
    x = rng.standard_normal(n + pad)
    if lo is None and hi is not None:
        b, a = signal.butter(order, hi / (fs / 2), "low")
    elif hi is None and lo is not None:
        b, a = signal.butter(order, lo / (fs / 2), "high")
    elif lo is not None and hi is not None:
        b, a = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], "band")
    else:
        return x[pad:] * sd
    y = signal.lfilter(b, a, x)[pad:]
    return y / (y.std() + 1e-12) * sd


def _intervals_mask(t, intervals):
    m = np.zeros_like(t, dtype=bool)
    for a, b in intervals:
        m |= (t >= a) & (t < b)
    return m


def _poisson_times(rng, rate_fn, t0, t1, max_rate):
    """Thinning: event times of an inhomogeneous Poisson process on [t0, t1)."""
    out, t = [], t0
    if max_rate <= 0:
        return out
    while True:
        t += rng.exponential(1.0 / max_rate)
        if t >= t1:
            return out
        if rng.random() < rate_fn(t) / max_rate:
            out.append(t)


def _smoothstep(t, a, dur):
    x = np.clip((t - a) / max(dur, 1e-6), 0, 1)
    return x * x * (3 - 2 * x)


def _transmissibility(fs, f0, zeta):
    """Base-excitation transmissibility of a 1-DOF seated body (bilinear)."""
    w = 2 * np.pi * f0
    b, a = signal.bilinear([2 * zeta * w, w * w], [1, 2 * zeta * w, w * w], fs)
    return b, a


def sample_driver(rng):
    return dict(
        ear_open=rng.normal(0.30, 0.03),
        blink_rate=rng.uniform(12, 22) / 60.0,
        asym_base=abs(rng.normal(0.03, 0.012)),
        asym_sd=rng.uniform(0.006, 0.012),
        steer_micro=rng.uniform(0.6, 1.6),
        pedal_mean=rng.uniform(0.18, 0.35),
        pitch0=rng.normal(0, 3),
        f_res=rng.uniform(*SIGNATURES["nd_resonance_hz"][0]),
        zeta=rng.uniform(0.25, 0.4),
        breath=rng.uniform(0.2, 0.33),
        fidget=rng.uniform(0.05, 0.15),
        gaze_sd=rng.uniform(4, 9),
        lane_sd=rng.uniform(0.10, 0.20),
    )


# --------------------------------------------------------------------------------------
# main generator
# --------------------------------------------------------------------------------------
def generate_trial(cls: str, seed: int, variant: str | None = None, duration: float | None = None):
    """Return a dict with all streams + ground truth for one trial."""
    rng = np.random.default_rng(seed)
    drv = sample_driver(rng)
    SG = SIGNATURES

    # ---------------- timeline ----------------
    if cls == "normal":
        T = duration or 240.0
        onset = None
    else:
        onset = CALIB_S + rng.uniform(30, 60)
        if cls == "hypoglycemia":
            ramp = rng.uniform(*SG["hy_ramp_s"][0])
            T = onset + ramp + 60
        else:
            T = onset + 60
    ev = dict(cls=cls, onset=onset, seed=seed)

    t_imu = np.arange(0, T, 1 / FS_IMU)
    t_cam = np.arange(0, T, 1 / FS_CAM)
    t_tel = np.arange(0, T, 1 / FS_TEL)

    # ---------------- event parameters ----------------
    conscious_end = np.inf     # time behaviour (blinks, fidget, corrections...) stops
    recover_t = np.inf
    if cls == "seizure":
        mild = variant == "mild" if variant else rng.random() < SG["sz_mild_frac"][0]
        ev["variant"] = "mild" if mild else "gtcs"
        tonic = 0.0 if mild else rng.uniform(*SG["sz_tonic_s"][0])
        clonic = rng.uniform(20, 40) if mild else rng.uniform(*SG["sz_clonic_s"][0])
        ev.update(tonic=tonic, clonic=clonic)
        if not mild:
            conscious_end = onset
        ev["pedal_press"] = (not mild) and rng.random() < SG["sz_pedal_press_frac"][0]
        ev["version"] = rng.choice([-1, 1]) * rng.uniform(*SG["sz_version_deg"][0])
        ev["belt_amp"] = rng.uniform(*SG["sz_belt_amp"][0]) * (0.3 if mild else 1.0)
    elif cls == "syncope":
        if variant:
            ev["variant"] = variant
        ev["collapse"] = rng.random() < SG["sy_collapse_frac"][0] if variant != "upright" else False
        if variant == "upright":
            ev["collapse"] = False
        ev["eyes_open"] = rng.random() < SG["sy_eyes_open_frac"][0]
        ev["myoclonus"] = rng.random() < SG["sy_myoclonus_frac"][0]
        ev["pedal_sustained"] = rng.random() < SG["sy_pedal_sustained_frac"][0]
        persistent = rng.random() < SG["sy_persistent_frac"][0]
        ev["persistent"] = persistent
        dur = np.inf if persistent else float(np.clip(rng.normal(*SG["sy_transient_dur_s"][0]), 4.5, 21.7))
        ev["loc_dur"] = dur
        conscious_end = onset
        recover_t = onset + dur
        ev["variant"] = ev.get("variant") or ("upright" if not ev["collapse"] else "collapse")
    elif cls == "stroke":
        clear = rng.random() < SG["st_clear_frac"][0] if variant is None else variant == "clear"
        ev["variant"] = "clear" if clear else "mild"
        ev["d_asym"] = rng.uniform(*SG["st_asym_clear" if clear else "st_asym_mild"][0])
        if variant and variant.startswith("delta:"):          # droop-magnitude sweep
            ev["d_asym"] = float(variant.split(":")[1]); ev["variant"] = variant
        ev["gaze_dev"] = (rng.random() < SG["st_gaze_dev_frac"][0]) * rng.choice([-1, 1]) * rng.uniform(10, 20)
        ev["drift"] = (rng.random() < SG["st_drift_frac"][0]) * rng.choice([-1, 1]) * rng.uniform(0.03, 0.08)
        ev["rise"] = rng.uniform(1, 5)
    elif cls == "hypoglycemia":
        ev["variant"] = "progressive"
        ev["ramp"] = ramp

    def h_of(t):  # hypoglycaemia impairment level 0..1
        if cls != "hypoglycemia":
            return np.zeros_like(np.asarray(t, dtype=float))
        return np.clip((np.asarray(t) - onset) / ev["ramp"], 0, 1)

    def conscious(t):
        t = np.asarray(t)
        return (t < conscious_end) | (t >= recover_t)

    # ---------------- context: road + driver-behaviour confounders ----------------
    # road segments (after calibration): rough road, steady straight road
    rough, steady = [], []
    t = CALIB_S
    while t < T:
        seg = rng.uniform(15, 45)
        r = rng.random()
        if r < 0.15:
            rough.append((t, t + seg))
        elif r < 0.35:
            steady.append((t, t + seg))
        t += seg
    potholes = _poisson_times(rng, lambda tt: (1.0 if any(a <= tt < b for a, b in rough) else 0.15) / 60,
                              CALIB_S, T, 1.0 / 60)

    # drowsiness episode (normal trials only)
    drowsy = []
    if cls == "normal" and rng.random() < SG["nd_drowsy_frac"][0]:
        a = rng.uniform(CALIB_S + 20, T - 150)
        drowsy.append((a, a + rng.uniform(60, 150)))
    ev["drowsy"] = drowsy
    in_drowsy = lambda tt: any(a <= tt < b for a, b in drowsy)

    # talking episodes (not in calibration; not while unconscious)
    talking = []
    t = CALIB_S + rng.uniform(0, 20)
    while t < T:
        d = rng.uniform(5, 25)
        if rng.random() < 0.35:
            talking.append((t, t + d))
        t += d + rng.uniform(5, 30)
    ev["talking"] = talking

    # ================================================================ CAMERA 30 Hz
    n = len(t_cam)
    cons_c = conscious(t_cam)
    hc = h_of(t_cam)
    ear = drv["ear_open"] + _filt_noise(rng, n, FS_CAM, hi=2, sd=0.008)
    pitch = drv["pitch0"] + _filt_noise(rng, n, FS_CAM, hi=0.5, sd=2.0)
    yaw = _filt_noise(rng, n, FS_CAM, hi=0.5, sd=3.0)
    asym = drv["asym_base"] + _filt_noise(rng, n, FS_CAM, hi=1.0, sd=drv["asym_sd"])
    motion = 1.0 + _filt_noise(rng, n, FS_CAM, hi=3, sd=0.15)

    def add_blink(tb, dur, depth=0.85):
        m = (t_cam >= tb) & (t_cam < tb + dur)
        ph = (t_cam[m] - tb) / dur
        ear[m] -= depth * drv["ear_open"] * np.sin(np.pi * ph) ** 0.7

    # blinks / microsleeps / nods
    blink_rate = lambda tt: drv["blink_rate"] * (1.5 if in_drowsy(tt) else 1.0) * float(conscious(tt))
    for tb in _poisson_times(rng, blink_rate, 0, T, drv["blink_rate"] * 1.5):
        dur = rng.uniform(0.3, 0.8) if in_drowsy(tb) else rng.uniform(*SG["nd_blink_dur_s"][0])
        add_blink(tb, dur)
    for a, b in drowsy:
        for tm in _poisson_times(rng, lambda tt: 1.5 / 60, a, b, 1.5 / 60):
            d = rng.uniform(*SG["nd_microsleep_s"][0])
            m = (t_cam >= tm) & (t_cam < tm + d)
            ear[m] = drv["ear_open"] * 0.1 + rng.normal(0, 0.005, m.sum())
            nod = rng.uniform(8, 20)
            pitch -= nod * np.exp(-0.5 * ((t_cam - (tm + d * 0.7)) / (0.35 + d / 3)) ** 2)
    # mirror checks
    mirror_rate = lambda tt: 3 / 60 * float(conscious(tt)) * (1 - 0.7 * float(h_of(tt))) * (0.3 if in_drowsy(tt) else 1)
    for tm in _poisson_times(rng, mirror_rate, CALIB_S, T, 3 / 60):
        d = rng.uniform(0.6, 1.5)
        amp = rng.choice([-1, 1]) * rng.uniform(35, 60)
        yaw += amp * (_smoothstep(t_cam, tm, 0.25) - _smoothstep(t_cam, tm + d, 0.3))
    # gaze = head yaw + fixations/saccades
    gaze = yaw.copy()
    fix_t = 0.0
    gsd_fn = lambda tt: drv["gaze_sd"] * (1 - SG["hy_gaze_drop"][0] * float(h_of(tt)))
    while fix_t < T:
        d = rng.uniform(0.2, 0.6)
        m = (t_cam >= fix_t) & (t_cam < fix_t + d)
        if conscious(fix_t):
            gaze[m] += rng.normal(0, gsd_fn(fix_t))
        fix_t += d
    if cls == "hypoglycemia":  # abrupt gaze shifts
        for tg in _poisson_times(rng, lambda tt: 2 * float(h_of(tt)) / 60, onset, T, 2 / 60):
            d = rng.uniform(0.5, 1.2)
            gaze += rng.choice([-1, 1]) * rng.uniform(20, 30) * (_smoothstep(t_cam, tg, 0.08) - _smoothstep(t_cam, tg + d, 0.1))
    # talking: rhythmic syllabic face motion + asymmetry fluctuation + smiles
    for a, b in talking:
        f = rng.uniform(*SG["nd_speech_hz"][0])
        m = (t_cam >= a) & (t_cam < b) & cons_c & (hc < 0.5)
        ph = 2 * np.pi * np.cumsum(f * (1 + 0.15 * _filt_noise(rng, n, FS_CAM, hi=0.5))) / FS_CAM
        motion[m] += (1.2 * (0.5 + 0.5 * np.sin(ph[m])) + 0.3 * rng.random(m.sum()))
        asym[m] += 0.6 * drv["asym_sd"] * np.sin(ph[m])
        pitch[m] += 1.0 * np.sin(ph[m])
    for ts in _poisson_times(rng, lambda tt: 1.0 / 60 * float(conscious(tt)), CALIB_S, T, 1.0 / 60):
        d = rng.uniform(1, 4)
        asym += rng.uniform(0.01, 0.05) * (_smoothstep(t_cam, ts, 0.3) - _smoothstep(t_cam, ts + d, 0.4))
        motion += 0.8 * np.exp(-0.5 * ((t_cam - ts) / 0.2) ** 2)
    motion += 0.04 * np.abs(np.gradient(yaw) * FS_CAM) ** 0.8
    # apparent (pose-induced) asymmetry: landmark projection error grows with head yaw (ASSUMED 0.04*|sin yaw|)
    asym += SIGNATURES["nd_pose_asym"][0] * np.abs(np.sin(np.deg2rad(yaw)))
    face_ok = np.ones(n, dtype=bool)
    face_ok &= np.abs(yaw) < 55

    # ================================================================ TELEMETRY 20 Hz
    nt = len(t_tel)
    cons_t = conscious(t_tel)
    ht = h_of(t_tel)
    steady_t = _intervals_mask(t_tel, steady)
    drowsy_t = _intervals_mask(t_tel, drowsy)
    curv = _filt_noise(rng, nt, FS_TEL, hi=0.05, sd=5.0) * np.where(steady_t, 0.2, 1.0)
    micro_gain = np.where(steady_t, 0.3, 1.0) * np.where(drowsy_t, 0.5, 1.0) * (1 - SG["hy_micro_drop"][0] * ht)
    micro = _filt_noise(rng, nt, FS_TEL, lo=0.2, hi=2.0, sd=drv["steer_micro"]) * micro_gain
    abrupt = np.zeros(nt)
    rate_ab = lambda tt: (1.5 / 60 if in_drowsy(tt) else 0) + SG["hy_abrupt_per_min"][0] / 60 * float(h_of(tt))
    for ta in _poisson_times(rng, rate_ab, CALIB_S, T, 3.5 / 60):
        abrupt += rng.choice([-1, 1]) * rng.uniform(4, 10) * np.exp(-0.5 * ((t_tel - ta) / 0.15) ** 2)
    steer = curv + micro + abrupt
    pedal = drv["pedal_mean"] + _filt_noise(rng, nt, FS_TEL, hi=0.1, sd=0.05) * (1 + ht)
    pedal = np.where(steady_t, drv["pedal_mean"] + 0.003 * rng.standard_normal(nt), pedal)
    lane = _filt_noise(rng, nt, FS_TEL, hi=0.1, sd=drv["lane_sd"]) * (1 + SG["hy_lane_gain"][0] * ht) * np.where(drowsy_t, 1.5, 1.0)

    # ================================================================ IMU 100 Hz
    ni = len(t_imu)
    rough_i = _intervals_mask(t_imu, rough)
    level = np.where(rough_i, 0.55, 0.15)
    level = signal.lfilter([0.02], [1, -0.98], level) / 1.0  # smooth transitions
    frame = np.zeros((ni, 3))
    frame[:, 2] = _filt_noise(rng, ni, FS_IMU, lo=1, hi=25, sd=1.0) * level
    frame[:, 0] = _filt_noise(rng, ni, FS_IMU, lo=1, hi=25, sd=0.5) * level
    frame[:, 1] = _filt_noise(rng, ni, FS_IMU, lo=1, hi=25, sd=0.5) * level
    for tp in potholes:
        m = (t_imu >= tp) & (t_imu < tp + 0.6)
        tau = t_imu[m] - tp
        frame[m, 2] += rng.uniform(2, 4) * np.exp(-tau / 0.08) * np.sin(2 * np.pi * 12 * tau)
    b, a = _transmissibility(FS_IMU, drv["f_res"], drv["zeta"])
    bl, al = _transmissibility(FS_IMU, drv["f_res"] * 0.45, drv["zeta"])
    belt = np.zeros((ni, 3))
    belt[:, 2] = signal.lfilter(b, a, frame[:, 2])
    belt[:, 0] = signal.lfilter(bl, al, frame[:, 0])
    belt[:, 1] = signal.lfilter(bl, al, frame[:, 1])
    cons_i = conscious(t_imu)
    fid = np.stack([_filt_noise(rng, ni, FS_IMU, hi=1.5, sd=drv["fidget"]) for _ in range(3)], 1)
    belt += fid * cons_i[:, None]
    belt[:, 0] += 0.03 * np.sin(2 * np.pi * drv["breath"] * t_imu)
    for a_, b_ in talking:  # gestures while talking
        m = (t_imu >= a_) & (t_imu < b_) & cons_i
        belt[m, :2] += _filt_noise(rng, m.sum(), FS_IMU, lo=0.5, hi=2.0, sd=0.15)[:, None]
    trunk_pitch = np.full(ni, 0.0)   # deg forward flexion
    trunk_roll = np.zeros(ni)

    # ================================================================ EVENT OVERLAYS
    if cls == "seizure":
        t0, Tt, Tc = onset, ev["tonic"], ev["clonic"]
        c0, c1 = t0 + Tt, t0 + Tt + Tc
        mild = ev["variant"] == "mild"
        A = ev["belt_amp"]
        if not mild:
            # tonic phase: stiffening, head/eye version, onset jerk, low-amplitude high-freq tremor
            vers = ev["version"] * _smoothstep(t_cam, t0, 1.0) * (1 - _smoothstep(t_cam, c1, 3.0))
            yaw += vers
            gaze = np.where((t_cam >= t0), yaw + 0.3 * rng.standard_normal(n), gaze)
            ear = np.where((t_cam >= t0) & (t_cam < c0), drv["ear_open"] * 1.12 + 0.005 * rng.standard_normal(n), ear)
            mt = (t_imu >= t0) & (t_imu < c0)
            belt[mt] += _filt_noise(rng, mt.sum(), FS_IMU, lo=8, hi=20, sd=0.25)[:, None] * np.array([1, 0.6, 1])
            mj = (t_imu >= t0) & (t_imu < t0 + 0.3)
            belt[mj, 0] += 2.0 * np.sin(np.pi * (t_imu[mj] - t0) / 0.3)
            trunk_pitch += rng.choice([-1, 1]) * rng.uniform(8, 15) * _smoothstep(t_imu, t0, 0.8)
            face_tremor = (t_cam >= t0) & (t_cam < c0)
            motion[face_tremor] += 0.6 + 0.3 * rng.random(face_tremor.sum())
            # steering: arms stiffen -> jerk then held; pedal
            jump = rng.choice([-1, 1]) * rng.uniform(5, 20)
            idx0 = np.searchsorted(t_tel, t0)
            held = steer[idx0] + jump * _smoothstep(t_tel, t0, 0.3)
            steer = np.where(t_tel >= t0, held + 0.05 * rng.standard_normal(nt), steer)
            if ev["pedal_press"]:
                target = rng.uniform(0.6, 1.0)
            else:
                target = 0.0
            p0 = pedal[idx0]
            pedal = np.where(t_tel >= t0, p0 + (target - p0) * _smoothstep(t_tel, t0, 0.5), pedal)
        # clonic phase: rhythmic pulse train, frequency falls, amplitude rises
        f_hi, f_lo = (3.0, 2.0) if mild else SG["sz_clonic_f_hz"][0]

        def pulse_train(tt, fs, width_s=0.06):
            m = (tt >= c0) & (tt < c1)
            x = np.zeros_like(tt)
            prog = np.clip((tt - c0) / Tc, 0, 1)
            f = f_hi + (f_lo - f_hi) * prog
            phase = np.cumsum(f * m) / fs
            amp = (0.4 + 0.6 * prog) * np.where(prog > 0.85, (1 - prog) / 0.15 + 0.1, 1.0)
            frac = phase % 1.0
            width = width_s * f  # pulse width expressed in cycles
            x = np.where(m & (frac < width), np.sin(np.pi * frac / np.maximum(width, 1e-6)), 0.0) * amp
            return x, m
        pi_, mi = pulse_train(t_imu, FS_IMU)
        belt[:, 0] += A * pi_ * rng.uniform(0.7, 1.0)
        belt[:, 2] += A * pi_ * rng.uniform(0.5, 1.0)
        belt[:, 1] += A * pi_ * rng.uniform(0.2, 0.5)
        pc, mc = pulse_train(t_cam, FS_CAM, 0.15)   # head/face jerk incl. recoil (assumed 150 ms)
        k = 0.35 if mild else 1.0
        pitch += k * rng.uniform(3, 8) * pc
        yaw += k * rng.uniform(1, 4) * pc
        motion += (1.5 if mild else 3.0) * pc
        asym += (0.05 if mild else 0.04) * pc
        if not mild:
            ear = np.where(mc, ear - 0.25 * drv["ear_open"] * pc, ear)
        ptl, mtl = pulse_train(t_tel, FS_TEL, 0.15)
        steer += k * rng.uniform(2, 6) * ptl
        pedal += (0.03 if mild else 0.06) * ptl
        if not mild:
            # post-ictal: collapsed, eyes closed, quiet
            post_c = t_cam >= c1
            ear = np.where(post_c, drv["ear_open"] * 0.2 + 0.005 * rng.standard_normal(n), ear)
            pitch -= 25 * _smoothstep(t_cam, c1, 3)
    elif cls == "syncope":
        t0 = onset
        rec_c = recover_t
        # posture
        if ev["collapse"]:
            drop = rng.uniform(*SG["sy_head_drop_deg"][0])
            fall = rng.uniform(1, 3)
            pitch -= drop * _smoothstep(t_cam, t0, fall) * (1 - _smoothstep(t_cam, rec_c, 2))
            tp = rng.uniform(10, 25)
            trunk_pitch += tp * _smoothstep(t_imu, t0, fall) * (1 - _smoothstep(t_imu, rec_c, 2))
            trunk_roll += rng.uniform(-10, 10) * _smoothstep(t_imu, t0, fall) * (1 - _smoothstep(t_imu, rec_c, 2))
            face_ok &= ~((t_cam >= t0 + fall) & (t_cam < rec_c) & (rng.random() < 0.3))
        else:
            pitch -= rng.uniform(0, 5) * _smoothstep(t_cam, t0, 2) * (1 - _smoothstep(t_cam, rec_c, 2))
        loc_c = (t_cam >= t0) & (t_cam < rec_c)
        # eyes: open & fixed (upward deviation) or closed; gaze stops scanning
        if ev["eyes_open"]:
            ear = np.where(loc_c, drv["ear_open"] * rng.uniform(0.8, 0.95) + 0.004 * rng.standard_normal(n), ear)
        else:
            ear = np.where(loc_c, drv["ear_open"] * (1 - 0.85 * _smoothstep(t_cam, t0, 1.0)) + 0.004 * rng.standard_normal(n), ear)
        g_hold = gaze[np.searchsorted(t_cam, t0)]
        gaze = np.where(loc_c, g_hold + 0.3 * rng.standard_normal(n), gaze)
        motion = np.where(loc_c, 0.6 + 0.1 * rng.random(n), motion)
        # myoclonus: arrhythmic, multifocal jerks
        if ev["myoclonus"]:
            ms = t0 + max(0.5, rng.normal(*SG["sy_myoclonus_start_s"][0]))
            md = max(1.0, rng.normal(*SG["sy_myoclonus_dur_s"][0]))
            me = min(ms + md, rec_c)
            ev["myoclonus_window"] = (ms, me)
            for tj in _poisson_times(rng, lambda tt: 3.0, ms, me, 3.0):
                amp = rng.uniform(0.3, 1.2)
                ax = rng.integers(0, 3)
                mj = (t_imu >= tj) & (t_imu < tj + 0.06)
                belt[mj, ax] += amp * np.sin(np.pi * (t_imu[mj] - tj) / 0.06)
                mcj = (t_cam >= tj) & (t_cam < tj + 0.1)
                motion[mcj] += rng.uniform(0.5, 1.5)
                pitch[mcj] += rng.uniform(-2, 2)
                mtj = (t_tel >= tj) & (t_tel < tj + 0.1)
                steer[mtj] += rng.uniform(-1.5, 1.5)
        # steering frozen, slowly self-centring (assumption); pedal sustained or released
        idx0 = np.searchsorted(t_tel, t0)
        s0, p0 = steer[idx0], pedal[idx0]
        loc_t = (t_tel >= t0) & (t_tel < rec_c)
        steer = np.where(loc_t, s0 * np.exp(-(t_tel - t0) / 8.0) + 0.05 * rng.standard_normal(nt) + (steer - (curv + micro + abrupt)), steer)
        if ev["pedal_sustained"]:
            pedal = np.where(loc_t, p0 + rng.uniform(0, 0.15) * _smoothstep(t_tel, t0, 1.5) + 0.002 * rng.standard_normal(nt), pedal)
        else:
            pedal = np.where(loc_t, p0 * (1 - _smoothstep(t_tel, t0, 1.0)), pedal)
        lane = np.where(t_tel >= t0, lane + np.clip(t_tel - t0, 0, None) * rng.uniform(0.1, 0.4) * rng.choice([-1, 1]) * loc_t, lane)
    elif cls == "stroke":
        t0 = onset
        asym += ev["d_asym"] * _smoothstep(t_cam, t0, ev["rise"])
        if ev["gaze_dev"]:
            gaze += ev["gaze_dev"] * _smoothstep(t_cam, t0, 2)
            yaw += 0.5 * ev["gaze_dev"] * _smoothstep(t_cam, t0, 2)
        if ev["drift"]:
            lane += ev["drift"] * np.clip(t_tel - t0, 0, None) * (1 - 0.5 * np.abs(np.sin(2 * np.pi * (t_tel - t0) / 20)))
            steer = np.where(t_tel >= t0, curv + micro * 0.6 + abrupt, steer)

    # trunk posture -> gravity projection on belt IMU
    th, ph = np.deg2rad(trunk_pitch), np.deg2rad(trunk_roll)
    belt[:, 0] += G * np.sin(th)
    belt[:, 1] += G * np.sin(ph)
    belt[:, 2] += G * np.cos(th) * np.cos(ph)
    frame[:, 2] += G
    # sensor noise (MPU6050-like)
    belt += 0.02 * rng.standard_normal(belt.shape)
    frame += 0.02 * rng.standard_normal(frame.shape)
    ear += 0.004 * rng.standard_normal(n)
    pedal = np.clip(pedal, 0, 1)
    steer += 0.02 * rng.standard_normal(nt)

    return dict(
        t_imu=t_imu, belt=belt, frame=frame,
        t_cam=t_cam, ear=ear, pitch=pitch, yaw=yaw, gaze=gaze, asym=asym, motion=motion, face_ok=face_ok,
        t_tel=t_tel, steer=steer, pedal=pedal, lane=lane,
        T=T, truth=ev, driver=drv, context=dict(rough=rough, steady=steady, potholes=potholes, drowsy=drowsy, talking=talking),
    )
