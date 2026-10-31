"""
CabinGuard-ADI - sensor-fault injection (fail-safe test, added 7 Oct 2026).

Faults are applied to a generated trial *after* generation, so the generator is unchanged.
Each fault reproduces what a real stream looks like to the detector when the hardware fails:

  belt_stuck   belt IMU (MPU6050) stops updating: the last register value is repeated
               (typical I2C hang). Detected from the data itself (bit-identical samples).
  frame_stuck  same failure on the seat-frame reference IMU.
  tel_loss     simulator telemetry messages stop arriving. The receive buffer repeats the last
               value AND the timestamp of the last message received (tel_rx_t); the detector's
               watchdog must notice that this timestamp is getting old.
  cam_loss     webcam frames stop arriving (USB disconnect): the last frame is repeated with
               its old timestamp (cam_rx_t) and no face is tracked.

The faulty interval is written to tr["faults"] = [(kind, t0, t1)] for scoring.
"""
from __future__ import annotations

import numpy as np

FAULT_KINDS = ("belt_stuck", "frame_stuck", "tel_loss", "cam_loss")


def _hold(x, m):
    """Repeat the last value before the masked interval (stuck sensor / stale buffer)."""
    x = x.copy()
    idx = np.flatnonzero(m)
    if len(idx):
        starts = idx[np.r_[True, np.diff(idx) > 1]]          # each contiguous faulty run holds its own last value
        run_start = starts[np.searchsorted(starts, idx, side="right") - 1]
        x[idx] = x[np.maximum(run_start - 1, 0)]
    return x


def inject(tr, kind, t0, t1):
    tr = dict(tr)
    if kind == "belt_stuck":
        m = (tr["t_imu"] >= t0) & (tr["t_imu"] < t1)
        tr["belt"] = _hold(tr["belt"], m)
    elif kind == "frame_stuck":
        m = (tr["t_imu"] >= t0) & (tr["t_imu"] < t1)
        tr["frame"] = _hold(tr["frame"], m)
    elif kind == "tel_loss":
        m = (tr["t_tel"] >= t0) & (tr["t_tel"] < t1)
        for k in ("steer", "pedal", "lane"):
            tr[k] = _hold(tr[k], m)
        # the receive buffer keeps the timestamp of the last message it actually received
        tr["tel_rx_t"] = _hold(np.asarray(tr.get("tel_rx_t", tr["t_tel"]), float), m)
    elif kind == "cam_loss":
        m = (tr["t_cam"] >= t0) & (tr["t_cam"] < t1)
        for k in ("ear", "pitch", "yaw", "gaze", "asym", "motion"):
            tr[k] = _hold(tr[k], m)
        tr["face_ok"] = tr["face_ok"] & ~m
        tr["cam_rx_t"] = _hold(np.asarray(tr.get("cam_rx_t", tr["t_cam"]), float), m)
    elif kind == "cam_stall":
        # benign: one 133-ms webcam stall (4 frames) every 60 s from t0 to t1 - must NOT cause DEGRADED
        m = np.zeros(len(tr["t_cam"]), bool)
        for ts in np.arange(t0, min(t1, tr["t_cam"][-1]), 60.0):
            m |= (tr["t_cam"] >= ts) & (tr["t_cam"] < ts + 0.133)
        tr["cam_rx_t"] = _hold(np.asarray(tr.get("cam_rx_t", tr["t_cam"]), float), m)
    else:
        raise ValueError(kind)
    tr["faults"] = list(tr.get("faults", [])) + [(kind, t0, t1)]
    return tr
