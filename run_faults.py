"""
CabinGuard-ADI - fault-injection test (fail-safe behaviour), added 7 Oct 2026.

    python run_faults.py            # -> results/faults/results.json + summary printed

Fresh seed range (500 000+), never used for tuning. Thresholds are unchanged from test-2;
the health parameters are fixed by design (0.1 s stuck IMU, 3 s / 60 s window exclusion),
not tuned.

Questions answered
  1. Normal driving: does a sensor fault ever produce a Stage-2 verification?
     Compared with the same detector with health gating disabled (the test-2 behaviour).
  2. Is every fault flagged DEGRADED, and how quickly?
  3. Events during a fault: which classes are still detected, and which are reported as
     not assessable instead of silently missed?
"""
from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")

from generator import generate_trial, EVENT_CLASSES, CALIB_S
from detector import features, decide, run_trial, first_true, DT
from faults import inject, FAULT_KINDS
from run_validation import clopper, poisson_upper

SEED = 500_000
N_NORMAL = 60     # normal trials per fault kind
N_EVENT = 25      # event trials per class per fault kind
KINDS = list(FAULT_KINDS) + ["belt_stuck+tel_loss", "belt_stuck@calib", "cam_loss@calib", "cam_stall"]
# cam_stall is benign (one 133-ms webcam stall per minute, incl. calibration): it must NOT raise DEGRADED
LSB = 9.81 * 4 / 65536   # MPU6050 at +-2 g, 16 bit


def _quantize(tr):
    """Realistic 16-bit MPU6050 output (checks that the stuck-IMU test does not fire on real resolution)."""
    tr = dict(tr)
    for k in ("belt", "frame"):
        tr[k] = np.round(tr[k] / LSB) * LSB
    return tr


def _apply(tr, kind, t0, t1):
    for k in kind.split("@")[0].split("+"):
        tr = inject(tr, k, t0, t1)
    return tr


def _job(args):
    cls, seed, kind = args
    rng = np.random.default_rng(seed + 7)
    tr = generate_trial(cls, seed)
    onset = tr["truth"]["onset"]
    if kind == "cam_stall":
        t0, t1 = 2.0, 1e9
    elif kind.endswith("@calib"):
        t0, t1 = 5.0, 15.0                                   # fault during calibration only
    else:
        if onset is None:
            t0 = CALIB_S + rng.uniform(20, 60)
        else:
            t0 = max(CALIB_S + 5, onset - rng.uniform(0, 10))   # fault covers the event onset
        t1 = t0 + rng.uniform(20, 40)
    trf = _apply(tr, kind, t0, t1)
    F = features(trf)
    D = decide(F)
    r = run_trial(F, D, onset)
    # same trial, health gating disabled (= test-2 detector behaviour)
    F0 = {k: v for k, v in F.items() if not k.startswith("h_")}
    r0 = run_trial(F0, decide(F0), onset)
    # same trial without any fault (reference for this fresh seed range)
    Fc = features(tr)
    Dc = decide(Fc)
    rc = run_trial(Fc, Dc, onset)
    clean_degraded = float(Dc["degraded"].mean())
    quant_degraded = None
    if cls == "normal":
        Fq = features(_quantize(tr))
        quant_degraded = float(decide(Fq)["degraded"].mean())
    t = D["t"]
    if kind.endswith("@calib"):
        # no valid baseline until the modality is recalibrated (8 Oct 2026); without a
        # recalibration window it stays degraded for the whole trial
        mod = {"belt_stuck": "belt", "cam_loss": "cam", "tel_loss": "tel"}.get(kind.split("@")[0])
        w = F["recal"].get(mod)
        in_fault = t < (w[1] if w else np.inf)
        td = t[0] if D["degraded"][0] else None
        t0_flag = t[0]
    elif kind == "cam_stall":
        in_fault = np.ones(len(t), bool)                     # fraction of the drive wrongly flagged
        _, td = first_true(D["degraded"], t, t[0])
        t0_flag = t[0]
    else:
        in_fault = (t >= t0) & (t < t1)
        _, td = first_true(D["degraded"], t, t0)
        t0_flag = t0
    out = dict(cls=cls, seed=seed, kind=kind, t0=t0, t1=t1,
               fault_flagged_frac=float(D["degraded"][in_fault].mean()),
               flag_delay=(td - t0_flag) if td is not None else None,
               verified=r.get("verified"), label=r.get("final_label"),
               verified_nogate=r0.get("verified"), verified_clean=rc.get("verified"),
               fv=r["false_verifications"], fv_nogate=r0["false_verifications"], fv_clean=rc["false_verifications"],
               exposure_h=r["exposure_h"], clean_degraded=clean_degraded, quant_degraded=quant_degraded,
               label_clean=rc.get("final_label"),
               recal=F["recal"])
    if onset is not None:
        i = int(np.searchsorted(t, onset))
        key = "loc" if cls == "syncope" else cls
        out["not_assessable_at_onset"] = bool(D["not_assessable"][key][min(i, len(t) - 1)])
    return out


def main():
    os.makedirs("results/faults", exist_ok=True)
    jobs = []
    for j, kind in enumerate(KINDS):
        jobs += [("normal", SEED + 10_000 * j + i, kind) for i in range(N_NORMAL)]
        for c, cls in enumerate(EVENT_CLASSES):
            jobs += [(cls, SEED + 10_000 * j + 1000 * (c + 1) + i, kind) for i in range(N_EVENT)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        rows = list(ex.map(_job, jobs, chunksize=4))

    S = {}
    for kind in KINDS:
        R = [r for r in rows if r["kind"] == kind]
        Rn = [r for r in R if r["cls"] == "normal"]
        hours = sum(r["exposure_h"] for r in Rn)
        fv = sum(r["fv"] for r in Rn)
        fv0 = sum(r["fv_nogate"] for r in Rn)
        delays = [r["flag_delay"] for r in R if r["flag_delay"] is not None]
        S[kind] = dict(
            normal_trials=len(Rn), normal_hours=hours,
            false_verif=fv, false_verif_upper95_per_h=poisson_upper(fv, hours),
            trials_with_false_verif=sum(r["fv"] > 0 for r in Rn),
            false_verif_without_health=fv0, trials_with_false_verif_without_health=sum(r["fv_nogate"] > 0 for r in Rn),
            false_verif_no_fault=sum(r["fv_clean"] for r in Rn),
            degraded_on_clean=float(np.mean([r["clean_degraded"] for r in R])),
            degraded_on_quantized_clean=float(np.mean([r["quant_degraded"] for r in Rn])),
            fault_time_flagged=float(np.mean([r["fault_flagged_frac"] for r in R])),
            flag_delay_median_s=float(np.median(delays)) if delays else None,
            flag_delay_max_s=float(np.max(delays)) if delays else None,
            events={})
        for cls in EVENT_CLASSES:
            Re = [r for r in R if r["cls"] == cls]
            k = sum(bool(r["verified"]) for r in Re)
            S[kind]["events"][cls] = dict(
                n=len(Re), verified=k, ci=clopper(k, len(Re)),
                verified_without_health=sum(bool(r["verified_nogate"]) for r in Re),
                verified_no_fault=sum(bool(r["verified_clean"]) for r in Re),
                correct_label=sum(r["label"] == cls for r in Re),
                wrong_label=sum(bool(r["verified"]) and r["label"] not in (cls, "ADI-unspecified") for r in Re),
                correct_label_no_fault=sum(r["label_clean"] == cls for r in Re),
                not_assessable_at_onset=sum(r["not_assessable_at_onset"] for r in Re))
    json.dump(dict(summary=S, rows=rows), open("results/faults/results.json", "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for kind, s in S.items():
        print(f"{kind:22s} normal: FV {s['false_verif']} in {s['normal_hours']:.1f} h "
              f"(trials {s['trials_with_false_verif']}/{s['normal_trials']}) | without health: FV {s['false_verif_without_health']} "
              f"(trials {s['trials_with_false_verif_without_health']}) | no fault: FV {s['false_verif_no_fault']} | clean-degraded {s['degraded_on_clean']:.4f} quant {s['degraded_on_quantized_clean']:.4f} | flagged {s['fault_time_flagged']*100:.0f}% delay {s['flag_delay_median_s']} max {s['flag_delay_max_s']}")
        for cls, e in s["events"].items():
            print(f"    {cls:13s} {e['verified']}/{e['n']} (no health {e['verified_without_health']}, no fault {e['verified_no_fault']}) label ok {e['correct_label']} (no fault {e['correct_label_no_fault']}) wrong {e['wrong_label']} "
                  f"not-assessable {e['not_assessable_at_onset']}")


if __name__ == "__main__":
    main()
