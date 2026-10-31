"""
CabinGuard-ADI - Phase 1 validation on synthetic data.

    python run_validation.py --split dev     # tune thresholds here
    python run_validation.py --split test2   # held-out split reported in the paper

Outputs (results/<split>/): results.json, summary.md, figures/*.png
"""
from __future__ import annotations

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

from generator import generate_trial, EVENT_CLASSES, CALIB_S
from detector import features, decide, run_trial, baselines, PARAMS, DT, _roll

N = dict(dev=dict(event=60, normal=200, upright=30), test=dict(event=100, normal=150, upright=30),
         test2=dict(event=100, normal=150, upright=30))
SEED_BASE = dict(dev=0, test=100_000, test2=200_000)  # test = first held-out run (consumed), test2 = final
CLASS_ID = {c: k for k, c in enumerate(["normal"] + EVENT_CLASSES + ["syncope_upright"])}


def seeds(split, cls, n):
    return [SEED_BASE[split] + 1000 * CLASS_ID[cls] + i for i in range(n)]


def _job(args):
    cls, seed, variant = args
    tr = generate_trial(cls, seed, variant=variant)
    F = features(tr)
    out = dict(cls=cls, seed=seed, truth={k: (v if not isinstance(v, (np.floating, np.bool_)) else v.item())
                                        for k, v in tr["truth"].items()}, F=F)
    # ablation: no seat-frame reference IMU (only for classes where IMU matters)
    if cls in ("normal", "seizure", "syncope"):
        F0 = features(tr, use_reference_imu=False)
        out["F_noref"] = {k: F0[k] for k in ("band_ratio", "imu_rhythm", "quiet_ratio", "hf_ratio", "tilt")}
    # rough-road mask on decision grid (for IMU false-evidence rate)
    t = F["t"]
    rough = np.zeros(len(t), bool)
    for a, b in tr["context"]["rough"]:
        rough |= (t >= a) & (t < b)
    out["rough"] = rough
    drowsy = np.zeros(len(t), bool)
    for a, b in tr["context"]["drowsy"]:
        drowsy |= (t >= a) & (t < b)
    out["drowsy"] = drowsy
    return out


def clopper(k, n, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - a / 2, k + 1, n - k)
    return lo, hi


def poisson_upper(k, T, a=0.05):
    return stats.chi2.ppf(1 - a, 2 * (k + 1)) / 2 / T


def evaluate(trials, P=None, use=("imu", "cam", "tel"), noref=False):
    rows = []
    for tr in trials:
        F = tr["F"]
        if noref and "F_noref" in tr:
            F = {**F, **tr["F_noref"]}
        D = decide(F, P, use)
        r = run_trial(F, D, tr["truth"]["onset"])
        r.update(cls=tr["cls"], variant=tr["truth"].get("variant"), seed=tr["seed"])
        # prompts inside drowsiness episodes
        if tr["cls"] == "normal":
            s1_edges = D["any_s1"] & ~np.r_[False, D["any_s1"][:-1]]
            r["prompts_in_drowsy"] = int((s1_edges & tr["drowsy"]).sum())
            r["prompts_edges"] = int(s1_edges.sum())
            r["drowsy_eps"] = len(tr["truth"]["drowsy"])
            r["drowsy_flagged"] = int(bool((D["fatigue"] & tr["drowsy"]).any())) if tr["truth"]["drowsy"] else 0
            r["imu_rh_rough"] = float(((F["imu_rhythm"] > PARAMS["rhythm"]) &
                                       (F["band_ratio"] > PARAMS["band_ratio"]))[tr["rough"]].mean()) if tr["rough"].any() else np.nan
        rows.append(r)
    return rows


def summarise(rows, trials_by_seed=None):
    S = {}
    for cls in EVENT_CLASSES:
        R = [r for r in rows if r["cls"] == cls]
        k = sum(r["verified"] for r in R); n = len(R)
        lat = np.array([r["latency"] for r in R if r["verified"]])
        s1 = np.array([r["s1_latency"] for r in R if not np.isnan(r["s1_latency"])])
        correct_v = sum(r["label"] == cls for r in R)
        correct_f = sum(r["final_label"] == cls for r in R)
        unspec = sum(r["final_label"] == "ADI-unspecified" for r in R)
        wrong = sum(r["verified"] and r["final_label"] not in (cls, "ADI-unspecified") for r in R)
        llat = np.array([r["label_latency"] for r in R if r["final_label"] == cls])
        S[cls] = dict(n=n, verified=k, rate=k / n, ci=clopper(k, n),
                      label_lat_median=float(np.median(llat)) if len(llat) else None,
                      lat_median=float(np.median(lat)) if len(lat) else None,
                      lat_q1=float(np.percentile(lat, 25)) if len(lat) else None,
                      lat_q3=float(np.percentile(lat, 75)) if len(lat) else None,
                      s1_median=float(np.median(s1)) if len(s1) else None,
                      label_at_verification_correct=correct_v, final_label_correct=correct_f,
                      unspecified=unspec, wrong_class=wrong)
        variants = sorted(set(r["variant"] for r in R if r["variant"]))
        S[cls]["by_variant"] = {}
        for v in variants:
            Rv = [r for r in R if r["variant"] == v]
            kv = sum(r["verified"] for r in Rv)
            latv = [r["latency"] for r in Rv if r["verified"]]
            S[cls]["by_variant"][v] = dict(n=len(Rv), verified=kv, rate=kv / len(Rv), ci=clopper(kv, len(Rv)),
                                           lat_median=float(np.median(latv)) if latv else None,
                                           final_correct=sum(r["final_label"] == cls for r in Rv))
    Rn = [r for r in rows if r["cls"] == "normal"]
    Tn = sum(r["exposure_h"] for r in Rn)
    fv = sum(r["false_verifications"] for r in Rn)
    Re = [r for r in rows if r["cls"] != "normal"]
    Te = sum(r["exposure_h"] for r in Re)
    fve = sum(r["false_verifications"] for r in Re)
    pr = sum(r["prompts"] for r in Rn)
    S["normal"] = dict(n=len(Rn), hours=Tn, false_verifications=fv, fv_rate=fv / Tn, fv_upper95=poisson_upper(fv, Tn),
                       preonset_hours=Te, preonset_false_verifications=fve,
                       total_hours=Tn + Te, total_fv=fv + fve, total_fv_upper95=poisson_upper(fv + fve, Tn + Te),
                       prompts=pr, prompts_per_h=pr / Tn,
                       prompts_in_drowsy=sum(r["prompts_in_drowsy"] for r in Rn),
                       prompt_edges=sum(r["prompts_edges"] for r in Rn),
                       drowsy_episodes=sum(r["drowsy_eps"] for r in Rn),
                       drowsy_flagged=sum(r["drowsy_flagged"] for r in Rn),
                       imu_rhythmic_on_rough_road=float(np.nanmean([r["imu_rh_rough"] for r in Rn])))
    # confusion (final label)
    labels = EVENT_CLASSES + ["ADI-unspecified", "missed"]
    conf = {c: {l: 0 for l in labels} for c in EVENT_CLASSES}
    for r in rows:
        if r["cls"] in EVENT_CLASSES:
            conf[r["cls"]][r["final_label"] if r["final_label"] else "missed"] += 1
    S["confusion"] = conf
    return S


def run(split, outdir):
    os.makedirs(os.path.join(outdir, "figures"), exist_ok=True)
    jobs = [("normal", s, None) for s in seeds(split, "normal", N[split]["normal"])]
    for c in EVENT_CLASSES:
        jobs += [(c, s, None) for s in seeds(split, c, N[split]["event"])]
    jobs += [("syncope", s, "upright") for s in seeds(split, "syncope_upright", N[split]["upright"])]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        trials = list(ex.map(_job, jobs, chunksize=4))
    n_up = N[split]["upright"]
    main, stress = trials[:-n_up], trials[-n_up:]

    res = {}
    rows = evaluate(main)
    res["main"] = summarise(rows)
    res["rows"] = [{k: (v if not isinstance(v, (np.floating,)) else float(v)) for k, v in r.items()} for r in rows]
    # upright-unconscious stress set
    rows_u = evaluate(stress)
    ku = sum(r["verified"] for r in rows_u)
    res["upright_stress"] = dict(n=len(rows_u), verified=ku, ci=clopper(ku, len(rows_u)),
                                 lat_median=float(np.median([r["latency"] for r in rows_u if r["verified"]])) if ku else None,
                                 final_correct=sum(r["final_label"] == "syncope" for r in rows_u))

    # ---- baselines on LOC-type events (syncope, incl. upright stress set) ----
    base = {}
    for name, subset in (("syncope_all", [t for t in main if t["cls"] == "syncope"]),
                         ("syncope_pedal_held", [t for t in main if t["cls"] == "syncope" and t["truth"]["pedal_sustained"]]),
                         ("syncope_eyes_open", [t for t in main if t["cls"] == "syncope" and t["truth"]["eyes_open"]]),
                         ("syncope_upright", stress),
                         ("seizure_gtcs", [t for t in main if t["cls"] == "seizure" and t["truth"]["variant"] == "gtcs"])):
        b = {"B1": [], "B2": [], "CG": []}
        lat = {"B1": [], "B2": [], "CG": []}
        for t in subset:
            bl = baselines(t["F"], t["truth"]["onset"])
            D = decide(t["F"]); r = run_trial(t["F"], D, t["truth"]["onset"])
            for key, v in (("B1", bl["B1"]), ("B2", bl["B2"])):
                b[key].append(v["verified"]); lat[key].append(v["latency"])
            b["CG"].append(r["verified"]); lat["CG"].append(r["latency"])
        base[name] = {k: dict(n=len(v), rate=float(np.mean(v)) if v else None, verified=int(np.sum(v)),
                              lat_median=float(np.nanmedian(lat[k])) if np.any(v) else None) for k, v in b.items()}
    nb = {"B1": 0, "B2": 0}
    for t in main:
        if t["cls"] == "normal":
            bl = baselines(t["F"])
            for k in nb:
                nb[k] += bl[k]["false"]
    base["normal_false"] = nb
    res["baselines"] = base

    # ---- ablations ----
    abl = {}
    for name, kw in (("full", {}), ("no_reference_imu", dict(noref=True)), ("no_telemetry", dict(use=("imu", "cam"))),
                     ("camera_only", dict(use=("cam",))), ("imu_telemetry_no_camera", dict(use=("imu", "tel")))):
        S = summarise(evaluate(main, **kw))
        abl[name] = {c: dict(rate=S[c]["rate"], lat=S[c]["lat_median"], final_correct=S[c]["final_label_correct"])
                     for c in EVENT_CLASSES}
        abl[name]["false_verifications"] = S["normal"]["false_verifications"]
        abl[name]["fv_per_h"] = S["normal"]["fv_rate"]
        abl[name]["imu_rhythmic_on_rough_road"] = S["normal"]["imu_rhythmic_on_rough_road"]
    # recompute rough-road rhythmic fraction without reference explicitly
    fr = []
    for t in main:
        if t["cls"] == "normal" and t["rough"].any():
            F0 = t["F_noref"]
            fr.append(((F0["imu_rhythm"] > PARAMS["rhythm"]) & (F0["band_ratio"] > PARAMS["band_ratio"]))[t["rough"]].mean())
    abl["no_reference_imu"]["imu_rhythmic_on_rough_road"] = float(np.mean(fr))
    res["ablations"] = abl

    # ---- sensitivity sweep of the LOC verification window ----
    sweep = []
    for v in (1.0, 2.0, 3.0, 4.0, 5.0):
        S = summarise(evaluate(main, P=dict(loc_verify_s=v)))
        sweep.append(dict(loc_verify_s=v, syncope_rate=S["syncope"]["rate"], syncope_lat=S["syncope"]["lat_median"],
                          fv=S["normal"]["false_verifications"], fv_h=S["normal"]["fv_rate"], hours=S["normal"]["hours"],
                          prompts_h=S["normal"]["prompts_per_h"]))
    res["sweep_loc_verify"] = sweep

    # ---- generator self-check: realised proportions vs. literature targets ----
    sy = [t["truth"] for t in main if t["cls"] == "syncope"]
    res["generator_check"] = dict(
        syncope_n=len(sy),
        eyes_open=float(np.mean([s["eyes_open"] for s in sy])),
        myoclonus=float(np.mean([s["myoclonus"] for s in sy])),
        pedal_sustained=float(np.mean([s["pedal_sustained"] for s in sy])),
        collapse=float(np.mean([s["collapse"] for s in sy])),
        transient_loc_median_s=float(np.median([s["loc_dur"] for s in sy if np.isfinite(s["loc_dur"])])),
    )
    res["params"] = PARAMS
    res["split"] = split
    with open(os.path.join(outdir, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    return res, main, stress


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test2", choices=["dev", "test", "test2"])  # "test" = retired test-1 seeds
    a = ap.parse_args()
    out = os.path.join("results", a.split)
    res, main, stress = run(a.split, out)
    M = res["main"]
    for c in EVENT_CLASSES:
        m = M[c]
        print(f"{c:13s} {m['verified']:3d}/{m['n']} ({m['rate']*100:.0f}% [{m['ci'][0]*100:.0f}-{m['ci'][1]*100:.0f}]) "
              f"lat med {m['lat_median']} IQR {m['lat_q1']}-{m['lat_q3']}  final-correct {m['final_label_correct']} "
              f"unspec {m['unspecified']} wrong {m['wrong_class']}  variants { {k:(v['verified'],v['n'],v['lat_median']) for k,v in m['by_variant'].items()} }")
    print("normal", {k: v for k, v in M["normal"].items()})
    print("upright stress", res["upright_stress"])
    print("baselines", json.dumps(res["baselines"], indent=0, default=str))
    print("ablations", json.dumps(res["ablations"], indent=0, default=str))
    print("sweep", res["sweep_loc_verify"])
    print("gen check", res["generator_check"])
    print("confusion", M["confusion"])
