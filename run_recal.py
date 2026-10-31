"""
CabinGuard-ADI - recalibration test (added 8 Oct 2026).

    python run_recal.py             # -> results/recal/results.json + summary.md

Fresh seed range (700 000+), never used before. The recalibration has no tuned parameter
(window = 20 s, the length of the initial calibration; safety rule = no Stage-1 alert).

Scenarios (each on 60 normal trials + 25 trials per event class):
  <fault>@calib       sensor faulty from 5 to 15 s (inside the initial calibration), then back.
  <fault>@calib_long  sensor faulty from 5 s until shortly before the event (0-25 s before onset),
                      so that the first recalibration window can OVERLAP the event: this is the
                      "medical trap" (learning a sick driver as normal).
For every trial: the detector with recalibration (v4), without it (v3 behaviour) and on the
same trial without fault.

Resume en francais : on verifie (1) combien de malaises la recalibration fait regagner,
(2) qu'elle ne cree pas de fausses alarmes en conduite normale, (3) combien de fois la fenetre
de recalibration a contenu le debut du malaise, et ce qui se passe alors.
"""
from __future__ import annotations

import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")

from generator import generate_trial, EVENT_CLASSES
from detector import features, decide, run_trial
from faults import inject
from run_validation import clopper, poisson_upper

SEED = 700_000
N_NORMAL = 60
N_EVENT = 25
FAULTS = ("belt_stuck", "cam_loss", "tel_loss")
KINDS = [f + "@calib" for f in FAULTS] + [f + "@calib_long" for f in FAULTS]
MOD = {"belt_stuck": "belt", "cam_loss": "cam", "tel_loss": "tel"}


def _job(args):
    cls, seed, kind = args
    rng = np.random.default_rng(seed + 11)
    tr = generate_trial(cls, seed)
    onset = tr["truth"]["onset"]
    fault = kind.split("@")[0]
    if kind.endswith("@calib"):
        t0, t1 = 5.0, 15.0
    elif onset is None:
        t0, t1 = 5.0, float(rng.uniform(30, 80))
    else:
        t0, t1 = 5.0, float(max(15.0, onset - rng.uniform(0, 25)))
    trf = inject(tr, fault, t0, t1)
    F = features(trf)                       # with recalibration
    r = run_trial(F, decide(F), onset)
    F0 = features(trf, recal=False)         # v3 behaviour: modality lost for the whole drive
    r0 = run_trial(F0, decide(F0), onset)
    Fc = features(tr)                       # same trial, no fault
    rc = run_trial(Fc, decide(Fc), onset)
    w = F["recal"].get(MOD[fault])
    out = dict(cls=cls, seed=seed, kind=kind, t0=t0, t1=t1, onset=onset,
               recal_start=w[0] if w else None, recal_end=w[1] if w else None,
               verified=r.get("verified"), label=r.get("final_label"), latency=r.get("latency"),
               verified_norecal=r0.get("verified"), label_norecal=r0.get("final_label"),
               verified_clean=rc.get("verified"), label_clean=rc.get("final_label"),
               fv=r["false_verifications"], fv_norecal=r0["false_verifications"], fv_clean=rc["false_verifications"],
               exposure_h=r["exposure_h"])
    # trap: the recalibration window contains the event onset (baseline learnt partly on a sick driver)
    out["window_contains_event"] = bool(w and onset is not None and w[1] > onset)
    return out


def main():
    os.makedirs("results/recal", exist_ok=True)
    jobs = []
    for j, kind in enumerate(KINDS):
        jobs += [("normal", SEED + 10_000 * j + i, kind) for i in range(N_NORMAL)]
        for c, cls in enumerate(EVENT_CLASSES):
            jobs += [(cls, SEED + 10_000 * j + 1000 * (c + 1) + i, kind) for i in range(N_EVENT)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        rows = list(ex.map(_job, jobs, chunksize=4))

    S = {}
    L = ["# Recalibration test (fresh seeds 700 000+)", "",
         "Verified events: with recalibration / without (v3) / same trials without fault.", ""]
    for kind in KINDS:
        R = [r for r in rows if r["kind"] == kind]
        Rn = [r for r in R if r["cls"] == "normal"]
        hours = sum(r["exposure_h"] for r in Rn)
        ev = {}
        for cls in EVENT_CLASSES:
            Re = [r for r in R if r["cls"] == cls]
            k = sum(bool(r["verified"]) for r in Re)
            ev[cls] = dict(n=len(Re), verified=k, ci=clopper(k, len(Re)),
                           verified_norecal=sum(bool(r["verified_norecal"]) for r in Re),
                           verified_clean=sum(bool(r["verified_clean"]) for r in Re),
                           correct_label=sum(r["label"] == cls for r in Re),
                           wrong_label=sum(bool(r["verified"]) and r["label"] not in (cls, "ADI-unspecified") for r in Re),
                           window_contains_event=sum(r["window_contains_event"] for r in Re),
                           missed_when_window_contains_event=sum(r["window_contains_event"] and not r["verified"] for r in Re),
                           missed_by_recal_but_found_without=sum(bool(r["verified_norecal"]) and not r["verified"] for r in Re))
        recal_found = [r for r in R if r["recal_end"] is not None]
        S[kind] = dict(
            normal_hours=hours, fv=sum(r["fv"] for r in Rn), fv_norecal=sum(r["fv_norecal"] for r in Rn),
            fv_clean=sum(r["fv_clean"] for r in Rn), fv_upper95_per_h=poisson_upper(sum(r["fv"] for r in Rn), hours),
            fv_preonset=sum(r["fv"] for r in R if r["cls"] != "normal"),
            fv_preonset_norecal=sum(r["fv_norecal"] for r in R if r["cls"] != "normal"),
            recal_found=len(recal_found), trials=len(R),
            recal_delay_after_fault_median_s=float(np.median([r["recal_end"] - r["t1"] for r in recal_found])) if recal_found else None,
            events=ev)
        s = S[kind]
        tot = lambda key: sum(e[key] for e in ev.values())
        L += [f"## {kind}", "",
              f"- events verified: **{tot('verified')}/{tot('n')}** with recalibration, {tot('verified_norecal')}/{tot('n')} without, "
              f"{tot('verified_clean')}/{tot('n')} without fault",
              f"- normal driving: {s['fv']} false verifications in {hours:.1f} h with recalibration "
              f"({s['fv_norecal']} without, {s['fv_clean']} without fault); before onset in event trials: "
              f"{s['fv_preonset']} ({s['fv_preonset_norecal']} without)",
              f"- recalibration window found in {s['recal_found']}/{s['trials']} trials, median "
              f"{s['recal_delay_after_fault_median_s']} s after the sensor came back",
              f"- window containing the event onset: {tot('window_contains_event')} trials, of which "
              f"{tot('missed_when_window_contains_event')} missed; events lost because of recalibration "
              f"(found without it): {tot('missed_by_recal_but_found_without')}; wrong labels {tot('wrong_label')}", "",
              "| Class | With recal | Without (v3) | No fault | Window contains event |", "|---|---|---|---|---|"]
        L += [f"| {c} | {e['verified']}/{e['n']} | {e['verified_norecal']}/{e['n']} | {e['verified_clean']}/{e['n']} | "
              f"{e['window_contains_event']} |" for c, e in ev.items()]
        L.append("")
    json.dump(dict(summary=S, rows=rows), open("results/recal/results.json", "w"), indent=1,
              default=lambda o: o.item() if hasattr(o, "item") else str(o))
    open("results/recal/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
