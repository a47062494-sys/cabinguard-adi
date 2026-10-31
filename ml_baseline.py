"""
CabinGuard-ADI - machine-learning comparison baseline (Random Forest).

    python ml_baseline.py            # train on dev seeds, evaluate on test2 seeds
    python ml_baseline.py --quick    # small run (smoke test, numbers NOT for the paper)

Outputs: results/ml_baseline/results.json, summary.md, feature_importance.png

EN BREF (francais)
------------------
Ce fichier ajoute un modele de machine learning (foret aleatoire, scikit-learn) A COTE du
detecteur a regles. Il ne le remplace pas : le systeme principal reste detector.py.
  1. On genere les memes essais synthetiques que run_validation.py (seeds dev et test2).
  2. On calcule les memes indices (detector.features), puis on construit un tableau :
     une ligne = un instant (toutes les 0,5 s), colonnes = indices, etiquette = "normal"
     avant le debut du malaise, la classe du malaise apres.
  3. La foret apprend sur les seeds dev, puis predit sur les seeds test2 (jamais vues).
  4. Pour comparer equitablement avec les regles, on transforme les predictions en alarmes :
     meme classe de malaise predite avec une probabilite >= seuil pendant VERIFY_S secondes.
     Le seuil est choisi sur dev pour ne pas depasser le taux de fausses alarmes des regles sur dev.
  5. On mesure exactement comme pour les regles : detection par classe, latence, bonne etiquette,
     fausses verifications par heure de conduite normale.

LIMITE IMPORTANTE : les etiquettes viennent de generator.py. La foret apprend donc les hypotheses
de NOTRE generateur. Un bon score ici ne prouve rien sur de vrais conducteurs.

Differences with the rule-based detector (stated in the summary):
- same features, same trials, same counting rules (run_trial semantics);
- pedal and raw steering rate are excluded (the rules do not use them as evidence either);
- no sensor-health gating (fault-free trials only), no ADI-unspecified output, no label refinement.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report

from generator import generate_trial, EVENT_CLASSES
from detector import features, decide, run_trial, events_per_run, first_true, _roll, DT
from run_validation import N, seeds, clopper, poisson_upper

CLASSES = ["normal"] + EVENT_CLASSES
# features the rules use as evidence; pedal / steer_rate (baseline-only signals) are left out
FEATURES = ["band_ratio", "quiet_ratio", "hf_ratio", "imu_rhythm", "tilt", "res_norm",
            "closed_run", "stare", "perclos", "pitch_drop", "yaw_dev_run", "face_ok",
            "asym_z", "asym_d", "energy_z", "face_rhythm", "gaze", "gaze_ratio",
            "frozen_run", "rate_ratio", "steer_rhythm", "micro_ratio", "abrupt", "lane_ratio"]
SUBSAMPLE = 5          # training rows every 5 x 0.1 s = 0.5 s
VERIFY_S = 3.0         # an alarm needs the same event class for 3 s (as the LOC verification window)
P_GRID = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
RF_KW = dict(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
             random_state=0, n_jobs=-1)


# ------------------------------------------------------------------------------------------
# data
# ------------------------------------------------------------------------------------------
def _job(args):
    cls, seed = args
    tr = generate_trial(cls, seed)
    F = features(tr)
    X = np.column_stack([np.nan_to_num(np.asarray(F[k], float), nan=0.0, posinf=1e6, neginf=-1e6)
                         for k in FEATURES])
    onset = tr["truth"]["onset"]
    y = np.zeros(len(F["t"]), int)
    if onset is not None:
        y[F["t"] >= onset] = CLASSES.index(cls)
    return dict(cls=cls, seed=seed, onset=onset, F=F, X=X, y=y)


def load(split, frac=1.0):
    n = {k: max(2, int(v * frac)) for k, v in N[split].items()}
    jobs = [("normal", s) for s in seeds(split, "normal", n["normal"])]
    for c in EVENT_CLASSES:
        jobs += [(c, s) for s in seeds(split, c, n["event"])]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        return list(ex.map(_job, jobs, chunksize=4))


# ------------------------------------------------------------------------------------------
# from per-step probabilities to alarms (same structure as the rule-based Stage 2)
# ------------------------------------------------------------------------------------------
def alarms(proba, p_thr, verify_s=VERIFY_S):
    """proba: (T, n_classes). Returns (any_alarm mask, label index per step or -1)."""
    w = round(verify_s / DT)
    lab = np.full(len(proba), -1)
    any_alarm = np.zeros(len(proba), bool)
    for k in range(1, len(CLASSES)):
        ok = proba[:, k] >= p_thr
        held = _roll(ok.astype(float), w, "sum") >= w - 1e-9      # true for the last verify_s seconds
        newer = held & (lab == -1)
        lab[newer] = k
        any_alarm |= held
    return any_alarm, lab


def score_trial(tr, proba, p_thr):
    t = tr["F"]["t"]
    a, lab = alarms(proba, p_thr)
    D = dict(t=t, any_s2=a, any_s1=a, fatigue=np.zeros(len(t), bool),
             label_at=lambda i: CLASSES[lab[i]] if lab[i] > 0 else None)
    return run_trial(tr["F"], D, tr["onset"])


def _f(x):
    return "-" if x is None else f"{x:.1f} s"


def summarise(rows):
    S = {}
    for c in EVENT_CLASSES:
        R = [r for r in rows if r["cls"] == c]
        k = sum(r["verified"] for r in R)
        lat = [r["latency"] for r in R if r["verified"]]
        S[c] = dict(n=len(R), verified=k, rate=k / len(R), ci=clopper(k, len(R)),
                    lat_median=float(np.median(lat)) if lat else None,
                    final_label_correct=sum(r["final_label"] == c for r in R))
    Rn = [r for r in rows if r["cls"] == "normal"]
    T = sum(r["exposure_h"] for r in Rn)
    fv = sum(r["false_verifications"] for r in Rn)
    Re = [r for r in rows if r["cls"] != "normal"]
    fve = sum(r["false_verifications"] for r in Re)
    Te = sum(r["exposure_h"] for r in Re)
    S["normal"] = dict(n=len(Rn), hours=T, false_verifications=fv, fv_rate=fv / T,
                       fv_upper95=poisson_upper(fv, T),
                       total_fv=fv + fve, total_hours=T + Te, total_fv_rate=(fv + fve) / (T + Te))
    return S


def rules_rows(trials):
    rows = []
    for tr in trials:
        D = decide(tr["F"])
        r = run_trial(tr["F"], D, tr["onset"]); r["cls"] = tr["cls"]
        rows.append(r)
    return rows


def ml_rows(trials, probas, p_thr):
    rows = []
    for tr, pr in zip(trials, probas):
        r = score_trial(tr, pr, p_thr); r["cls"] = tr["cls"]
        rows.append(r)
    return rows


# ------------------------------------------------------------------------------------------
def main(quick=False):
    t0 = time.time()
    frac = 0.15 if quick else 1.0
    out = os.path.join("results", "ml_baseline" + ("_quick" if quick else ""))
    os.makedirs(out, exist_ok=True)

    dev, test = load("dev", frac), load("test2", frac)
    print(f"trials: dev {len(dev)}, test2 {len(test)}  ({time.time() - t0:.0f} s)")

    # ---- train on dev (subsampled rows) ----
    Xtr = np.vstack([d["X"][::SUBSAMPLE] for d in dev])
    ytr = np.concatenate([d["y"][::SUBSAMPLE] for d in dev])
    rf = RandomForestClassifier(**RF_KW).fit(Xtr, ytr)

    # ---- choose the probability threshold on dev only ----
    # Out-of-sample probabilities on dev would be better (cross-validation by trial); to keep it
    # simple and conservative we refit on half of dev trials and pick the threshold on the other half.
    half = np.arange(len(dev)) % 2 == 0
    rf_half = RandomForestClassifier(**RF_KW).fit(
        np.vstack([d["X"][::SUBSAMPLE] for d, h in zip(dev, half) if h]),
        np.concatenate([d["y"][::SUBSAMPLE] for d, h in zip(dev, half) if h]))
    dev_b = [d for d, h in zip(dev, half) if not h]
    pr_dev = [rf_half.predict_proba(d["X"]) for d in dev_b]
    rules_dev = summarise(rules_rows(dev_b))
    budget = rules_dev["normal"]["fv_rate"]
    grid = []
    for p in P_GRID:
        S = summarise(ml_rows(dev_b, pr_dev, p))
        mean_det = float(np.mean([S[c]["rate"] for c in EVENT_CLASSES]))
        grid.append(dict(p_thr=p, mean_detection=mean_det, fv_rate=S["normal"]["fv_rate"]))
    ok = [g for g in grid if g["fv_rate"] <= budget + 1e-12]
    p_thr = max(ok, key=lambda g: g["mean_detection"])["p_thr"] if ok else max(P_GRID)
    print(f"threshold chosen on dev: p >= {p_thr} (rules dev false-verification budget {budget:.2f}/h)")

    # ---- evaluate both systems on the same test2 trials ----
    pr_test = [rf.predict_proba(d["X"]) for d in test]
    ML = summarise(ml_rows(test, pr_test, p_thr))
    RU = summarise(rules_rows(test))

    # per-step (frame-level) metrics, for information
    yte = np.concatenate([d["y"] for d in test])
    ypr = np.concatenate([p.argmax(1) for p in pr_test])
    cm = confusion_matrix(yte, ypr, labels=range(len(CLASSES)))
    rep = classification_report(yte, ypr, labels=range(len(CLASSES)), target_names=CLASSES,
                                output_dict=True, zero_division=0)
    imp = sorted(zip(FEATURES, rf.feature_importances_), key=lambda x: -x[1])

    res = dict(quick=quick, p_thr=p_thr, threshold_grid_dev=grid, rules_dev_fv_budget=budget,
               ml=ML, rules=RU, step_confusion=dict(labels=CLASSES, matrix=cm.tolist()),
               step_report=rep, feature_importance=imp, features=FEATURES,
               settings=dict(subsample=SUBSAMPLE, verify_s=VERIFY_S, rf=RF_KW))
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    # ---- summary table ----
    L = ["# ML baseline (Random Forest) vs rule-based detector - test2 seeds",
         "", "Synthetic data only: both systems inherit the generator's assumptions.", "",
         f"Threshold p >= {p_thr} chosen on dev (rules dev false-verification rate {budget:.2f}/h).", "",
         "| Class | Rules: verified | Rules: median latency | ML: verified | ML: median latency | ML: correct label |",
         "|---|---|---|---|---|---|"]
    for c in EVENT_CLASSES:
        r, m = RU[c], ML[c]
        L.append(f"| {c} | {r['verified']}/{r['n']} ({r['rate']*100:.0f}%) | {_f(r['lat_median'])} | "
                 f"{m['verified']}/{m['n']} ({m['rate']*100:.0f}%) | {_f(m['lat_median'])} | {m['final_label_correct']} |")
    L += ["", "| Normal driving | Rules | ML |", "|---|---|---|",
          f"| False verifications | {RU['normal']['false_verifications']} in {RU['normal']['hours']:.1f} h "
          f"({RU['normal']['fv_rate']:.2f}/h) | {ML['normal']['false_verifications']} in {ML['normal']['hours']:.1f} h "
          f"({ML['normal']['fv_rate']:.2f}/h) |",
          "", "Top features (Random Forest importance):", ""]
    L += [f"- {k}: {v:.3f}" for k, v in imp[:10]]
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k, v = zip(*imp[:12][::-1])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(k, v, color="#2E75B6")
        ax.set_xlabel("Importance (mean decrease in impurity)")
        ax.set_title("Random Forest - most used features")
        fig.tight_layout(); fig.savefig(os.path.join(out, "feature_importance.png"), dpi=150)
    except Exception as e:  # figure is optional
        print("figure skipped:", e)
    print(f"done in {time.time() - t0:.0f} s -> {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    main(ap.parse_args().quick)
