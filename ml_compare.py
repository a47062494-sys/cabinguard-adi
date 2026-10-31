"""
CabinGuard-ADI - comparison of the rule-based detector with two ML models
(Random Forest and Gradient Boosting) on the same features and the same held-out trials.

    python ml_compare.py            # train on dev seeds, evaluate on test2 seeds
    python ml_compare.py --quick    # smoke test (numbers NOT for the paper)

Outputs (results/ml_compare/):
    results.json                    every number below
    summary.md                      tables for the "Primary results" section
    tables.tex                      the same tables in LaTeX (IEEE style)
    fig_cm_trial_<system>.png       trial-level confusion matrix (true event -> final label / missed)
    fig_cm_trial_all.png            the three trial-level matrices side by side
    fig_cm_step_<model>.png         frame-level confusion matrix (row-normalised) for each ML model
    fig_detection.png               detection rate per class with 95 % Clopper-Pearson CI
    fig_latency.png                 latency distribution of verified detections

EN BREF (francais)
------------------
Meme protocole que ml_baseline.py (memes essais, memes indices, memes regles de comptage),
avec deux modeles ML au lieu d'un : foret aleatoire (RF) et gradient boosting (HGB).
Pour chaque modele, le seuil de probabilite est choisi sur la moitie "hors apprentissage" de dev,
avec la meme contrainte : ne pas depasser le taux de fausses alarmes des regles sur dev.
La matrice de confusion "par essai" est celle a mettre dans le papier : elle compare les trois
systemes avec la meme unite que le tableau de detection (un essai = un malaise).
La matrice "par instant" (pas de 0,1 s) est donnee pour information sur les modeles ML.
LIMITE : donnees synthetiques ; les modeles ML apprennent les hypotheses du generateur.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

from generator import EVENT_CLASSES
from ml_baseline import (CLASSES, FEATURES, SUBSAMPLE, VERIFY_S, P_GRID, RF_KW,
                         load, summarise, rules_rows, ml_rows)

HGB_KW = dict(max_iter=300, learning_rate=0.1, max_leaf_nodes=31, min_samples_leaf=20,
              l2_regularization=1.0, class_weight="balanced", early_stopping=False, random_state=0)
MODELS = {"Random Forest": (RandomForestClassifier, RF_KW),
          "Gradient Boosting": (HistGradientBoostingClassifier, HGB_KW)}
# finer, higher grid than ml_baseline.P_GRID: boosted trees are over-confident and exceed the
# false-alarm budget at every p <= 0.9 on dev. Same grid for both models; RF still selects 0.9.
GRID = P_GRID + [0.95, 0.97, 0.98, 0.99, 0.995, 0.999]
SHORT = {"Rule-based (ours)": "rules", "Random Forest": "rf", "Gradient Boosting": "hgb"}
NICE = dict(seizure="Seizure", syncope="Syncope/LOC", stroke="Stroke", hypoglycemia="Hypoglycaemia")
TRIAL_COLS = EVENT_CLASSES + ["ADI-unspecified", "missed"]


def stack(trials, sub=SUBSAMPLE):
    return (np.vstack([d["X"][::sub] for d in trials]), np.concatenate([d["y"][::sub] for d in trials]))


def choose_threshold(cls, kw, dev):
    """Refit on half of dev, pick p on the other half under the rules' dev false-alarm budget."""
    half = np.arange(len(dev)) % 2 == 0
    m = cls(**kw).fit(*stack([d for d, h in zip(dev, half) if h]))
    dev_b = [d for d, h in zip(dev, half) if not h]
    pr = [m.predict_proba(d["X"]) for d in dev_b]
    budget = summarise(rules_rows(dev_b))["normal"]["fv_rate"]
    grid = []
    for p in GRID:
        S = summarise(ml_rows(dev_b, pr, p))
        grid.append(dict(p_thr=p, mean_detection=float(np.mean([S[c]["rate"] for c in EVENT_CLASSES])),
                         fv_rate=S["normal"]["fv_rate"]))
    ok = [g for g in grid if g["fv_rate"] <= budget + 1e-12]
    p_thr = max(ok, key=lambda g: g["mean_detection"])["p_thr"] if ok else max(GRID)
    return p_thr, grid, budget


def trial_confusion(rows):
    """Rows: true event class. Columns: final label of the first verification after onset, or missed."""
    M = np.zeros((len(EVENT_CLASSES), len(TRIAL_COLS)), int)
    for r in rows:
        if r["cls"] == "normal":
            continue
        j = TRIAL_COLS.index(r["final_label"]) if (r["verified"] and r["final_label"] in TRIAL_COLS) else \
            (TRIAL_COLS.index("ADI-unspecified") if r["verified"] else TRIAL_COLS.index("missed"))
        M[EVENT_CLASSES.index(r["cls"]), j] += 1
    return M


def _f(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


# ------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------
def figures(out, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    systems = list(R["systems"])
    colors = {"Rule-based (ours)": "#1F4E79", "Random Forest": "#C55A11", "Gradient Boosting": "#548235"}
    rows_lab = [NICE[c] for c in EVENT_CLASSES]
    cols_lab = [NICE[c] for c in EVENT_CLASSES] + ["Unspec.", "Missed"]

    def draw_cm(ax, M, title, rlab, clab, norm_rows=True, fmt_counts=True):
        M = np.asarray(M, float)
        P = M / np.maximum(M.sum(1, keepdims=True), 1) if norm_rows else M
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
        for i in range(P.shape[0]):
            for j in range(P.shape[1]):
                txt = f"{int(M[i, j])}" if fmt_counts else f"{P[i, j]*100:.1f}%"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if P[i, j] > 0.55 else "black")
        ax.set_xticks(range(len(clab))); ax.set_xticklabels(clab, rotation=35, ha="right")
        ax.set_yticks(range(len(rlab))); ax.set_yticklabels(rlab)
        ax.set_xlabel("Predicted (final label)"); ax.set_ylabel("True event")
        ax.set_title(title, fontsize=9.5)

    # trial-level confusion matrices
    for s in systems:
        fig, ax = plt.subplots(figsize=(4.6, 3.6))
        draw_cm(ax, R["systems"][s]["trial_confusion"], f"{s} - test2 (100 trials/class)", rows_lab, cols_lab)
        fig.tight_layout(); fig.savefig(f"{out}/fig_cm_trial_{SHORT[s]}.png", dpi=200); plt.close(fig)
    fig, axs = plt.subplots(1, len(systems), figsize=(4.2 * len(systems), 3.7))
    for ax, s in zip(axs, systems):
        draw_cm(ax, R["systems"][s]["trial_confusion"], s, rows_lab, cols_lab)
        if ax is not axs[0]:
            ax.set_ylabel("")
    fig.suptitle("Trial-level confusion matrices on held-out test2 trials (counts; shading = row %)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/fig_cm_trial_all.png", dpi=200); plt.close(fig)

    # frame-level confusion matrices (ML only)
    lab5 = ["Normal"] + rows_lab
    for s in systems:
        sc = R["systems"][s].get("step_confusion")
        if sc is None:
            continue
        fig, ax = plt.subplots(figsize=(4.4, 3.8))
        draw_cm(ax, sc, f"{s} - frame level (0.1 s steps, row %)", lab5, lab5, fmt_counts=False)
        ax.set_xlabel("Predicted class (argmax)"); ax.set_ylabel("True class")
        fig.tight_layout(); fig.savefig(f"{out}/fig_cm_step_{SHORT[s]}.png", dpi=200); plt.close(fig)

    # detection rate with CI
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    w = 0.8 / len(systems); x = np.arange(len(EVENT_CLASSES))
    for k, s in enumerate(systems):
        S = R["systems"][s]["trial"]
        v = np.array([S[c]["rate"] for c in EVENT_CLASSES]) * 100
        lo = v - np.array([S[c]["ci"][0] for c in EVENT_CLASSES]) * 100
        hi = np.array([S[c]["ci"][1] for c in EVENT_CLASSES]) * 100 - v
        ax.bar(x + (k - (len(systems) - 1) / 2) * w, v, w, yerr=[lo, hi], capsize=2.5,
               color=colors[s], error_kw=dict(lw=0.8),
               label=f"{s} - {R['systems'][s]['trial']['normal']['fv_rate']:.2f} false verif./h")
    ax.set_xticks(x); ax.set_xticklabels(rows_lab); ax.set_ylim(0, 105)
    ax.set_ylabel("Verified detection (%)")
    ax.set_title("Verified detection on held-out test2 trials (95% Clopper-Pearson CI)", fontsize=9)
    ax.legend(fontsize=7.5, loc="lower left", framealpha=0.95); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}/fig_detection.png", dpi=200); plt.close(fig)

    # latency
    fig, axs = plt.subplots(1, len(EVENT_CLASSES), figsize=(8, 2.8))
    for ax, c in zip(axs, EVENT_CLASSES):
        data = [R["systems"][s]["latencies"][c] or [np.nan] for s in systems]
        bp = ax.boxplot(data, widths=0.6, patch_artist=True, showfliers=False)
        for patch, s in zip(bp["boxes"], systems):
            patch.set_facecolor(colors[s]); patch.set_alpha(0.75)
        ax.set_xticks(range(1, len(systems) + 1)); ax.set_xticklabels(["Rules", "RF", "GB"][:len(systems)])
        ax.set_title(NICE[c], fontsize=9)
        if c == "hypoglycemia":
            ax.set_ylabel("")
    axs[0].set_ylabel("Latency after onset (s)")
    fig.suptitle("Detection latency of verified events (test2)", fontsize=10)
    fig.tight_layout(); fig.savefig(f"{out}/fig_latency.png", dpi=200); plt.close(fig)


# ------------------------------------------------------------------------------------------
def write_tables(out, R):
    systems = list(R["systems"])
    md = ["# Primary results - rule-based detector vs. ML models (held-out test2 seeds)", "",
          "Synthetic data only: all systems inherit the generator's assumptions. Same 24 features, same",
          "trials, same counting rules for all systems. ML thresholds chosen on dev only, under the rules'",
          "dev false-verification budget "
          f"({R['rules_dev_fv_budget']:.2f}/h).", "",
          "## Table A - Verified detection per event (trial level)", "",
          "| Event | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for c in EVENT_CLASSES:
        cells = []
        for s in systems:
            S = R["systems"][s]["trial"][c]
            cells.append(f"{S['verified']}/{S['n']} ({S['rate']*100:.0f}% [{S['ci'][0]*100:.0f}-{S['ci'][1]*100:.0f}]), "
                         f"{_f(S['lat_median'])} s")
        md.append(f"| {NICE[c]} | " + " | ".join(cells) + " |")
    cells = []
    for s in systems:
        S = R["systems"][s]["trial"]
        k = sum(S[c]["verified"] for c in EVENT_CLASSES); n = sum(S[c]["n"] for c in EVENT_CLASSES)
        cells.append(f"{k}/{n} ({k/n*100:.1f}%)")
    md.append("| **All events** | " + " | ".join(cells) + " |")
    md.append("| False verifications, normal driving | " + " | ".join(
        f"{R['systems'][s]['trial']['normal']['false_verifications']} in {R['systems'][s]['trial']['normal']['hours']:.1f} h "
        f"({R['systems'][s]['trial']['normal']['fv_rate']:.2f}/h, 95% up. {R['systems'][s]['trial']['normal']['fv_upper95']:.2f}/h)"
        for s in systems) + " |")
    md.append("| Probability threshold | " + " | ".join(
        "-" if R["systems"][s].get("p_thr") is None else f"p >= {R['systems'][s]['p_thr']}" for s in systems) + " |")
    md += ["", "Cells: verified/n (rate [95% Clopper-Pearson CI]), median latency after onset.", "",
           "## Table B - Correct final label among all event trials", "",
           "| Event | " + " | ".join(systems) + " |", "|---|" + "---|" * len(systems)]
    for c in EVENT_CLASSES:
        md.append(f"| {NICE[c]} | " + " | ".join(
            f"{R['systems'][s]['trial'][c]['final_label_correct']}/{R['systems'][s]['trial'][c]['n']}" for s in systems) + " |")
    md.append("| Macro accuracy (4 events) | " + " | ".join(f"{R['systems'][s]['trial_macro_acc']*100:.1f}%" for s in systems) + " |")
    md += ["", "## Table C - Trial-level confusion matrices (rows: true event; columns: final label)", ""]
    for s in systems:
        M = R["systems"][s]["trial_confusion"]
        md += [f"**{s}**", "", "| True \\ Predicted | " + " | ".join(TRIAL_COLS) + " |", "|---|" + "---|" * len(TRIAL_COLS)]
        md += [f"| {NICE[c]} | " + " | ".join(str(v) for v in M[i]) + " |" for i, c in enumerate(EVENT_CLASSES)]
        md.append("")
    md += ["## Table D - Frame-level metrics of the ML models (0.1 s steps, argmax, for information)", "",
           "| Model | Accuracy | Macro F1 | " + " | ".join(f"F1 {c}" for c in CLASSES) + " |",
           "|---|---|---|" + "---|" * len(CLASSES)]
    for s in systems:
        rep = R["systems"][s].get("step_report")
        if rep:
            md.append(f"| {s} | {rep['accuracy']:.3f} | {rep['macro avg']['f1-score']:.3f} | " +
                      " | ".join(f"{rep[c]['f1-score']:.3f}" for c in CLASSES) + " |")
    md += ["", "## Figures", "",
           "- `fig_cm_trial_all.png` - trial-level confusion matrices, three systems (recommended for the paper)",
           "- `fig_cm_trial_rules.png`, `fig_cm_trial_rf.png`, `fig_cm_trial_hgb.png` - same, one per system",
           "- `fig_cm_step_rf.png`, `fig_cm_step_hgb.png` - frame-level matrices, row-normalised",
           "- `fig_detection.png` - detection rate per event with 95% CI",
           "- `fig_latency.png` - latency of verified detections", "",
           "## Limits to state with these results", "",
           "- Synthetic trials from `generator.py`; the ML labels are the generator's labels.",
           "- ML models: fault-free trials only, no sensor-health gating, no ADI-unspecified output.",
           "- Latencies are offline (zero-phase filters).", ""]
    with open(f"{out}/summary.md", "w") as f:
        f.write("\n".join(md))

    # LaTeX
    tx = [r"% generated by ml_compare.py - held-out test2 seeds, synthetic data", r"\begin{table}[t]", r"\centering",
          r"\caption{Verified detection (95\% CI) and median latency on held-out trials: rule-based detector vs. ML models on the same features.}",
          r"\label{tab:ml_compare}", r"\footnotesize", r"\begin{tabular}{l" + "c" * len(systems) + "}", r"\hline",
          "Event & " + " & ".join(s.replace("(ours)", "(ours)") for s in systems) + r" \\", r"\hline"]
    for c in EVENT_CLASSES:
        cells = []
        for s in systems:
            S = R["systems"][s]["trial"][c]
            cells.append(f"{S['rate']*100:.0f}\\% [{S['ci'][0]*100:.0f}--{S['ci'][1]*100:.0f}], {_f(S['lat_median'])} s")
        tx.append(f"{NICE[c].replace('/', '/')} & " + " & ".join(cells) + r" \\")
    tx.append(r"\hline")
    tx.append("Correct label (4 events) & " + " & ".join(f"{R['systems'][s]['trial_macro_acc']*100:.1f}\\%" for s in systems) + r" \\")
    tx.append("False verif./h (normal) & " + " & ".join(f"{R['systems'][s]['trial']['normal']['fv_rate']:.2f}" for s in systems) + r" \\")
    tx += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    for s in systems:
        M = R["systems"][s]["trial_confusion"]
        tx += [r"\begin{table}[t]", r"\centering", r"\footnotesize",
               f"\\caption{{Trial-level confusion matrix, {s}, test2 (100 trials per event).}}",
               f"\\label{{tab:cm_{SHORT[s]}}}", r"\begin{tabular}{l" + "c" * len(TRIAL_COLS) + "}", r"\hline",
               r"True $\backslash$ Pred. & " + " & ".join(["Seiz.", "Sync.", "Stroke", "Hypo.", "Unspec.", "Missed"]) + r" \\", r"\hline"]
        tx += [f"{NICE[c]} & " + " & ".join(str(v) for v in M[i]) + r" \\" for i, c in enumerate(EVENT_CLASSES)]
        tx += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    with open(f"{out}/tables.tex", "w") as f:
        f.write("\n".join(tx))


# ------------------------------------------------------------------------------------------
def main(quick=False):
    t0 = time.time()
    frac = 0.15 if quick else 1.0
    out = os.path.join("results", "ml_compare" + ("_quick" if quick else ""))
    os.makedirs(out, exist_ok=True)
    dev, test = load("dev", frac), load("test2", frac)
    print(f"trials: dev {len(dev)}, test2 {len(test)}  ({time.time() - t0:.0f} s)")

    R = dict(quick=quick, features=FEATURES, settings=dict(subsample=SUBSAMPLE, verify_s=VERIFY_S,
             rf=RF_KW, hgb=HGB_KW, p_grid=GRID), systems={})

    def pack(rows, extra=None):
        S = summarise(rows)
        M = trial_confusion(rows)
        d = dict(trial=S, trial_confusion=M.tolist(),
                 trial_macro_acc=float(np.mean([M[i, i] / max(M[i].sum(), 1) for i in range(len(EVENT_CLASSES))])),
                 latencies={c: [float(r["latency"]) for r in rows if r["cls"] == c and r["verified"]] for c in EVENT_CLASSES})
        d.update(extra or {})
        return d

    rules_test = rules_rows(test)
    R["systems"]["Rule-based (ours)"] = pack(rules_test)

    yte = np.concatenate([d["y"] for d in test])
    for name, (cls, kw) in MODELS.items():
        t1 = time.time()
        p_thr, grid, budget = choose_threshold(cls, kw, dev)
        R["rules_dev_fv_budget"] = budget
        m = cls(**kw).fit(*stack(dev))
        pr = [m.predict_proba(d["X"]) for d in test]
        ypr = np.concatenate([p.argmax(1) for p in pr])
        extra = dict(p_thr=p_thr, threshold_grid_dev=grid,
                     step_confusion=confusion_matrix(yte, ypr, labels=range(len(CLASSES))).tolist(),
                     step_report=classification_report(yte, ypr, labels=range(len(CLASSES)), target_names=CLASSES,
                                                       output_dict=True, zero_division=0))
        if hasattr(m, "feature_importances_"):
            extra["feature_importance"] = sorted(zip(FEATURES, m.feature_importances_.tolist()), key=lambda x: -x[1])
        R["systems"][name] = pack(ml_rows(test, pr, p_thr), extra)
        print(f"{name}: p >= {p_thr}  ({time.time() - t1:.0f} s)")

    with open(f"{out}/results.json", "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_tables(out, R)
    figures(out, R)
    print(open(f"{out}/summary.md").read())
    print(f"done in {time.time() - t0:.0f} s -> {out}")
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    main(ap.parse_args().quick)
