"""
CabinGuard-ADI - extra paper figure built from existing result files (no new simulation).

    python paper_figures.py     ->  results/paper/fig_faults.png, fig_cm_cabinguard.png, fig_cm_all.png, fig_detection.png

fig_faults.png: sensor-fault injection (results/faults/results.json, run_faults.py).
  (a) false Stage-2 verifications on normal driving with vs. without sensor-health gating;
  (b) verified events under each fault vs. the same trials without the fault.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KINDS = [("belt_stuck", "Belt IMU\nstuck"), ("frame_stuck", "Ref. IMU\nstuck"), ("tel_loss", "Telemetry\nlost"),
         ("cam_loss", "Camera\nlost"), ("belt_stuck+tel_loss", "Belt IMU +\ntelemetry"), ("cam_stall", "Camera\nstall"),
         ("belt_stuck@calib", "Belt IMU\n@calib."), ("cam_loss@calib", "Camera\n@calib.")]


def main():
    S = json.load(open(os.path.join("results", "faults", "results.json")))["summary"]
    out = os.path.join("results", "paper"); os.makedirs(out, exist_ok=True)
    lab = [l for _, l in KINDS]
    fv_g = [S[k]["false_verif"] for k, _ in KINDS]
    fv_n = [S[k]["false_verif_without_health"] for k, _ in KINDS]
    ev = lambda k, f: sum(e[f] for e in S[k]["events"].values())
    ev_g = [ev(k, "verified") for k, _ in KINDS]
    ev_c = [ev(k, "verified_no_fault") for k, _ in KINDS]
    hours = S[KINDS[0][0]]["normal_hours"]

    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(2, 1, figsize=(7.0, 4.4), sharex=True)
    x = np.arange(len(KINDS)); w = 0.38
    a = axs[0]
    b1 = a.bar(x - w / 2, fv_n, w, color="#C55A11", label="without sensor-health gating")
    b2 = a.bar(x + w / 2, fv_g, w, color="#1F4E79", label="CabinGuard (health gating + DEGRADED)")
    for bars in (b1, b2):
        for r in bars:
            a.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.6, f"{int(r.get_height())}", ha="center", fontsize=7)
    a.set_ylabel(f"False verifications\n(normal driving, {hours:.1f} h each)")
    a.set_ylim(0, max(fv_n) * 1.2); a.legend(fontsize=7, frameon=False, loc="upper left")
    a.set_title("(a) Unsafe false alarms caused by sensor faults", fontsize=8.5, loc="left")
    a.grid(axis="y", alpha=0.3)
    a = axs[1]
    c1 = a.bar(x - w / 2, ev_c, w, color="#9DC3E6", label="same trials, no fault")
    c2 = a.bar(x + w / 2, ev_g, w, color="#1F4E79", label="with fault (CabinGuard)")
    for bars in (c1, c2):
        for r in bars:
            a.text(r.get_x() + r.get_width() / 2, r.get_height() + 1.0, f"{int(r.get_height())}", ha="center", fontsize=7)
    a.set_ylabel("Events verified (/100)"); a.set_ylim(0, 128)
    a.set_title("(b) Detection kept under faults (25 trials per event class)", fontsize=8.5, loc="left")
    a.legend(fontsize=7, frameon=False, loc="upper left", ncol=2); a.grid(axis="y", alpha=0.3)
    a.set_xticks(x); a.set_xticklabels(lab, fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(out, "fig_faults.png"), dpi=300); plt.close(fig)
    print("saved", os.path.join(out, "fig_faults.png"))
    print("fv gated", fv_g, sum(fv_g), "ungated", fv_n, sum(fv_n), "events", ev_g, ev_c)


def fig_cm_rules():
    """Trial-level confusion matrix of the rule-based detector (results/ml_compare/results.json)."""
    R = json.load(open(os.path.join("results", "ml_compare", "results.json")))
    M = np.array(R["systems"]["Rule-based (ours)"]["trial_confusion"], float)
    assert M[:, 4].sum() == 0          # no final "ADI-unspecified" label on test2 -> column omitted
    M = np.delete(M, 4, axis=1)
    rl = ["Seizure", "Syncope/LOC", "Stroke", "Hypoglycaemia"]
    cl = rl + ["Missed"]
    P = M / M.sum(1, keepdims=True)
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    ax.imshow(P, cmap="Blues", vmin=0, vmax=1)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if P[i, j] > 0.55 else "black")
    ax.set_xticks(range(len(cl))); ax.set_xticklabels(cl, rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(rl))); ax.set_yticklabels(rl, fontsize=7)
    ax.set_xlabel("Transmitted label"); ax.set_ylabel("True event")
    fig.tight_layout(); fig.savefig(os.path.join("results", "paper", "fig_cm_cabinguard.png"), dpi=300); plt.close(fig)
    print("saved results/paper/fig_cm_cabinguard.png")


def fig_cm_all():
    """Trial-level confusion matrices, CabinGuard-ADI vs. the two ML models, stacked vertically
    (single-column figure for the paper)."""
    R = json.load(open(os.path.join("results", "ml_compare", "results.json")))["systems"]
    names = [("Rule-based (ours)", "(a) CabinGuard-ADI (rules)"), ("Random Forest", "(b) Random Forest"),
             ("Gradient Boosting", "(c) Gradient Boosting")]
    rl = ["Seizure", "Syncope/LOC", "Stroke", "Hypoglycaemia"]
    cl = rl + ["Missed"]
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, axs = plt.subplots(3, 1, figsize=(3.4, 6.0), sharex=True)
    for ax, (key, title) in zip(axs, names):
        M = np.array(R[key]["trial_confusion"], float)
        assert M[:, 4].sum() == 0          # no final "ADI-unspecified" label -> column omitted
        M = np.delete(M, 4, axis=1)
        P = M / M.sum(1, keepdims=True)
        ax.imshow(P, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=7.5,
                        color="white" if P[i, j] > 0.55 else "black")
        ax.set_yticks(range(len(rl))); ax.set_yticklabels(rl, fontsize=6.8)
        ax.set_ylabel("True event", fontsize=7)
        ax.set_title(title, fontsize=8, loc="left", fontweight="bold" if key.startswith("Rule") else "normal")
    axs[-1].set_xticks(range(len(cl))); axs[-1].set_xticklabels(cl, rotation=30, ha="right", fontsize=6.8)
    axs[-1].set_xlabel("Transmitted label", fontsize=7)
    fig.tight_layout(h_pad=0.6); fig.savefig(os.path.join("results", "paper", "fig_cm_all.png"), dpi=300); plt.close(fig)
    print("saved results/paper/fig_cm_all.png")


def fig_detection():
    """Detection rate per event (95% CI) + false-alarm rate, CabinGuard-ADI vs. ML, paper version."""
    R = json.load(open(os.path.join("results", "ml_compare", "results.json")))["systems"]
    names = [("Rule-based (ours)", "CabinGuard-ADI (rules)", "#1F4E79"), ("Random Forest", "Random Forest", "#C55A11"),
             ("Gradient Boosting", "Gradient Boosting", "#8FAADC")]
    ev = ["seizure", "syncope", "stroke", "hypoglycemia"]
    lab = ["Seizure", "Syncope\n/LOC", "Stroke", "Hypo-\nglycaemia", "All\nevents"]
    plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    x = np.arange(len(lab)); w = 0.27
    for k, (key, name, col) in enumerate(names):
        S = R[key]["trial"]
        v = [S[c]["rate"] * 100 for c in ev]
        lo = [S[c]["ci"][0] * 100 for c in ev]; hi = [S[c]["ci"][1] * 100 for c in ev]
        kk = sum(S[c]["verified"] for c in ev); nn = sum(S[c]["n"] for c in ev)
        v.append(kk / nn * 100); lo.append(np.nan); hi.append(np.nan)
        v, lo, hi = map(np.array, (v, lo, hi))
        fv = S["normal"]["fv_rate"]
        bars = ax.bar(x + (k - 1) * w, v, w, color=col, label=f"{name}: {fv:.2f} false alarms/h")
        ax.errorbar(x[:4] + (k - 1) * w, v[:4], yerr=np.vstack([v[:4] - lo[:4], hi[:4] - v[:4]]),
                    fmt="none", ecolor="black", elinewidth=0.6, capsize=1.8)
        for r, val in zip(bars, v):
            ax.text(r.get_x() + r.get_width() / 2, 2, f"{val:.0f}", ha="center", va="bottom", fontsize=5.5,
                    color="white", rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7)
    ax.set_ylabel("Verified detection (%)"); ax.set_ylim(0, 135); ax.set_yticks(range(0, 101, 20))
    ax.legend(fontsize=6.3, frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join("results", "paper", "fig_detection.png"), dpi=300); plt.close(fig)
    print("saved results/paper/fig_detection.png")


if __name__ == "__main__":
    main()
    fig_cm_rules()
    fig_cm_all()
    fig_detection()
