"""Figures + extra analyses for the paper (reads results/test2/results.json)."""
import json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor

from generator import generate_trial
from detector import features, decide, run_trial, _bp
from generator import FS_IMU

OUT = "results/test2/figures"
os.makedirs(OUT, exist_ok=True)
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.titlesize": 10,
                     "axes.titlecolor": INK, "figure.dpi": 200})
R = json.load(open("results/test2/results.json"))


# ---------------------------------------------------------------- 1. baselines
def fig_baselines():
    B = R["baselines"]
    groups = [("syncope_all", "All syncope"), ("syncope_pedal_held", "Accelerator\nheld"),
              ("syncope_eyes_open", "Eyes open"), ("syncope_upright", "Upright\n(stress set)")]
    names = [("B1", "B1: Euro NCAP eyes-closed ≥6 s / gaze-away", C2), ("B2", "B2: B1 + input override", C3),
             ("CG", "CabinGuard fusion", C1)]
    x = np.arange(len(groups)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for k, (key, lab, col) in enumerate(names):
        v = [B[g][key]["rate"] * 100 for g, _ in groups]
        bars = ax.bar(x + (k - 1) * w, v, w - 0.02, color=col, label=lab, zorder=3)
        for xi, vi in zip(x + (k - 1) * w, v):
            ax.text(xi, vi + 1.5, f"{vi:.0f}", ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, [g[1] for g in groups]); ax.set_ylim(0, 112); ax.set_ylabel("Episodes verified (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_baselines.png"); plt.close(fig)


# ---------------------------------------------------------------- 2. latency
def fig_latency():
    rows = [r for r in R["rows"] if r["cls"] != "normal" and r["verified"]]
    order = [("syncope", "Syncope / LOC"), ("seizure", "Seizure (ADI response)"), ("stroke", "Stroke"), ("hypoglycemia", "Hypoglycaemia")]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    rng = np.random.default_rng(0)
    for i, (c, lab) in enumerate(order):
        v = np.array([r["latency"] for r in rows if r["cls"] == c])
        y = i + rng.uniform(-0.18, 0.18, len(v))
        ax.scatter(v, y, s=9, color=C1, alpha=0.55, edgecolors="none", zorder=3)
        med = np.median(v)
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, lw=2, zorder=4)
        ax.text(med * 1.08, i + 0.32, f"median {med:.1f} s", fontsize=7.5, color=INK2, va="bottom")
    # seizure label latency
    lv = np.array([r["label_latency"] for r in R["rows"] if r["cls"] == "seizure" and r["final_label"] == "seizure"])
    ax.scatter(lv, 1 + rng.uniform(-0.18, 0.18, len(lv)) - 0.0, s=9, marker="x", color=C2, alpha=0.6, zorder=3,
               label=f"Seizure label assigned (median {np.median(lv):.1f} s)")
    ax.set_xscale("log"); ax.set_yticks(range(len(order)), [o[1] for o in order]); ax.invert_yaxis()
    ax.set_xlabel("Time from first physical sign to Stage-2 verification (s, log scale)")
    ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 500], ["1", "2", "5", "10", "20", "50", "100", "200", "500"])
    ax.grid(axis="y", visible=False); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_latency.png"); plt.close(fig)


# ---------------------------------------------------------------- 3. traces
def _trace(cls, seed, variant, fname, title, window):
    tr = generate_trial(cls, seed, variant=variant)
    F = features(tr); D = decide(F); on = tr["truth"]["onset"]
    r = run_trial(F, D, on)
    a, b = on + window[0], on + window[1]
    fig, axs = plt.subplots(5, 1, figsize=(6.6, 5.4), sharex=True)
    def seg(t, x):
        m = (t >= a) & (t <= b); return t[m] - on, x[m]
    ax = axs[0]; ax.plot(*seg(tr["t_tel"], tr["steer"]), color=C1, lw=1.2); ax.set_ylabel("Steering\n(deg)")
    ax = axs[1]; ax.plot(*seg(tr["t_tel"], tr["pedal"] * 100), color=C1, lw=1.2); ax.set_ylabel("Accelerator\n(%)"); ax.set_ylim(-5, 105)
    ax = axs[2]; ax.plot(*seg(tr["t_cam"], tr["ear"]), color=C1, lw=1.0); ax.set_ylabel("Eye aspect\nratio")
    ax2 = axs[3]; t_, p_ = seg(tr["t_cam"], np.where(tr["face_ok"], tr["pitch"], np.nan)); ax2.plot(t_, p_, color=C1, lw=1.0, label="head pitch")
    t_, y_ = seg(tr["t_cam"], np.where(tr["face_ok"], tr["yaw"], np.nan)); ax2.plot(t_, y_, color=C2, lw=1.0, label="head yaw")
    ax2.set_ylabel("Head (deg)"); ax2.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    from detector import _hp
    res = _bp(tr["belt"], FS_IMU, 1.5, 8)
    ax = axs[4]; ax.plot(*seg(tr["t_imu"], np.linalg.norm(res, axis=1)), color=C1, lw=0.6); ax.set_ylabel("Belt IMU\n1.5–8 Hz (m/s²)")
    ax.set_xlabel("Time from event onset (s)")
    t = D["t"]
    for ax in axs:
        ax.axvline(0, color=INK, lw=1, ls="--")
        if not np.isnan(r["s1_latency"]):
            ax.axvline(r["s1_latency"], color=C2, lw=1)
        if r["verified"]:
            ax.axvline(r["latency"], color=C3, lw=1.6)
        if r.get("label_latency") and r["label_latency"] != r["latency"]:
            ax.axvline(r["label_latency"], color=C3, lw=1.0, ls=":")
    axs[0].set_title(title, loc="left")
    ymax = axs[0].get_ylim()[1]
    axs[0].text(0, ymax, " onset", fontsize=7, color=INK, va="top")
    if not np.isnan(r["s1_latency"]):
        axs[0].text(r["s1_latency"], ymax, " Stage 1", fontsize=7, color=INK2, va="bottom")
    if r["verified"]:
        axs[0].text(r["latency"], ymax, f" Stage 2", fontsize=7, color=INK2, va="top")
        axs[1].text(r["latency"], 100, f" label: {r['label']}", fontsize=7, color=INK2, va="top")
    if r.get("label_latency") and r["label_latency"] != r["latency"]:
        axs[1].text(r["label_latency"], 70, f" label → {r['final_label']}", fontsize=7, color=INK2, va="top")
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}"); plt.close(fig)
    return r, tr["truth"]


def pick_trace_seeds():
    # upright syncope with eyes open and accelerator held (the case baselines miss)
    for s in range(300_000, 300_400):
        tr = generate_trial("syncope", s, variant="upright")
        t = tr["truth"]
        if t["eyes_open"] and t["pedal_sustained"] and t["persistent"]:
            F = features(tr); r = run_trial(F, decide(F), t["onset"])
            if r["verified"]:
                sy = s; break
    for s in range(301_000, 301_400):
        tr = generate_trial("seizure", s, variant="gtcs")
        if tr["truth"]["tonic"] < 13:
            sz = s; break
    return sy, sz


# ---------------------------------------------------------------- 4. droop sweep
def _droop_job(args):
    d, s = args
    tr = generate_trial("stroke", s, variant=f"delta:{d}")
    F = features(tr); r = run_trial(F, decide(F), tr["truth"]["onset"])
    return d, r["verified"] and r["final_label"] == "stroke", r["latency"]


def fig_droop():
    deltas = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.06, 0.09]
    jobs = [(d, 310_000 + 1000 * i + k) for i, d in enumerate(deltas) for k in range(40)]
    with ProcessPoolExecutor(2) as ex:
        out = list(ex.map(_droop_job, jobs, chunksize=8))
    rate = [np.mean([o[1] for o in out if o[0] == d]) for d in deltas]
    lat = [float(np.nanmedian([o[2] for o in out if o[0] == d and o[1]])) if any(o[1] for o in out if o[0] == d) else None for d in deltas]
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.plot(deltas, np.array(rate) * 100, color=C1, lw=2, marker="o", ms=5, zorder=3)
    for d, v in zip(deltas, rate):
        ax.text(d, v * 100 + 4, f"{v*100:.0f}", ha="center", fontsize=7.5, color=INK2)
    ax.axvspan(0.02, 0.045, color=GRID, alpha=0.6, zorder=1); ax.text(0.0325, 8, "'mild' range\nin main set", ha="center", fontsize=7, color=INK2)
    ax.set_ylim(0, 112); ax.set_xlabel("Facial-asymmetry increase Δ (landmark index units)"); ax.set_ylabel("Verified as stroke (%)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_droop.png"); plt.close(fig)
    return dict(deltas=deltas, rate=rate, lat=lat, n_each=40)


def syncope_miss_analysis():
    out = []
    for r in R["rows"]:
        if r["cls"] == "syncope":
            t = generate_trial("syncope", r["seed"])["truth"]
            out.append(dict(verified=r["verified"], persistent=t["persistent"], loc=t["loc_dur"], collapse=t["collapse"],
                            eyes_open=t["eyes_open"], myoclonus=t["myoclonus"], pedal=t["pedal_sustained"]))
    miss = [o for o in out if not o["verified"]]
    return dict(n=len(out), missed=len(miss),
                missed_transient=sum(not m["persistent"] for m in miss),
                missed_loc_s=[round(m["loc"], 1) if np.isfinite(m["loc"]) else "persistent" for m in miss],
                missed_upright=sum(not m["collapse"] for m in miss),
                persistent_rate=np.mean([o["verified"] for o in out if o["persistent"]]),
                transient_rate=np.mean([o["verified"] for o in out if not o["persistent"]]),
                transient_ge10_rate=np.mean([o["verified"] for o in out if not o["persistent"] and o["loc"] >= 10]),
                n_persistent=sum(o["persistent"] for o in out))


if __name__ == "__main__":
    import sys
    fig_baselines(); fig_latency()
    if "--quick" in sys.argv:
        _trace("syncope", 300001, "upright", "fig_trace_syncope_upright.png", "Upright syncope, eyes open, accelerator held (synthetic)", (-6, 18))
        _trace("seizure", 301003, "gtcs", "fig_trace_seizure.png", "Generalised tonic-clonic seizure (synthetic)", (-5, 30))
        sys.exit()
    sy, sz = pick_trace_seeds()
    r1, t1 = _trace("syncope", sy, "upright", "fig_trace_syncope_upright.png",
                    "Upright syncope, eyes open, accelerator held (synthetic)", (-6, 18))
    r2, t2 = _trace("seizure", sz, "gtcs", "fig_trace_seizure.png",
                    "Generalised tonic-clonic seizure (synthetic)", (-5, 30))
    droop = fig_droop()
    miss = syncope_miss_analysis()
    extra = dict(trace_syncope=dict(seed=sy, result=r1, truth={k: v for k, v in t1.items() if k != "talking"}),
                 trace_seizure=dict(seed=sz, result=r2, truth={k: v for k, v in t2.items() if k != "talking"}),
                 droop=droop, syncope_misses=miss)
    json.dump(extra, open("results/test2/extra.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(extra, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
