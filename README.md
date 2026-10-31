# CabinGuard-ADI — detection logic and validation code (Phase 1)

TSYP14 Technical Challenge (IEEE VTS / EMBS Tunisia).
This code produces **every number in the Phase-1 paper**. It contains the detection logic of
CabinGuard-ADI and the synthetic test bench used to evaluate it. All data are synthetic:
no video, no personal data.

## Install

```
pip install -r requirements.txt
```

Every script is deterministic (fixed random seeds): re-running it gives the same numbers.

## Reproduce the paper

| Paper result | Command | Output | Time* |
|---|---|---|---|
| Table II (detection per event), comparison with Euro NCAP-style rules, sensor ablations | `python run_validation.py --split test2` | `results/test2/results.json` | ~2 min |
| Figures | `python figures.py` | `results/test2/figures/` | ~2 min |
| Fail-safe behaviour: 7 sensor-fault types, DEGRADED delay, calibration faults (174/200) | `python run_faults.py` | `results/faults/results.json` | ~8 min |
| Recalibration test, incl. sensor back just before the event | `python run_recal.py` | `results/recal/summary.md` | ~10 min |
| Rules vs. Random Forest | `python ml_baseline.py` | `results/ml_baseline/summary.md` | ~7 min |
| Rules vs. Random Forest vs. Gradient Boosting, confusion matrices | `python ml_compare.py` | `results/ml_compare/` | ~8 min |
| Paper figures (detection, confusion matrices, sensor faults) | `python paper_figures.py` | `results/paper/` | <1 s |

\*on a 2-core machine. Normal-driving false-alarm rate in the paper (13 false verifications in 34.8 h):
1 in 9.2 h from `results/test2` + 12 in 25.7 h on the fault-free copies of the normal trials in
`results/faults` (field `fv_clean`).

## Files

| File | Role |
|---|---|
| `generator.py` | Synthetic multimodal trials (belt + seat IMU 100 Hz, camera features 30 Hz, telemetry 20 Hz). `SIGNATURES` gives every event parameter with its provenance: LIT (literature), LIT-DIR (direction from literature), ASSUMED. |
| `detector.py` | **The detector.** Calibration (20 s), sensor health, features, binary evidence, corroboration vote, Stage 1 / Stage 2, labels, recalibration. All thresholds are in `PARAMS`. |
| `faults.py` | Sensor-fault injection (stuck IMU, lost telemetry, lost camera). |
| `run_validation.py` | Main evaluation, baselines (B1 Euro NCAP-style, B2 + input override), ablations. |
| `run_faults.py`, `run_recal.py` | Fail-safe and recalibration tests. |
| `ml_baseline.py` | Random Forest trained on the same features — a comparison, not the detector. |
| `ml_compare.py` | Rules vs. Random Forest vs. Gradient Boosting: trial-level and frame-level confusion matrices, detection/latency figures, LaTeX tables. |
| `tune.py` | Threshold exploration, run on the development split only. |
| `figures.py` | Paper figures. |
| `CHANGELOG.md` | Development history. |

## Evaluation discipline

- Thresholds were set on the **development** seeds (`--split dev`); the paper reports the
  **held-out** seeds (`--split test2`) once. `results/test_v1` is a first held-out run that was
  retired after it revealed failures; it is kept for transparency.
- The fault and recalibration tests use fresh seed ranges (500 000+ and 700 000+).

## Limits (stated in the paper)

- The generator and the detector encode the same team's assumptions: results show that the logic
  works when the assumptions hold, not clinical performance.
- Filters are zero-phase (offline), so latencies are offline estimates.
- Pedal position is not used as evidence; stroke and hypoglycaemia rely on one sensor;
  the Safety Supervisor, vehicle response and driver-response check are designed for Phase 2.
