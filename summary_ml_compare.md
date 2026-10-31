# Primary results - rule-based detector vs. ML models (held-out test2 seeds)

Synthetic data only: all systems inherit the generator's assumptions. Same 24 features, same
trials, same counting rules for all systems. ML thresholds chosen on dev only, under the rules'
dev false-verification budget (0.16/h).

## Table A - Verified detection per event (trial level)

| Event | Rule-based (ours) | Random Forest | Gradient Boosting |
|---|---|---|---|
| Seizure | 86/100 (86% [78-92]), 6.5 s | 81/100 (81% [72-88]), 3.4 s | 93/100 (93% [86-97]), 3.1 s |
| Syncope/LOC | 91/100 (91% [84-96]), 4.5 s | 95/100 (95% [89-98]), 4.2 s | 97/100 (97% [91-99]), 3.9 s |
| Stroke | 94/100 (94% [87-98]), 13.8 s | 66/100 (66% [56-75]), 6.5 s | 73/100 (73% [63-81]), 7.2 s |
| Hypoglycaemia | 85/100 (85% [76-91]), 185.5 s | 99/100 (99% [95-100]), 123.6 s | 99/100 (99% [95-100]), 120.2 s |
| **All events** | 356/400 (89.0%) | 341/400 (85.2%) | 362/400 (90.5%) |
| False verifications, normal driving | 1 in 9.2 h (0.11/h, 95% up. 0.52/h) | 1 in 9.2 h (0.11/h, 95% up. 0.52/h) | 4 in 9.2 h (0.44/h, 95% up. 1.00/h) |
| Probability threshold | - | p >= 0.9 | p >= 0.999 |

Cells: verified/n (rate [95% Clopper-Pearson CI]), median latency after onset.

## Table B - Correct final label among all event trials

| Event | Rule-based (ours) | Random Forest | Gradient Boosting |
|---|---|---|---|
| Seizure | 86/100 | 81/100 | 93/100 |
| Syncope/LOC | 91/100 | 95/100 | 97/100 |
| Stroke | 93/100 | 66/100 | 73/100 |
| Hypoglycaemia | 84/100 | 99/100 | 99/100 |
| Macro accuracy (4 events) | 88.5% | 85.2% | 90.5% |

## Table C - Trial-level confusion matrices (rows: true event; columns: final label)

**Rule-based (ours)**

| True \ Predicted | seizure | syncope | stroke | hypoglycemia | ADI-unspecified | missed |
|---|---|---|---|---|---|---|
| Seizure | 86 | 0 | 0 | 0 | 0 | 14 |
| Syncope/LOC | 0 | 91 | 0 | 0 | 0 | 9 |
| Stroke | 0 | 0 | 93 | 1 | 0 | 6 |
| Hypoglycaemia | 1 | 0 | 0 | 84 | 0 | 15 |

**Random Forest**

| True \ Predicted | seizure | syncope | stroke | hypoglycemia | ADI-unspecified | missed |
|---|---|---|---|---|---|---|
| Seizure | 81 | 0 | 0 | 0 | 0 | 19 |
| Syncope/LOC | 0 | 95 | 0 | 0 | 0 | 5 |
| Stroke | 0 | 0 | 66 | 0 | 0 | 34 |
| Hypoglycaemia | 0 | 0 | 0 | 99 | 0 | 1 |

**Gradient Boosting**

| True \ Predicted | seizure | syncope | stroke | hypoglycemia | ADI-unspecified | missed |
|---|---|---|---|---|---|---|
| Seizure | 93 | 0 | 0 | 0 | 0 | 7 |
| Syncope/LOC | 0 | 97 | 0 | 0 | 0 | 3 |
| Stroke | 0 | 0 | 73 | 0 | 0 | 27 |
| Hypoglycaemia | 0 | 0 | 0 | 99 | 0 | 1 |

## Table D - Frame-level metrics of the ML models (0.1 s steps, argmax, for information)

| Model | Accuracy | Macro F1 | F1 normal | F1 seizure | F1 syncope | F1 stroke | F1 hypoglycemia |
|---|---|---|---|---|---|---|---|
| Random Forest | 0.857 | 0.867 | 0.884 | 0.915 | 0.917 | 0.850 | 0.768 |
| Gradient Boosting | 0.842 | 0.855 | 0.870 | 0.904 | 0.920 | 0.831 | 0.752 |

## Figures

- `fig_cm_trial_all.png` - trial-level confusion matrices, three systems (recommended for the paper)
- `fig_cm_trial_rules.png`, `fig_cm_trial_rf.png`, `fig_cm_trial_hgb.png` - same, one per system
- `fig_cm_step_rf.png`, `fig_cm_step_hgb.png` - frame-level matrices, row-normalised
- `fig_detection.png` - detection rate per event with 95% CI
- `fig_latency.png` - latency of verified detections

## Limits to state with these results

- Synthetic trials from `generator.py`; the ML labels are the generator's labels.
- ML models: fault-free trials only, no sensor-health gating, no ADI-unspecified output.
- Latencies are offline (zero-phase filters).
