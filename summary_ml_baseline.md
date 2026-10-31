# ML baseline (Random Forest) vs rule-based detector - test2 seeds

Synthetic data only: both systems inherit the generator's assumptions.

Threshold p >= 0.9 chosen on dev (rules dev false-verification rate 0.16/h).

| Class | Rules: verified | Rules: median latency | ML: verified | ML: median latency | ML: correct label |
|---|---|---|---|---|---|
| seizure | 86/100 (86%) | 6.5 s | 81/100 (81%) | 3.4 s | 81 |
| syncope | 91/100 (91%) | 4.5 s | 95/100 (95%) | 4.2 s | 95 |
| stroke | 94/100 (94%) | 13.8 s | 66/100 (66%) | 6.5 s | 66 |
| hypoglycemia | 85/100 (85%) | 185.5 s | 99/100 (99%) | 123.6 s | 99 |

| Normal driving | Rules | ML |
|---|---|---|
| False verifications | 1 in 9.2 h (0.11/h) | 1 in 9.2 h (0.11/h) |

Top features (Random Forest importance):

- lane_ratio: 0.165
- asym_d: 0.126
- asym_z: 0.109
- hf_ratio: 0.078
- energy_z: 0.077
- tilt: 0.063
- gaze_ratio: 0.060
- perclos: 0.050
- micro_ratio: 0.041
- rate_ratio: 0.032
