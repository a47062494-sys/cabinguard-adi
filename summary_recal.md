# Recalibration test (fresh seeds 700 000+)

Verified events: with recalibration / without (v3) / same trials without fault.

## belt_stuck@calib

- events verified: **89/100** with recalibration, 83/100 without, 90/100 without fault
- normal driving: 2 false verifications in 3.7 h with recalibration (0 without, 3 without fault); before onset in event trials: 0 (0 without)
- recalibration window found in 160/160 trials, median 25.0 s after the sensor came back
- window containing the event onset: 0 trials, of which 0 missed; events lost because of recalibration (found without it): 0; wrong labels 1

| Class | With recal | Without (v3) | No fault | Window contains event |
|---|---|---|---|---|
| seizure | 22/25 | 21/25 | 23/25 | 0 |
| syncope | 24/25 | 19/25 | 24/25 | 0 |
| stroke | 21/25 | 21/25 | 21/25 | 0 |
| hypoglycemia | 22/25 | 22/25 | 22/25 | 0 |

## cam_loss@calib

- events verified: **84/100** with recalibration, 34/100 without, 90/100 without fault
- normal driving: 1 false verifications in 3.7 h with recalibration (0 without, 1 without fault); before onset in event trials: 1 (0 without)
- recalibration window found in 160/160 trials, median 25.0 s after the sensor came back
- window containing the event onset: 0 trials, of which 0 missed; events lost because of recalibration (found without it): 0; wrong labels 1

| Class | With recal | Without (v3) | No fault | Window contains event |
|---|---|---|---|---|
| seizure | 20/25 | 18/25 | 22/25 | 0 |
| syncope | 23/25 | 16/25 | 23/25 | 0 |
| stroke | 20/25 | 0/25 | 24/25 | 0 |
| hypoglycemia | 21/25 | 0/25 | 21/25 | 0 |

## tel_loss@calib

- events verified: **90/100** with recalibration, 65/100 without, 92/100 without fault
- normal driving: 3 false verifications in 3.7 h with recalibration (2 without, 2 without fault); before onset in event trials: 0 (0 without)
- recalibration window found in 160/160 trials, median 25.0 s after the sensor came back
- window containing the event onset: 0 trials, of which 0 missed; events lost because of recalibration (found without it): 0; wrong labels 1

| Class | With recal | Without (v3) | No fault | Window contains event |
|---|---|---|---|---|
| seizure | 23/25 | 20/25 | 22/25 | 0 |
| syncope | 22/25 | 21/25 | 23/25 | 0 |
| stroke | 24/25 | 24/25 | 24/25 | 0 |
| hypoglycemia | 21/25 | 0/25 | 23/25 | 0 |

## belt_stuck@calib_long

- events verified: **89/100** with recalibration, 86/100 without, 91/100 without fault
- normal driving: 2 false verifications in 3.7 h with recalibration (1 without, 3 without fault); before onset in event trials: 0 (0 without)
- recalibration window found in 123/160 trials, median 20.15321250116404 s after the sensor came back
- window containing the event onset: 44 trials, of which 10 missed; events lost because of recalibration (found without it): 0; wrong labels 12

| Class | With recal | Without (v3) | No fault | Window contains event |
|---|---|---|---|---|
| seizure | 21/25 | 19/25 | 23/25 | 8 |
| syncope | 23/25 | 22/25 | 23/25 | 9 |
| stroke | 24/25 | 24/25 | 24/25 | 7 |
| hypoglycemia | 21/25 | 21/25 | 21/25 | 20 |

## cam_loss@calib_long

- events verified: **71/100** with recalibration, 40/100 without, 94/100 without fault
- normal driving: 1 false verifications in 3.7 h with recalibration (0 without, 2 without fault); before onset in event trials: 0 (0 without)
- recalibration window found in 146/160 trials, median 20.116773810634584 s after the sensor came back
- window containing the event onset: 66 trials, of which 26 missed; events lost because of recalibration (found without it): 0; wrong labels 0

| Class | With recal | Without (v3) | No fault | Window contains event |
|---|---|---|---|---|
| seizure | 20/25 | 20/25 | 22/25 | 12 |
| syncope | 20/25 | 20/25 | 25/25 | 13 |
| stroke | 8/25 | 0/25 | 25/25 | 23 |
| hypoglycemia | 23/25 | 0/25 | 22/25 | 18 |

## tel_loss@calib_long

- events verified: **84/100** with recalibration, 62/100 without, 91/100 without fault
- normal driving: 0 false verifications in 3.7 h with recalibration (0 without, 0 without fault); before onset in event trials: 0 (0 without)
- recalibration window found in 139/160 trials, median 20.111613302546317 s after the sensor came back
- window containing the event onset: 58 trials, of which 14 missed; events lost because of recalibration (found without it): 0; wrong labels 0

| Class | With recal | Without (v3) | No fault | Window contains event |
|---|---|---|---|---|
| seizure | 24/25 | 23/25 | 24/25 | 9 |
| syncope | 17/25 | 16/25 | 24/25 | 16 |
| stroke | 23/25 | 23/25 | 23/25 | 12 |
| hypoglycemia | 20/25 | 0/25 | 20/25 | 21 |

