| held-out | loss legacy | loss jesse | ratio | cascade PR-AUC legacy (t6/24/48/96) | cascade PR-AUC jesse | Δ pp (t6/24/48/96) |
|---|---|---|---|---|---|---|
| geoclaw_2050 | 0.1406 | 0.1807 | 1.29 | 0.955/0.983/0.959/0.981 | 0.956/0.968/0.941/0.969 | +0.1/-1.5/-1.9/-1.2 |
| syn_ts_914_6_1p2955 | 0.0855 | 0.0900 | 1.05 | 0.967/0.863/0.858/0.856 | 0.969/0.851/0.855/0.845 | +0.2/-1.2/-0.3/-1.1 |
| syn_ts_258_9_3p0022 | 0.1248 | 0.1491 | 1.19 | 0.966/0.981/0.955/0.978 | 0.971/0.969/0.945/0.971 | +0.5/-1.2/-1.0/-0.7 |
| syn_ts_605_5_3p7563 | 0.2052 | 0.3441 | 1.68 | 0.933/0.975/0.969/0.988 | 0.884/0.921/0.909/0.947 | -5.0/-5.4/-5.9/-4.1 |
| syn_ts_808_27_3p7885 | 0.1400 | 0.2072 | 1.48 | 0.955/0.978/0.963/0.984 | 0.957/0.952/0.926/0.957 | +0.2/-2.6/-3.6/-2.7 |

Best validation loss: legacy 0.10824 (epoch 5), jesse 0.11969 (epoch 5). Task: given the direct (t=0) failure mask as input, predict failure by t in [6, 24, 48, 96]; cascade-only metrics score nodes that were not direct failures. Scenario set jesse22, 5-way scenario holdout, identical seed/architecture/graph across arms.