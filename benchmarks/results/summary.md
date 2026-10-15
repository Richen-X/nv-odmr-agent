Seeds 0-99 (100 samples per configuration).

Target (median |dB| <= 40 mG with >= 90% of runs finding all 8 lines within 0.5 MHz): GP two-stage needs 538 points, the best uniform sweep 1315 (59% fewer). Bootstrap over samples: saving 59% (95% CI 30-69%).

Median field error with bootstrap 95% CI of the median:

| method | setting | points | median field error (95% CI) |
|---|---|---|---|
| GP two-stage (this work) | 96 refine pts @ 0 dBm | 250 | 122 mG (103-156) |
| GP two-stage (this work) | 160 refine pts @ 0 dBm | 314 | 64 mG (54-73) |
| GP two-stage (this work) | 256 refine pts @ 0 dBm | 410 | 37 mG (34-45) |
| GP two-stage (this work) | 384 refine pts @ 0 dBm | 538 | 31 mG (25-35) |
| GP two-stage (this work) | 576 refine pts @ 0 dBm | 730 | 27 mG (23-34) |
| GP two-stage (this work) | 768 refine pts @ 0 dBm | 922 | 22 mG (19-24) |
| Two-stage, no GP (ablation) | 96 refine pts @ 0 dBm | 250 | 115 mG (100-134) |
| Two-stage, no GP (ablation) | 160 refine pts @ 0 dBm | 314 | 67 mG (59-74) |
| Two-stage, no GP (ablation) | 256 refine pts @ 0 dBm | 410 | 50 mG (43-57) |
| Two-stage, no GP (ablation) | 384 refine pts @ 0 dBm | 538 | 40 mG (35-45) |
| Two-stage, no GP (ablation) | 576 refine pts @ 0 dBm | 730 | 31 mG (27-35) |
| Two-stage, no GP (ablation) | 768 refine pts @ 0 dBm | 922 | 28 mG (24-32) |
| Uniform sweep, best power | 1.4 MHz step @ +10 dBm | 329 | 77 mG (67-88) |
| Uniform sweep, best power | 1.0 MHz step @ +10 dBm | 461 | 58 mG (53-67) |
| Uniform sweep, best power | 0.7 MHz step @ +10 dBm | 658 | 48 mG (40-54) |
| Uniform sweep, best power | 0.5 MHz step @ +0 dBm | 921 | 41 mG (38-48) |
| Uniform sweep, best power | 0.35 MHz step @ +6 dBm | 1315 | 33 mG (29-39) |
| Uniform sweep, best power | 0.25 MHz step @ +8 dBm | 1841 | 27 mG (23-32) |

GP vs. the same budget spread uniformly over the same windows (paired by sample, Wilcoxon signed-rank):

| points | GP median | no-GP median | GP better in | p |
|---|---|---|---|---|
| 250 | 122 mG | 115 mG | 46% | 0.132 |
| 314 | 64 mG | 67 mG | 53% | 0.781 |
| 410 | 37 mG | 50 mG | 61% | 0.041 |
| 538 | 31 mG | 40 mG | 61% | 0.036 |
| 730 | 27 mG | 31 mG | 54% | 0.970 |
| 922 | 22 mG | 28 mG | 62% | 0.005 |

All configurations:

| method | setting | points | success | median worst-line error | median / p90 field error |
|---|---|---|---|---|---|
| GP two-stage (this work) | 96 refine pts @ 0 dBm | 250 | 34% | 769 kHz | 122 / 455 mG |
| GP two-stage (this work) | 160 refine pts @ 0 dBm | 314 | 61% | 338 kHz | 64 / 210 mG |
| GP two-stage (this work) | 256 refine pts @ 0 dBm | 410 | 85% | 207 kHz | 37 / 91 mG |
| GP two-stage (this work) | 384 refine pts @ 0 dBm | 538 | 94% | 194 kHz | 31 / 75 mG |
| GP two-stage (this work) | 576 refine pts @ 0 dBm | 730 | 92% | 137 kHz | 27 / 66 mG |
| GP two-stage (this work) | 768 refine pts @ 0 dBm | 922 | 94% | 119 kHz | 22 / 54 mG |
| Two-stage, no GP (ablation) | 96 refine pts @ 0 dBm | 250 | 41% | 554 kHz | 115 / 303 mG |
| Two-stage, no GP (ablation) | 160 refine pts @ 0 dBm | 314 | 65% | 391 kHz | 67 / 177 mG |
| Two-stage, no GP (ablation) | 256 refine pts @ 0 dBm | 410 | 82% | 259 kHz | 50 / 102 mG |
| Two-stage, no GP (ablation) | 384 refine pts @ 0 dBm | 538 | 90% | 203 kHz | 40 / 78 mG |
| Two-stage, no GP (ablation) | 576 refine pts @ 0 dBm | 730 | 94% | 156 kHz | 31 / 61 mG |
| Two-stage, no GP (ablation) | 768 refine pts @ 0 dBm | 922 | 95% | 152 kHz | 28 / 54 mG |
| Uniform sweep, best power | 1.4 MHz step @ +10 dBm | 329 | 72% | 378 kHz | 77 / 191 mG |
| Uniform sweep, best power | 1.0 MHz step @ +10 dBm | 461 | 80% | 319 kHz | 58 / 120 mG |
| Uniform sweep, best power | 0.7 MHz step @ +10 dBm | 658 | 90% | 239 kHz | 48 / 120 mG |
| Uniform sweep, best power | 0.5 MHz step @ +0 dBm | 921 | 84% | 266 kHz | 41 / 179 mG |
| Uniform sweep, best power | 0.35 MHz step @ +6 dBm | 1315 | 95% | 195 kHz | 33 / 70 mG |
| Uniform sweep, best power | 0.25 MHz step @ +8 dBm | 1841 | 97% | 154 kHz | 27 / 72 mG |
