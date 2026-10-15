Seeds 0-29 (30 samples per configuration).

Target (median |dB| <= 40 mG with >= 90% success): GP two-stage needs 410 points, the best uniform sweep 921 (55% fewer).

| method | setting | points | success | median worst-line error | median / p90 field error |
|---|---|---|---|---|---|
| GP two-stage (this work) | 96 refine pts @ 0 dBm | 250 | 37% | 644 kHz | 127 / 419 mG |
| GP two-stage (this work) | 160 refine pts @ 0 dBm | 314 | 77% | 297 kHz | 57 / 92 mG |
| GP two-stage (this work) | 256 refine pts @ 0 dBm | 410 | 93% | 205 kHz | 35 / 89 mG |
| GP two-stage (this work) | 384 refine pts @ 0 dBm | 538 | 97% | 197 kHz | 30 / 61 mG |
| GP two-stage (this work) | 576 refine pts @ 0 dBm | 730 | 100% | 111 kHz | 20 / 30 mG |
| GP two-stage (this work) | 768 refine pts @ 0 dBm | 922 | 100% | 117 kHz | 23 / 39 mG |
| Two-stage, no GP (ablation) | 96 refine pts @ 0 dBm | 250 | 47% | 522 kHz | 132 / 429 mG |
| Two-stage, no GP (ablation) | 160 refine pts @ 0 dBm | 314 | 77% | 333 kHz | 65 / 114 mG |
| Two-stage, no GP (ablation) | 256 refine pts @ 0 dBm | 410 | 80% | 264 kHz | 43 / 92 mG |
| Two-stage, no GP (ablation) | 384 refine pts @ 0 dBm | 538 | 97% | 209 kHz | 41 / 67 mG |
| Two-stage, no GP (ablation) | 576 refine pts @ 0 dBm | 730 | 97% | 142 kHz | 29 / 56 mG |
| Two-stage, no GP (ablation) | 768 refine pts @ 0 dBm | 922 | 97% | 134 kHz | 22 / 51 mG |
| Uniform sweep, best power | 1.4 MHz step @ +4 dBm | 329 | 63% | 430 kHz | 99 / 184 mG |
| Uniform sweep, best power | 1.0 MHz step @ +4 dBm | 461 | 80% | 324 kHz | 60 / 105 mG |
| Uniform sweep, best power | 0.7 MHz step @ +4 dBm | 658 | 90% | 236 kHz | 46 / 84 mG |
| Uniform sweep, best power | 0.5 MHz step @ +0 dBm | 921 | 90% | 233 kHz | 36 / 98 mG |
| Uniform sweep, best power | 0.35 MHz step @ +4 dBm | 1315 | 97% | 156 kHz | 38 / 76 mG |
| Uniform sweep, best power | 0.25 MHz step @ +4 dBm | 1841 | 100% | 161 kHz | 23 / 52 mG |
