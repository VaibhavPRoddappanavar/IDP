# Canonical Results Ledger

This ledger separates measured, derived, simulated, and unavailable values.
Only measured and derived values are used as hardware claims in the revised
manuscript.

## Measured Raspberry Pi 5 results

| Quantity | Value | Evidence | Classification |
|---|---:|---|---|
| Phase 0 duration | 213 s | `phase_0_baseline/output/logs_rpi5.csv` | Measured |
| Phase 0 processed FPS | 3.31 | Same log | Measured |
| Phase 0 latency mean | 301.8 ms | Same log | Measured |
| Phase 0 latency P95 | 388.5 ms | Same log | Derived |
| Phase 0 CPU | 53.2% | Same log | Measured |
| Phase 0 RAM | 29.5% | Same log | Measured |
| Phase 0 mean temperature | 56.6 C | Same log | Measured |
| Phase 0 maximum temperature | 60.6 C | Same log | Measured |
| Phase 0 thermal throttle event | None observed | Same run; no throttle flag artifact | Observed, duration-bounded |
| Phase 1 M0 processed FPS | 3.08 | Raspberry Pi-only rows | Measured |
| Phase 1 M1 processed FPS | 4.98 | Raspberry Pi-only rows | Measured |
| Phase 1 M2 processed FPS | 7.09 | Raspberry Pi-only rows | Measured |
| Phase 1 M0/M1/M2 source coverage | 3.08/9.96/21.28 FPS | Derived from skip configuration | Derived |
| Phase 2-3 processed FPS | 5.02 | Raspberry Pi log | Measured |
| Phase 2-3 source coverage | 8.5 FPS | Raspberry Pi log | Derived |
| Phase 2-3 maximum temperature | 60.0 C | Raspberry Pi log | Measured |
| Phase 4 transitions | 3,033 | Parquet dataset | Measured |

## Preliminary offline policy comparison

| Quantity | Value | Classification |
|---|---:|---|
| Policy-selected size in archived accuracy run | 320 px | Measured artifact |
| Pixel-area reduction from 640 to 320 | 75% | Derived |
| Baseline precision/recall/F1 | 61.18/70.27/65.41% | Exploratory artifact |
| Policy-condition precision/recall/F1 | 67.24/52.70/59.09% | Exploratory artifact |
| Telemetry input | Simulated state schedule | Simulated |
| Standard AP@0.5 | Not available | Not evaluated |
| Small-object metrics | Not available | Not evaluated |

## Explicitly withdrawn claims

The following values are not supported by committed source records and are
not used in the revised manuscript: 6.3/10.4 FPS, 169.6/100.4 ms latency,
99.9/99.8% CPU, 75.1/74.2% RAM, 400 px average size, 61% pixel saving, and
the corresponding learned-controller superiority claim.
