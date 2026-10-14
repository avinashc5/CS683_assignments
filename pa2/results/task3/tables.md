**IPC**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (no pf, LRU) | 0.2322 | 0.2511 | 0.3821 | 0.2885 |
| SPP | 0.2344 | 0.2548 | 0.3873 | 0.2922 |
| Dynamic IRRIP | 0.2320 | 0.2588 | 0.3828 | 0.2912 |
| SPP + Dyn IRRIP (pf-unaware) | 0.2341 | 0.2628 | 0.3882 | 0.2951 |
| SPP + Dyn IRRIP (pf-aware) | 0.2341 | 0.2629 | 0.3883 | 0.2951 |
| PF insert RRPV 1 | 0.2340 | 0.2624 | 0.3875 | 0.2947 |
| PF insert RRPV 2 | 0.2341 | 0.2625 | 0.3875 | 0.2947 |
| PF insert RRPV 3 (final) | 0.2341 | 0.2629 | 0.3883 | 0.2951 |

**Speedup over baseline**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (no pf, LRU) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SPP | 1.0093 | 1.0148 | 1.0138 | 1.0126 |
| Dynamic IRRIP | 0.9992 | 1.0304 | 1.0019 | 1.0104 |
| SPP + Dyn IRRIP (pf-unaware) | 1.0081 | 1.0466 | 1.0162 | 1.0235 |
| SPP + Dyn IRRIP (pf-aware) | 1.0081 | 1.0467 | 1.0164 | 1.0236 |
| PF insert RRPV 1 | 1.0079 | 1.0450 | 1.0143 | 1.0223 |
| PF insert RRPV 2 | 1.0082 | 1.0452 | 1.0144 | 1.0225 |
| PF insert RRPV 3 (final) | 1.0081 | 1.0467 | 1.0164 | 1.0236 |

**L1D load MPKI**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (no pf, LRU) | 10.679 | 13.534 | 3.403 | 9.205 |
| SPP | 9.649 | 12.900 | 3.013 | 8.520 |
| Dynamic IRRIP | 10.679 | 13.544 | 3.403 | 9.209 |
| SPP + Dyn IRRIP (pf-unaware) | 9.666 | 12.916 | 3.017 | 8.533 |
| SPP + Dyn IRRIP (pf-aware) | 9.663 | 12.916 | 3.018 | 8.532 |
| PF insert RRPV 1 | 9.663 | 12.915 | 3.013 | 8.530 |
| PF insert RRPV 2 | 9.659 | 12.912 | 3.020 | 8.530 |
| PF insert RRPV 3 (final) | 9.663 | 12.916 | 3.018 | 8.532 |

**L1D prefetch accuracy (%)**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (no pf, LRU) | – | – | – | – |
| SPP | 52.01 | 60.02 | 36.44 | 49.49 |
| Dynamic IRRIP | – | – | – | – |
| SPP + Dyn IRRIP (pf-unaware) | 52.42 | 59.33 | 36.08 | 49.28 |
| SPP + Dyn IRRIP (pf-aware) | 52.52 | 59.89 | 36.00 | 49.47 |
| PF insert RRPV 1 | 52.84 | 59.75 | 36.17 | 49.59 |
| PF insert RRPV 2 | 52.21 | 59.59 | 36.08 | 49.29 |
| PF insert RRPV 3 (final) | 52.52 | 59.89 | 36.00 | 49.47 |

**L1D load coverage (%)**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (no pf, LRU) | 0.00 | 0.00 | 0.00 | 0.00 |
| SPP | 9.65 | 4.69 | 11.48 | 8.60 |
| Dynamic IRRIP | -0.00 | -0.08 | 0.00 | -0.03 |
| SPP + Dyn IRRIP (pf-unaware) | 9.49 | 4.56 | 11.34 | 8.46 |
| SPP + Dyn IRRIP (pf-aware) | 9.51 | 4.56 | 11.32 | 8.47 |
| PF insert RRPV 1 | 9.52 | 4.57 | 11.47 | 8.52 |
| PF insert RRPV 2 | 9.55 | 4.60 | 11.27 | 8.47 |
| PF insert RRPV 3 (final) | 9.51 | 4.56 | 11.32 | 8.47 |

**L2C load MPKI**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (no pf, LRU) | 17.730 | 27.923 | 3.959 | 16.538 |
| SPP | 16.887 | 27.361 | 3.456 | 15.901 |
| Dynamic IRRIP | 17.679 | 21.272 | 3.931 | 14.294 |
| SPP + Dyn IRRIP (pf-unaware) | 16.891 | 20.675 | 3.393 | 13.653 |
| SPP + Dyn IRRIP (pf-aware) | 16.890 | 20.663 | 3.391 | 13.648 |
| PF insert RRPV 1 | 16.902 | 20.749 | 3.432 | 13.694 |
| PF insert RRPV 2 | 16.885 | 20.739 | 3.430 | 13.685 |
| PF insert RRPV 3 (final) | 16.890 | 20.663 | 3.391 | 13.648 |

**L2C instruction load MPKI**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (no pf, LRU) | 10.384 | 19.391 | 1.126 | 10.300 |
| SPP | 10.414 | 19.437 | 1.143 | 10.331 |
| Dynamic IRRIP | 10.525 | 10.396 | 0.801 | 7.241 |
| SPP + Dyn IRRIP (pf-unaware) | 10.564 | 10.411 | 0.807 | 7.261 |
| SPP + Dyn IRRIP (pf-aware) | 10.563 | 10.393 | 0.806 | 7.254 |
| PF insert RRPV 1 | 10.574 | 10.498 | 0.859 | 7.310 |
| PF insert RRPV 2 | 10.570 | 10.497 | 0.854 | 7.307 |
| PF insert RRPV 3 (final) | 10.563 | 10.393 | 0.806 | 7.254 |

**L2C data load MPKI**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (no pf, LRU) | 7.346 | 8.533 | 2.833 | 6.237 |
| SPP | 6.472 | 7.924 | 2.313 | 5.570 |
| Dynamic IRRIP | 7.154 | 10.876 | 3.130 | 7.053 |
| SPP + Dyn IRRIP (pf-unaware) | 6.327 | 10.264 | 2.586 | 6.392 |
| SPP + Dyn IRRIP (pf-aware) | 6.327 | 10.270 | 2.585 | 6.394 |
| PF insert RRPV 1 | 6.328 | 10.251 | 2.574 | 6.384 |
| PF insert RRPV 2 | 6.315 | 10.242 | 2.576 | 6.378 |
| PF insert RRPV 3 (final) | 6.327 | 10.270 | 2.585 | 6.394 |
