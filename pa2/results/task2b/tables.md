**IPC**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (LRU) | 0.2322 | 0.2513 | 0.3819 | 0.2885 |
| SRRIP | 0.2322 | 0.2518 | 0.3816 | 0.2885 |
| I-RRIP (Task 2A) | 0.2319 | 0.2597 | 0.3826 | 0.2914 |
| I-RRIP mode, static | 0.2317 | 0.2600 | 0.3826 | 0.2914 |
| Dynamic I-RRIP | 0.2321 | 0.2597 | 0.3825 | 0.2914 |

**Speedup over baseline**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (LRU) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SRRIP | 0.9999 | 1.0022 | 0.9993 | 1.0004 |
| I-RRIP (Task 2A) | 0.9986 | 1.0334 | 1.0020 | 1.0112 |
| I-RRIP mode, static | 0.9974 | 1.0348 | 1.0020 | 1.0113 |
| Dynamic I-RRIP | 0.9993 | 1.0335 | 1.0017 | 1.0114 |

**L2C load MPKI**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (LRU) | 17.703 | 27.560 | 3.960 | 16.408 |
| SRRIP | 17.619 | 26.012 | 4.037 | 15.889 |
| I-RRIP (Task 2A) | 18.578 | 20.104 | 3.928 | 14.204 |
| I-RRIP mode, static | 18.700 | 19.927 | 3.931 | 14.186 |
| Dynamic I-RRIP | 17.665 | 20.108 | 3.937 | 13.903 |

**L2C instruction load MPKI**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (LRU) | 10.368 | 18.892 | 1.124 | 10.128 |
| SRRIP | 10.563 | 17.980 | 1.190 | 9.911 |
| I-RRIP (Task 2A) | 9.316 | 9.435 | 0.811 | 6.520 |
| I-RRIP mode, static | 9.242 | 9.029 | 0.792 | 6.354 |
| Dynamic I-RRIP | 10.528 | 9.296 | 0.804 | 6.876 |

**L2C data load MPKI**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (LRU) | 7.334 | 8.668 | 2.837 | 6.280 |
| SRRIP | 7.056 | 8.032 | 2.846 | 5.978 |
| I-RRIP (Task 2A) | 9.263 | 10.669 | 3.118 | 7.683 |
| I-RRIP mode, static | 9.459 | 10.897 | 3.139 | 7.832 |
| Dynamic I-RRIP | 7.137 | 10.811 | 3.133 | 7.027 |

**Dynamic I-RRIP selector (after warmup)**

| Trace | Demand misses, SRRIP leaders | Demand misses, I-RRIP leaders | Final PSEL (of 1023) | Follower updates in I-RRIP mode | Mode switches |
|---|---|---|---|---|---|
| Trace1 | 13403 | 15120 | 0 | 0.0% | 0 |
| Trace2 | 19435 | 16414 | 1017 | 100.0% | 0 |
| Trace3 | 3202 | 3006 | 1020 | 100.0% | 0 |

**Sensitivity (speedup over baseline); last column: share of Trace1 follower updates in I-RRIP mode**

| Leader sets / spread / PSEL | Trace1 | Trace2 | Trace3 | Mean | Trace1 I-RRIP share |
|---|---|---|---|---|---|
| 8 / contiguous / 8-bit | 0.9997 | 1.0343 | 1.0019 | 1.0118 | 6.6% |
| 8 / contiguous / 10-bit | 0.9998 | 1.0343 | 1.0019 | 1.0119 | 0.0% |
| 8 / contiguous / 12-bit | 0.9998 | 1.0343 | 1.0019 | 1.0119 | 0.0% |
| 8 / hashed / 8-bit | 0.9973 | 1.0346 | 1.0020 | 1.0112 | 100.0% |
| 8 / hashed / 10-bit | 0.9973 | 1.0346 | 1.0020 | 1.0112 | 100.0% |
| 8 / hashed / 12-bit | 0.9973 | 1.0346 | 1.0020 | 1.0112 | 100.0% |
| 16 / contiguous / 8-bit | 0.9994 | 1.0328 | 1.0019 | 1.0112 | 23.4% |
| 16 / contiguous / 10-bit | 0.9999 | 1.0328 | 1.0019 | 1.0114 | 2.5% |
| 16 / contiguous / 12-bit | 0.9999 | 1.0328 | 1.0019 | 1.0114 | 0.0% |
| 16 / hashed / 8-bit | 0.9974 | 1.0343 | 1.0019 | 1.0110 | 100.0% |
| 16 / hashed / 10-bit | 0.9974 | 1.0343 | 1.0019 | 1.0110 | 100.0% |
| 16 / hashed / 12-bit | 0.9974 | 1.0343 | 1.0019 | 1.0110 | 100.0% |
| 32 / contiguous / 8-bit | 0.9986 | 1.0329 | 0.9986 | 1.0099 | 58.6% |
| 32 / contiguous / 10-bit | 0.9990 | 1.0329 | 1.0019 | 1.0111 | 60.3% |
| 32 / contiguous / 12-bit | 0.9990 | 1.0329 | 1.0019 | 1.0111 | 60.3% |
| 32 / hashed / 8-bit | 0.9993 | 1.0335 | 1.0017 | 1.0114 | 2.6% |
| 32 / hashed / 10-bit | 0.9993 | 1.0335 | 1.0017 | 1.0114 | 0.0% |
| 32 / hashed / 12-bit | 0.9993 | 1.0335 | 1.0017 | 1.0114 | 0.0% |
| Upper bound (best static per trace) | 0.9999 | 1.0348 | 1.0020 | 1.0121 | |

