**IPC**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (LRU) | 0.2322 | 0.2513 | 0.3819 | 0.2885 |
| SRRIP | 0.2322 | 0.2518 | 0.3816 | 0.2885 |
| I-RRIP | 0.2319 | 0.2597 | 0.3826 | 0.2914 |

**Speedup over baseline**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (LRU) | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| SRRIP | 0.9999 | 1.0022 | 0.9993 | 1.0004 |
| I-RRIP | 0.9986 | 1.0334 | 1.0020 | 1.0112 |

**L2C load MPKI**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (LRU) | 17.703 | 27.560 | 3.960 | 16.408 |
| SRRIP | 17.619 | 26.012 | 4.037 | 15.889 |
| I-RRIP | 18.578 | 20.104 | 3.928 | 14.204 |

**L2C instruction load MPKI**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (LRU) | 10.368 | 18.892 | 1.124 | 10.128 |
| SRRIP | 10.563 | 17.980 | 1.190 | 9.911 |
| I-RRIP | 9.316 | 9.435 | 0.811 | 6.520 |

**L2C data load MPKI**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| Baseline (LRU) | 7.334 | 8.668 | 2.837 | 6.280 |
| SRRIP | 7.056 | 8.032 | 2.846 | 5.978 |
| I-RRIP | 9.263 | 10.669 | 3.118 | 7.683 |

**Stage 1 (one knob at a time), speedup over baseline**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| SRRIP (all knobs off) | 0.9999 | 1.0022 | 0.9993 | 1.0004 |
| Insertion only (instr RRPV 0) | 1.0000 | 1.0130 | 0.9999 | 1.0043 |
| Promotion only (data decrement) | 1.0008 | 1.0085 | 0.9995 | 1.0029 |
| Eviction only (prefer data) | 0.9998 | 1.0185 | 0.9993 | 1.0058 |
| Eviction only (age data faster) | 1.0004 | 1.0088 | 0.9994 | 1.0029 |
| Insertion + promotion (I+P) | 0.9998 | 1.0176 | 0.9999 | 1.0058 |
| I+P + prefer data | 0.9994 | 1.0229 | 1.0000 | 1.0074 |
| I+P + age data faster | 0.9993 | 1.0196 | 1.0001 | 1.0063 |
| I+P + both eviction options | 0.9983 | 1.0251 | 1.0000 | 1.0077 |
| I+P, instr insert RRPV 1 | 1.0003 | 1.0154 | 0.9999 | 1.0052 |
| I+P, data insert RRPV 3 | 0.9968 | 1.0291 | 1.0013 | 1.0089 |
| I+P, data insert bimodal | 0.9971 | 1.0283 | 1.0008 | 1.0087 |
| I+P, data not promoted | 0.9978 | 1.0202 | 1.0000 | 1.0059 |
| I+P, instr decrement on hit | 1.0001 | 1.0176 | 1.0001 | 1.0059 |

**Stage 2 (full factorial of the knobs that mattered), speedup over baseline**

| Config | Trace1 | Trace2 | Trace3 | Mean |
|---|---|---|---|---|
| instr 1 / data RRPV 3 / prefer on / age on | 0.9974 | 1.0348 | 1.0020 | 1.0113 |
| instr 1 / data RRPV 3 / prefer on / age off | 0.9986 | 1.0334 | 1.0020 | 1.0112 |
| instr 1 / data bimodal / prefer on / age on | 0.9980 | 1.0343 | 1.0015 | 1.0111 |
| instr 1 / data RRPV 3 / prefer off / age on | 0.9982 | 1.0337 | 1.0018 | 1.0111 |
| instr 1 / data bimodal / prefer on / age off | 0.9990 | 1.0332 | 1.0014 | 1.0111 |
| instr 1 / data bimodal / prefer off / age on | 0.9986 | 1.0327 | 1.0012 | 1.0107 |
| instr 1 / data RRPV 3 / prefer off / age off | 0.9993 | 1.0313 | 1.0017 | 1.0107 |
| instr 1 / data bimodal / prefer off / age off | 0.9998 | 1.0305 | 1.0012 | 1.0104 |
| instr 0 / data RRPV 3 / prefer on / age off | 0.9961 | 1.0300 | 1.0014 | 1.0091 |
| instr 0 / data RRPV 3 / prefer off / age off | 0.9968 | 1.0291 | 1.0013 | 1.0089 |
| instr 0 / data bimodal / prefer on / age on | 0.9954 | 1.0299 | 1.0011 | 1.0087 |
| instr 0 / data bimodal / prefer off / age on | 0.9963 | 1.0291 | 1.0011 | 1.0087 |
| instr 0 / data bimodal / prefer on / age off | 0.9964 | 1.0290 | 1.0010 | 1.0087 |
| instr 0 / data bimodal / prefer off / age off | 0.9971 | 1.0283 | 1.0008 | 1.0087 |
| instr 0 / data RRPV 3 / prefer off / age on | 0.9955 | 1.0297 | 1.0011 | 1.0087 |
| instr 0 / data RRPV 3 / prefer on / age on | 0.9946 | 1.0300 | 1.0014 | 1.0085 |
