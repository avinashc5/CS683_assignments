# Task 2 — Status: what's done, what's left

Tracks the **assignment PDF** (`CS683 PA1 2026.pdf`, Task 2 = 9 pts) rather than
`HOW_TO_RUN.md`, which is the repo's build runbook and is narrower than the PDF.

Measurement environment for every number below:
i5-11400H (Tiger Lake-H, 6 P-cores, AVX-512F/DQ/BW/VL), pinned `taskset -c 3`,
`performance` governor, `perf_event_paranoid=-1`, machine otherwise idle.

---

## 1. Done

### Kernels + graded harness

All three graded kernels implemented and numerically correct. `make && taskset -c 3 ./bin/matmul`:

| stage | median speedup (3 reps) |
|---|---|
| simd | 34.72x |
| prefetch | 24.59x |
| optimized | **27.74x** |

**Score: 100/100** (30/30 correctness + 70/70 speedup tier). Baseline spread across
reps 0.3%, so this is reproducible. Edge case `./bin/matmul optimized 1 4096 64`
(llama token-gen shape) is correct at 0.92x — served by the scalar tail, as designed.

### Data collected (all in `bench/`)

| file | covers | PDF item |
|---|---|---|
| `size_sweep_results.txt` | 8 sizes (128–2048) x 3 stages | 2A.1, 2A.3, 2B.2, 2C |
| `prefetch_degree_results.txt` | 12 distances (0–512 floats) @1536 | 2A.4 |
| `prefetch_level_results.txt` | T0 / T1 / T2 / NTA @1536 | 2A.4 |
| `simd_width_results.txt` | 128/256/512-bit x 5 sizes + instr counts | 2B.1, 2B.2, 2B.3 |
| `perf_stage_results.txt` | instr/L1D/L2/LLC per stage @1024 | 2A.2 (partial) |

### Key findings already in hand

- **Size crossover (2A.1):** `simd` peaks 88.5 GF/s @384 then collapses to 36.8 @1536;
  `prefetch`/`optimized` stay flat ~55/~63 GF/s across 128→2048. `optimized` goes from
  19% *slower* than `simd` @1024 to 72% *faster* @1536.
- **Cache mechanism (2A.2):** blocking cuts L2 misses **13.1x** (17.1M → 1.31M) while
  *raising* L1D misses (17.1M → 35.4M; prefetch instructions pull lines in). This is why
  the size crossover happens. `simd` also cuts instructions **28.2x** vs naive, and IPC
  rises 1.52 → 2.19.
- **Cache fill level (2A.4):** T0 57.8, T1 55.9, T2 56.1, **NTA 15.8 GF/s**. The 3.7x
  NTA cliff is the strongest single result — non-temporal bypass destroys exactly the
  panel reuse that blocking creates.
- **Prefetch distance (2A.4):** flat, 25.2–26.8x across 0→512, ~6% spread. Honest
  reading: *no significant effect* on this core. **See gap #1 — this needs the hardware
  prefetcher experiment to be a defensible conclusion rather than a shrug.**
- **SIMD width (2B):** AVX-512 112 GF/s vs AVX2 91 @512. Kernel-only instruction counts
  @2048 are 24.7G / 12.6G / 6.5G for SSE / AVX2 / AVX-512 — almost exactly 4:2:1, ideal
  width scaling.

### llama.cpp integration

`make llama-demo` passes both gates: kernel confirmed used by ggml, output matches stock
**token-for-token**. Timings (median of 3, 901-token prompt, `-t 1`):

| path | stock | student |
|---|---|---|
| prompt eval (batched, fast path) | 3225 tok/s | **3341 tok/s** |
| token gen (M=1, scalar tail) | 1854 tok/s | 1473 tok/s |

Parity-to-slightly-ahead on the batched matmuls; ~21% behind on M=1, which matches the
0.92x microbenchmark edge case.

### Housekeeping

- `.gitignore` now excludes `task2/llama/llama.cpp/` (158 MB clone) and `task2/llama/models/`.
- `bench/perf_stage.cpp` added (ungraded, not submitted) — see note in §3.

---

## 2. Remaining

Ordered by scoring impact.

### Gap 1 — Hardware prefetcher enable/disable study  (2A item 5, REQUIRED, not started)

PDF: *"Enable and disable hardware prefetchers, and analyze their impact on the
performance of the SGEMM operation."* Nothing in `bench/` does this.

**Why this matters most:** plot 2 (prefetch distance) came out flat. Without this
experiment the report can only say "software prefetch distance didn't matter." With it,
the claim becomes "the L2 hardware prefetcher already covers the sequential panel access
— disable it and the software prefetch distance curve reappears." That converts the
weakest result into a real finding.

Mechanism: MSR `0x1a4`, bits 0–3 = disable flags for the four prefetchers
(0 = L2 HW, 1 = L2 adjacent-line, 2 = DCU, 3 = DCU IP). `rdmsr`/`wrmsr` are installed and
the `msr` module is loaded. **Needs sudo.** Restore to `0x0` when finished.

Suggested runs: `prefetch` and `optimized` across sizes with all HW prefetchers ON vs
OFF, plus a re-run of the distance sweep with them OFF.

### Gap 2 — `sw_prefetch_access` counters  (2A, named explicitly, not started)

PDF: *"use the `sw_prefetch_access` event in perf to track the number of software
prefetch requests issued."* Available here (`.nta`, `.t0`, `.t1`, `.prefetchw`).
Should be recorded per prefetch distance and per fill level to confirm the kernel is
issuing the prefetches you think it is.

### Gap 3 — Cache metrics across sizes and parameters  (2A.2 / 2C, partially done)

`perf_stage_results.txt` covers all four stages but **only at 1024³**. The PDF asks for
L1D/L2/LLC trends *"for different matrix sizes and different software prefetching
parameters."* Needs extending to a size sweep (and ideally a distance sweep) rather than
one point.

### Gap 4 — Plot PNGs not rendered

matplotlib is not installed anywhere on this machine (neither system python nor
`~/.venv`). All underlying data exists as `.txt`. PDF names the required files:

- `plots/task2_speedup_vs_size.png`
- `plots/task2_prefetch_degree.png`
- `plots/task2_prefetch_level.png`
- plus a SIMD-width plot and the 2C summary plot (best of prefetch / SIMD / both vs size)

The 2C summary plot needs no new measurements — it's a re-plot of `size_sweep_results.txt`.

### Gap 5 — `task2_report.pdf` not written

Needs plots 1–4 + the 2C summary, and answers to the PDF's explicit questions:

- 2A: trend of speedup vs matrix size? best prefetch distance? best cache fill level?
- 2B: trends across size × width? which width gives max speedup?
- 2C: all of the above with both optimizations on, + the final comparison plot.

Table 2.1 in the PDF gives the expected format for the 2B table (No-SIMD vs SIMD
instructions and execution time, per size, per width).

### Gap 6 — Submission tarball (deliberately not built)

Per PDF: `rollno1_rollno2_..._pa1.tar.gz`, roll numbers ascending, one member submits.
Structure: `task2/src/{matmul_simd,matmul_prefetch,matmul_optimized}.cpp` +
`task2/task2_report.pdf`, PNGs in top-level `plots/`. **`bench/` must stay out.**

**Deadline: 07/09/2026.** Commits pushed after it are not evaluated.

---

## 3. Notes / caveats for whoever writes the report

**Do not use any "53.7x" figure.** An earlier run was contaminated: two benchmark chains
were timesharing core 3, which halved the `naive` baseline and inflated every ratio. All
numbers in this file come from a single-chain run on an idle machine. Always check
nothing else is running before timing.

**The PDF's own perf command doesn't measure what it looks like it measures.**
`./bin/matmul <stage>` always runs `matmul_naive` too, as the correctness reference, so
`perf stat -- ./bin/matmul optimized ...` counts naive + optimized, and naive dominates —
all four stages come out within 4% of each other (~5.8 s each). `bench/perf_stage.cpp`
was added to call exactly one kernel so the counters separate.

`perf_stage` now uses **`perf_event_open(2)` region counting**: it opens the counters
itself and brackets only the kernel call with `IOC_ENABLE`/`IOC_DISABLE`, so setup is
never counted rather than counted-then-cancelled. Run it directly, no `perf stat`
wrapper:

```
taskset -c 3 ./bench/bin/perf_stage optimized 1024 1024 1024 1234 1 1
#                                    stage     M    N    K  seed reps flush
```

This replaced an earlier version that called the kernel once and was measured with
`perf stat` over the whole process. That version silently included its own setup
(allocation + `mt19937` fill, ~1.2e8 instructions) — negligible against naive (1.8%) but
a 35–50% inflation on the fast kernels. It was caught by cross-checking against
`perf record`, not by inspection. Four methods now agree on simd within 1.4%:
region counting 2.2886e8, rep-subtraction 2.2923e8, `perf record` sampling 2.2940e8,
harness subtraction 2.32e8.

`flush=1` streams a 64 MiB buffer before the region so the call starts cold. Measured
cold vs warm differ by <2% on every counter here, so cache carryover is not a concern at
1024³ — but the option exists for the Gap 3 size sweep, where smaller matrices will fit
in cache and the distinction starts to matter.

Alternative needing no extra file: `perf record -e <event> -- ./bin/matmul <stage> ...`
then `perf report --sort symbol`, which attributes per function (naive ~96.5%, simd
~2.9%). Sampled rather than exact, but it needs no code change and makes a good
cross-check — it is what caught the setup-overhead error above.

The same caveat applies to the `instructions` column in `simd_width_results.txt` — those
totals include the naive reference run. Subtract the `naive (ref)` row of the same size
to get kernel-only counts (that's how the 4:2:1 ratio above was derived).

**`bench/` is ungraded and must not be submitted** — including `perf_stage.cpp`.
