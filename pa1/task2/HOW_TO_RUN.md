# Task 2 — How to Run (build, benchmark, plots, llama.cpp)

Everything needed to reproduce the Task 2 numbers and the report plots. The three graded
kernels (`src/matmul_simd.cpp`, `src/matmul_prefetch.cpp`, `src/matmul_optimized.cpp`) are
implemented; `bench/` holds the **ungraded** harnesses that produce the report plots.

---

## 0. Requirements

| Need | For | Notes |
|------|-----|-------|
| x86‑64 CPU with **AVX2 + FMA** | graded microbenchmark, `make llama-demo` | `grep -o 'avx2\|fma' /proc/cpuinfo \| sort -u` |
| CPU with **AVX‑512F** | plot 4's 512‑bit row only | optional; `grep avx512f /proc/cpuinfo`. Without it the 512‑bit row prints `n/a`. |
| `g++` (C++17), `make` | everything | gcc ≥ 9; gcc 11 is fine |
| `git`, `cmake`, `python3`, `curl`, network | `make llama-demo` only | cmake ≥ 3.14 |
| `perf` with `kernel.perf_event_paranoid ≤ 1` | instruction‑count / cache‑miss columns | `sudo sysctl kernel.perf_event_paranoid=1` (session‑only) |

**Pinning & frequency.** The lab machines are hybrid (P + E cores) and clock‑unstable.
Pin every measurement to one P‑core and, if you can, fix the governor:

```bash
sudo cpupower frequency-set -g performance     # optional, big noise reduction
taskset -c 0 ./bin/matmul                      # core 0 is a P-core on the 12450H
```

All scripts below accept `PIN="taskset -c <cpu>"` to override the core.

---

## 1. Graded microbenchmark

```bash
cd task2
make
taskset -c 0 ./bin/matmul                 # smoke test + graded 1024^3 table + score
```

The score line at the bottom is the autograder's internal 100‑pt scale: 10 pts each for
`simd`/`prefetch`/`optimized` being numerically correct (30), plus a speedup tier on
`optimized` vs `naive` (`5×→20, 10×→40, 18×→55, 25×→70`). A correct, unthrottled run on an
Alder‑Lake‑class P‑core lands `optimized` at ~30–35× → 100/100. Under thermal throttling
the ratio sags to ~20× (the machine, not the kernel — `naive` and `optimized` scale
differently with frequency); re‑run pinned with the performance governor for a clean
number.

### Checking one stage while developing

```bash
./bin/matmul simd                         # naive vs simd, default 1024^3
./bin/matmul prefetch 1024 1024 1024      # custom M N K
./bin/matmul all 2048 2048 2048           # every stage, no score (watch prefetch/optimized pull ahead of simd)
./bin/matmul optimized 1 4096 64          # M=1 edge case (llama token-gen shape)
./bin/matmul help
```

Correctness is checked against `matmul_naive` at a relative tolerance for every call.

### What each kernel does

- **`matmul_simd.cpp`** — 4×2 register‑blocked AVX2 micro‑kernel. The dot‑product over `K`
  (the only unit‑stride axis in the NT layout) is vectorised into 8 independent `__m256`
  FMA accumulators; the 2 B‑vectors are reused across the 4 A‑rows. No cache blocking, so
  it goes memory‑bound once the panels leave cache.
- **`matmul_prefetch.cpp`** — same micro‑kernel + `MC×NC` cache blocking (loop order
  `jc→ic→jr→ir`, A/B sub‑panels held in L2) + `_mm_prefetch` ahead of the FMA cursor.
  Compile‑time knobs `MC`, `NC`, `PF_DIST`, `PF_HINT` (see plots 2–3).
- **`matmul_optimized.cpp`** — blocking + 2× K‑unroll + prefetch + tuned block sizes
  (`MMO_MC`, `MMO_NC`, `MMO_PF_DIST`, `MMO_PF_HINT`). This is the graded kernel **and** the
  one injected into llama.cpp.

All three handle arbitrary `M, N, K` and arbitrary `lda/ldb/ldc` (scalar cleanup for the
`M%tile / N%2 / K%vector` edges), so the same code runs in the microbenchmark and inside
ggml.

---

## 2. Report plots

All plot drivers live in `bench/` and write a `*_results.txt` table next to themselves.
They are **ungraded** — nothing in `bench/` is part of the submission.

### Plot 1 — speedup vs matrix size

```bash
bench/run_size_sweep.sh
# tweak the size list / core:
SIZES="128 256 512 1024 2048 3072" PIN="taskset -c 0" bench/run_size_sweep.sh
```

Runs `simd`, `prefetch`, `optimized` vs `naive` at each square size → `size_sweep_results.txt`
(`naive_GF/s`, `stage_GF/s`, `speedup`). Use a range from "fits in L2" to "well past L3"
(≈128 → ≥2048) so the plot shows `prefetch`/`optimized` diverging from `simd` once the
working set stops fitting in cache. **Large sizes are slow** because `naive` is re‑timed
for every point (2048³ ≈ a few minutes); trim `SIZES` if needed.

### Plot 2 — speedup vs prefetch distance (degree)

```bash
bench/run_prefetch_degree.sh
SIZE=1536 DISTS="0 16 32 64 128 256 512" bench/run_prefetch_degree.sh
```

Rebuilds `matmul_prefetch` with `-DPF_DIST=<floats>` for each distance (pinned project
flags otherwise unchanged), times `prefetch` vs `naive` at a fixed size →
`prefetch_degree_results.txt`. `PF_DIST` is in **floats** (64 floats = 4 cache lines).
`DIST=0` ≈ "prefetch the line you're already on" ≈ prefetch disabled — the left end of the
curve. Hold the hint fixed with `HINT=_MM_HINT_T0`.

### Plot 3 — speedup vs prefetch level (locality hint)

```bash
bench/run_prefetch_level.sh
SIZE=1536 DIST=64 bench/run_prefetch_level.sh
```

Rebuilds `matmul_prefetch` with `-DPF_HINT=` each of `_MM_HINT_T0`, `_MM_HINT_T1`,
`_MM_HINT_T2`, `_MM_HINT_NTA` (distance fixed), times `prefetch` vs `naive` →
`prefetch_level_results.txt`. `T0` = into L1/all levels, `T1`/`T2` = skip L1 / into L3,
`NTA` = non‑temporal (bypass, minimise pollution).

### Plot 4 — speedup vs SIMD width (128 / 256 / 512‑bit)

```bash
bench/run_simd_width.sh
SIZES="256 512 1024 2048" bench/run_simd_width.sh
AVX512=0 bench/run_simd_width.sh          # only if the toolchain rejects -mavx512f
```

Builds `bench/bin/bench_simd_width`, which links three self‑contained variants of the
Stage‑1 kernel — identical algorithm, different vector width:

| variant file | width | build flag |
|---|---|---|
| `bench/matmul_simd_sse.cpp` | 128‑bit `__m128` (4 f) | `-msse4.2 -mfma` |
| `bench/matmul_simd_avx2.cpp` | 256‑bit `__m256` (8 f) | `-mavx2 -mfma` |
| `bench/matmul_simd_avx512.cpp` | 512‑bit `__m512` (16 f) | `-mavx512f` |

→ `simd_width_results.txt` (`GFLOP/s`, `speedup`, and `instructions` when perf is usable).
The 512‑bit row is `n/a` on a CPU without AVX‑512F (checked at runtime with
`__builtin_cpu_supports`). Run this one **on the AVX‑512 machine** for the real plot.

Direct use / single width (this is what the script wraps for `perf`):

```bash
cd task2/bench && make
taskset -c 0 ./bin/bench_simd_width all 1024 1024 1024
taskset -c 0 perf stat -e instructions -- ./bin/bench_simd_width avx512 1024 1024 1024
```

### Instruction‑count / cache‑miss columns

Task 2A/2B also ask for `perf` metrics (instructions, L1D/L2/LLC misses). Enable counters:

```bash
sudo sysctl kernel.perf_event_paranoid=1        # 1 is enough for 'instructions'; use -1 for cache events
```

then, per stage / width / size:

```bash
taskset -c 0 perf stat -e instructions,L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses \
    -- ./bin/matmul optimized 1024 1024 1024
```

`run_simd_width.sh` auto‑fills its `instructions` column when paranoid ≤ 1; otherwise it
prints `-`.

---

## 3. Run the kernel inside llama.cpp

```bash
cd task2
make llama-demo            # needs git, cmake, python3, curl, network
```

This clones a pinned llama.cpp (`b5731`), builds it **stock**, then injects
`src/matmul_optimized.cpp` into ggml's single‑threaded F32 `mul_mat`
(`llama/inject.py` + `llama/ggml_student_sgemm.cpp`), rebuilds, downloads a ~1.2 MB F32
model, and runs the same prompt on both builds (`-t 1`, fixed seed). It reports:

- `[ok] student matmul_optimized was actually used by ggml`
- whether the generated text matches the stock build **token‑for‑token** (kernel correct
  inside a real model)
- `tokens/sec` for llama.cpp's native matmul vs yours

Note: token *generation* calls `mul_mat` with `M=1` → served by the kernel's scalar tail
(correct, not the fast path). The **prompt‑eval** matmuls are the batched, fast ones —
compare those rows of the timing output. Overridable: `LLAMA_TAG`, `MODEL_URL`, `PROMPT`,
`NGEN`, `OPT_SRC`.

If `cmake` is missing: `sudo apt install cmake` (Debian/Ubuntu).

---

## 4. Submission checklist (Task 2)

Submit only these three edited files:

- `src/matmul_simd.cpp`
- `src/matmul_prefetch.cpp`
- `src/matmul_optimized.cpp`

Do **not** submit or modify anything else in `task2/` (Makefile, `src/main.cpp`,
`include/`, `llama/`). `bench/` is a local convenience for the plots — keep it out of the
tarball. The report PDF (with plots 1–4 and the analysis) goes in as `task2_report.pdf`
per the assignment's folder layout; the `plots/` PNGs referenced by the report go in the
top‑level `plots/` directory.
