# Task 1 — Status: what's done, what's left

Tracks the **assignment PDF** (`CS683 PA1 2026.pdf`, Task 1 = 6 pts: 1A unroll/reorder 1,
1B tiling 1, 1C SIMD 2, 1D combined 2).

Measurement environment for every number below:
i5-11400H (Tiger Lake-H, 6 uniform P-cores, SMT off, AVX-512F/DQ/BW/VL),
L1-D 48 KiB/core, L2 1.25 MiB/core, L3 12 MiB shared. Pinned `taskset -c 3`,
`performance` governor, `perf_event_paranoid=-1`, machine otherwise idle.

---

## 1. Done

### `conv_optimized.cpp` rewritten

The previous version was a copy of `conv_simd_256` — naive loop order, AVX2, and a
**single accumulator**, so all K*K FMAs formed one dependency chain and stalled on ~4-cycle
FMA latency. It combined nothing, which is why it lost to plain `simd`.

Now: **8 independent accumulators** over adjacent column blocks (this is `unroll_ox`, the
strategy the sweep ranks best at 2.99x mean), plus **runtime AVX-512 dispatch** via
`__builtin_cpu_supports` with an AVX2 fallback. The 512-bit function carries
`__attribute__((target("avx512f")))` so the pinned Makefile flags are untouched — same
pattern as `conv_simd.cpp`. Reorder and tiling are deliberately excluded; the sweeps show
both hurt this access pattern, and the file header records why.

Accumulators cover *different output columns*, so each output pixel still sums its K*K
terms in the original order — no floating-point reassociation, no rounding change.
Verified correct on both dispatch paths across 9 shapes including `W % 16 == 8`, which
exercises the masked AVX-512 tail.

### Measurements — all five sweeps re-run in one session with the new kernel

| file | rows | covers |
|---|---|---|
| `reorder_results.txt` | 10 | 5 sizes x 2 kernels |
| `unroll_results.txt` | 80 | 8 strategies x 5 sizes x 2 kernels |
| `tile_results.txt` | 60 | 6 tile sizes x 5 sizes x 2 kernels, **with L1-D MPKI** |
| `simd_results.txt` | 30 | 128 / 256 / 512-bit x 5 sizes x 2 kernels, **with instruction counts** |
| `optimized_results.txt` | 10 | 5 sizes x 2 kernels |

Pinning is uniform: all five scripts use `PIN="${PIN:-taskset -c 3}"`. Previously
`test_tile/simd/optimized` hardcoded core 0 while `test_reorder/test_unroll` did not pin at
all.

**Cross-check for thermal drift:** `naive` is identical code in all five scripts and
measured 3.99 / 4.08 / 4.12 / 3.75 / 4.15 GF/s across them (2048^2, K=3) — ±5%, no
downward trend, despite the machine running 65 → 91 → 70 °C over the suite. The five files
are therefore mutually comparable.

### Plots + report regenerated

20 PNGs (18 in `task1/plots/`, plus `plots/task1_speedup_vs_size.png` and
`plots/task1_speedup_vs_kernel.png` which the PDF's folder layout names), `report/data.tex`
(300 macros), `report/task1_report.pdf` rebuilt.

Report corrections applied: the System Configuration section had described a different
machine (Core Ultra 5 125U) and the SIMD section asserted AVX-512 was unavailable — both
contradicted the data.

### Results

- **1A reorder:** helps only while the image fits in L3 — 1.37x @512², 1.06x @1024²,
  then **0.61x** @2048², 0.55x @4096², 0.57x @8192² (K=3). The flip sits between 4 MiB and
  16 MiB against a 12 MiB L3. K=5 is worse still, because it makes 25 passes over the image
  instead of 9. This is the README's intended teaching point, not a bug.
- **1A unroll:** `unroll_ox` best at 2.99x mean (K=3), then `unroll_ox_oy` 2.68x. Every
  *reordered* variant is below 1.0x, consistent with the reorder result above.
- **1B tiling:** best tile is 128 at 512², and even that reaches only **1.42x**; MPKI moves
  the wrong way (naive ~1.9 vs tiled ~8.3). Tiling does not pay for this access pattern.
- **1C SIMD:** AVX-512 11.99x @512² K=3, holding 6.98–7.46x at the larger sizes; AVX2 and
  SSE below that. Instruction counts fall as width rises.
- **1D combined:** `optimized` now beats every individual technique at every size —
  **20.85x** @512² K=3, **18.06x** @1024², 7.84x @2048², and **24.75x** @512² K=5,
  17.99x @2048² K=5, 15.72x @8192² K=5. Peak throughput 112.9 GF/s (was ~29).

---

## 2. Remaining

### Blocker — the graded score is thermally capped on this laptop

Three graded reps scored **85 / 70 / 85** (`optimized` 6.04x / 5.72x / 6.32x), essentially
the same range the old kernel gave here. The graded workload is 2048² K=3, where the sweep
— run at a cooler moment — records 7.84x. This laptop idles at 58–65 °C, reaches 83 °C
within a single run, and drops to 800 MHz under sustained load, so that particular number
is limited by the machine, not the kernel.

Evidence the kernel improvement is real, from an interleaved A/B run of old vs new binaries
under identical conditions: 512² K=3 **3.36x faster**, 2048² K=5 **2.28x faster**, 2048²
K=3 1.12x faster. The graded shape shows the least benefit because it is memory-bound.

Nothing further to do in code; the autograder's hardware will decide the score.

### ~~Remaining writing~~  DONE — report fully written by the team

`report/task1_report.tex` has zero open TODOs (commits "Report written", "report updated
with the tiling results"). One optional commented-out note remains at line ~510: a
suggested discussion of why AVX-512's advantage is size-dependent.

### NEW — tiling re-swept with 4 modes; the compiled-in default is a poor one

`conv_tile.cpp` now supports 4 modes (naive/reordered x column/square tiles) and
`tile_results.txt` covers all of them. Best at the graded shape (2048² K=3):
`reorder+sq 8x32` at **1.17x**. But the shipped default (`TILE_MODE=1`, 32x32) measures
**0.79x** in the team's own sweep and ~0.96x in the graded table — the tile row prints
*slower than naive*. Score-neutral (tile earns correctness points only), but it invites
the viva question "why is your tiled version slower than naive?". Consider changing the
defaults to the sweep's best (`-DTILE_MODE=3 -DTILE_H=8 -DTILE_W=32` equivalents in the
`#ifndef` defaults) — zero risk, purely presentational.

### Submission

Per PDF: `rollno1_rollno2_..._pa1.tar.gz`, roll numbers ascending, one member submits.
`task1/src/conv_{reorder,unroll,tile,simd,optimized}.cpp` + `task1/task1_report.pdf`,
PNGs in the top-level `plots/`. Do not modify anything else in `task1/`.

**Deadline: 07/09/2026 — TODAY.** Commits pushed after it are not evaluated. The
tarball (`rollno1_..._pa1.tar.gz`, ascending roll numbers, one member submits) has not
been built yet.

### Machine standardisation caveat

Task 2's numbers are being re-collected on a teammate's machine. **Task 1 cannot move
with them wholesale:** its 1C results include real AVX-512 rows, which need this
i5-11400H — a hybrid (Alder Lake-class) machine has AVX-512 fused off and would report
the 512-bit variant as unavailable. Either keep all Task 1 numbers from this machine
(the report's System Configuration already describes it), or re-run only the
non-AVX-512 parts elsewhere and say so explicitly. Mixed-machine numbers in one table
need a footnote at minimum.

---

## 3. Notes for whoever writes the report

**`conv_optimized.cpp` still contains a `CONV_NO_AVX512` environment-variable hook** that
forces the AVX2 path, added so both dispatch paths could be measured on this machine. It
only selects a path and never affects correctness. Strip before the tarball (two lines:
drop the `getenv` clause and the `#include <cstdlib>`).

**`stash@{0}` holds 6 wording edits to the report that are now STALE** — they were made
against the 649-line pre-"Report written" tex, and the file has since been heavily
expanded by the team. Popping it will conflict. Either re-apply the wording by hand to
the current tex or drop the stash.

**The governor reverts to `powersave` on every reboot.** The latest graded run (70/100,
optimized 5.89x) was taken under powersave at 73 °C — not comparable to the 8.61x
cold-machine result. Before any number that matters:
`sudo sh -c 'cpupower frequency-set -g performance && sysctl kernel.perf_event_paranoid=-1'`

**`src/main.cpp` and `include/timer.h` were modified in earlier commits** (mean-timing
helper, median reporting). The README says not to modify files outside the five kernels.
It does not affect grading — the autograder supplies its own harness — but it is what a
diff against the boilerplate would show.

**Absolute speedups depend heavily on thermal state.** The same binary measured 8.5x on a
cool machine and 6.0x when hot. Comparisons *between stages within one run* are safe;
comparisons against numbers from another session are not.

**Regenerating plots** (matplotlib lives in `~/.venv-plots`, created for this):

```
~/.venv-plots/bin/python plot_results.py && ~/.venv-plots/bin/python generate_data.py
cd report && latexmk -pdf task1_report.tex
```

**`perf` settings do not survive a reboot.** Both are session-only:

```
sudo sysctl kernel.perf_event_paranoid=-1
sudo cpupower frequency-set -g performance
```

Without the first, `tile_results.txt`'s MPKI columns and `simd_results.txt`'s instruction
columns come out empty.
