# Task 1 — Status: what's done, what's left

Tracks the **assignment PDF** (`CS683 PA1 2026.pdf`, Task 1 = 6 pts: 1A unroll/reorder 1,
1B tiling 1, 1C SIMD 2, 1D combined 2).

Measurement environment for every number below:
i5-11400H (Tiger Lake-H, 6 uniform P-cores, SMT off, AVX-512F/DQ/BW/VL),
L1-D 48 KiB/core, L2 1.25 MiB/core, L3 12 MiB shared. Pinned `taskset -c 3`,
`performance` governor, `perf_event_paranoid=-1`, machine otherwise idle.

---

## 1. Done

### Measurements — all five sweeps re-run on this one machine

| file | rows | covers |
|---|---|---|
| `reorder_results.txt` | 10 | 5 sizes x 2 kernels |
| `unroll_results.txt` | 80 | 8 strategies x 5 sizes x 2 kernels |
| `tile_results.txt` | 60 | 6 tile sizes x 5 sizes x 2 kernels, **with L1-D MPKI** |
| `simd_results.txt` | 30 | 128 / 256 / 512-bit x 5 sizes x 2 kernels, **with instruction counts** |
| `optimized_results.txt` | 10 | 5 sizes x 2 kernels |

Pinning is now uniform: all five scripts use `PIN="${PIN:-taskset -c 3}"`. Previously
`test_tile/simd/optimized` hardcoded core 0 while `test_reorder/test_unroll` did not pin
at all, so the older files were not measured under comparable conditions.

### Plots + report regenerated from the new data

20 PNGs (18 in `task1/plots/`, plus `plots/task1_speedup_vs_size.png` and
`plots/task1_speedup_vs_kernel.png` which the PDF's folder layout names), `report/data.tex`
(300 macros, via `generate_data.py`), and `report/task1_report.pdf` rebuilt.

Report corrections: the System Configuration section described a different machine
(Core Ultra 5 125U), and the SIMD section asserted AVX-512 was unavailable — both
contradicted the data. Fixed to this machine's specs and to AVX-512 being present.

### Results in hand

- **1A reorder:** helps only while the image fits in L3, then loses badly —
  1.36x @512², 1.31x @1024², **0.83x** @2048², 0.69x @4096², 0.64x @8192² (K=3).
  The flip sits exactly between 4 MiB and 16 MiB against a 12 MiB L3. K=5 is worse
  (0.60x @2048²) because it makes 25 passes over the image instead of 9. This is the
  README's intended teaching point, not a bug.
- **1A unroll:** `unroll_ox` is clearly best (mean 3.19x over sizes, K=3), then
  `unroll_ox_oy` 2.64x. Every *reordered* variant is below 1.0x, consistent with the
  reorder finding above.
- **1B tiling:** best tile is 16–32 depending on size, but the peak is only **1.44x**,
  and MPKI moves the wrong way — naive ~1.9 vs tiled ~8.3. Tile 256 at 4096² is 0.59x.
- **1C SIMD:** AVX-512 12.32x @512² K=3 but 5.03x @8192²; at K=5 it holds 11.5–12.6x at
  every size. AVX2 7.9x → 6.7x, SSE ~4.4–5.1x. Instruction counts fall as width rises,
  as expected.
- **1D combined:** `optimized` 6.59x / 5.99x / 5.93x across three graded reps.

### Graded harness

**85 / 70 / 70** across three reps — see the blocker below.

---

## 2. Remaining

### Blocker 1 — `optimized` straddles the 6x tier boundary

Three graded runs scored 85, 70, 70 because `optimized` averages ~6.1x and the tier
boundary is at 6x. The score is effectively a coin flip; clearing 6x comfortably is worth
15 points on its own.

### Blocker 2 — `optimized` is slower than `simd` alone

| workload | simd | optimized |
|---|---|---|
| 2048² K=3 (graded) | **7.65x** | 6.59x |
| 2048² K=5 | **7.11x** | 6.29x |

A kernel that combines reorder + unroll + tile + SIMD should never lose to SIMD alone.
The evidence points at the tiling component: standalone tiling peaks at 1.44x and
*raises* L1-D MPKI (~1.9 → ~8.3) instead of lowering it. Note the graded workload is
2048² K=3, which is also where `optimized` is weakest in the sweep (4.30x there).

Two things worth checking: whether `conv_optimized.cpp`'s tile size matches the 16–32
that `tile_results.txt` finds best, and whether the tiling is worth including at all at
the graded shape. Fixing this addresses both blockers at once.

Also unexplored: `conv_simd.cpp` builds AVX2 by default, but this machine has AVX-512,
which is 1.5–1.7x faster than AVX2 in the width sweep. Using it in `conv_optimized`
would need the extra `-mavx512f` flag, which the pinned project flags do not include —
so check whether the autograder would accept it before relying on it.

### Remaining writing — 21 `[TODO:` blocks in `report/task1_report.tex`

All the analysis prose. The PDF asks specific questions per subtask:

- **1A:** what motivated the changes, what you considered, how effective (and which
  metric, and why).
- **1B:** L1-D cache size on this system; naive MPKI; MPKI change naive → tiled with
  justification; how MPKI varies across matrix and tile sizes in terms of the cache
  hierarchy and working-set sizes; did you get a speedup, and if not what limits it.
  *(The honest answer here is the 1.44x / rising-MPKI result above — worth analysing
  rather than hiding.)*
- **1C:** instruction-count change naive → SIMD with justification; speedup and what
  contributed; which intrinsics and why.
- **1D:** a final plot comparing the best of each technique across matrix sizes, showing
  that combining beats each in isolation. **Right now the data shows the opposite**
  (`optimized` < `simd`), so either fix Blocker 2 or explain it honestly.

### Submission

Per PDF: `rollno1_rollno2_..._pa1.tar.gz`, roll numbers ascending, one member submits.
`task1/src/conv_{reorder,unroll,tile,simd,optimized}.cpp` + `task1/task1_report.pdf`,
PNGs in the top-level `plots/`. Do not modify anything else in `task1/`.

**Deadline: 07/09/2026.** Commits pushed after it are not evaluated.

---

## 3. Notes for whoever writes the report

**Re-running changes the numbers.** The graded speedup moved between 5.93x and 6.59x
across three consecutive reps on an idle, pinned, fixed-frequency machine. Quote medians
over several reps rather than a single run, and never quote a number from a run where
something else was active.

**Regenerating plots needs matplotlib**, which is not installed system-wide. It was
installed into a throwaway venv for this run. For a durable one:

```
uv venv ~/.venv-plots && uv pip install --python ~/.venv-plots/bin/python matplotlib numpy
~/.venv-plots/bin/python plot_results.py && ~/.venv-plots/bin/python generate_data.py
```

**`perf` settings do not survive a reboot.** Both are session-only:

```
sudo sysctl kernel.perf_event_paranoid=-1
sudo cpupower frequency-set -g performance
```

Without the first, `tile_results.txt`'s MPKI columns and `simd_results.txt`'s instruction
columns come out empty.
