# PA2 – How to build, run and reproduce (Task 1)

## Requirements
- **GCC/G++ 7** (the old ChampSim code crashes with "stack smashing" under GCC 13). See README.md.
- Python 3 with matplotlib (plots); `pdflatex` (report).
- Traces (not included): put `Trace1.gz Trace2.gz Trace3.gz` in a directory, e.g. `../traces`.

## Build (from this directory)
    ./build_champsim.sh no        lru baseline    # no prefetcher
    ./build_champsim.sh ip_stride lru ipstride    # provided IP-stride
    ./build_champsim.sh spp       lru spp         # our Stride Pattern Prefetcher (prefetcher/spp.l1d_pref)
Binaries land in `bin/` as `*-1core-<name>`.

## Run (25M warmup + 25M simulation)
    scripts/run_all.sh baseline  [trace_dir]
    scripts/run_all.sh ipstride  [trace_dir]
    scripts/run_all.sh spp       [trace_dir]
`trace_dir` defaults to `../traces`; `JOBS=n` sets parallelism (default nproc/2; each run takes ~5 min).
Outputs: `results/<name>/<Trace>.txt`.

## Tables and plots
    python3 scripts/analyze_task1.py
Writes `results/task1/{tables.md,results.csv,speedup.png,mpki.png,accuracy.png,coverage.png}`.

## Report
    cd report && pdflatex Summary.tex && pdflatex Summary.tex
(`report/tables.tex` holds the tables included by Summary.tex.)

## Contents
- `prefetcher/spp.l1d_pref` – SPP implementation; header comment has the storage breakdown; all tunables are `#define`s at the top.
- `results/task1/synthetic_tests.txt` – synthetic pattern tests (coverage, page-boundary violations).
- `results/task1/ablation_sweeps.txt` – ablation/tuning sweeps (speedup, coverage, accuracy per config).
- `report/Summary.pdf` – Task 1 summary (Tasks 2/3 pending).

## Notes
- Speedup = IPC / baseline IPC; coverage = (baseline − pf L1D load misses) / baseline; accuracy is the simulator's (useful / prefetches issued to lower level).
- Do not commit `bin/`, `obj/` or traces (see `.gitignore`).
