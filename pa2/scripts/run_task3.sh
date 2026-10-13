#!/bin/bash
# Usage: scripts/run_task3.sh [trace_dir]
# Builds (scripts/build_task3.sh) and runs all Task 3 configurations on every trace
# (25M warmup + 25M sim), writing results/<config>/<trace>.txt. JOBS=n sets parallelism.
cd "$(dirname "$0")/.."
TD=${1:-../traces}
scripts/build_task3.sh || exit 1
for c in baseline spp dynirrip spp_dynnopf spp_dyn spp_dyn_pfr1 spp_dyn_pfr2 spp_dyn_pfr3; do
    echo "== $c"
    scripts/run_all.sh $c "$TD"
done
python3 scripts/analyze_task3.py
