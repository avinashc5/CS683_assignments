#!/bin/bash
# Usage: scripts/run_all.sh <binary-suffix> [trace_dir]
# Runs bin/*-<suffix> on every trace in trace_dir (default ../traces), 25M warmup + 25M sim,
# writing results/<suffix>/<trace>.txt. Runs up to $JOBS in parallel (default nproc/2).
cd "$(dirname "$0")/.."
SUF=$1; TD=${2:-../traces}; JOBS=${JOBS:-$(( $(nproc) / 2 ))}
BIN=$(ls bin/*-1core-$SUF) || exit 1
mkdir -p results/$SUF
for t in "$TD"/*; do
  n=$(basename "$t"); n=${n%%.champsimtrace*}; n=${n%.gz}; n=${n%.xz}
  echo "$BIN -warmup_instructions 25000000 -simulation_instructions 25000000 -traces $t > results/$SUF/$n.txt 2>&1"
done | xargs -P "$JOBS" -I{} sh -c "{}"
