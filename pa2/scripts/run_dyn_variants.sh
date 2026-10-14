#!/bin/bash
# Builds and runs the dynamic I-RRIP knob variants (Task 2B) on every trace, without an L1D prefetcher.
# Knob order: NUM_LEADER_SETS LEADER_SPREAD PSEL_BITS PSEL_FORCE
# Results: results/<trace>/<subdir>/<config>.txt
# Usage: scripts/run_dyn_variants.sh [config_file] [subdir]
#        defaults: scripts/dyn_sweep.cfg dyn_sweep
# Config names must be unique across config files (binaries are named dyn_<config>).

set -u
cd "$(dirname "$0")/.."

JOBS=10
WARMUP=25000000
SIM=25000000
SRC=replacement/dyn_irrip_policies.l2c_repl
KNOBS=(NUM_LEADER_SETS LEADER_SPREAD PSEL_BITS PSEL_FORCE)

CFG_FILE=${1:-scripts/dyn_sweep.cfg}
SUBDIR=${2:-dyn_sweep}
mapfile -t CONFIGS < <(grep -v '^\s*#' "$CFG_FILE" | grep -v '^\s*$')

# build one binary per config (sequential: builds share obj/), restoring the source afterwards
BACKUP=$(mktemp)
cp "$SRC" "$BACKUP"
trap 'cp "$BACKUP" "$SRC"; rm -f "$BACKUP"' EXIT

for cfg in "${CONFIGS[@]}"; do
    read -r name vals <<< "$cfg"
    read -r -a v <<< "$vals"
    cp "$BACKUP" "$SRC"
    for i in "${!KNOBS[@]}"; do
        sed -i -E "s/^#define ${KNOBS[$i]} [0-9]+/#define ${KNOBS[$i]} ${v[$i]}/" "$SRC"
    done
    if CXX=g++-7 ./build_champsim.sh no dyn_irrip_policies "dyn_$name" > /dev/null 2>&1; then
        echo "built dyn_$name"
    else
        echo "BUILD FAILED: dyn_$name"; exit 1
    fi
done
cp "$BACKUP" "$SRC"

# run every (config, trace) pair, JOBS at a time
for t in traces/*.champsimtrace.gz; do
    tn=$(basename "$t" .champsimtrace.gz)
    mkdir -p "results/$tn/$SUBDIR"
    for cfg in "${CONFIGS[@]}"; do
        read -r name _ <<< "$cfg"
        echo "$name $tn $t $SUBDIR"
    done
done | xargs -P "$JOBS" -L 1 bash -c '
    bin=$(ls bin/*-1core-dyn_$0)
    ./$bin -warmup_instructions '"$WARMUP"' -simulation_instructions '"$SIM"' -traces $2 > results/$1/$3/$0.txt 2>&1
    echo "done $0 $1"
'
