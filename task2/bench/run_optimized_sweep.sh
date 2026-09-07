#!/usr/bin/env bash
# run_optimized_sweep.sh  Sweep the optimized (SIMD + blocking) kernel across matrix sizes.
#
# Builds bin/matmul and runs `./bin/matmul optimized S S S` for each size, parsing
# naive and optimized GFLOP/s, speedup, and hardware counters (instructions, L1-D misses,
# L2 misses, LLC misses, IPC).
#
#   bench/run_optimized_sweep.sh
#   SIZES="128 256 512 1024 2048" bench/run_optimized_sweep.sh
#
# Output: bench/optimized_sweep_results.txt
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> task2/

SIZES="${SIZES:-128 256 512 768 1024 1536 2048}"
PIN="${PIN:-taskset -c 3}"
OUT="bench/optimized_sweep_results.txt"
CORE=3

PERF_EVENTS="cpu_core/cycles/,cpu_core/instructions/,cpu_core/L1-dcache-load-misses/,cpu_core/l2_rqsts.miss/,cpu_core/LLC-load-misses/"

echo ">>> building bin/matmul"
make -j >/dev/null 2>&1

{
    printf "%-16s  %-12s  %12s  %12s  %10s  %14s  %14s  %14s  %14s  %8s\n" \
        "size" "stage" "naive_GF/s" "stage_GF/s" "speedup" "insns" "L1D_miss" "L2_miss" "LLC_miss" "IPC"
    printf '%s\n' "$(printf '=%.0s' {1..140})"
} > "$OUT"

for S in $SIZES; do
    echo "  size=${S}"
    raw=$(taskset -c $CORE perf stat -e "$PERF_EVENTS" ./bin/matmul optimized "$S" "$S" "$S" 2>&1)

    # naive (ref) has extra "(ref)" token: $1=naive $2=(ref) $3=correct $4=time $5=GF/s $6=speedup
    naive_gf=$(echo "$raw" | grep -E '^\s*naive' | awk '{print $5}')

    # optimized: $1=optimized $2=correct $3=time $4=GF/s $5=speedup
    opt_gf=$(echo "$raw" | grep -E '^\s*optimized' | awk '{print $4}')
    opt_spd=$(echo "$raw" | grep -E '^\s*optimized' | awk '{print $5}')

    # Parse perf counters — cpu_core only (hybrid CPU)
    cycles=$(echo "$raw" | grep 'cpu_core/cycles/' | awk '{gsub(/,/,"",$1); print $1}')
    insns=$(echo "$raw" | grep 'cpu_core/instructions/' | awk '{gsub(/,/,"",$1); print $1}')
    l1d=$(echo "$raw" | grep 'cpu_core/L1-dcache-load-misses/' | awk '{gsub(/,/,"",$1); print $1}')
    l2=$(echo "$raw" | grep 'cpu_core/l2_rqsts.miss/' | awk '{gsub(/,/,"",$1); print $1}')
    llc=$(echo "$raw" | grep 'cpu_core/LLC-load-misses/' | awk '{gsub(/,/,"",$1); print $1}')

    # Compute IPC manually
    if [[ -n "$cycles" && -n "$insns" && "$cycles" -gt 0 ]]; then
        ipc=$(awk "BEGIN {printf \"%.2f\", $insns / $cycles}")
    else
        ipc="--"
    fi

    printf "%-16s  %-12s  %12s  %12s  %10s  %14s  %14s  %14s  %14s  %8s\n" \
        "${S}x${S}x${S}" "optimized" \
        "${naive_gf:---}" "${opt_gf:---}" "${opt_spd:---}" \
        "${insns:---}" "${l1d:---}" "${l2:---}" "${llc:---}" "${ipc:---}" >> "$OUT"
    echo "" >> "$OUT"
done

echo
echo ">>> wrote $OUT"
cat "$OUT"
