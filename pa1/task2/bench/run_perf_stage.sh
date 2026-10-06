#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"/..

PIN="${PIN:-taskset -c 3}"
SIZE=1024
OUT="bench/perf_stage_results.txt"
EVENTS="cpu_core/cycles/,cpu_core/instructions/,cpu_core/L1-dcache-load-misses/,cpu_core/l2_rqsts.miss/,cpu_core/LLC-load-misses/"

parse_counter() {
    local label=$1 output=$2
    echo "$output" | grep -E "$label" | grep -v "not supported\|not counted" | head -1 | awk '{gsub(/,/,"",$1); print $1}'
}

{
    printf "# Per-kernel hardware counters, M=N=K=%d, taskset -c 3\n" "$SIZE"
    printf "%-12s  %-5s  %16s  %14s  %14s  %12s  %6s\n" \
        "stage" "cache" "instructions" "L1D-miss" "L2-miss" "LLC-miss" "IPC"
} > "$OUT"

for stage in naive simd prefetch optimized; do
    echo ">>> $stage (cold)"
    sync
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true

    perf_out=$($PIN perf stat -e "$EVENTS" -- ./bin/matmul "$stage" $SIZE $SIZE $SIZE 2>&1)

    cycles=$(parse_counter "cycles" "$perf_out")
    insns=$(parse_counter "instructions" "$perf_out")
    l1d=$(parse_counter "L1-dcache-load-misses" "$perf_out")
    l2=$(parse_counter "l2_rqsts" "$perf_out")
    llc=$(parse_counter "LLC-load-misses" "$perf_out")
    ipc=$(awk "BEGIN {printf \"%.2f\", $insns / $cycles}")

    printf "%-12s  %-5s  %16s  %14s  %14s  %12s  %6s\n" \
        "$stage" "cold" "$insns" "$l1d" "$l2" "$llc" "$ipc" >> "$OUT"
done

echo
echo ">>> wrote $OUT"
cat "$OUT"
