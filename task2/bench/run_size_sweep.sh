#!/usr/bin/env bash
# run_size_sweep.sh  Report plot 1: speedup vs matrix size.
#
# For each square size M=N=K, times matmul_{simd,prefetch,optimized} against matmul_naive
# using the project harness (../bin/matmul) and records GFLOP/s and speedup.
#
#   bench/run_size_sweep.sh                 # default size list
#   SIZES="128 256 512 1024" bench/run_size_sweep.sh
#   PIN="taskset -c 2" bench/run_size_sweep.sh
#
# Output: bench/size_sweep_results.txt
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> task2/

SIZES="${SIZES:-128 256 384 512 768 1024 1536 2048}"
PIN="${PIN:-taskset -c 0}"
OUT="bench/size_sweep_results.txt"

echo ">>> building ../bin/matmul (pinned flags)"
make >/dev/null

{
    printf "%-12s  %-10s  %12s  %12s  %10s\n" \
        "size" "stage" "naive_GF/s" "stage_GF/s" "speedup"
    printf '%s\n' "--------------------------------------------------------------------"
} > "$OUT"

for S in $SIZES; do
    for stage in simd prefetch optimized; do
        echo "  size=${S}  stage=${stage}"
        o=$($PIN ./bin/matmul "$stage" "$S" "$S" "$S" 2>&1)
        ngf=$(echo "$o" | grep "naive (ref)" | awk '{print $5}')
        sgf=$(echo "$o" | grep "^${stage} "  | awk '{print $4}')
        spd=$(echo "$o" | grep "^${stage} "  | awk '{print $5}')
        printf "%-12s  %-10s  %12s  %12s  %10s\n" \
            "${S}x${S}x${S}" "$stage" "$ngf" "$sgf" "$spd" >> "$OUT"
    done
    echo "" >> "$OUT"
done

echo
echo ">>> wrote $OUT"
cat "$OUT"
