#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SIZES=(512 1024 2048 4096 8192)
KERNELS=(3 5)
OUT=optimized_results.txt

PIN="taskset -c 0"

echo "Running optimized benchmarks..."
echo ""

{
    printf "%-10s  %-3s  %10s  %10s  %10s  %10s  %8s\n" \
        "HxW" "K" "naive(ms)" "optim(ms)" "naive_GF/s" "optim_GF/s" "speedup"
    printf "%s\n" "$(printf '=%.0s' {1..80})"
} > "$OUT"

make clean > /dev/null 2>&1
make > /dev/null 2>&1

for K in "${KERNELS[@]}"; do
    for SIZE in "${SIZES[@]}"; do
        echo "  K=$K  size=${SIZE}x${SIZE}"

        output=$($PIN ./bin/conv optimized "$SIZE" "$SIZE" "$K" 2>&1)
        naive_ms=$(echo "$output" | grep "naive (ref)" | awk '{print $4}')
        naive_gf=$(echo "$output" | grep "naive (ref)" | awk '{print $5}')
        optim_ms=$(echo "$output" | grep "^optimized" | awk '{print $3}')
        optim_gf=$(echo "$output" | grep "^optimized" | awk '{print $4}')
        speedup=$(echo "$output" | grep "^optimized" | awk '{print $5}')

        printf "%-10s  %-3s  %10s  %10s  %10s  %10s  %8s\n" \
            "${SIZE}x${SIZE}" "$K" "$naive_ms" "$optim_ms" \
            "$naive_gf" "$optim_gf" "$speedup" >> "$OUT"
    done

    echo "" >> "$OUT"
done

echo ""
echo "Results written to $OUT"
echo ""
cat "$OUT"
