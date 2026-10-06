#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SIZES=(512 1024 2048 4096 8192)
KERNELS=(3 5)
OUT=reorder_results.txt
PIN="${PIN:-taskset -c 3}"

echo "Running reorder benchmarks..."
echo ""

{
    printf "%-10s  %-3s  %10s  %10s  %10s  %10s  %8s\n" \
        "HxW" "K" "naive(ms)" "reorder(ms)" "naive_GF/s" "reord_GF/s" "speedup"
    printf "%s\n" "$(printf '=%.0s' {1..78})"
} > "$OUT"

make clean > /dev/null 2>&1
make > /dev/null 2>&1

for K in "${KERNELS[@]}"; do
    for SIZE in "${SIZES[@]}"; do
        echo "  K=$K  size=${SIZE}x${SIZE}"

        output=$($PIN ./bin/conv reorder "$SIZE" "$SIZE" "$K" 2>&1)

        naive_ms=$(echo "$output" | grep "naive (ref)" | awk '{print $4}')
        naive_gf=$(echo "$output" | grep "naive (ref)" | awk '{print $5}')
        reorder_ms=$(echo "$output" | grep "^reorder" | awk '{print $3}')
        reorder_gf=$(echo "$output" | grep "^reorder" | awk '{print $4}')
        speedup=$(echo "$output" | grep "^reorder" | awk '{print $5}')

        printf "%-10s  %-3s  %10s  %10s  %10s  %10s  %8s\n" \
            "${SIZE}x${SIZE}" "$K" "$naive_ms" "$reorder_ms" \
            "$naive_gf" "$reorder_gf" "$speedup" >> "$OUT"
    done

    echo "" >> "$OUT"
done

echo ""
echo "Results written to $OUT"
echo ""
cat "$OUT"
