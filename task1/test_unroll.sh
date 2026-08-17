#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SIZES=(512 1024 2048 4096 8192)
KERNELS=(3 5)
ALL_STRATEGIES=(1 2 3 4)
ALL_STRAT_NAMES=("unroll_ox" "unroll_kx" "unroll_ky" "unroll_ox_oy")
SRC=src/conv_unroll.cpp
OUT=unroll_results.txt

usage() {
    echo "Usage: $0 [strategy]"
    echo ""
    echo "  strategy: 1 = unroll_ox     (unroll output columns by 4)"
    echo "            2 = unroll_kx     (unroll kernel columns by 4)"
    echo "            3 = unroll_ky     (unroll kernel rows by 4)"
    echo "            4 = unroll_ox_oy  (2D output unroll, 4x4 block)"
    echo "            (omit to run all strategies)"
    exit 0
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

if [[ -n "${1:-}" ]]; then
    if [[ "$1" -lt 1 || "$1" -gt 4 ]]; then
        echo "Error: strategy must be 1-4"
        usage
    fi
    STRATEGIES=("$1")
    STRAT_NAMES=("${ALL_STRAT_NAMES[$(($1 - 1))]}")
else
    STRATEGIES=("${ALL_STRATEGIES[@]}")
    STRAT_NAMES=("${ALL_STRAT_NAMES[@]}")
fi

echo "Running unroll benchmarks..."
echo ""

{
    printf "%-10s  %-3s  %-12s  %10s  %10s  %10s  %10s  %8s\n" \
        "HxW" "K" "strategy" "naive(ms)" "unroll(ms)" "naive_GF/s" "unrl_GF/s" "speedup"
    printf "%s\n" "$(printf '=%.0s' {1..90})"
} > "$OUT"

for K in "${KERNELS[@]}"; do
    for S in "${!STRATEGIES[@]}"; do
        strat=${STRATEGIES[$S]}
        name=${STRAT_NAMES[$S]}

        sed -i "s/^#define UNROLL_STRATEGY.*/#define UNROLL_STRATEGY $strat/" "$SRC"
        make clean > /dev/null 2>&1
        make > /dev/null 2>&1

        for SIZE in "${SIZES[@]}"; do
            echo "  K=$K  strategy=$name  size=${SIZE}x${SIZE}"

            output=$(./bin/conv unroll "$SIZE" "$SIZE" "$K" 2>&1)

            # Fields: name correct time(ms) GFLOP/s speedup
            naive_ms=$(echo "$output" | grep "naive (ref)" | awk '{print $4}')
            naive_gf=$(echo "$output" | grep "naive (ref)" | awk '{print $5}')
            unroll_ms=$(echo "$output" | grep "^unroll" | awk '{print $3}')
            unroll_gf=$(echo "$output" | grep "^unroll" | awk '{print $4}')
            speedup=$(echo "$output" | grep "^unroll" | awk '{print $5}')

            printf "%-10s  %-3s  %-12s  %10s  %10s  %10s  %10s  %8s\n" \
                "${SIZE}x${SIZE}" "$K" "$name" "$naive_ms" "$unroll_ms" \
                "$naive_gf" "$unroll_gf" "$speedup" >> "$OUT"
        done
    done

    echo "" >> "$OUT"
done

# Reset to strategy 1
sed -i "s/^#define UNROLL_STRATEGY.*/#define UNROLL_STRATEGY 1/" "$SRC"
make clean > /dev/null 2>&1
make > /dev/null 2>&1

echo ""
echo "Results written to $OUT"
echo ""
cat "$OUT"
