#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SIZES=(512 1024 2048 4096 8192)
KERNELS=(3 5)
TILE_SIZES=(8 16 32 64 128 256)
SRC=src/conv_tile.cpp
OUT=tile_results.txt

usage() {
    echo "Usage: $0 [tile_size]"
    echo ""
    echo "  tile_size: one of ${TILE_SIZES[*]}"
    echo "             (omit to run all tile sizes)"
    echo ""
    echo "  Requires perf_event_paranoid <= 1 for MPKI data."
    echo "  Set with: sudo sysctl kernel.perf_event_paranoid=-1"
    exit 0
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# Pin to P-core 0 for consistent perf counter access on hybrid CPUs
PIN="taskset -c 0"

# Check if perf is available for MPKI
HAS_PERF=0
if command -v perf &>/dev/null; then
    perf_test=$($PIN perf stat -e L1-dcache-load-misses,instructions -- true 2>&1)
    if echo "$perf_test" | grep -v "not supported" | grep -v "not counted" | grep -q "L1-dcache-load-misses"; then
        HAS_PERF=1
    fi
fi

if [[ "$HAS_PERF" -eq 0 ]]; then
    echo "WARNING: perf not available or counters not supported on this core."
    echo "         MPKI columns will be empty. To enable:"
    echo "         sudo sysctl kernel.perf_event_paranoid=-1"
    echo ""
fi

# Filter tile sizes if argument given
if [[ -n "${1:-}" ]]; then
    found=0
    for t in "${TILE_SIZES[@]}"; do
        if [[ "$1" -eq "$t" ]]; then found=1; break; fi
    done
    if [[ "$found" -eq 0 ]]; then
        echo "Error: tile_size must be one of: ${TILE_SIZES[*]}"
        exit 1
    fi
    TILE_SIZES=("$1")
fi

echo "Running tiling benchmarks..."
echo ""

# Helper to get MPKI from perf for a specific stage
get_mpki() {
    local size=$1 K=$2 stage=$3
    if [[ "$HAS_PERF" -eq 0 ]]; then
        echo "-"
        return
    fi
    local perf_out
    perf_out=$($PIN perf stat -e L1-dcache-load-misses,instructions -- ./bin/conv "$stage" "$size" "$size" "$K" 2>&1)
    local misses instructions
    # On hybrid CPUs, pick the cpu_core line (not cpu_atom); on normal CPUs, match the bare event name
    misses=$(echo "$perf_out" | grep "L1-dcache-load-misses" | grep -v "not supported" | grep -v "not counted" | head -1 | awk '{gsub(/,/,"",$1); print $1}' || true)
    instructions=$(echo "$perf_out" | grep -w "instructions" | grep -v "not supported" | grep -v "not counted" | head -1 | awk '{gsub(/,/,"",$1); print $1}' || true)
    if [[ -n "${misses:-}" && -n "${instructions:-}" ]] && [[ "$instructions" =~ ^[0-9]+$ ]] && [[ "$instructions" -gt 0 ]]; then
        awk "BEGIN { printf \"%.2f\", ($misses / ($instructions / 1000.0)) }"
    else
        echo "-"
    fi
}

{
    printf "%-10s  %-3s  %-8s  %10s  %10s  %10s  %10s  %8s  %10s  %10s\n" \
        "HxW" "K" "tile_h" "naive(ms)" "tile(ms)" "naive_GF/s" "tile_GF/s" "speedup" "naive_MPKI" "tile_MPKI"
    printf "%s\n" "$(printf '=%.0s' {1..110})"
} > "$OUT"

for K in "${KERNELS[@]}"; do
    for tile in "${TILE_SIZES[@]}"; do
        # Set tile size in source
        sed -i "s/^#define TILE_H.*/#define TILE_H $tile/" "$SRC"
        make clean > /dev/null 2>&1
        make > /dev/null 2>&1

        for SIZE in "${SIZES[@]}"; do
            echo "  K=$K  tile_h=$tile  size=${SIZE}x${SIZE}"

            # Timing from harness
            output=$($PIN ./bin/conv tile "$SIZE" "$SIZE" "$K" 2>&1)
            naive_ms=$(echo "$output" | grep "naive (ref)" | awk '{print $4}')
            naive_gf=$(echo "$output" | grep "naive (ref)" | awk '{print $5}')
            tile_ms=$(echo "$output" | grep "^tile" | awk '{print $3}')
            tile_gf=$(echo "$output" | grep "^tile" | awk '{print $4}')
            speedup=$(echo "$output" | grep "^tile" | awk '{print $5}')

            # MPKI from perf
            naive_mpki=$(get_mpki "$SIZE" "$K" "naive")
            tile_mpki=$(get_mpki "$SIZE" "$K" "tile")

            printf "%-10s  %-3s  %-8s  %10s  %10s  %10s  %10s  %8s  %10s  %10s\n" \
                "${SIZE}x${SIZE}" "$K" "$tile" "$naive_ms" "$tile_ms" \
                "$naive_gf" "$tile_gf" "$speedup" "$naive_mpki" "$tile_mpki" >> "$OUT"
        done
    done

    echo "" >> "$OUT"
done

# Reset to default tile size
sed -i "s/^#define TILE_H.*/#define TILE_H 32/" "$SRC"
make clean > /dev/null 2>&1
make > /dev/null 2>&1

echo ""
echo "Results written to $OUT"
echo ""
cat "$OUT"
