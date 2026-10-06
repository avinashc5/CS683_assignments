#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SIZES=(512 1024 2048 4096 8192)
KERNELS=(3 5)
TILE_H_SIZES=(8 16 32 64 128 256)
TILE_W_SIZES=(32 64 128 256)
TILE_MODES=(0 1 2 3)
MODE_NAMES=("naive+col" "naive+sq" "reorder+col" "reorder+sq")
SRC=src/conv_tile.cpp
OUT=tile_results.txt

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "  --mode MODE    one of 0,1,2,3 (omit to run all)"
    echo "    0 = naive order + column tiling"
    echo "    1 = naive order + square tiling"
    echo "    2 = reordered  + column tiling"
    echo "    3 = reordered  + square tiling"
    echo "  --tile_h SIZE  one of ${TILE_H_SIZES[*]} (omit to run all)"
    echo "  --tile_w SIZE  one of ${TILE_W_SIZES[*]} (omit to run all, only used for sq modes)"
    echo ""
    echo "  Requires perf_event_paranoid <= 1 for MPKI data."
    echo "  Set with: sudo sysctl kernel.perf_event_paranoid=-1"
    exit 0
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

# Parse optional args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) TILE_MODES=("$2"); shift 2 ;;
        --tile_h) TILE_H_SIZES=("$2"); shift 2 ;;
        --tile_w) TILE_W_SIZES=("$2"); shift 2 ;;
        *) shift ;;
    esac
done

# Pin to P-core 0 for consistent perf counter access on hybrid CPUs
PIN="${PIN:-taskset -c 3}"

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
    misses=$(echo "$perf_out" | grep "L1-dcache-load-misses" | grep -v "not supported" | grep -v "not counted" | head -1 | awk '{gsub(/,/,"",$1); print $1}' || true)
    instructions=$(echo "$perf_out" | grep -w "instructions" | grep -v "not supported" | grep -v "not counted" | head -1 | awk '{gsub(/,/,"",$1); print $1}' || true)
    if [[ -n "${misses:-}" && -n "${instructions:-}" ]] && [[ "$instructions" =~ ^[0-9]+$ ]] && [[ "$instructions" -gt 0 ]]; then
        awk "BEGIN { printf \"%.2f\", ($misses / ($instructions / 1000.0)) }"
    else
        echo "-"
    fi
}

echo "Running tiling benchmarks (4 mode sweep)..."
echo ""

{
    printf "%-12s  %-10s  %-3s  %-6s  %-6s  %10s  %10s  %10s  %10s  %8s  %10s  %10s\n" \
        "mode" "HxW" "K" "tile_h" "tile_w" "naive(ms)" "tile(ms)" "naive_GF/s" "tile_GF/s" "speedup" "naive_MPKI" "tile_MPKI"
    printf "%s\n" "$(printf '=%.0s' {1..130})"
} > "$OUT"

for K in "${KERNELS[@]}"; do
    for mode in "${TILE_MODES[@]}"; do
        mode_name="${MODE_NAMES[$mode]}"
        # For column-tiling modes (0,2), TILE_W is irrelevant — use W (set very large)
        if [[ "$mode" -eq 0 || "$mode" -eq 2 ]]; then
            tw_list=(99999)
        else
            tw_list=("${TILE_W_SIZES[@]}")
        fi

        for tile_h in "${TILE_H_SIZES[@]}"; do
            for tile_w in "${tw_list[@]}"; do
                # Set defines in source
                sed -i "s/^#define TILE_H.*/#define TILE_H $tile_h/" "$SRC"
                sed -i "s/^#define TILE_W.*/#define TILE_W $tile_w/" "$SRC"
                sed -i "s/^#define TILE_MODE.*/#define TILE_MODE $mode/" "$SRC"
                make clean > /dev/null 2>&1
                make > /dev/null 2>&1

                for SIZE in "${SIZES[@]}"; do
                    tw_display="$tile_w"
                    if [[ "$mode" -eq 0 || "$mode" -eq 2 ]]; then
                        tw_display="W"
                    fi
                    echo "  mode=$mode_name  K=$K  tile=${tile_h}x${tw_display}  size=${SIZE}x${SIZE}"

                    output=$($PIN ./bin/conv tile "$SIZE" "$SIZE" "$K" 2>&1)
                    naive_ms=$(echo "$output" | grep "naive (ref)" | awk '{print $4}')
                    naive_gf=$(echo "$output" | grep "naive (ref)" | awk '{print $5}')
                    tile_ms=$(echo "$output" | grep "^tile" | awk '{print $3}')
                    tile_gf=$(echo "$output" | grep "^tile" | awk '{print $4}')
                    speedup=$(echo "$output" | grep "^tile" | awk '{print $5}')

                    naive_mpki=$(get_mpki "$SIZE" "$K" "naive")
                    tile_mpki=$(get_mpki "$SIZE" "$K" "tile")

                    printf "%-12s  %-10s  %-3s  %-6s  %-6s  %10s  %10s  %10s  %10s  %8s  %10s  %10s\n" \
                        "$mode_name" "${SIZE}x${SIZE}" "$K" "$tile_h" "$tw_display" \
                        "$naive_ms" "$tile_ms" "$naive_gf" "$tile_gf" "$speedup" \
                        "$naive_mpki" "$tile_mpki" >> "$OUT"
                done
            done
        done
    done

    echo "" >> "$OUT"
done

# Reset to defaults
sed -i "s/^#define TILE_H.*/#define TILE_H 32/" "$SRC"
sed -i "s/^#define TILE_W.*/#define TILE_W 32/" "$SRC"
sed -i "s/^#define TILE_MODE.*/#define TILE_MODE 1/" "$SRC"
make clean > /dev/null 2>&1
make > /dev/null 2>&1

echo ""
echo "Results written to $OUT"
echo ""
cat "$OUT"
