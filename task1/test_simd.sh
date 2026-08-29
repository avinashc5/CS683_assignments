#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

SIZES=(512 1024 2048 4096 8192)
KERNELS=(3 5)
ALL_WIDTHS=(128 256 512)
ALL_WIDTH_NAMES=("SSE_128" "AVX2_256" "AVX512_512")
SRC=src/conv_simd.cpp
OUT=simd_results.txt

usage() {
    echo "Usage: $0 [width]"
    echo ""
    echo "  width: 128 = SSE     (4 floats per vector)"
    echo "         256 = AVX2    (8 floats per vector)"
    echo "         512 = AVX-512 (16 floats per vector)"
    echo "         (omit to run all widths)"
    exit 0
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
fi

PIN="taskset -c 0"

# Check if perf is available for instruction count
HAS_PERF=0
if command -v perf &>/dev/null; then
    perf_test=$($PIN perf stat -e instructions -- true 2>&1 || true)
    if echo "$perf_test" | grep -v "not supported" | grep -v "not counted" | grep -qw "instructions"; then
        HAS_PERF=1
    fi
fi

if [[ "$HAS_PERF" -eq 0 ]]; then
    echo "WARNING: perf not available or restricted."
    echo "         Instruction count columns will be empty."
    echo ""
fi

# Filter widths if argument given
if [[ -n "${1:-}" ]]; then
    case "$1" in
        128) WIDTHS=(128); WIDTH_NAMES=("SSE_128")    ;;
        256) WIDTHS=(256); WIDTH_NAMES=("AVX2_256")   ;;
        512) WIDTHS=(512); WIDTH_NAMES=("AVX512_512") ;;
        *)   echo "Error: width must be 128, 256, or 512"; exit 1 ;;
    esac
else
    WIDTHS=("${ALL_WIDTHS[@]}")
    WIDTH_NAMES=("${ALL_WIDTH_NAMES[@]}")
fi

get_insn_count() {
    local size=$1 K=$2 stage=$3
    if [[ "$HAS_PERF" -eq 0 ]]; then
        echo "-"
        return
    fi
    local perf_out
    perf_out=$($PIN perf stat -e instructions -- ./bin/conv "$stage" "$size" "$size" "$K" 2>&1)
    local instructions
    instructions=$(echo "$perf_out" | grep -w "instructions" | grep -v "not supported" | grep -v "not counted" | head -1 | awk '{gsub(/,/,"",$1); print $1}' || true)
    if [[ -n "${instructions:-}" ]] && [[ "$instructions" =~ ^[0-9]+$ ]]; then
        echo "$instructions"
    else
        echo "-"
    fi
}

echo "Running SIMD benchmarks..."
echo ""

{
    printf "%-10s  %-3s  %-10s  %10s  %10s  %10s  %10s  %8s  %14s  %14s\n" \
        "HxW" "K" "width" "naive(ms)" "simd(ms)" "naive_GF/s" "simd_GF/s" "speedup" "naive_insns" "simd_insns"
    printf "%s\n" "$(printf '=%.0s' {1..115})"
} > "$OUT"

for K in "${KERNELS[@]}"; do
    for W in "${!WIDTHS[@]}"; do
        width=${WIDTHS[$W]}
        name=${WIDTH_NAMES[$W]}

        sed -i "s/^#define SIMD_WIDTH.*/#define SIMD_WIDTH $width/" "$SRC"
        make clean > /dev/null 2>&1
        make > /dev/null 2>&1

        for SIZE in "${SIZES[@]}"; do
            echo "  K=$K  width=$name  size=${SIZE}x${SIZE}"

            output=$($PIN ./bin/conv simd "$SIZE" "$SIZE" "$K" 2>&1)
            naive_ms=$(echo "$output" | grep "naive (ref)" | awk '{print $4}')
            naive_gf=$(echo "$output" | grep "naive (ref)" | awk '{print $5}')
            simd_ms=$(echo "$output" | grep "^simd" | awk '{print $3}')
            simd_gf=$(echo "$output" | grep "^simd" | awk '{print $4}')
            speedup=$(echo "$output" | grep "^simd" | awk '{print $5}')

            naive_insns=$(get_insn_count "$SIZE" "$K" "naive")
            simd_insns=$(get_insn_count "$SIZE" "$K" "simd")

            printf "%-10s  %-3s  %-10s  %10s  %10s  %10s  %10s  %8s  %14s  %14s\n" \
                "${SIZE}x${SIZE}" "$K" "$name" "$naive_ms" "$simd_ms" \
                "$naive_gf" "$simd_gf" "$speedup" "$naive_insns" "$simd_insns" >> "$OUT"
        done
    done

    echo "" >> "$OUT"
done

# Reset to default
sed -i "s/^#define SIMD_WIDTH.*/#define SIMD_WIDTH 256/" "$SRC"
make clean > /dev/null 2>&1
make > /dev/null 2>&1

echo ""
echo "Results written to $OUT"
echo ""
cat "$OUT"
