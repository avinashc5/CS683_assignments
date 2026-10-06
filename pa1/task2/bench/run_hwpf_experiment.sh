#!/usr/bin/env bash
# run_hwpf_experiment.sh  Measure naive SGEMM with hardware prefetchers ON vs OFF.
#
# The prefetch_sweep_results.txt was collected with HW prefetchers DISABLED.
# This script measures the naive (no SW prefetch) kernel with HW prefetchers
# ON and OFF to quantify how much the hardware prefetcher helps.
#
# MSR 0x1A4 bits (1 = disabled):
#   bit 0: L2 hardware prefetcher (streamer)
#   bit 1: L2 adjacent cache line prefetcher
#   bit 2: DCU (L1) prefetcher
#   bit 3: DCU IP prefetcher
#
# Requires: sudo, msr-tools (sudo apt install msr-tools), modprobe msr
#
# Usage:  cd task2 && sudo bash bench/run_hwpf_experiment.sh
# Output: bench/hwpf_results.txt

set -euo pipefail
cd "$(dirname "$0")/.."

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: must run as root (sudo) to write MSR registers"
    exit 1
fi

modprobe msr 2>/dev/null || true

if ! command -v wrmsr &>/dev/null; then
    echo "ERROR: msr-tools not installed. Run: sudo apt install msr-tools"
    exit 1
fi

CXX="g++"
CXXFLAGS="-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall"
SRC="src/matmul_naive.cpp src/matmul_simd.cpp src/matmul_prefetch.cpp src/matmul_optimized.cpp src/main.cpp"
BIN_DIR="bench/bin_hwpf"
OUT="bench/hwpf_results.txt"
CORE=3

SIZES=(128 256 512 1024 2048)

mkdir -p "$BIN_DIR"
echo ">>> building kernel (mode=0, no SW prefetch)"
$CXX $CXXFLAGS -DPF_MODE=0 $SRC -o "$BIN_DIR/matmul_nopf"

PERF_EVENTS="cpu_core/L1-dcache-load-misses/,cpu_core/l2_rqsts.miss/,cpu_core/LLC-load-misses/"

# Two configurations: HW prefetchers ON vs OFF
declare -a CONFIGS=(
    "hw_on    0x0"
    "hw_off   0xF"
)

set_hwpf() { wrmsr -a 0x1A4 "$1"; }
get_hwpf() { rdmsr -c 0x1A4 | head -1; }

cleanup() {
    echo ">>> restoring hardware prefetchers (MSR 0x1A4 = 0x0)"
    set_hwpf 0x0
}
trap cleanup EXIT

{
    printf "%-10s  %6s  %5s  %10s  %10s  %14s  %14s  %14s\n" \
        "hw_config" "size" "MSR" "time_ms" "naive_GF/s" "L1D_miss" "L2_miss" "LLC_miss"
    printf '%s\n' "$(printf '=%.0s' {1..100})"
} > "$OUT"

for cfg_line in "${CONFIGS[@]}"; do
    cfg_name=$(echo "$cfg_line" | awk '{print $1}')
    cfg_msr=$(echo "$cfg_line" | awk '{print $2}')

    echo ""
    echo "=== HW config: $cfg_name (MSR 0x1A4 = $cfg_msr) ==="
    set_hwpf "$cfg_msr"
    actual=$(get_hwpf)
    echo "    MSR readback: $actual"

    for sz in "${SIZES[@]}"; do
        echo "  $cfg_name / size=$sz"

        # Run naive only — use "prefetch" stage with mode=0 (identical to naive)
        # but we only care about the naive (ref) line
        perf_out=$(taskset -c $CORE perf stat -e "$PERF_EVENTS" \
            "$BIN_DIR/matmul_nopf" prefetch "$sz" "$sz" "$sz" 2>&1)

        # naive (ref): $1=naive $2=(ref) $3=yes $4=time(ms) $5=GF/s
        naive_time=$(echo "$perf_out" | grep -E '^\s*naive' | awk '{print $4}')
        naive_gf=$(echo "$perf_out" | grep -E '^\s*naive' | awk '{print $5}')

        # Parse perf counters — cpu_core only (hybrid CPU)
        l1d=$(echo "$perf_out" | grep 'cpu_core/L1-dcache-load-misses/' | awk '{gsub(/,/,"",$1); print $1}')
        l2=$(echo "$perf_out" | grep 'cpu_core/l2_rqsts.miss/' | awk '{gsub(/,/,"",$1); print $1}')
        llc=$(echo "$perf_out" | grep 'cpu_core/LLC-load-misses/' | awk '{gsub(/,/,"",$1); print $1}')

        printf "%-10s  %6s  %5s  %10s  %10s  %14s  %14s  %14s\n" \
            "$cfg_name" "$sz" "$cfg_msr" \
            "${naive_time:---}" "${naive_gf:---}" "${l1d:---}" "${l2:---}" "${llc:---}" >> "$OUT"
    done
done

echo ""
echo ">>> wrote $OUT"
cat "$OUT"
