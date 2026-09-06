#!/usr/bin/env bash
# Sweep prefetch distance and cache fill level across matrix sizes.
# Recompiles matmul_prefetch.cpp with each (PF_DIST, PF_HINT) combo,
# then runs the harness on multiple sizes.
#
# Usage:  cd task2 && bash bench/run_prefetch_sweep.sh
# Output: bench/prefetch_sweep_results.txt

set -euo pipefail

cd "$(dirname "$0")/.."

CXX="g++"
CXXFLAGS="-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall"
SRC="src/matmul_naive.cpp src/matmul_simd.cpp src/matmul_prefetch.cpp src/matmul_optimized.cpp src/main.cpp"
BIN="bench/bin_prefetch_sweep"
OUT="bench/prefetch_sweep_results.txt"

SIZES=(128 256 512 1024 2048)
DISTS=(0 8 16 32 64 128 256 512)
HINTS=("_MM_HINT_T0" "_MM_HINT_T1" "_MM_HINT_T2" "_MM_HINT_NTA")
HINT_NAMES=("T0" "T1" "T2" "NTA")

WARMUP=1
REPS=3

mkdir -p "$BIN"

printf "%-12s  %-6s  %-8s  %10s  %10s  %9s\n" \
       "size" "hint" "pf_dist" "naive_GF/s" "pf_GF/s" "speedup" > "$OUT"
printf "==========================================================================\n" >> "$OUT"

for h_idx in "${!HINTS[@]}"; do
    hint="${HINTS[$h_idx]}"
    hint_name="${HINT_NAMES[$h_idx]}"

    for dist in "${DISTS[@]}"; do
        echo "--- Compiling: hint=$hint_name  dist=$dist ---"
        $CXX $CXXFLAGS -DPF_DIST=$dist -DPF_HINT=$hint $SRC -o "$BIN/matmul_pf" 2>&1

        for sz in "${SIZES[@]}"; do
            line=$(taskset -c 3 "$BIN/matmul_pf" prefetch "$sz" "$sz" "$sz" 2>&1 \
                   | grep -E "^(naive|prefetch)" | head -2)

            naive_gf=$(echo "$line" | grep "naive" | awk '{print $5}')
            pf_gf=$(echo "$line" | grep "^prefetch" | awk '{print $4}')
            speedup=$(echo "$line" | grep "^prefetch" | awk '{print $5}')

            printf "%-12s  %-6s  %-8s  %10s  %10s  %9s\n" \
                   "${sz}x${sz}" "$hint_name" "$dist" "$naive_gf" "$pf_gf" "$speedup" >> "$OUT"
            echo "  ${sz}x${sz}  hint=$hint_name  dist=$dist  =>  $speedup"
        done
    done
done

rm -rf "$BIN"

echo ""
echo "Done. Results in $OUT"
