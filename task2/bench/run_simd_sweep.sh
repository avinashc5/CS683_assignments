#!/usr/bin/env bash
# Task 2B: Sweep SIMD width x matrix size.
# Collects: instruction count (perf), execution time, GF/s, speedup.
#
# Usage:  cd task2 && bash bench/run_simd_sweep.sh
# Output: bench/simd_sweep_results.txt

set -euo pipefail

cd "$(dirname "$0")/.."

CXX="g++"
BASE_FLAGS="-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall"
SRC_NAIVE="src/matmul_naive.cpp"
SRC_SIMD="src/matmul_simd.cpp"
SRC_PREFETCH="src/matmul_prefetch.cpp"
SRC_OPTIMIZED="src/matmul_optimized.cpp"
SRC_MAIN="src/main.cpp"
BIN="bench/bin_simd_sweep"
OUT="bench/simd_sweep_results.txt"

SIZES=(128 256 512 1024 2048)
WIDTHS=(128 256 512)
WIDTH_NAMES=("SSE_128" "AVX2_256" "AVX512_512")

# AVX-512 needs the extra flag
WIDTH_EXTRA_FLAGS=("" "" "-mavx512f")

mkdir -p "$BIN"

printf "%-16s  %-12s  %12s  %12s  %12s  %9s\n" \
       "size" "width" "naive_insns" "simd_insns" "naive_GF/s" "simd_GF/s" > "$OUT"
# Add speedup column header
sed -i "s/$/ speedup/" "$OUT"

# Rewrite header cleanly
cat > "$OUT" <<'EOF'
size              width         naive_insns    simd_insns   naive_GF/s    simd_GF/s   speedup
================================================================================================
EOF

# Check AVX-512 support
HAS_AVX512=0
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
    HAS_AVX512=1
fi

for w_idx in "${!WIDTHS[@]}"; do
    width="${WIDTHS[$w_idx]}"
    wname="${WIDTH_NAMES[$w_idx]}"
    extra="${WIDTH_EXTRA_FLAGS[$w_idx]}"

    if [[ "$width" == "512" && "$HAS_AVX512" == "0" ]]; then
        echo "--- Skipping AVX-512 (not supported on this CPU) ---"
        continue
    fi

    echo "--- Compiling: SIMD_WIDTH=$width ($wname) ---"
    $CXX $BASE_FLAGS $extra -DSIMD_WIDTH=$width \
        $SRC_NAIVE $SRC_SIMD $SRC_PREFETCH $SRC_OPTIMIZED $SRC_MAIN \
        -o "$BIN/matmul_simd_$width" 2>&1

    for sz in "${SIZES[@]}"; do
        echo "  Running ${sz}x${sz}x${sz} width=$wname ..."

        # Get timing and GF/s from the harness
        harness_out=$(taskset -c 2 "$BIN/matmul_simd_$width" simd "$sz" "$sz" "$sz" 2>&1)

        # "naive (ref)" has 2-word name, so fields shift by 1
        naive_gfs=$(echo "$harness_out" | grep "naive" | awk '{print $5}')
        simd_gfs=$(echo "$harness_out" | grep "^simd" | awk '{print $4}')
        speedup=$(echo "$harness_out" | grep "^simd" | awk '{print $5}')

        # Get instruction counts via perf stat.
        # In `perf stat -x,` output the event name is field 3. It is
        # "cpu_core/instructions/" on hybrid CPUs (Alder Lake and newer, where the
        # cpu_atom line must be skipped) but plain "instructions" on non-hybrid parts
        # such as Tiger Lake. Matching field 3 on /instructions/ handles both, and
        # `exit` keeps only the first match.
        # Run naive alone
        naive_insns=$(taskset -c 2 perf stat -e instructions -x, \
            "$BIN/matmul_simd_$width" naive "$sz" "$sz" "$sz" 2>&1 \
            | awk -F, '$3 ~ /instructions/ {print $1; exit}')

        # Run simd alone
        simd_insns=$(taskset -c 2 perf stat -e instructions -x, \
            "$BIN/matmul_simd_$width" simd "$sz" "$sz" "$sz" 2>&1 \
            | awk -F, '$3 ~ /instructions/ {print $1; exit}')

        printf "%-16s  %-12s  %12s  %12s  %12s  %9s  %9s\n" \
               "${sz}x${sz}x${sz}" "$wname" "$naive_insns" "$simd_insns" \
               "$naive_gfs" "$simd_gfs" "$speedup" >> "$OUT"
    done
    echo "" >> "$OUT"
done

rm -rf "$BIN"

echo ""
echo "Done. Results in $OUT"
