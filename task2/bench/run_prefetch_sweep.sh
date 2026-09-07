#!/usr/bin/env bash
# Sweep prefetch mode, distance, cache fill level, and matrix size.
# Collects: timing, GF/s, speedup, L1D/L2/LLC misses, sw_prefetch_access.
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
CORE=3

SIZES=(128 256 512 1024 2048)
MODES=(0 1 2 3 4)
MODE_NAMES=("none" "within_row" "next_B" "C_output" "all")
DISTS=(0 16 32 64 128)
HINTS=("_MM_HINT_T0" "_MM_HINT_T1" "_MM_HINT_T2" "_MM_HINT_NTA")
HINT_NAMES=("T0" "T1" "T2" "NTA")

PERF_EVENTS="L1-dcache-load-misses,l2_rqsts.miss,LLC-load-misses,sw_prefetch_access.any"

mkdir -p "$BIN"

cat > "$OUT" <<'EOF'
size          mode         hint    pf_dist   naive_GF/s    pf_GF/s   speedup      L1D_miss      L2_miss     LLC_miss    sw_pf_acc
==================================================================================================================================
EOF

extract_perf() {
    local val
    val=$(echo "$1" | grep "cpu_core/$2/" | head -1 | cut -d',' -f1) || true
    echo "${val:---}"
}

for m_idx in "${!MODES[@]}"; do
    mode="${MODES[$m_idx]}"
    mode_name="${MODE_NAMES[$m_idx]}"

    # Mode 0 (no prefetch) doesn't depend on hint or distance, run once
    if [[ "$mode" == "0" ]]; then
        echo "=== Mode: $mode_name ==="
        $CXX $CXXFLAGS -DPF_MODE=0 $SRC -o "$BIN/matmul_pf" 2>&1

        for sz in "${SIZES[@]}"; do
            echo -n "  ${sz}x${sz}  mode=$mode_name  =>  "

            harness_out=$(taskset -c $CORE "$BIN/matmul_pf" prefetch "$sz" "$sz" "$sz" 2>&1)
            naive_gfs=$(echo "$harness_out" | grep "naive" | awk '{print $5}') || true
            pf_gfs=$(echo "$harness_out" | grep "^prefetch" | awk '{print $4}') || true
            speedup=$(echo "$harness_out" | grep "^prefetch" | awk '{print $5}') || true

            perf_out=$(taskset -c $CORE perf stat -e "$PERF_EVENTS" -x, \
                "$BIN/matmul_pf" prefetch "$sz" "$sz" "$sz" 2>&1)

            l1d_miss=$(extract_perf "$perf_out" "L1-dcache-load-misses")
            l2_miss=$(extract_perf "$perf_out" "l2_rqsts.miss")
            llc_miss=$(extract_perf "$perf_out" "LLC-load-misses")
            sw_pf=$(extract_perf "$perf_out" "sw_prefetch_access.any")

            printf "%-12s  %-11s  %-6s  %7s  %10s  %9s  %8s  %12s  %11s  %11s  %11s\n" \
                   "${sz}x${sz}" "$mode_name" "--" "--" "$naive_gfs" "$pf_gfs" "$speedup" \
                   "$l1d_miss" "$l2_miss" "$llc_miss" "$sw_pf" >> "$OUT"

            echo "$speedup"
        done
        echo "" >> "$OUT"
        continue
    fi

    # Modes 1-4: sweep hint and distance
    for h_idx in "${!HINTS[@]}"; do
        hint="${HINTS[$h_idx]}"
        hint_name="${HINT_NAMES[$h_idx]}"

        for dist in "${DISTS[@]}"; do
            echo "=== Mode: $mode_name  hint=$hint_name  dist=$dist ==="
            $CXX $CXXFLAGS -DPF_MODE=$mode -DPF_DIST=$dist -DPF_HINT=$hint \
                $SRC -o "$BIN/matmul_pf" 2>&1

            for sz in "${SIZES[@]}"; do
                echo -n "  ${sz}x${sz}  =>  "

                harness_out=$(taskset -c $CORE "$BIN/matmul_pf" prefetch "$sz" "$sz" "$sz" 2>&1)
                naive_gfs=$(echo "$harness_out" | grep "naive" | awk '{print $5}') || true
                pf_gfs=$(echo "$harness_out" | grep "^prefetch" | awk '{print $4}') || true
                speedup=$(echo "$harness_out" | grep "^prefetch" | awk '{print $5}') || true

                perf_out=$(taskset -c $CORE perf stat -e "$PERF_EVENTS" -x, \
                    "$BIN/matmul_pf" prefetch "$sz" "$sz" "$sz" 2>&1)

                l1d_miss=$(extract_perf "$perf_out" "L1-dcache-load-misses")
                l2_miss=$(extract_perf "$perf_out" "l2_rqsts.miss")
                llc_miss=$(extract_perf "$perf_out" "LLC-load-misses")
                sw_pf=$(extract_perf "$perf_out" "sw_prefetch_access.any")

                printf "%-12s  %-11s  %-6s  %7s  %10s  %9s  %8s  %12s  %11s  %11s  %11s\n" \
                       "${sz}x${sz}" "$mode_name" "$hint_name" "$dist" "$naive_gfs" "$pf_gfs" \
                       "$speedup" "$l1d_miss" "$l2_miss" "$llc_miss" "$sw_pf" >> "$OUT"

                echo "$speedup  L1D=$l1d_miss  L2=$l2_miss  sw_pf=$sw_pf"
            done
        done
    done
    echo "" >> "$OUT"
done

rm -rf "$BIN"

echo ""
echo "Done. Results in $OUT"
