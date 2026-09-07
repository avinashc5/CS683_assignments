#!/usr/bin/env bash
# run_optimized_sweep.sh  Sweep the optimized (SIMD + blocking) kernel across matrix sizes.
#
# Builds bin/matmul and runs `./bin/matmul optimized S S S` for each size, parsing
# naive and optimized GFLOP/s and speedup.
#
#   bench/run_optimized_sweep.sh
#   SIZES="128 256 512 1024 2048" bench/run_optimized_sweep.sh
#
# Output: bench/optimized_sweep_results.txt
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> task2/

SIZES="${SIZES:-128 256 512 768 1024 1536 2048}"
PIN="${PIN:-taskset -c 3}"
OUT="bench/optimized_sweep_results.txt"

echo ">>> building bin/matmul"
make -j >/dev/null 2>&1

{
    printf "%-16s  %-12s  %12s  %12s  %10s\n" \
        "size" "stage" "naive_GF/s" "stage_GF/s" "speedup"
    printf '%s\n' "--------------------------------------------------------------------"
} > "$OUT"

for S in $SIZES; do
    echo "  size=${S}"
    raw=$($PIN ./bin/matmul optimized "$S" "$S" "$S" 2>&1)

    # naive (ref) has extra "(ref)" token: $1=naive $2=(ref) $3=correct $4=time $5=GF/s $6=speedup
    naive_gf=$(echo "$raw" | grep -E '^\s*naive' | awk '{print $5}')

    # optimized: $1=optimized $2=correct $3=time $4=GF/s $5=speedup
    opt_gf=$(echo "$raw" | grep -E '^\s*optimized' | awk '{print $4}')
    opt_spd=$(echo "$raw" | grep -E '^\s*optimized' | awk '{print $5}')

    printf "%-16s  %-12s  %12s  %12s  %10s\n" \
        "${S}x${S}x${S}" "optimized" "$naive_gf" "$opt_gf" "$opt_spd" >> "$OUT"
    echo "" >> "$OUT"
done

echo
echo ">>> wrote $OUT"
cat "$OUT"
