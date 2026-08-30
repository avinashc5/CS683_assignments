#!/usr/bin/env bash
# run_prefetch_degree.sh  Report plot 2: speedup vs software-prefetch distance.
#
# Rebuilds matmul_prefetch with -DPF_DIST=<d> for a list of distances (in floats) and
# records the speedup of matmul_prefetch over matmul_naive at a fixed size. The rest of
# the pinned project flags are kept identical to task2/Makefile.
#
#   bench/run_prefetch_degree.sh                    # defaults: size 1024, distances below
#   SIZE=1536 DISTS="0 16 32 64 128 256 512" bench/run_prefetch_degree.sh
#   HINT=_MM_HINT_T0 bench/run_prefetch_degree.sh   # locality hint held fixed
#
# Output: bench/prefetch_degree_results.txt
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> task2/

SIZE="${SIZE:-1024}"
DISTS="${DISTS:-0 8 16 32 48 64 96 128 192 256 384 512}"
HINT="${HINT:-_MM_HINT_T0}"
MC="${MC:-96}"
NC="${NC:-192}"
PIN="${PIN:-taskset -c 0}"
OUT="bench/prefetch_degree_results.txt"

CXXFLAGS="-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall"
SRC="src/matmul_naive.cpp src/matmul_simd.cpp src/matmul_prefetch.cpp src/matmul_optimized.cpp src/main.cpp"
BIN="bench/bin/matmul_pf"
mkdir -p bench/bin

{
    printf "# size=%sx%s  hint=%s  MC=%s NC=%s\n" "$SIZE" "$SIZE" "$HINT" "$MC" "$NC"
    printf "%-14s  %12s  %12s  %10s\n" "pf_dist(floats)" "naive_GF/s" "pf_GF/s" "speedup"
    printf '%s\n' "----------------------------------------------------------"
} > "$OUT"

for D in $DISTS; do
    echo "  PF_DIST=$D"
    # shellcheck disable=SC2086
    g++ $CXXFLAGS -DPF_DIST=$D -DPF_HINT=$HINT -DMC=$MC -DNC=$NC $SRC -o "$BIN"
    o=$($PIN "$BIN" prefetch "$SIZE" "$SIZE" "$SIZE" 2>&1)
    ngf=$(echo "$o" | grep "naive (ref)" | awk '{print $5}')
    pgf=$(echo "$o" | grep "^prefetch "  | awk '{print $4}')
    spd=$(echo "$o" | grep "^prefetch "  | awk '{print $5}')
    printf "%-14s  %12s  %12s  %10s\n" "$D" "$ngf" "$pgf" "$spd" >> "$OUT"
done

echo
echo ">>> wrote $OUT"
cat "$OUT"
