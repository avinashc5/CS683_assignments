#!/usr/bin/env bash
# run_prefetch_level.sh  Report plot 3: speedup vs prefetch locality hint.
#
# Rebuilds matmul_prefetch with -DPF_HINT set to each of _MM_HINT_T0 / _MM_HINT_T1 /
# _MM_HINT_T2 / _MM_HINT_NTA (prefetch distance held fixed) and records the speedup of
# matmul_prefetch over matmul_naive at a fixed size. Pinned project flags unchanged.
#
#   bench/run_prefetch_level.sh
#   SIZE=1536 DIST=64 bench/run_prefetch_level.sh
#
# Output: bench/prefetch_level_results.txt
set -euo pipefail
cd "$(dirname "$0")/.."                       # -> task2/

SIZE="${SIZE:-1024}"
DIST="${DIST:-64}"
MC="${MC:-96}"
NC="${NC:-192}"
HINTS="${HINTS:-_MM_HINT_T0 _MM_HINT_T1 _MM_HINT_T2 _MM_HINT_NTA}"
PIN="${PIN:-taskset -c 0}"
OUT="bench/prefetch_level_results.txt"

CXXFLAGS="-std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -Iinclude -Wall"
SRC="src/matmul_naive.cpp src/matmul_simd.cpp src/matmul_prefetch.cpp src/matmul_optimized.cpp src/main.cpp"
BIN="bench/bin/matmul_pf"
mkdir -p bench/bin

{
    printf "# size=%sx%s  pf_dist=%s floats  MC=%s NC=%s\n" "$SIZE" "$SIZE" "$DIST" "$MC" "$NC"
    printf "%-14s  %12s  %12s  %10s\n" "hint" "naive_GF/s" "pf_GF/s" "speedup"
    printf '%s\n' "----------------------------------------------------------"
} > "$OUT"

for H in $HINTS; do
    echo "  PF_HINT=$H"
    # shellcheck disable=SC2086
    g++ $CXXFLAGS -DPF_DIST=$DIST -DPF_HINT=$H -DMC=$MC -DNC=$NC $SRC -o "$BIN"
    o=$($PIN "$BIN" prefetch "$SIZE" "$SIZE" "$SIZE" 2>&1)
    ngf=$(echo "$o" | grep "naive (ref)" | awk '{print $5}')
    pgf=$(echo "$o" | grep "^prefetch "  | awk '{print $4}')
    spd=$(echo "$o" | grep "^prefetch "  | awk '{print $5}')
    printf "%-14s  %12s  %12s  %10s\n" "${H#_MM_HINT_}" "$ngf" "$pgf" "$spd" >> "$OUT"
done

echo
echo ">>> wrote $OUT"
cat "$OUT"
