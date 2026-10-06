#!/usr/bin/env bash
# run_simd_width.sh  Report plot 4: speedup vs SIMD width (128 / 256 / 512-bit).
#
# Builds bench/bin/bench_simd_width (SSE / AVX2 / AVX-512 variants of the Stage-1 kernel,
# each compiled with its own ISA flag) and runs it across a list of square sizes. If perf
# is usable it also records the retired-instruction count per width (Task 2B deliverable).
#
#   bench/run_simd_width.sh
#   SIZES="256 512 1024 2048" bench/run_simd_width.sh
#   AVX512=0 bench/run_simd_width.sh      # toolchain without -mavx512f
#
# Needs an AVX-512-capable CPU for the 512-bit row to be non-"n/a".
# For the instruction-count column: sudo sysctl kernel.perf_event_paranoid=1
#
# Output: bench/simd_width_results.txt
set -euo pipefail
cd "$(dirname "$0")"                          # -> task2/bench/

SIZES="${SIZES:-128 256 512 1024 2048}"
AVX512="${AVX512:-1}"
PIN="${PIN:-taskset -c 0}"
OUT="simd_width_results.txt"
BIN="bin/bench_simd_width"

echo ">>> building $BIN (AVX512=$AVX512)"
make AVX512="$AVX512" >/dev/null

# Detect whether perf can count instructions here.
HAS_PERF=0
if command -v perf >/dev/null 2>&1; then
    if $PIN perf stat -e instructions -- true 2>&1 | grep -qE "[0-9].*instructions"; then
        HAS_PERF=1
    fi
fi
[ "$HAS_PERF" -eq 1 ] || echo ">>> perf instruction counts unavailable (need perf_event_paranoid<=1); insns column = '-'"

insns_for() {  # <width-key> <size>
    [ "$HAS_PERF" -eq 1 ] || { echo "-"; return; }
    $PIN perf stat -e instructions -- "$BIN" "$1" "$2" "$2" "$2" 2>&1 \
        | grep -E "instructions" \
        | grep -E "[0-9]" \
        | grep -v "not counted\|not supported" \
        | head -1 | awk '{gsub(/,/,"",$1); print $1}'
}

{
    printf "%-12s  %-16s  %12s  %10s  %18s\n" \
        "size" "width" "GFLOP/s" "speedup" "instructions"
    printf '%s\n' "----------------------------------------------------------------------------"
} > "$OUT"

for S in $SIZES; do
    echo "  size=${S}"
    o=$($PIN "$BIN" all "$S" "$S" "$S" 2>&1)
    echo "$o" | grep -E "naive \(ref\)|SSE 128|AVX2 256|AVX-512 512" | while read -r line; do
        name=$(echo "$line" | sed -E 's/  +/\t/g' | cut -f1)
        gf=$(  echo "$line" | awk '{print $(NF-1)}')
        spd=$( echo "$line" | awk '{print $NF}')
        case "$name" in
            naive*)     key=naive ;;
            "SSE 128-bit")     key=sse ;;
            "AVX2 256-bit")    key=avx2 ;;
            "AVX-512 512-bit") key=avx512 ;;
            *) key="" ;;
        esac
        ic="-"
        if [ -n "$key" ] && echo "$line" | grep -qv "skip\|n/a"; then
            ic=$(insns_for "$key" "$S")
        fi
        printf "%-12s  %-16s  %12s  %10s  %18s\n" \
            "${S}x${S}x${S}" "$name" "$gf" "$spd" "$ic" >> "$OUT"
    done
    echo "" >> "$OUT"
done

echo
echo ">>> wrote $OUT"
cat "$OUT"
