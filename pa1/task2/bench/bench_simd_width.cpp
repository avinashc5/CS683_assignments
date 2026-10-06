// bench_simd_width.cpp  UNGRADED driver for the "speedup vs SIMD width" plot (Task 2B).
//
// Times matmul_naive against the 128-bit / 256-bit / 512-bit variants of the Stage-1
// micro-kernel (bench/matmul_simd_{sse,avx2,avx512}.cpp) on a square M=N=K workload and
// prints, per width: correctness vs naive, time, GFLOP/s, and speedup over naive.
//
//   ./bench_simd_width                 # all widths, default 1024^3
//   ./bench_simd_width all 512 512 512  # all widths, custom size
//   ./bench_simd_width avx2 1024 1024 1024 [seed]   # ONE width (for `perf stat` runs)
//   ./bench_simd_width naive 1024 1024 1024         # just the baseline
//
// The single-width form is what run_simd_width.sh wraps in `perf stat -e instructions`
// to fill the instruction-count column of the report table.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul.h"        // matmul_naive + MatMulFn
#include "matmul_width.h"  // the width variants
#include "timer.h"
#include "utils.h"

// Set to 0 by `make AVX512=0` when the toolchain cannot compile -mavx512f. The 512-bit
// translation unit is then not linked, so it must not be referenced here either.
#ifndef BENCH_HAVE_AVX512
#define BENCH_HAVE_AVX512 1
#endif

namespace {

constexpr int kWarmup = 1;
constexpr int kReps = 5;
constexpr float kRelTol = 1e-4f;

struct Variant {
    const char* key;
    const char* name;
    MatMulFn fn;
    bool needs_avx512;
};

const Variant kVariants[] = {
    {"sse", "SSE 128-bit", matmul_simd_sse, false},
    {"avx2", "AVX2 256-bit", matmul_simd_avx2, false},
#if BENCH_HAVE_AVX512
    {"avx512", "AVX-512 512-bit", matmul_simd_avx512, true},
#endif
};
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

bool has_avx512() {
#if BENCH_HAVE_AVX512
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

void run(const char* only, int M, int N, int K, unsigned seed) {
    const long szA = static_cast<long>(M) * K;
    const long szB = static_cast<long>(N) * K;
    const long szC = static_cast<long>(M) * N;
    float* A = pa1::alloc_floats(szA);
    float* B = pa1::alloc_floats(szB);
    float* C = pa1::alloc_floats(szC);
    float* ref = pa1::alloc_floats(szC);
    pa1::fill_random(A, szA, seed);
    pa1::fill_random(B, szB, seed + 1u);

    const int lda = K, ldb = K, ldc = N;
    matmul_naive(A, B, ref, M, N, K, lda, ldb, ldc);
    const float tol = kRelTol * (pa1::max_abs(ref, szC) + 1e-30f);
    const double flops = pa1::matmul_flops(M, N, K);

    std::printf("\n=== SIMD width sweep  M=%d N=%d K=%d seed=%u ===\n", M, N, K, seed);
    std::printf("%-16s  %-7s  %12s  %10s  %9s\n", "width", "correct", "time(ms)",
                "GFLOP/s", "speedup");
    std::printf("----------------------------------------------------------------\n");

    auto naive_run = [&]() { matmul_naive(A, B, C, M, N, K, lda, ldb, ldc); };
    const double naive_ms = pa1::time_median_ms(naive_run, kWarmup, kReps);
    std::printf("%-16s  %-7s  %12.3f  %10.2f  %8.2fx\n", "naive (ref)", "yes", naive_ms,
                flops / (naive_ms * 1e6), 1.0);

    const bool all = (only == nullptr) || std::strcmp(only, "all") == 0;
    for (int v = 0; v < kNumVariants; ++v) {
        if (!all && std::strcmp(kVariants[v].key, only) != 0) continue;
        if (kVariants[v].needs_avx512 && !has_avx512()) {
            std::printf("%-16s  %-7s  %12s  %10s  %9s\n", kVariants[v].name, "skip",
                        "n/a", "n/a", "n/a");
            continue;
        }
        auto r = [&]() { kVariants[v].fn(A, B, C, M, N, K, lda, ldb, ldc); };
        r();
        const bool ok = pa1::max_abs_diff(C, ref, szC) <= tol;
        const double ms = pa1::time_median_ms(r, kWarmup, kReps);
        std::printf("%-16s  %-7s  %12.3f  %10.2f  %8.2fx\n", kVariants[v].name,
                    ok ? "yes" : "NO", ms, flops / (ms * 1e6),
                    ms > 0.0 ? naive_ms / ms : 0.0);
    }

    pa1::free_floats(A);
    pa1::free_floats(B);
    pa1::free_floats(C);
    pa1::free_floats(ref);
}

}  // namespace

int main(int argc, char** argv) {
    const char* only = "all";
    int M = 1024, N = 1024, K = 1024;
    unsigned seed = 1234u;

    if (argc >= 2) only = argv[1];
    if (argc >= 5) {
        M = std::atoi(argv[2]);
        N = std::atoi(argv[3]);
        K = std::atoi(argv[4]);
    } else if (argc != 1 && argc != 2) {
        std::printf("usage: %s [naive|sse|avx2|avx512|all] [M N K] [seed]\n", argv[0]);
        return 1;
    }
    if (argc >= 6) seed = static_cast<unsigned>(std::strtoul(argv[5], nullptr, 10));
    if (M <= 0 || N <= 0 || K <= 0) {
        std::printf("error: M, N, K must be positive\n");
        return 1;
    }

    if (std::strcmp(only, "naive") == 0) {
        // baseline-only: reuse run() with a filter that matches nothing but naive prints.
        run("__none__", M, N, K, seed);
        return 0;
    }
    run(only, M, N, K, seed);
    return 0;
}
