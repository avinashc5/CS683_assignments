// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING
//
// The Stage-1 SIMD micro-kernel is compute-bound only while its operand panels sit in
// cache. Unblocked, every 4x2 output tile re-streams its 4 A rows and 2 B rows straight
// from DRAM, and for a large matmul the A rows alone are read N/2 times. This stage makes
// the kernel memory-friendly in two steps:
//
//   1. Cache blocking (loop reordering). Traverse the output in MC x NC blocks with order
//      jc -> ic -> jr -> ir. Within one (jc, ic) block the A sub-panel [MC x K] is reused
//      across all NC columns and the B sub-panel [NC x K] across all MC rows, so both
//      stay resident in L2. MC and NC are chosen so (MC + NC) * K * 4B is on the order of
//      the L2 capacity.
//
//   2. Software prefetch. Inside the micro-kernel's reduction loop, _mm_prefetch requests
//      the A and B streams PF_DIST floats ahead of the FMA cursor, so a line is already
//      in cache by the time the FMAs consume it. PF_DIST (distance) and PF_HINT (target
//      level: _MM_HINT_T0/T1/T2/NTA) are compile-time knobs so the report can sweep them.
//
// Tunables (override with -D... on the command line for the plot sweeps):
#ifndef MC
#define MC 96           // rows of A per cache block
#endif
#ifndef NC
#define NC 192          // rows of B per cache block
#endif
#ifndef PF_DIST
#define PF_DIST 64      // software-prefetch distance, in floats
#endif
#ifndef PF_HINT
#define PF_HINT _MM_HINT_T0   // _MM_HINT_T0 | _MM_HINT_T1 | _MM_HINT_T2 | _MM_HINT_NTA
#endif

#include <immintrin.h>

#include "matmul.h"

namespace {

inline int imin(int a, int b) { return a < b ? a : b; }

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 sh = _mm_movehl_ps(lo, lo);
    lo = _mm_add_ps(lo, sh);
    sh = _mm_shuffle_ps(lo, lo, 0x1);
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}

inline void micro_scalar(const float* A, const float* B, float* C,
                         int i0, int j0, int im, int jn, int K,
                         int lda, int ldb, int ldc) {
    for (int ii = 0; ii < im; ++ii) {
        const float* a = A + static_cast<long>(i0 + ii) * lda;
        float* crow = C + static_cast<long>(i0 + ii) * ldc + j0;
        for (int jj = 0; jj < jn; ++jj) {
            const float* b = B + static_cast<long>(j0 + jj) * ldb;
            float acc = 0.0f;
            for (int p = 0; p < K; ++p) acc += a[p] * b[p];
            crow[jj] = acc;
        }
    }
}

// 4 rows of A x 2 rows of B, reduction over p, with software prefetch ahead of use.
inline void micro_4x2(const float* A, const float* B, float* C,
                      int i0, int j0, int K,
                      int lda, int ldb, int ldc) {
    const float* a0 = A + static_cast<long>(i0 + 0) * lda;
    const float* a1 = A + static_cast<long>(i0 + 1) * lda;
    const float* a2 = A + static_cast<long>(i0 + 2) * lda;
    const float* a3 = A + static_cast<long>(i0 + 3) * lda;
    const float* b0 = B + static_cast<long>(j0 + 0) * ldb;
    const float* b1 = B + static_cast<long>(j0 + 1) * ldb;

    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();

    int p = 0;
    for (; p + 8 <= K; p += 8) {
        _mm_prefetch(reinterpret_cast<const char*>(a0 + p + PF_DIST), PF_HINT);
        _mm_prefetch(reinterpret_cast<const char*>(a1 + p + PF_DIST), PF_HINT);
        _mm_prefetch(reinterpret_cast<const char*>(a2 + p + PF_DIST), PF_HINT);
        _mm_prefetch(reinterpret_cast<const char*>(a3 + p + PF_DIST), PF_HINT);
        _mm_prefetch(reinterpret_cast<const char*>(b0 + p + PF_DIST), PF_HINT);
        _mm_prefetch(reinterpret_cast<const char*>(b1 + p + PF_DIST), PF_HINT);

        __m256 vb0 = _mm256_loadu_ps(b0 + p);
        __m256 vb1 = _mm256_loadu_ps(b1 + p);
        __m256 va = _mm256_loadu_ps(a0 + p);
        c00 = _mm256_fmadd_ps(va, vb0, c00);
        c01 = _mm256_fmadd_ps(va, vb1, c01);
        va = _mm256_loadu_ps(a1 + p);
        c10 = _mm256_fmadd_ps(va, vb0, c10);
        c11 = _mm256_fmadd_ps(va, vb1, c11);
        va = _mm256_loadu_ps(a2 + p);
        c20 = _mm256_fmadd_ps(va, vb0, c20);
        c21 = _mm256_fmadd_ps(va, vb1, c21);
        va = _mm256_loadu_ps(a3 + p);
        c30 = _mm256_fmadd_ps(va, vb0, c30);
        c31 = _mm256_fmadd_ps(va, vb1, c31);
    }

    float r00 = hsum256(c00), r01 = hsum256(c01);
    float r10 = hsum256(c10), r11 = hsum256(c11);
    float r20 = hsum256(c20), r21 = hsum256(c21);
    float r30 = hsum256(c30), r31 = hsum256(c31);
    for (; p < K; ++p) {
        float x0 = b0[p], x1 = b1[p];
        r00 += a0[p] * x0; r01 += a0[p] * x1;
        r10 += a1[p] * x0; r11 += a1[p] * x1;
        r20 += a2[p] * x0; r21 += a2[p] * x1;
        r30 += a3[p] * x0; r31 += a3[p] * x1;
    }

    float* d0 = C + static_cast<long>(i0 + 0) * ldc + j0;
    float* d1 = C + static_cast<long>(i0 + 1) * ldc + j0;
    float* d2 = C + static_cast<long>(i0 + 2) * ldc + j0;
    float* d3 = C + static_cast<long>(i0 + 3) * ldc + j0;
    d0[0] = r00; d0[1] = r01;
    d1[0] = r10; d1[1] = r11;
    d2[0] = r20; d2[1] = r21;
    d3[0] = r30; d3[1] = r31;
}

}  // namespace

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int jc = 0; jc < N; jc += NC) {
        const int nc = imin(NC, N - jc);
        const int nr = nc - (nc & 1);

        for (int ic = 0; ic < M; ic += MC) {
            const int mc = imin(MC, M - ic);
            const int mr = mc - (mc & 3);

            for (int j = 0; j < nr; j += 2) {
                int i = 0;
                for (; i < mr; i += 4)
                    micro_4x2(A, B, C, ic + i, jc + j, K, lda, ldb, ldc);
                if (i < mc)
                    micro_scalar(A, B, C, ic + i, jc + j, mc - i, 2, K,
                                 lda, ldb, ldc);
            }
            if (nr < nc)
                micro_scalar(A, B, C, ic, jc + nr, mc, nc - nr, K,
                             lda, ldb, ldc);
        }
    }
}
