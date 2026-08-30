// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// The graded kernel, and the one injected into llama.cpp by `make llama-demo`. It layers
// every technique from the assignment on top of the Stage-1 micro-kernel:
//
//   * SIMD + register blocking  -- a 4x2 AVX2 micro-kernel with 8 independent __m256 FMA
//     accumulators (enough to saturate FMA throughput at 2/cycle); the 2 B vectors are
//     reused across the 4 A rows, so the inner step issues 6 loads for 8 FMAs.
//   * Loop unrolling            -- the K (reduction) loop is 2x unrolled, 16 floats/step,
//     so loop overhead is amortised and the two halves feed the 8 accumulators back to
//     back.
//   * Loop reordering + cache blocking -- the (i, j) space is traversed in MC x NC blocks
//     with order jc -> ic -> jr -> ir. Inside one block the A sub-panel [MC x K] is
//     reused across all NC columns and the B sub-panel [NC x K] across all MC rows, so
//     both stay resident in L2 (MC and NC are sized so (MC + NC) * K * 4B ~ L2) instead
//     of being re-fetched from DRAM for every output tile.
//   * Software prefetch         -- _mm_prefetch pulls the A and B streams PF_DIST floats
//     ahead of the FMA cursor (T0 = into L1).
//
// Arbitrary M, N, K and arbitrary lda/ldb/ldc are supported: the 4x2 kernel takes the
// bulk of every block and the M%4 / N%2 / K%8 edges fall to scalar cleanup. llama's
// M = 1 token-generation call is served entirely by the scalar path (correct, not fast;
// the batched prompt-eval matmuls are the fast ones).

#include <immintrin.h>

#include "matmul.h"

// ---- tunables (overridable with -D... for local experiments) ----------------
#ifndef MMO_MC
#define MMO_MC 96             // rows of A per L2 block
#endif
#ifndef MMO_NC
#define MMO_NC 192            // rows of B per L2 block
#endif
#ifndef MMO_PF_DIST
#define MMO_PF_DIST 64        // software-prefetch distance, in floats
#endif
#ifndef MMO_PF_HINT
#define MMO_PF_HINT _MM_HINT_T0
#endif

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

// 4 rows of A x 2 rows of B, reduction over p, 2x unrolled, with software prefetch.
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
    for (; p + 16 <= K; p += 16) {
        _mm_prefetch(reinterpret_cast<const char*>(a0 + p + MMO_PF_DIST), MMO_PF_HINT);
        _mm_prefetch(reinterpret_cast<const char*>(a1 + p + MMO_PF_DIST), MMO_PF_HINT);
        _mm_prefetch(reinterpret_cast<const char*>(a2 + p + MMO_PF_DIST), MMO_PF_HINT);
        _mm_prefetch(reinterpret_cast<const char*>(a3 + p + MMO_PF_DIST), MMO_PF_HINT);
        _mm_prefetch(reinterpret_cast<const char*>(b0 + p + MMO_PF_DIST), MMO_PF_HINT);
        _mm_prefetch(reinterpret_cast<const char*>(b1 + p + MMO_PF_DIST), MMO_PF_HINT);

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

        vb0 = _mm256_loadu_ps(b0 + p + 8);
        vb1 = _mm256_loadu_ps(b1 + p + 8);
        va = _mm256_loadu_ps(a0 + p + 8);
        c00 = _mm256_fmadd_ps(va, vb0, c00);
        c01 = _mm256_fmadd_ps(va, vb1, c01);
        va = _mm256_loadu_ps(a1 + p + 8);
        c10 = _mm256_fmadd_ps(va, vb0, c10);
        c11 = _mm256_fmadd_ps(va, vb1, c11);
        va = _mm256_loadu_ps(a2 + p + 8);
        c20 = _mm256_fmadd_ps(va, vb0, c20);
        c21 = _mm256_fmadd_ps(va, vb1, c21);
        va = _mm256_loadu_ps(a3 + p + 8);
        c30 = _mm256_fmadd_ps(va, vb0, c30);
        c31 = _mm256_fmadd_ps(va, vb1, c31);
    }
    for (; p + 8 <= K; p += 8) {
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

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    const int mc_blk = MMO_MC;   // plain locals (not macro-named) so a stray -DMC / -DNC
    const int nc_blk = MMO_NC;   // from the prefetch-sweep scripts cannot clobber them

    for (int jc = 0; jc < N; jc += nc_blk) {
        const int nc = imin(nc_blk, N - jc);
        const int nr = nc - (nc & 1);

        for (int ic = 0; ic < M; ic += mc_blk) {
            const int mc = imin(mc_blk, M - ic);
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
