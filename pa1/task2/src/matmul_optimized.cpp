// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// The graded kernel, and the one injected into llama.cpp by `make llama-demo`. It layers
// every technique from the assignment on top of the Stage-1 micro-kernel:
//
//   * SIMD + register blocking  -- a 6x16 AVX2 micro-kernel. Each C row is held in two
//     __m256 accumulators spanning 16 CONTIGUOUS columns, so there is NO horizontal
//     reduction -- the tile is written with plain vector stores. 12 accumulator chains
//     hide FMA latency; per K step 2 packed-B vector loads + 6 A broadcasts -> 12 FMAs.
//   * Packing                   -- B is stored "transposed" (K contiguous), so the 16
//     columns a K step needs are strided. The active NC-column block of B is packed once
//     into a [K x NC] scratch panel so every B load in the micro-kernel is unit-stride.
//   * Loop reordering + cache blocking -- traversal order jc -> ic -> jr -> ir over
//     MC x NC blocks. The packed B panel is reused across all MC rows and the A
//     sub-panel across all NC columns, so both stay resident in L2.
//
// Software prefetch was measured (bench/prefetch_degree_results.txt): on this hardware the
// L2 hardware streamer already covers the sequential panel access, so an explicit
// _mm_prefetch gives no speedup (flat across distance 0..512) and only adds instructions.
// It is therefore not used here.
//
// Arbitrary M, N, K and arbitrary lda/ldb/ldc are supported: the 6x16 kernel takes the
// bulk of every block; the M%6 / N%16 edges fall to scalar cleanup (any K). llama's
// M = 1 token-generation call is served entirely by the scalar path.

#include <cstdlib>

#include <immintrin.h>

#include "matmul.h"

// ---- tunables (overridable with -D... for local experiments) ----------------
#ifndef MMO_MC
#define MMO_MC 96             // rows of A per L2 block (multiple of 6)
#endif
#ifndef MMO_NC
#define MMO_NC 64             // columns of B per L2 block (packed panel is K x NC)
#endif
#ifndef MMO_MR
#define MMO_MR 6              // micro-kernel row count
#endif

namespace {

inline int imin(int a, int b) { return a < b ? a : b; }

// Scalar dot-product fallback for a small im x jn block (reads B directly, unpacked).
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

// 6 rows of A x 16 columns of C, reduction over p.
//   a       : &A[i0][0]          rows i0..i0+5 via lda
//   bp      : &Bpack[0][jrel]    packed panel, row p at bp + p*bstride (16 wide used)
//   bstride : packed-panel row stride (= nc of the current block)
//   c       : &C[i0][j0]         rows via ldc, 16 contiguous columns
inline void micro_6x16(const float* a, int lda,
                       const float* bp, int bstride,
                       float* c, int ldc, int K) {
    const float* a0 = a + 0 * lda; const float* a1 = a + 1 * lda;
    const float* a2 = a + 2 * lda; const float* a3 = a + 3 * lda;
    const float* a4 = a + 4 * lda; const float* a5 = a + 5 * lda;

    __m256 c0l = _mm256_setzero_ps(), c0h = _mm256_setzero_ps();
    __m256 c1l = _mm256_setzero_ps(), c1h = _mm256_setzero_ps();
    __m256 c2l = _mm256_setzero_ps(), c2h = _mm256_setzero_ps();
    __m256 c3l = _mm256_setzero_ps(), c3h = _mm256_setzero_ps();
    __m256 c4l = _mm256_setzero_ps(), c4h = _mm256_setzero_ps();
    __m256 c5l = _mm256_setzero_ps(), c5h = _mm256_setzero_ps();

    for (int p = 0; p < K; ++p) {
        const float* r_ = bp + static_cast<long>(p) * bstride;
        __m256 bl_ = _mm256_loadu_ps(r_);
        __m256 bh_ = _mm256_loadu_ps(r_ + 8);
        __m256 s_;
        s_ = _mm256_broadcast_ss(a0 + p); c0l = _mm256_fmadd_ps(s_, bl_, c0l); c0h = _mm256_fmadd_ps(s_, bh_, c0h);
        s_ = _mm256_broadcast_ss(a1 + p); c1l = _mm256_fmadd_ps(s_, bl_, c1l); c1h = _mm256_fmadd_ps(s_, bh_, c1h);
        s_ = _mm256_broadcast_ss(a2 + p); c2l = _mm256_fmadd_ps(s_, bl_, c2l); c2h = _mm256_fmadd_ps(s_, bh_, c2h);
        s_ = _mm256_broadcast_ss(a3 + p); c3l = _mm256_fmadd_ps(s_, bl_, c3l); c3h = _mm256_fmadd_ps(s_, bh_, c3h);
        s_ = _mm256_broadcast_ss(a4 + p); c4l = _mm256_fmadd_ps(s_, bl_, c4l); c4h = _mm256_fmadd_ps(s_, bh_, c4h);
        s_ = _mm256_broadcast_ss(a5 + p); c5l = _mm256_fmadd_ps(s_, bl_, c5l); c5h = _mm256_fmadd_ps(s_, bh_, c5h);
    }

    _mm256_storeu_ps(c + 0 * ldc, c0l); _mm256_storeu_ps(c + 0 * ldc + 8, c0h);
    _mm256_storeu_ps(c + 1 * ldc, c1l); _mm256_storeu_ps(c + 1 * ldc + 8, c1h);
    _mm256_storeu_ps(c + 2 * ldc, c2l); _mm256_storeu_ps(c + 2 * ldc + 8, c2h);
    _mm256_storeu_ps(c + 3 * ldc, c3l); _mm256_storeu_ps(c + 3 * ldc + 8, c3h);
    _mm256_storeu_ps(c + 4 * ldc, c4l); _mm256_storeu_ps(c + 4 * ldc + 8, c4h);
    _mm256_storeu_ps(c + 5 * ldc, c5l); _mm256_storeu_ps(c + 5 * ldc + 8, c5h);
}

}  // namespace

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    const int mc_blk = MMO_MC;
    const int nc_blk = MMO_NC;

    const std::size_t pack_elems = static_cast<std::size_t>(K) * nc_blk;
    const std::size_t pack_bytes = ((pack_elems * sizeof(float)) + 63) & ~std::size_t(63);
    float* Bpack = static_cast<float*>(std::aligned_alloc(64, pack_bytes));
    if (!Bpack) {
        micro_scalar(A, B, C, 0, 0, M, N, K, lda, ldb, ldc);
        return;
    }

    for (int jc = 0; jc < N; jc += nc_blk) {
        const int nc = imin(nc_blk, N - jc);
        const int nr = nc - (nc & 15);       // largest multiple of 16 <= nc

        for (int p = 0; p < K; ++p) {         // pack B[jc..jc+nc)[p]  ->  Bpack[p*nc + jj]
            float* dst = Bpack + static_cast<long>(p) * nc;
            for (int jj = 0; jj < nc; ++jj)
                dst[jj] = B[static_cast<long>(jc + jj) * ldb + p];
        }

        for (int ic = 0; ic < M; ic += mc_blk) {
            const int mc = imin(mc_blk, M - ic);
            const int mr = mc - mc % MMO_MR;  // largest multiple of MMO_MR <= mc

            for (int i = 0; i < mr; i += MMO_MR) {
                const float* a = A + static_cast<long>(ic + i) * lda;
                for (int j = 0; j < nr; j += 16)
                    micro_6x16(a, lda, Bpack + j, nc,
                               C + static_cast<long>(ic + i) * ldc + (jc + j), ldc, K);
                if (nr < nc)
                    micro_scalar(A, B, C, ic + i, jc + nr, MMO_MR, nc - nr, K,
                                 lda, ldb, ldc);
            }
            if (mr < mc)
                micro_scalar(A, B, C, ic + mr, jc, mc - mr, nc, K, lda, ldb, ldc);
        }
    }

    std::free(Bpack);
}
