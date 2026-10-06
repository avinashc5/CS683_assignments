// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
//
// Layout (see matmul.h): C[i][j] = sum_p A[i][p] * B[j][p], with p (the K axis) the
// contiguous dimension of BOTH operands. The only unit-stride axis is p, so the SIMD
// strategy is a *vectorized dot product* over p: accumulate 8 partial products per step
// in a __m256, then horizontally reduce once at the end.
//
// A bare 1x1 dot product has a single length-K FMA dependency chain (FMA latency ~4c,
// throughput 2/c) and reuses nothing. We fix both with a REGISTER TILE: compute an
// MR x NR block of C at once (here 4 rows of A x 2 rows of B = 8 independent __m256
// accumulators). 8 independent FMA chains saturate FMA throughput, and each pair of
// B vectors is reused across the 4 A rows, so the inner step issues 6 loads for 8 FMAs
// (compute-bound while the panels sit in cache).
//
// This kernel is deliberately NOT cache-blocked: for large K the A/B panels spill out of
// L1/L2 and it becomes memory-bound. Stage 2 (matmul_prefetch) adds the blocking.
//
// Correctness for arbitrary shapes: the 4x2 micro-kernel runs on the bulk; leftover rows
// (M % 4), columns (N % 2) and K elements (K % 8) fall to scalar cleanup. M = 1 (llama's
// token-generation call) is handled entirely by the scalar path.

#include <immintrin.h>

#include "matmul.h"

namespace {

// Horizontal sum of the 8 lanes of a __m256.
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);                 // 4 partial sums
    __m128 sh = _mm_movehl_ps(lo, lo);       // [2 3 . .]
    lo = _mm_add_ps(lo, sh);                 // [0+2 1+3 . .]
    sh = _mm_shuffle_ps(lo, lo, 0x1);        // [1+3 . . .]
    lo = _mm_add_ss(lo, sh);
    return _mm_cvtss_f32(lo);
}

// Scalar dot-product fallback for a small im x jn block (im, jn small; any K).
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

// 4 rows of A x 2 rows of B, vectorized over p. Writes an exact 4x2 block of C.
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
        __m256 vb0 = _mm256_loadu_ps(b0 + p);
        __m256 vb1 = _mm256_loadu_ps(b1 + p);
        __m256 va;
        va = _mm256_loadu_ps(a0 + p);
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

    for (; p < K; ++p) {                      // K % 8 scalar tail
        float x0 = b0[p], x1 = b1[p];
        r00 += a0[p] * x0; r01 += a0[p] * x1;
        r10 += a1[p] * x0; r11 += a1[p] * x1;
        r20 += a2[p] * x0; r21 += a2[p] * x1;
        r30 += a3[p] * x0; r31 += a3[p] * x1;
    }

    float* c0 = C + static_cast<long>(i0 + 0) * ldc + j0;
    float* c1 = C + static_cast<long>(i0 + 1) * ldc + j0;
    float* c2 = C + static_cast<long>(i0 + 2) * ldc + j0;
    float* c3 = C + static_cast<long>(i0 + 3) * ldc + j0;
    c0[0] = r00; c0[1] = r01;
    c1[0] = r10; c1[1] = r11;
    c2[0] = r20; c2[1] = r21;
    c3[0] = r30; c3[1] = r31;
}

}  // namespace

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    const int Mr = M - (M & 3);   // largest multiple of 4 <= M
    const int Nr = N - (N & 1);   // largest multiple of 2 <= N

    for (int i = 0; i < Mr; i += 4) {
        int j = 0;
        for (; j < Nr; j += 2)
            micro_4x2(A, B, C, i, j, K, lda, ldb, ldc);
        if (j < N)                                   // trailing odd column
            micro_scalar(A, B, C, i, j, 4, N - j, K, lda, ldb, ldc);
    }
    if (Mr < M)                                      // trailing rows (incl. M < 4)
        micro_scalar(A, B, C, Mr, 0, M - Mr, N, K, lda, ldb, ldc);
}
