// matmul_simd_avx2.cpp  256-bit AVX2 variant of the Stage-1 micro-kernel (UNGRADED).
// Build with -mavx2 -mfma. See matmul_width.h.
//
// This mirrors the graded src/matmul_simd.cpp exactly (4x2 register block, 8 __m256 FMA
// accumulators, reduction over p). It is duplicated here only so the SIMD-width benchmark
// links one self-contained translation unit per width.

#include <immintrin.h>

#include "matmul_width.h"

namespace {

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

inline void micro_4x2(const float* A, const float* B, float* C,
                      int i0, int j0, int K, int lda, int ldb, int ldc) {
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

void matmul_simd_avx2(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    const int Mr = M - (M & 3);
    const int Nr = N - (N & 1);
    for (int i = 0; i < Mr; i += 4) {
        int j = 0;
        for (; j < Nr; j += 2)
            micro_4x2(A, B, C, i, j, K, lda, ldb, ldc);
        if (j < N)
            micro_scalar(A, B, C, i, j, 4, N - j, K, lda, ldb, ldc);
    }
    if (Mr < M)
        micro_scalar(A, B, C, Mr, 0, M - Mr, N, K, lda, ldb, ldc);
}
