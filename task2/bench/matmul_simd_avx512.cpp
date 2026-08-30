// matmul_simd_avx512.cpp  512-bit AVX-512 variant of the Stage-1 micro-kernel (UNGRADED).
// Build with -mavx512f. See matmul_width.h.
//
// 4 rows of A x 2 rows of B, reduction over p in 512-bit lanes (16 floats), 8 independent
// __m512 FMA accumulators. Same structure as src/matmul_simd.cpp; only the vector width
// changes. AVX-512F provides _mm512_reduce_add_ps for the final horizontal sum.
//
// Requires a CPU with the AVX-512F feature. The SIMD-width benchmark guards the call with
// __builtin_cpu_supports("avx512f"); this file itself must still be COMPILED on a
// toolchain that accepts -mavx512f (any recent gcc/clang does, regardless of the host
// CPU). If the toolchain cannot, build the bench with `make AVX512=0` to drop it.

#include <cstdint>

#include <immintrin.h>

#include "matmul_width.h"

namespace {

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

    __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
    __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
    __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
    __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();

    int p = 0;
    for (; p + 16 <= K; p += 16) {
        __m512 vb0 = _mm512_loadu_ps(b0 + p);
        __m512 vb1 = _mm512_loadu_ps(b1 + p);
        __m512 va = _mm512_loadu_ps(a0 + p);
        c00 = _mm512_fmadd_ps(va, vb0, c00);
        c01 = _mm512_fmadd_ps(va, vb1, c01);
        va = _mm512_loadu_ps(a1 + p);
        c10 = _mm512_fmadd_ps(va, vb0, c10);
        c11 = _mm512_fmadd_ps(va, vb1, c11);
        va = _mm512_loadu_ps(a2 + p);
        c20 = _mm512_fmadd_ps(va, vb0, c20);
        c21 = _mm512_fmadd_ps(va, vb1, c21);
        va = _mm512_loadu_ps(a3 + p);
        c30 = _mm512_fmadd_ps(va, vb0, c30);
        c31 = _mm512_fmadd_ps(va, vb1, c31);
    }

    float r00 = _mm512_reduce_add_ps(c00), r01 = _mm512_reduce_add_ps(c01);
    float r10 = _mm512_reduce_add_ps(c10), r11 = _mm512_reduce_add_ps(c11);
    float r20 = _mm512_reduce_add_ps(c20), r21 = _mm512_reduce_add_ps(c21);
    float r30 = _mm512_reduce_add_ps(c30), r31 = _mm512_reduce_add_ps(c31);
    for (; p < K; ++p) {                      // K % 16 scalar tail
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

void matmul_simd_avx512(const float* A, const float* B, float* C,
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
