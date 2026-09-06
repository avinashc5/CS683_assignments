// matmul_simd.cpp  STAGE 1: SIMD with intrinsics
//
// Default: AVX2 (256-bit). Override with -DSIMD_WIDTH=128 or -DSIMD_WIDTH=512
// for SSE or AVX-512 builds (used by the benchmark script).

#include <immintrin.h>

#include "matmul.h"

#ifndef SIMD_WIDTH
#define SIMD_WIDTH 256
#endif

#if SIMD_WIDTH == 128

// SSE: 4 floats per vector
void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;

            __m128 acc = _mm_setzero_ps();
            int p = 0;
            for (; p + 4 <= K; p += 4) {
                __m128 va = _mm_loadu_ps(a + p);
                __m128 vb = _mm_loadu_ps(b + p);
                acc = _mm_fmadd_ps(va, vb, acc);
            }

            __m128 sh = _mm_movehl_ps(acc, acc);
            acc = _mm_add_ps(acc, sh);
            sh = _mm_shuffle_ps(acc, acc, 0x1);
            acc = _mm_add_ss(acc, sh);
            float result = _mm_cvtss_f32(acc);

            for (; p < K; ++p)
                result += a[p] * b[p];

            C[static_cast<long>(i) * ldc + j] = result;
        }
    }
}

#elif SIMD_WIDTH == 256

// AVX2: 8 floats per vector
void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;

            __m256 acc = _mm256_setzero_ps();
            int p = 0;
            for (; p + 8 <= K; p += 8) {
                __m256 va = _mm256_loadu_ps(a + p);
                __m256 vb = _mm256_loadu_ps(b + p);
                acc = _mm256_fmadd_ps(va, vb, acc);
            }

            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            lo = _mm_add_ps(lo, hi);
            __m128 sh = _mm_movehl_ps(lo, lo);
            lo = _mm_add_ps(lo, sh);
            sh = _mm_shuffle_ps(lo, lo, 0x1);
            lo = _mm_add_ss(lo, sh);
            float result = _mm_cvtss_f32(lo);

            for (; p < K; ++p)
                result += a[p] * b[p];

            C[static_cast<long>(i) * ldc + j] = result;
        }
    }
}

#elif SIMD_WIDTH == 512

// AVX-512: 16 floats per vector
void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;

            __m512 acc = _mm512_setzero_ps();
            int p = 0;
            for (; p + 16 <= K; p += 16) {
                __m512 va = _mm512_loadu_ps(a + p);
                __m512 vb = _mm512_loadu_ps(b + p);
                acc = _mm512_fmadd_ps(va, vb, acc);
            }

            float result = _mm512_reduce_add_ps(acc);

            for (; p < K; ++p)
                result += a[p] * b[p];

            C[static_cast<long>(i) * ldc + j] = result;
        }
    }
}

#endif
