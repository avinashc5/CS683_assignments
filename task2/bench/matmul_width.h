// matmul_width.h  declarations for the ungraded SIMD-width kernel variants.
//
// These exist ONLY to produce the "speedup vs SIMD width" plot required by the Task 2
// report. They are NOT part of the graded submission (which is src/matmul_simd.cpp,
// src/matmul_prefetch.cpp, src/matmul_optimized.cpp, all AVX2 under the pinned flags).
//
// Every variant is the SAME algorithm as src/matmul_simd.cpp -- a 4x2 register-blocked
// dot-product micro-kernel with 8 independent FMA accumulators -- changed only in vector
// width, so the plot isolates the effect of the SIMD register width:
//
//   matmul_simd_sse     128-bit  __m128   (4 floats/lane)   build: -msse4.2 -mfma
//   matmul_simd_avx2    256-bit  __m256   (8 floats/lane)   build: -mavx2 -mfma
//   matmul_simd_avx512  512-bit  __m512   (16 floats/lane)  build: -mavx512f
//
// Same signature as matmul.h. Each supports arbitrary M, N, K and leading dimensions.

#ifndef CS683_PA1_MATMUL_WIDTH_H
#define CS683_PA1_MATMUL_WIDTH_H

void matmul_simd_sse(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc);
void matmul_simd_avx2(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc);
void matmul_simd_avx512(const float* A, const float* B, float* C,
                        int M, int N, int K, int lda, int ldb, int ldc);

#endif  // CS683_PA1_MATMUL_WIDTH_H
