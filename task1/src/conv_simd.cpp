// conv_simd.cpp  STAGE 4: SIMD with intrinsics
//
// Multiple SIMD widths. Change SIMD_WIDTH below to select.
//   128 = SSE   (4 floats per vector, _mm128)
//   256 = AVX2  (8 floats per vector, _mm256)

#define SIMD_WIDTH 256

#include <immintrin.h>
#include "convolution.h"
#include <cstring>

// ---------------------------------------------------------------------------
// 128-bit SSE: 4 floats per vector
// Uses _mm_fmadd_ps (FMA3). W must be a multiple of 4.
// ---------------------------------------------------------------------------
void conv_simd_128(const float* in, float* out, const float* ker,
                   int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 4) {
            __m128 acc = _mm_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    __m128 w = _mm_set1_ps(ker[ky * K + kx]);
                    __m128 v = _mm_loadu_ps(in_row + kx);
                    acc = _mm_fmadd_ps(v, w, acc);
                }
            }
            _mm_storeu_ps(out + oy * W + ox, acc);
        }
    }
}

// ---------------------------------------------------------------------------
// 256-bit AVX2: 8 floats per vector
// Uses _mm256_fmadd_ps (FMA3). W is guaranteed to be a multiple of 8.
// ---------------------------------------------------------------------------
void conv_simd_256(const float* in, float* out, const float* ker,
                   int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ox += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    __m256 w = _mm256_set1_ps(ker[ky * K + kx]);
                    __m256 v = _mm256_loadu_ps(in_row + kx);
                    acc = _mm256_fmadd_ps(v, w, acc);
                }
            }
            _mm256_storeu_ps(out + oy * W + ox, acc);
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
#if SIMD_WIDTH == 128
    conv_simd_128(in, out, ker, H, W, K);
#elif SIMD_WIDTH == 256
    conv_simd_256(in, out, ker, H, W, K);
#else
#error "SIMD_WIDTH must be 128 or 256"
#endif
}
