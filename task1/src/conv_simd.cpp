// conv_simd.cpp  STAGE 4: SIMD with intrinsics
//
// Multiple SIMD widths. Change SIMD_WIDTH below to select.
//   128 = SSE     (4 floats per vector, _mm128)
//   256 = AVX2    (8 floats per vector, _mm256)
//   512 = AVX-512 (16 floats per vector, _mm512)

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

#if SIMD_WIDTH == 512
// ---------------------------------------------------------------------------
// 512-bit AVX-512: 16 floats per vector
// Uses _mm512_fmadd_ps (AVX-512F).
//
// Built only when SIMD_WIDTH == 512, and carries an explicit target attribute
// so that this one function is compiled for AVX-512F while the rest of the
// project keeps the pinned -mavx2 -mfma flags from the Makefile untouched.
// No optimization flags are changed: -O2 -fno-tree-vectorize still apply.
//
// The harness only guarantees W % 8 == 0, so a bare 16-wide loop would run off
// the end of every row when W is 8 (mod 16). The trailing partial vector is
// therefore handled with a __mmask16: maskz loads suppress faults on the
// masked-off lanes (a plain load would read past the padded input on the last
// row) and the masked store writes only the live columns. When W % 16 == 0 --
// every benchmarked size -- this costs one predictable branch per row.
// ---------------------------------------------------------------------------
__attribute__((target("avx512f")))
void conv_simd_512(const float* in, float* out, const float* ker,
                   int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int W16 = W & ~15;          // largest multiple of 16 <= W
    const int rem = W - W16;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W16; ox += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    __m512 w = _mm512_set1_ps(ker[ky * K + kx]);
                    __m512 v = _mm512_loadu_ps(in_row + kx);
                    acc = _mm512_fmadd_ps(v, w, acc);
                }
            }
            _mm512_storeu_ps(out + oy * W + ox, acc);
        }

        if (rem) {
            const __mmask16 m = (__mmask16)((1u << rem) - 1u);
            __m512 acc = _mm512_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + W16;
                for (int kx = 0; kx < K; ++kx) {
                    __m512 w = _mm512_set1_ps(ker[ky * K + kx]);
                    __m512 v = _mm512_maskz_loadu_ps(m, in_row + kx);
                    acc = _mm512_fmadd_ps(v, w, acc);
                }
            }
            _mm512_mask_storeu_ps(out + oy * W + W16, m, acc);
        }
    }
}
#endif  // SIMD_WIDTH == 512

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------
void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
#if SIMD_WIDTH == 128
    conv_simd_128(in, out, ker, H, W, K);
#elif SIMD_WIDTH == 256
    conv_simd_256(in, out, ker, H, W, K);
#elif SIMD_WIDTH == 512
    conv_simd_512(in, out, ker, H, W, K);
#else
#error "SIMD_WIDTH must be 128, 256, or 512"
#endif
}
