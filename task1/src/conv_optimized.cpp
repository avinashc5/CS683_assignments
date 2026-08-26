// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Naive loop order (oy,ox,ky,kx) + AVX2 SIMD (single accumulator, 8 floats per step)

#include <immintrin.h>
#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        for (int ox = 0; ox < W; ox += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    __m256 w = _mm256_set1_ps(ker[ky * K + kx]);
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx), w, acc);
                }
            }
            _mm256_storeu_ps(out_row + ox, acc);
        }
    }
}
