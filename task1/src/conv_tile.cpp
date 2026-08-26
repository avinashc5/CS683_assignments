// conv_tile.cpp  STAGE 3: CACHE TILING
//
// Reordered (ky,kx,oy,ox) loop with output blocked into horizontal strips.
// Each strip of TILE_H rows x W columns fits in L1 cache, so the K*K passes
// over each strip hit cache instead of DRAM.
//
// Change TILE_H below to experiment with different tile sizes.

#define TILE_H 32

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int tile_h = TILE_H;

    for (int ty = 0; ty < H; ty += tile_h) {
        const int bh = (ty + tile_h <= H) ? tile_h : (H - ty);

        for (int oy = ty; oy < ty + bh; ++oy)
            for (int ox = 0; ox < W; ++ox)
                out[oy * W + ox] = 0.0f;

        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                float w = ker[ky * K + kx];
                for (int oy = ty; oy < ty + bh; ++oy) {
                    const float* in_row = in + (oy + ky) * in_stride + kx;
                    float* out_row = out + oy * W;
                    for (int ox = 0; ox < W; ++ox) {
                        out_row[ox] += in_row[ox] * w;
                    }
                }
            }
        }
    }
}
