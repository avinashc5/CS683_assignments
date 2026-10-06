// conv_tile.cpp  STAGE 3: CACHE TILING
//
// Four combinations controlled by TILE_MODE:
//   0 = naive order (oy,ox,ky,kx) + column tiling  (TILE_H x W)
//   1 = naive order (oy,ox,ky,kx) + square tiling  (TILE_H x TILE_W)
//   2 = reordered   (ky,kx,oy,ox) + column tiling  (TILE_H x W)
//   3 = reordered   (ky,kx,oy,ox) + square tiling  (TILE_H x TILE_W)

#ifndef TILE_H
#define TILE_H 8
#endif
#ifndef TILE_W
#define TILE_W 32
#endif
#ifndef TILE_MODE
#define TILE_MODE 3
#endif

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

#if TILE_MODE == 0
    // naive order + column tiling
    for (int ty = 0; ty < H; ty += TILE_H) {
        const int bh = (ty + TILE_H <= H) ? TILE_H : (H - ty);
        for (int oy = ty; oy < ty + bh; ++oy) {
            float* out_row = out + oy * W;
            for (int ox = 0; ox < W; ++ox) {
                float sum = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    const float* in_row = in + (oy + ky) * in_stride;
                    for (int kx = 0; kx < K; ++kx)
                        sum += in_row[ox + kx] * ker[ky * K + kx];
                }
                out_row[ox] = sum;
            }
        }
    }

#elif TILE_MODE == 1
    // naive order + square tiling
    for (int ty = 0; ty < H; ty += TILE_H) {
        const int bh = (ty + TILE_H <= H) ? TILE_H : (H - ty);
        for (int tx = 0; tx < W; tx += TILE_W) {
            const int bw = (tx + TILE_W <= W) ? TILE_W : (W - tx);
            for (int oy = ty; oy < ty + bh; ++oy) {
                float* out_row = out + oy * W;
                for (int ox = tx; ox < tx + bw; ++ox) {
                    float sum = 0.0f;
                    for (int ky = 0; ky < K; ++ky) {
                        const float* in_row = in + (oy + ky) * in_stride;
                        for (int kx = 0; kx < K; ++kx)
                            sum += in_row[ox + kx] * ker[ky * K + kx];
                    }
                    out_row[ox] = sum;
                }
            }
        }
    }

#elif TILE_MODE == 2
    // reordered + column tiling
    for (int ty = 0; ty < H; ty += TILE_H) {
        const int bh = (ty + TILE_H <= H) ? TILE_H : (H - ty);

        for (int oy = ty; oy < ty + bh; ++oy)
            for (int ox = 0; ox < W; ++ox)
                out[oy * W + ox] = 0.0f;

        for (int ky = 0; ky < K; ++ky) {
            for (int kx = 0; kx < K; ++kx) {
                float w = ker[ky * K + kx];
                for (int oy = ty; oy < ty + bh; ++oy) {
                    const float* in_row = in + (oy + ky) * in_stride + kx;
                    float* out_row = out + oy * W;
                    for (int ox = 0; ox < W; ++ox)
                        out_row[ox] += in_row[ox] * w;
                }
            }
        }
    }

#elif TILE_MODE == 3
    // reordered + square tiling
    for (int ty = 0; ty < H; ty += TILE_H) {
        const int bh = (ty + TILE_H <= H) ? TILE_H : (H - ty);
        for (int tx = 0; tx < W; tx += TILE_W) {
            const int bw = (tx + TILE_W <= W) ? TILE_W : (W - tx);

            for (int oy = ty; oy < ty + bh; ++oy)
                for (int ox = tx; ox < tx + bw; ++ox)
                    out[oy * W + ox] = 0.0f;

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float w = ker[ky * K + kx];
                    for (int oy = ty; oy < ty + bh; ++oy) {
                        const float* in_row = in + (oy + ky) * in_stride + kx;
                        float* out_row = out + oy * W;
                        for (int ox = tx; ox < tx + bw; ++ox)
                            out_row[ox] += in_row[ox] * w;
                    }
                }
            }
        }
    }

#endif
}
