// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
//
// Multiple unrolling strategies. 1-4 use the naive loop order, 5-8 use reordered.
// Change UNROLL_STRATEGY below to select which one conv_unroll dispatches to.
//   1 = unroll ox (output columns) by 4 with separate accumulators
//   2 = unroll kx (kernel columns) by 4
//   3 = unroll ky (kernel rows) by 4
//   4 = unroll both ox and oy (2D output unroll, 4x4 block)
//   5 = reordered (ky,kx,oy,ox) + unroll ox by 4
//   6 = reordered (ky,kx,oy,ox) + unroll oy by 4
//   7 = reordered (ky,kx,oy,ox) + unroll ox and oy (4x4 block)
//   8 = reordered (ky,kx,oy,ox) + unroll kx by 4

#define UNROLL_STRATEGY 1

#include "convolution.h"

// ---------------------------------------------------------------------------
// Strategy 1: Unroll ox by 4 with independent accumulators.
// Computes 4 adjacent output pixels simultaneously. Each accumulator is
// independent, exposing ILP. The full K*K sum stays in registers per pixel —
// output is written once, not read-modify-written K*K times.
// ---------------------------------------------------------------------------
void conv_unroll_ox(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        int ox = 0;
        for (; ox + 3 < W; ox += 4) {
            float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    float w = ker[ky * K + kx];
                    acc0 += in_row[kx]     * w;
                    acc1 += in_row[kx + 1] * w;
                    acc2 += in_row[kx + 2] * w;
                    acc3 += in_row[kx + 3] * w;
                }
            }
            out_row[ox]     = acc0;
            out_row[ox + 1] = acc1;
            out_row[ox + 2] = acc2;
            out_row[ox + 3] = acc3;
        }
        for (; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx)
                    acc += in_row[kx] * ker[ky * K + kx];
            }
            out_row[ox] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy 2: Unroll kx (kernel columns) by 4.
// Processes 4 kernel columns per inner-loop iteration, reducing loop overhead
// and giving the CPU 4 independent multiply-adds to schedule.
// Works best when K >= 5. Falls back to scalar for the tail.
// ---------------------------------------------------------------------------
void conv_unroll_kx(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                int kx = 0;
                for (; kx + 3 < K; kx += 4) {
                    acc += in_row[kx]     * ker_row[kx];
                    acc += in_row[kx + 1] * ker_row[kx + 1];
                    acc += in_row[kx + 2] * ker_row[kx + 2];
                    acc += in_row[kx + 3] * ker_row[kx + 3];
                }
                for (; kx < K; ++kx)
                    acc += in_row[kx] * ker_row[kx];
            }
            out[oy * W + ox] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy 3: Unroll ky (kernel rows) by 4.
// Processes 4 kernel rows per iteration of the ky loop, accumulating from
// 4 different input rows with their respective kernel weights. Reduces
// ky loop overhead and exposes cross-row ILP.
// ---------------------------------------------------------------------------
void conv_unroll_ky(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            int ky = 0;
            for (; ky + 3 < K; ky += 4) {
                const float* r0 = in + (oy + ky)     * in_stride + ox;
                const float* r1 = in + (oy + ky + 1) * in_stride + ox;
                const float* r2 = in + (oy + ky + 2) * in_stride + ox;
                const float* r3 = in + (oy + ky + 3) * in_stride + ox;
                const float* k0 = ker + (ky)     * K;
                const float* k1 = ker + (ky + 1) * K;
                const float* k2 = ker + (ky + 2) * K;
                const float* k3 = ker + (ky + 3) * K;
                for (int kx = 0; kx < K; ++kx) {
                    acc += r0[kx] * k0[kx];
                    acc += r1[kx] * k1[kx];
                    acc += r2[kx] * k2[kx];
                    acc += r3[kx] * k3[kx];
                }
            }
            for (; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                const float* ker_row = ker + ky * K;
                for (int kx = 0; kx < K; ++kx)
                    acc += in_row[kx] * ker_row[kx];
            }
            out[oy * W + ox] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy 4: 2D output unroll — process a 4x4 block of output pixels at once.
// Each of the 16 accumulators is independent, maximizing ILP and register
// utilization. The full kernel sum for each pixel stays in registers.
// ---------------------------------------------------------------------------
void conv_unroll_ox_oy(const float* in, float* out, const float* ker,
                       int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    int oy = 0;
    for (; oy + 3 < H; oy += 4) {
        int ox = 0;
        for (; ox + 3 < W; ox += 4) {
            float a00 = 0, a01 = 0, a02 = 0, a03 = 0;
            float a10 = 0, a11 = 0, a12 = 0, a13 = 0;
            float a20 = 0, a21 = 0, a22 = 0, a23 = 0;
            float a30 = 0, a31 = 0, a32 = 0, a33 = 0;
            for (int ky = 0; ky < K; ++ky) {
                const float* r0 = in + (oy + ky)     * in_stride + ox;
                const float* r1 = in + (oy + 1 + ky) * in_stride + ox;
                const float* r2 = in + (oy + 2 + ky) * in_stride + ox;
                const float* r3 = in + (oy + 3 + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx) {
                    float w = ker[ky * K + kx];
                    a00 += r0[kx]     * w; a01 += r0[kx + 1] * w;
                    a02 += r0[kx + 2] * w; a03 += r0[kx + 3] * w;
                    a10 += r1[kx]     * w; a11 += r1[kx + 1] * w;
                    a12 += r1[kx + 2] * w; a13 += r1[kx + 3] * w;
                    a20 += r2[kx]     * w; a21 += r2[kx + 1] * w;
                    a22 += r2[kx + 2] * w; a23 += r2[kx + 3] * w;
                    a30 += r3[kx]     * w; a31 += r3[kx + 1] * w;
                    a32 += r3[kx + 2] * w; a33 += r3[kx + 3] * w;
                }
            }
            out[(oy)     * W + ox] = a00; out[(oy)     * W + ox+1] = a01;
            out[(oy)     * W + ox+2] = a02; out[(oy)     * W + ox+3] = a03;
            out[(oy + 1) * W + ox] = a10; out[(oy + 1) * W + ox+1] = a11;
            out[(oy + 1) * W + ox+2] = a12; out[(oy + 1) * W + ox+3] = a13;
            out[(oy + 2) * W + ox] = a20; out[(oy + 2) * W + ox+1] = a21;
            out[(oy + 2) * W + ox+2] = a22; out[(oy + 2) * W + ox+3] = a23;
            out[(oy + 3) * W + ox] = a30; out[(oy + 3) * W + ox+1] = a31;
            out[(oy + 3) * W + ox+2] = a32; out[(oy + 3) * W + ox+3] = a33;
        }
        for (; ox < W; ++ox) {
            for (int r = 0; r < 4; ++r) {
                float acc = 0.0f;
                for (int ky = 0; ky < K; ++ky) {
                    const float* in_row = in + (oy + r + ky) * in_stride + ox;
                    for (int kx = 0; kx < K; ++kx)
                        acc += in_row[kx] * ker[ky * K + kx];
                }
                out[(oy + r) * W + ox] = acc;
            }
        }
    }
    for (; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in + (oy + ky) * in_stride + ox;
                for (int kx = 0; kx < K; ++kx)
                    acc += in_row[kx] * ker[ky * K + kx];
            }
            out[oy * W + ox] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy 5: Reordered (ky,kx,oy,ox) + unroll ox by 4.
// Kernel loops outermost for input reuse; inner ox loop unrolled by 4 to
// reduce loop overhead and amortize the read-modify-write on out[].
// ---------------------------------------------------------------------------
void conv_reorder_unroll_ox(const float* in, float* out, const float* ker,
                            int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int i = 0; i < H * W; ++i)
        out[i] = 0.0f;

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            float w = ker[ky * K + kx];
            for (int oy = 0; oy < H; ++oy) {
                const float* in_row = in + (oy + ky) * in_stride + kx;
                float* out_row = out + oy * W;
                int ox = 0;
                for (; ox + 3 < W; ox += 4) {
                    out_row[ox]     += in_row[ox]     * w;
                    out_row[ox + 1] += in_row[ox + 1] * w;
                    out_row[ox + 2] += in_row[ox + 2] * w;
                    out_row[ox + 3] += in_row[ox + 3] * w;
                }
                for (; ox < W; ++ox)
                    out_row[ox] += in_row[ox] * w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy 6: Reordered (ky,kx,oy,ox) + unroll oy by 4.
// Processes 4 output rows simultaneously for each kernel position, improving
// ILP on the out[] accumulation and amortizing kernel weight broadcast.
// ---------------------------------------------------------------------------
void conv_reorder_unroll_oy(const float* in, float* out, const float* ker,
                            int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int i = 0; i < H * W; ++i)
        out[i] = 0.0f;

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            float w = ker[ky * K + kx];
            int oy = 0;
            for (; oy + 3 < H; oy += 4) {
                const float* r0 = in + (oy + ky)     * in_stride + kx;
                const float* r1 = in + (oy + 1 + ky) * in_stride + kx;
                const float* r2 = in + (oy + 2 + ky) * in_stride + kx;
                const float* r3 = in + (oy + 3 + ky) * in_stride + kx;
                float* o0 = out + oy       * W;
                float* o1 = out + (oy + 1) * W;
                float* o2 = out + (oy + 2) * W;
                float* o3 = out + (oy + 3) * W;
                for (int ox = 0; ox < W; ++ox) {
                    o0[ox] += r0[ox] * w;
                    o1[ox] += r1[ox] * w;
                    o2[ox] += r2[ox] * w;
                    o3[ox] += r3[ox] * w;
                }
            }
            for (; oy < H; ++oy) {
                const float* in_row = in + (oy + ky) * in_stride + kx;
                float* out_row = out + oy * W;
                for (int ox = 0; ox < W; ++ox)
                    out_row[ox] += in_row[ox] * w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy 7: Reordered (ky,kx,oy,ox) + unroll ox by 4 AND oy by 4.
// 4x4 output block per iteration, combining benefits of strategies 5 and 6.
// ---------------------------------------------------------------------------
void conv_reorder_unroll_ox_oy(const float* in, float* out, const float* ker,
                               int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int i = 0; i < H * W; ++i)
        out[i] = 0.0f;

    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            float w = ker[ky * K + kx];
            int oy = 0;
            for (; oy + 3 < H; oy += 4) {
                const float* r0 = in + (oy + ky)     * in_stride + kx;
                const float* r1 = in + (oy + 1 + ky) * in_stride + kx;
                const float* r2 = in + (oy + 2 + ky) * in_stride + kx;
                const float* r3 = in + (oy + 3 + ky) * in_stride + kx;
                float* o0 = out + oy       * W;
                float* o1 = out + (oy + 1) * W;
                float* o2 = out + (oy + 2) * W;
                float* o3 = out + (oy + 3) * W;
                int ox = 0;
                for (; ox + 3 < W; ox += 4) {
                    o0[ox]   += r0[ox]   * w; o0[ox+1] += r0[ox+1] * w;
                    o0[ox+2] += r0[ox+2] * w; o0[ox+3] += r0[ox+3] * w;
                    o1[ox]   += r1[ox]   * w; o1[ox+1] += r1[ox+1] * w;
                    o1[ox+2] += r1[ox+2] * w; o1[ox+3] += r1[ox+3] * w;
                    o2[ox]   += r2[ox]   * w; o2[ox+1] += r2[ox+1] * w;
                    o2[ox+2] += r2[ox+2] * w; o2[ox+3] += r2[ox+3] * w;
                    o3[ox]   += r3[ox]   * w; o3[ox+1] += r3[ox+1] * w;
                    o3[ox+2] += r3[ox+2] * w; o3[ox+3] += r3[ox+3] * w;
                }
                for (; ox < W; ++ox) {
                    o0[ox] += r0[ox] * w;
                    o1[ox] += r1[ox] * w;
                    o2[ox] += r2[ox] * w;
                    o3[ox] += r3[ox] * w;
                }
            }
            for (; oy < H; ++oy) {
                const float* in_row = in + (oy + ky) * in_stride + kx;
                float* out_row = out + oy * W;
                for (int ox = 0; ox < W; ++ox)
                    out_row[ox] += in_row[ox] * w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Strategy 8: Reordered (ky,kx,oy,ox) + unroll kx by 4.
// Instead of iterating kx one at a time, process 4 kernel columns per
// iteration. Each uses a different weight but the same output row, so the
// out[] read-modify-write is shared across 4 kernel positions.
// ---------------------------------------------------------------------------
void conv_reorder_unroll_kx(const float* in, float* out, const float* ker,
                            int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;

    for (int i = 0; i < H * W; ++i)
        out[i] = 0.0f;

    for (int ky = 0; ky < K; ++ky) {
        int kx = 0;
        for (; kx + 3 < K; kx += 4) {
            float w0 = ker[ky * K + kx];
            float w1 = ker[ky * K + kx + 1];
            float w2 = ker[ky * K + kx + 2];
            float w3 = ker[ky * K + kx + 3];
            for (int oy = 0; oy < H; ++oy) {
                const float* in_row = in + (oy + ky) * in_stride;
                float* out_row = out + oy * W;
                for (int ox = 0; ox < W; ++ox) {
                    out_row[ox] += in_row[ox + kx]     * w0
                                +  in_row[ox + kx + 1] * w1
                                +  in_row[ox + kx + 2] * w2
                                +  in_row[ox + kx + 3] * w3;
                }
            }
        }
        for (; kx < K; ++kx) {
            float w = ker[ky * K + kx];
            for (int oy = 0; oy < H; ++oy) {
                const float* in_row = in + (oy + ky) * in_stride + kx;
                float* out_row = out + oy * W;
                for (int ox = 0; ox < W; ++ox)
                    out_row[ox] += in_row[ox] * w;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch: change UNROLL_STRATEGY at the top to select which one to use.
// ---------------------------------------------------------------------------
void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
#if UNROLL_STRATEGY == 1
    conv_unroll_ox(in, out, ker, H, W, K);
#elif UNROLL_STRATEGY == 2
    conv_unroll_kx(in, out, ker, H, W, K);
#elif UNROLL_STRATEGY == 3
    conv_unroll_ky(in, out, ker, H, W, K);
#elif UNROLL_STRATEGY == 4
    conv_unroll_ox_oy(in, out, ker, H, W, K);
#elif UNROLL_STRATEGY == 5
    conv_reorder_unroll_ox(in, out, ker, H, W, K);
#elif UNROLL_STRATEGY == 6
    conv_reorder_unroll_oy(in, out, ker, H, W, K);
#elif UNROLL_STRATEGY == 7
    conv_reorder_unroll_ox_oy(in, out, ker, H, W, K);
#elif UNROLL_STRATEGY == 8
    conv_reorder_unroll_kx(in, out, ker, H, W, K);
#else
#error "Unknown UNROLL_STRATEGY"
#endif
}
