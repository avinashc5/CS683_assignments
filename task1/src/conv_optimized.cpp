// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
//
// Combines the techniques that the per-stage sweeps showed actually pay off on this
// machine, plus the widest vector unit the CPU provides:
//
//   1. Loop order (oy, ox, ky, kx), i.e. the NAIVE order, deliberately kept.
//      The reorder sweep shows the "hoist the kernel loops out" variant is a loss once
//      the image stops fitting in L3 (0.83x at 2048^2, 0.64x at 8192^2, K=3) because it
//      makes K*K full passes over the image. The naive order instead touches only K
//      input rows per output row, which stays L2-resident at every size tested, so its
//      locality is already close to optimal. Tiling is likewise omitted: the tile sweep
//      peaks at 1.44x and *raises* L1-D MPKI (~1.9 -> ~8.3), i.e. it adds bookkeeping to
//      an access pattern that was not cache-limited to begin with.
//
//   2. Output-column unrolling with INDEPENDENT ACCUMULATORS -- the real win.
//      The previous version of this file used one accumulator:
//          acc = fmadd(v, w, acc);
//      so all K*K FMAs formed a single dependency chain. FMA latency is ~4 cycles while
//      throughput is 2/cycle, so that chain leaves the vector units ~87% idle. Four
//      accumulators over four adjacent column blocks give four independent chains and
//      let the FMAs actually pipeline. This is the same effect the unroll sweep found:
//      unroll_ox is the best strategy at 3.19x mean, while every reordered variant is
//      below 1.0x.
//
//   3. Widest available SIMD, chosen AT RUNTIME.
//      AVX-512 is 1.5-1.7x faster than AVX2 in the width sweep on this CPU. It is used
//      only when the CPU reports avx512f; otherwise the AVX2 path runs. This matters
//      because many Alder Lake and newer consumer parts have AVX-512 fused off, and a
//      blind 512-bit path would raise SIGILL there. The dispatch happens once.
//
// Compiler flags are NOT changed: the AVX-512 function carries an explicit
// __attribute__((target("avx512f"))) so it compiles under the project's pinned
// -mavx2 -mfma, exactly as conv_simd.cpp already does. -O2 -fno-tree-vectorize still
// apply, so every vector instruction here is one written by hand.
//
// Kernel weights are broadcast once per (ky,kx) and reused across all accumulators,
// so the K*K broadcasts are amortised over 4 vectors of output instead of 1.

#include <immintrin.h>
#include "convolution.h"
#include <cstdlib>

namespace {

// ---------------------------------------------------------------------------
// AVX2 path: 8 floats per vector, 4 accumulators -> 32 output columns per block.
// W is guaranteed to be a multiple of 8, so the 8-wide cleanup loop is exact and
// no masking is needed.
// ---------------------------------------------------------------------------
void conv_opt_avx2(const float* in, float* out, const float* ker,
                   int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int W64 = W & ~63;                 // largest multiple of 64 <= W
    const int W32 = W & ~31;                 // largest multiple of 32 <= W

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        const float* in_base = in + oy * in_stride;

        int ox = 0;
        for (; ox < W64; ox += 64) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
            __m256 a4 = _mm256_setzero_ps(), a5 = _mm256_setzero_ps();
            __m256 a6 = _mm256_setzero_ps(), a7 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in_base + ky * in_stride + ox;
                const float* krow = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(krow[kx]);
                    const float* src = in_row + kx;
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(src +  0), w, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(src +  8), w, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 16), w, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 24), w, a3);
                    a4 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 32), w, a4);
                    a5 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 40), w, a5);
                    a6 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 48), w, a6);
                    a7 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 56), w, a7);
                }
            }
            _mm256_storeu_ps(out_row + ox +  0, a0);
            _mm256_storeu_ps(out_row + ox +  8, a1);
            _mm256_storeu_ps(out_row + ox + 16, a2);
            _mm256_storeu_ps(out_row + ox + 24, a3);
            _mm256_storeu_ps(out_row + ox + 32, a4);
            _mm256_storeu_ps(out_row + ox + 40, a5);
            _mm256_storeu_ps(out_row + ox + 48, a6);
            _mm256_storeu_ps(out_row + ox + 56, a7);
        }
        for (; ox < W32; ox += 32) {
            __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
            __m256 a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in_base + ky * in_stride + ox;
                const float* krow = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m256 w = _mm256_set1_ps(krow[kx]);
                    const float* src = in_row + kx;
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(src +  0), w, a0);
                    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(src +  8), w, a1);
                    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 16), w, a2);
                    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(src + 24), w, a3);
                }
            }
            _mm256_storeu_ps(out_row + ox +  0, a0);
            _mm256_storeu_ps(out_row + ox +  8, a1);
            _mm256_storeu_ps(out_row + ox + 16, a2);
            _mm256_storeu_ps(out_row + ox + 24, a3);
        }

        for (; ox < W; ox += 8) {            // W % 8 == 0, so this is exact
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in_base + ky * in_stride + ox;
                const float* krow = ker + ky * K;
                for (int kx = 0; kx < K; ++kx)
                    acc = _mm256_fmadd_ps(_mm256_loadu_ps(in_row + kx),
                                          _mm256_set1_ps(krow[kx]), acc);
            }
            _mm256_storeu_ps(out_row + ox, acc);
        }
    }
}

// ---------------------------------------------------------------------------
// AVX-512 path: 16 floats per vector, 4 accumulators -> 64 output columns per block.
// Cleanup is a 16-wide loop plus one masked iteration, because the harness only
// guarantees W % 8 == 0: a bare 16-wide loop would run past the row when W is 8
// (mod 16). maskz loads suppress faults on the masked-off lanes.
// ---------------------------------------------------------------------------
__attribute__((target("avx512f")))
void conv_opt_avx512(const float* in, float* out, const float* ker,
                     int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;
    const int W128 = W & ~127;               // largest multiple of 128 <= W
    const int W64 = W & ~63;                 // largest multiple of 64 <= W
    const int W16 = W & ~15;                 // largest multiple of 16 <= W
    const int rem = W - W16;                 // 0 or 8 given W % 8 == 0

    for (int oy = 0; oy < H; ++oy) {
        float* out_row = out + oy * W;
        const float* in_base = in + oy * in_stride;

        int ox = 0;
        for (; ox < W128; ox += 128) {
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
            __m512 a4 = _mm512_setzero_ps(), a5 = _mm512_setzero_ps();
            __m512 a6 = _mm512_setzero_ps(), a7 = _mm512_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in_base + ky * in_stride + ox;
                const float* krow = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m512 w = _mm512_set1_ps(krow[kx]);
                    const float* src = in_row + kx;
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(src +   0), w, a0);
                    a1 = _mm512_fmadd_ps(_mm512_loadu_ps(src +  16), w, a1);
                    a2 = _mm512_fmadd_ps(_mm512_loadu_ps(src +  32), w, a2);
                    a3 = _mm512_fmadd_ps(_mm512_loadu_ps(src +  48), w, a3);
                    a4 = _mm512_fmadd_ps(_mm512_loadu_ps(src +  64), w, a4);
                    a5 = _mm512_fmadd_ps(_mm512_loadu_ps(src +  80), w, a5);
                    a6 = _mm512_fmadd_ps(_mm512_loadu_ps(src +  96), w, a6);
                    a7 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 112), w, a7);
                }
            }
            _mm512_storeu_ps(out_row + ox +   0, a0);
            _mm512_storeu_ps(out_row + ox +  16, a1);
            _mm512_storeu_ps(out_row + ox +  32, a2);
            _mm512_storeu_ps(out_row + ox +  48, a3);
            _mm512_storeu_ps(out_row + ox +  64, a4);
            _mm512_storeu_ps(out_row + ox +  80, a5);
            _mm512_storeu_ps(out_row + ox +  96, a6);
            _mm512_storeu_ps(out_row + ox + 112, a7);
        }
        for (; ox < W64; ox += 64) {
            __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
            __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();

            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in_base + ky * in_stride + ox;
                const float* krow = ker + ky * K;
                for (int kx = 0; kx < K; ++kx) {
                    const __m512 w = _mm512_set1_ps(krow[kx]);
                    const float* src = in_row + kx;
                    a0 = _mm512_fmadd_ps(_mm512_loadu_ps(src +  0), w, a0);
                    a1 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 16), w, a1);
                    a2 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 32), w, a2);
                    a3 = _mm512_fmadd_ps(_mm512_loadu_ps(src + 48), w, a3);
                }
            }
            _mm512_storeu_ps(out_row + ox +  0, a0);
            _mm512_storeu_ps(out_row + ox + 16, a1);
            _mm512_storeu_ps(out_row + ox + 32, a2);
            _mm512_storeu_ps(out_row + ox + 48, a3);
        }

        for (; ox < W16; ox += 16) {
            __m512 acc = _mm512_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in_base + ky * in_stride + ox;
                const float* krow = ker + ky * K;
                for (int kx = 0; kx < K; ++kx)
                    acc = _mm512_fmadd_ps(_mm512_loadu_ps(in_row + kx),
                                          _mm512_set1_ps(krow[kx]), acc);
            }
            _mm512_storeu_ps(out_row + ox, acc);
        }

        if (rem) {
            const __mmask16 m = (__mmask16)((1u << rem) - 1u);
            __m512 acc = _mm512_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                const float* in_row = in_base + ky * in_stride + W16;
                const float* krow = ker + ky * K;
                for (int kx = 0; kx < K; ++kx)
                    acc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(m, in_row + kx),
                                          _mm512_set1_ps(krow[kx]), acc);
            }
            _mm512_mask_storeu_ps(out_row + W16, m, acc);
        }
    }
}

// Resolved once, on first call. __builtin_cpu_supports consults CPUID, so a CPU
// without AVX-512 (or with it fused off, as on many hybrid parts) takes the AVX2 path
// instead of faulting.
bool have_avx512() {
    static const bool yes = (getenv("CONV_NO_AVX512") == nullptr) && __builtin_cpu_supports("avx512f") != 0;
    return yes;
}

}  // namespace

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    if (have_avx512()) conv_opt_avx512(in, out, ker, H, W, K);
    else               conv_opt_avx2(in, out, ker, H, W, K);
}
