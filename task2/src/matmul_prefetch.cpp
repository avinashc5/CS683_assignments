// matmul_prefetch.cpp  STAGE 2: SOFTWARE PREFETCHING
//
// PF_MODE selects the prefetching strategy:
//   0 = no prefetch (baseline for comparison)
//   1 = within-row A/B prefetch (once per cache line)
//   2 = next B row prefetch
//   3 = C output prefetch
//   4 = all three combined

#include <immintrin.h>

#include "matmul.h"

#ifndef PF_DIST
#define PF_DIST 0
#endif
#ifndef PF_HINT
#define PF_HINT _MM_HINT_T0
#endif
#ifndef PF_MODE
#define PF_MODE 2
#endif

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        float* crow = C + static_cast<long>(i) * ldc;

        for (int j = 0; j < N; ++j) {
            const float* b = B + static_cast<long>(j) * ldb;

#if PF_MODE == 2 || PF_MODE == 4
            // Prefetch the start of the next B row
            if (j + 1 < N) {
                const float* b_next = B + static_cast<long>(j + 1) * ldb;
                for (int pf = 0; pf < K && pf < PF_DIST; pf += 16)
                    _mm_prefetch(reinterpret_cast<const char*>(b_next + pf), PF_HINT);
            }
#endif

#if PF_MODE == 3 || PF_MODE == 4
            // Prefetch the C output location
            _mm_prefetch(reinterpret_cast<const char*>(crow + j), PF_HINT);
#endif

            float acc = 0.0f;
            for (int p = 0; p < K; ++p) {
#if PF_MODE == 1 || PF_MODE == 4
                // Prefetch ahead within current A and B rows (once per cache line)
                if (p + PF_DIST < K && (p % 16 == 0)) {
                    _mm_prefetch(reinterpret_cast<const char*>(a + p + PF_DIST), PF_HINT);
                    _mm_prefetch(reinterpret_cast<const char*>(b + p + PF_DIST), PF_HINT);
                }
#endif
                acc += a[p] * b[p];
            }
            crow[j] = acc;
        }
    }
}
