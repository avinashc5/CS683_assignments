// matmul_prefetch.cpp  STAGE 2: SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

#ifndef PF_DIST
#define PF_DIST 64
#endif
#ifndef PF_HINT
#define PF_HINT _MM_HINT_T0
#endif

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; ++p) {
                if (p + PF_DIST < K && (p % 16 == 0)) {
                    _mm_prefetch(reinterpret_cast<const char*>(a + p + PF_DIST), PF_HINT);
                    _mm_prefetch(reinterpret_cast<const char*>(b + p + PF_DIST), PF_HINT);
                }
                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
