// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; ++p) {
                if(p + 8 < K){
                    __builtin_prefetch(a+p+8, 0, 0);
                    __builtin_prefetch(b+p+8, 0, 0);
                }
                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
        }
    }
}
