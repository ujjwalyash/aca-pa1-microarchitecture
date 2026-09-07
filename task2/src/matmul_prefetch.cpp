// matmul_prefetch.cpp  STAGE 2: CACHE BLOCKING + SOFTWARE PREFETCHING

#include <immintrin.h>

#include "matmul.h"

void matmul_prefetch(const float* A, const float* B, float* C,
                     int M, int N, int K, int lda, int ldb, int ldc) {

    const int PFD = 2;
    for (int i = 0; i < M; ++i) {
        const float* a = A + static_cast<long>(i) * lda;
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            const float* b = B + static_cast<long>(j) * ldb;
            for (int p = 0; p < K; ++p) {
                // this made the code slow
                // if(p%16 == 8){
                    // _mm_prefetch(a+p+8, _MM_HINT_T0);
                    // _mm_prefetch(b+p+8, _MM_HINT_T0);
                // }
                acc += a[p] * b[p];
            }
            C[static_cast<long>(i) * ldc + j] = acc;
            
            _mm_prefetch(b+PFD*ldb , _MM_HINT_T0);
            _mm_prefetch(C+PFD*ldc , _MM_HINT_T0);
        }
        
        _mm_prefetch(a+PFD*lda , _MM_HINT_T0);
    }
}
