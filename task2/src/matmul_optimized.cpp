// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.

#include <immintrin.h>

#include "matmul.h"

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {

    // a tile is made up of 
    const int BS = 64;

    // just using a tile does not have register reuse -- lots of L1d loads
    // every a[p] * b[p] loads two vals into cpu regs -- 2 loads instructions but no reg reuse happens
    // TODO: register blocking
    // v1: 

    // we have 16 ymm regs 
    const int BR = 1;
    
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            C[static_cast<long>(i) * ldc + j] = 0;
        }
    }

    int ia = 0;
    for (; ia + BS - 1 < M; ia+=BS) {
        int ja = 0;
        for (; ja + BS - 1 < K; ja+=BS) {
            // current block of A starts from ia, ja
            // now iterate over all rows and B cols at a time of B
            // iterate over all rows in the A block

            //? why is this loop order(1.5x) better than ra, ib(1.12x) ???
            for (int ib = 0; ib < N; ++ib) {
                for (int ra = 0; ra < BS; ++ra) {
                
                    const float* b = B + static_cast<long>(ib) * ldb + ja;
                    const float* a = A + static_cast<long>(ia+ra) * lda + ja;
                    float acc = 0.0f;
                    for (int p = 0; p < BS; ++p) {
                        acc += a[p] * b[p];
                    }
                    C[static_cast<long>(ia+ra) * ldc + ib] += acc;
                }
            }
        }
        
        // TODO: handle leftover in K
        int left_over_cols = K-ja;
        for (int ib = 0; ib < N; ++ib) {
            // iterate over all rows in the A block
            const float* b = B + static_cast<long>(ib) * ldb + ja;

            for (int ra = 0; ra < BS; ++ra) {
                float acc = 0.0f;
                const float* a = A + static_cast<long>(ia+ra) * lda + ja;
                for (int p = 0; p < left_over_cols; ++p) {
                    acc += a[p] * b[p];
                }
                C[static_cast<long>(ia+ra) * ldc + ib] += acc;
            }
        }
    }            

    // TODO: handle leftover rows
    int left_over_rows = M-ia;
    int ja = 0;
    for (; ja+BS-1 < K; ja+=BS) {
        // current block of A starts from ia, ja
        // now iterate over all rows and B cols at a time of B
        for (int ib = 0; ib < N; ++ib) {
            // iterate over all rows in the A block
            const float* b = B + static_cast<long>(ib) * ldb + ja;

            for (int ra = 0; ra < left_over_rows; ++ra) {
                float acc = 0.0f;
                const float* a = A + static_cast<long>(ia+ra) * lda + ja;
                for (int p = 0; p < BS; ++p) {
                    acc += a[p] * b[p];
                }
                C[static_cast<long>(ia+ra) * ldc + ib] += acc;
            }
        }
    }
    
    // TODO: handle leftover
    int left_over_cols = K-ja;
    for (int ib = 0; ib < N; ++ib) {
        // iterate over all rows in the A block
        const float* b = B + static_cast<long>(ib) * ldb + ja;

        for (int ra = 0; ra < left_over_rows; ++ra) {
            float acc = 0.0f;
            const float* a = A + static_cast<long>(ia+ra) * lda + ja;
            for (int p = 0; p < left_over_cols; ++p) {
                acc += a[p] * b[p];
            }
            C[static_cast<long>(ia+ra) * ldc + ib] += acc;
        }
    }
}
