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
    static_assert(BS>=64, "simd assumes BS >= 64");

    // just using a tile does not have register reuse -- lots of L1d loads
    // every a[p] * b[p] loads two vals into cpu regs -- 2 loads instructions but no reg reuse happens
    // TODO: register blocking
    // v1: 

    auto hsum = [](__m256 reg, float temp[8]){
        _mm256_store_ps(temp, reg);
        return temp[0]+temp[1]+temp[2]+temp[3]+temp[4]+temp[5]+temp[6]+temp[7];
    };

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
            //* benchmark this hard to say otherwise

            for (int ib = 0; ib+3 < N; ib+=4) {
                for (int ra = 0; ra+3 < BS; ra+=4) {
                
                    const float* a = A + static_cast<long>(ia+ra) * lda + ja;
                    const float* b = B + static_cast<long>(ib) * ldb + ja;

                    __m256 acc00=_mm256_setzero_ps(), acc01=_mm256_setzero_ps(), acc02=_mm256_setzero_ps(), acc03=_mm256_setzero_ps();
                    __m256 acc10=_mm256_setzero_ps(), acc11=_mm256_setzero_ps(), acc12=_mm256_setzero_ps(), acc13=_mm256_setzero_ps();
                    __m256 acc20=_mm256_setzero_ps(), acc21=_mm256_setzero_ps(), acc22=_mm256_setzero_ps(), acc23=_mm256_setzero_ps();
                    __m256 acc30=_mm256_setzero_ps(), acc31=_mm256_setzero_ps(), acc32=_mm256_setzero_ps(), acc33=_mm256_setzero_ps();
                    
                    int p = 0;
                    for (; p+7 < BS; p+=8) {
                        __m256 av0=_mm256_loadu_ps(a      +p), av1=_mm256_loadu_ps(a+1*lda+p);
                        __m256 av2=_mm256_loadu_ps(a+2*lda+p), av3=_mm256_loadu_ps(a+3*lda+p);
                        __m256 bv0=_mm256_loadu_ps(b+      p), bv1=_mm256_loadu_ps(b+1*ldb+p);
                        __m256 bv2=_mm256_loadu_ps(b+2*lda+p), bv3=_mm256_loadu_ps(b+3*ldb+p);

                        acc00=_mm256_fmadd_ps(av0,bv0,acc00); acc01=_mm256_fmadd_ps(av0,bv1,acc01);
                        acc02=_mm256_fmadd_ps(av0,bv2,acc02); acc03=_mm256_fmadd_ps(av0,bv3,acc03);
                        acc10=_mm256_fmadd_ps(av1,bv0,acc10); acc11=_mm256_fmadd_ps(av1,bv1,acc11);
                        acc12=_mm256_fmadd_ps(av1,bv2,acc12); acc13=_mm256_fmadd_ps(av1,bv3,acc13);
                        acc20=_mm256_fmadd_ps(av2,bv0,acc20); acc21=_mm256_fmadd_ps(av2,bv1,acc21);
                        acc22=_mm256_fmadd_ps(av2,bv2,acc22); acc23=_mm256_fmadd_ps(av2,bv3,acc23);
                        acc30=_mm256_fmadd_ps(av3,bv0,acc30); acc31=_mm256_fmadd_ps(av3,bv1,acc31);
                        acc32=_mm256_fmadd_ps(av3,bv2,acc32); acc33=_mm256_fmadd_ps(av3,bv3,acc33);
                    }


                    float acc[4][4];
                    float temp[8];
                    acc[0][0]=hsum(acc00, temp); acc[0][1]=hsum(acc01, temp); acc[0][2]=hsum(acc02, temp); acc[0][3]=hsum(acc03, temp);
                    acc[1][0]=hsum(acc10, temp); acc[1][1]=hsum(acc11, temp); acc[1][2]=hsum(acc12, temp); acc[1][3]=hsum(acc13, temp);
                    acc[2][0]=hsum(acc20, temp); acc[2][1]=hsum(acc21, temp); acc[2][2]=hsum(acc22, temp); acc[2][3]=hsum(acc23, temp);
                    acc[3][0]=hsum(acc30, temp); acc[3][1]=hsum(acc31, temp); acc[3][2]=hsum(acc32, temp); acc[3][3]=hsum(acc33, temp);

                    for (; p < BS; ++p) {        // scalar tail for BS % 8 leftover
                        float av0=a[p],av1=a[lda+p],av2=a[2*lda+p],av3=a[3*lda+p];
                        float bv0=b[p],bv1=b[lda+p],bv2=b[2*lda+p],bv3=b[3*lda+p];
                        acc[0][0]+=av0*bv0; acc[0][1]+=av0*bv1; acc[0][2]+=av0*bv2; acc[0][3]+=av0*bv3;
                        acc[1][0]+=av1*bv0; acc[1][1]+=av1*bv1; acc[1][2]+=av1*bv2; acc[1][3]+=av1*bv3;
                        acc[2][0]+=av2*bv0; acc[2][1]+=av2*bv1; acc[2][2]+=av2*bv2; acc[2][3]+=av2*bv3;
                        acc[3][0]+=av3*bv0; acc[3][1]+=av3*bv1; acc[3][2]+=av3*bv2; acc[3][3]+=av3*bv3;
                    }
                    
                    int base = static_cast<long>(ia+ra) * ldc + ib;
                    C[base      ]+=acc[0][0]; C[base      +1]+=acc[0][1]; C[base      +2]+=acc[0][2]; C[base      +3]+=acc[0][3];
                    C[base+  lda]+=acc[1][0]; C[base+  lda+1]+=acc[1][1]; C[base+  lda+2]+=acc[1][2]; C[base+  lda+3]+=acc[1][3];
                    C[base+2*lda]+=acc[2][0]; C[base+2*lda+1]+=acc[2][1]; C[base+2*lda+2]+=acc[2][2]; C[base+2*lda+3]+=acc[2][3];
                    C[base+3*lda]+=acc[3][0]; C[base+3*lda+1]+=acc[3][1]; C[base+3*lda+2]+=acc[3][2]; C[base+3*lda+3]+=acc[3][3];
                    // C[static_cast<long>(ia+ra) * ldc + ib] += acc;
                }
            }

            //TODO: handle leftovers
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


                        // Aacc1 = _mm256_load_ps(A + p);
                        // Aacc2 = _mm256_load_ps(A + p +   lda);
                        // Aacc3 = _mm256_load_ps(A + p + 2*lda);
                        // Aacc4 = _mm256_load_ps(A + p + 3*lda);
                        // Aacc5 = _mm256_load_ps(A + p + 4*lda);
                        // Aacc6 = _mm256_load_ps(A + p + 5*lda);
                        // Aacc7 = _mm256_load_ps(A + p + 6*lda);
                        // Aacc8 = _mm256_load_ps(A + p + 7*lda);
                        
                        // Bacc1 = _mm256_fmadd_ps(Aacc1, Bacc1, Bacc1);
                        // Bacc2 = _mm256_fmadd_ps(Aacc2, Bacc2, Bacc2);
                        // Bacc3 = _mm256_fmadd_ps(Aacc3, Bacc3, Bacc3);
                        // Bacc4 = _mm256_fmadd_ps(Aacc4, Bacc4, Bacc4);
                        // Bacc5 = _mm256_fmadd_ps(Aacc5, Bacc5, Bacc5);
                        // Bacc6 = _mm256_fmadd_ps(Aacc6, Bacc6, Bacc6);
                        // Bacc7 = _mm256_fmadd_ps(Aacc7, Bacc7, Bacc7);
                        // Bacc8 = _mm256_fmadd_ps(Aacc8, Bacc8, Bacc8);

    //                         __m256 Aacc1 = _mm256_set1_ps(0); 
    // __m256 Aacc2 = _mm256_set1_ps(0); 
    // __m256 Aacc3 = _mm256_set1_ps(0); 
    // __m256 Aacc4 = _mm256_set1_ps(0); 
    // __m256 Aacc5 = _mm256_set1_ps(0); 
    // __m256 Aacc6 = _mm256_set1_ps(0); 
    // __m256 Aacc7 = _mm256_set1_ps(0); 
    // __m256 Aacc8 = _mm256_set1_ps(0); 

    // __m256 Bacc1 = _mm256_set1_ps(0); 
    // __m256 Bacc2 = _mm256_set1_ps(0); 
    // __m256 Bacc3 = _mm256_set1_ps(0); 
    // __m256 Bacc4 = _mm256_set1_ps(0); 
    // __m256 Bacc5 = _mm256_set1_ps(0); 
    // __m256 Bacc6 = _mm256_set1_ps(0); 
    // __m256 Bacc7 = _mm256_set1_ps(0); 
    // __m256 Bacc8 = _mm256_set1_ps(0); 