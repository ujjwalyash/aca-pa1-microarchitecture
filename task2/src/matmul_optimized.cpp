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

    const int BS = 256;
    const int NRA = 2;
    const int NRB = 4;
    const int PFD = 2;

    auto hsum = [](__m256 reg){
        __m128 sum = _mm_add_ps(_mm256_castps256_ps128(reg), _mm256_extractf128_ps(reg, 1));
        sum = _mm_hadd_ps(sum, sum);
        sum = _mm_hadd_ps(sum, sum);
        return _mm_cvtss_f32(sum);
    };

    #define min(a, b) (a < b ? a : b)
    
    for(int i = 0; i < M; ++i){
        for(int j = 0; j < N; ++j){
            C[static_cast<long>(i) * ldc + j] = 0.0f;
        }
    }

    for (int ia = 0; ia < M; ia += BS) {
        int i_max = min(ia + BS, M);
        
        for (int ja = 0; ja < K; ja += BS) {
            int k_max = min(ja + BS, K);
            int p_end = k_max - ja;
            
            for (int ib = 0; ib < N; ib += NRB) {
                const float* b = B + static_cast<long>(ib) * ldb + ja;
                
                for (int ra = 0; ra < (i_max - ia); ra += NRA) {
                    const float* a = A + static_cast<long>(ia+ra) * lda + ja;

                    if (ia + ra + NRA <= M && ib + NRB <= N && ja + BS <= K) {
                        __m256 acc00=_mm256_setzero_ps(), acc01=_mm256_setzero_ps(), acc02=_mm256_setzero_ps(), acc03=_mm256_setzero_ps();
                        __m256 acc10=_mm256_setzero_ps(), acc11=_mm256_setzero_ps(), acc12=_mm256_setzero_ps(), acc13=_mm256_setzero_ps();
                        
                        int p = 0;
                        for (; p+7 < BS; p+=8) {
                            __m256 av0=_mm256_loadu_ps(a      +p), av1=_mm256_loadu_ps(a+1*lda+p);
                            __m256 bv0=_mm256_loadu_ps(b+      p), bv1=_mm256_loadu_ps(b+1*ldb+p);
                            __m256 bv2=_mm256_loadu_ps(b+2*ldb+p), bv3=_mm256_loadu_ps(b+3*ldb+p);

                            acc00=_mm256_fmadd_ps(av0,bv0,acc00); acc01=_mm256_fmadd_ps(av0,bv1,acc01);
                            acc02=_mm256_fmadd_ps(av0,bv2,acc02); acc03=_mm256_fmadd_ps(av0,bv3,acc03);
                            acc10=_mm256_fmadd_ps(av1,bv0,acc10); acc11=_mm256_fmadd_ps(av1,bv1,acc11);
                            acc12=_mm256_fmadd_ps(av1,bv2,acc12); acc13=_mm256_fmadd_ps(av1,bv3,acc13);
                        }

                        float acc[2][4];
                        acc[0][0]=hsum(acc00); acc[0][1]=hsum(acc01); acc[0][2]=hsum(acc02); acc[0][3]=hsum(acc03);
                        acc[1][0]=hsum(acc10); acc[1][1]=hsum(acc11); acc[1][2]=hsum(acc12); acc[1][3]=hsum(acc13);

                        for (; p < BS; ++p) {
                            float av0=a[p],av1=a[lda+p];
                            float bv0=b[p],bv1=b[ldb+p],bv2=b[2*ldb+p],bv3=b[3*ldb+p];
                            acc[0][0]+=av0*bv0; acc[0][1]+=av0*bv1; acc[0][2]+=av0*bv2; acc[0][3]+=av0*bv3;
                            acc[1][0]+=av1*bv0; acc[1][1]+=av1*bv1; acc[1][2]+=av1*bv2; acc[1][3]+=av1*bv3;
                        }
                        
                        int base = static_cast<long>(ia+ra) * ldc + ib;
                        C[base      ]+=acc[0][0]; C[base      +1]+=acc[0][1]; C[base      +2]+=acc[0][2]; C[base      +3]+=acc[0][3];
                        C[base+  ldc]+=acc[1][0]; C[base+  ldc+1]+=acc[1][1]; C[base+  ldc+2]+=acc[1][2]; C[base+  ldc+3]+=acc[1][3];

                        _mm_prefetch(a+2*lda+PFD*lda, _MM_HINT_T0);
                        _mm_prefetch(a+3*lda+PFD*lda, _MM_HINT_T0);
                        _mm_prefetch(C+base+2*ldc+PFD*lda, _MM_HINT_T0);
                        _mm_prefetch(C+base+3*ldc+PFD*lda, _MM_HINT_T0);
                        
                    } 
                    else {
                        int r_end = min(NRA, M - (ia + ra));
                        int c_end = min(NRB, N - ib);

                        for (int r = 0; r < r_end; ++r) {
                            for (int c = 0; c < c_end; ++c) {
                                float acc = 0;
                                for (int p = 0; p < p_end; ++p) {
                                    acc += a[r * lda + p] * b[c * ldb + p];
                                }
                                C[(ia + ra + r) * ldc + (ib + c)] += acc;
                            }
                        }
                    }
                }

                _mm_prefetch(b+0*lda+PFD*lda, _MM_HINT_T0);
                _mm_prefetch(b+1*lda+PFD*lda, _MM_HINT_T0);
                _mm_prefetch(b+2*lda+PFD*lda, _MM_HINT_T0);
                _mm_prefetch(b+3*lda+PFD*lda, _MM_HINT_T0);
            }
        }
    }            
}