// matmul_simd.cpp  STAGE 1: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "matmul.h"

void matmul_simd(const float* A, const float* B, float* C,
                 int M, int N, int K, int lda, int ldb, int ldc) {

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            const float* a = A + static_cast<long>(i) * lda;
            const float* b = B + static_cast<long>(j) * ldb;
            __m256 acc1 = _mm256_set1_ps(0); 
            __m256 acc2 = _mm256_set1_ps(0); 
            __m256 acc3 = _mm256_set1_ps(0); 
            __m256 acc4 = _mm256_set1_ps(0); 
            __m256 acc5 = _mm256_set1_ps(0); 
            __m256 acc6 = _mm256_set1_ps(0); 
            __m256 acc7 = _mm256_set1_ps(0); 
            __m256 acc8 = _mm256_set1_ps(0); 
            float out[8];
            int p = 0;
            for (; p+63 < K; p+=64) {
                __m256 vec_a_1 = _mm256_loadu_ps(a+p); 
                __m256 vec_b_1 = _mm256_loadu_ps(b+p); 
                acc1 = _mm256_fmadd_ps(vec_a_1, vec_b_1, acc1);
                
                __m256 vec_a_2 = _mm256_loadu_ps(a+p+8); 
                __m256 vec_b_2 = _mm256_loadu_ps(b+p+8); 
                acc2 = _mm256_fmadd_ps(vec_a_2, vec_b_2, acc2);
                
                __m256 vec_a_3 = _mm256_loadu_ps(a+p+16); 
                __m256 vec_b_3 = _mm256_loadu_ps(b+p+16); 
                acc3 = _mm256_fmadd_ps(vec_a_3, vec_b_3, acc3);
                
                __m256 vec_a_4 = _mm256_loadu_ps(a+p+24); 
                __m256 vec_b_4 = _mm256_loadu_ps(b+p+24); 
                acc4 = _mm256_fmadd_ps(vec_a_4, vec_b_4, acc4);
                
                __m256 vec_a_5 = _mm256_loadu_ps(a+p+32); 
                __m256 vec_b_5 = _mm256_loadu_ps(b+p+32); 
                acc5 = _mm256_fmadd_ps(vec_a_5, vec_b_5, acc5);
                
                __m256 vec_a_6 = _mm256_loadu_ps(a+p+40); 
                __m256 vec_b_6 = _mm256_loadu_ps(b+p+40); 
                acc6 = _mm256_fmadd_ps(vec_a_6, vec_b_6, acc6);
                
                __m256 vec_a_7 = _mm256_loadu_ps(a+p+48); 
                __m256 vec_b_7 = _mm256_loadu_ps(b+p+48); 
                acc7 = _mm256_fmadd_ps(vec_a_7, vec_b_7, acc7);
                
                __m256 vec_a_8 = _mm256_loadu_ps(a+p+56); 
                __m256 vec_b_8 = _mm256_loadu_ps(b+p+56); 
                acc8 = _mm256_fmadd_ps(vec_a_8, vec_b_8, acc8);
            }
            
            for (; p+31 < K; p+=32) {
                __m256 vec_a_1 = _mm256_loadu_ps(a+p); 
                __m256 vec_b_1 = _mm256_loadu_ps(b+p); 
                acc1 = _mm256_fmadd_ps(vec_a_1, vec_b_1, acc1);

                __m256 vec_a_2 = _mm256_loadu_ps(a+p+8); 
                __m256 vec_b_2 = _mm256_loadu_ps(b+p+8); 
                acc2 = _mm256_fmadd_ps(vec_a_2, vec_b_2, acc2);

                __m256 vec_a_3 = _mm256_loadu_ps(a+p+16); 
                __m256 vec_b_3 = _mm256_loadu_ps(b+p+16); 
                acc3 = _mm256_fmadd_ps(vec_a_3, vec_b_3, acc3);

                __m256 vec_a_4 = _mm256_loadu_ps(a+p+24); 
                __m256 vec_b_4 = _mm256_loadu_ps(b+p+24); 
                acc4 = _mm256_fmadd_ps(vec_a_4, vec_b_4, acc4);
            }

            for (; p+7 < K; p+=8) {
                __m256 vec_a_1 = _mm256_loadu_ps(a+p); 
                __m256 vec_b_1 = _mm256_loadu_ps(b+p); 
                acc1 = _mm256_fmadd_ps(vec_a_1, vec_b_1, acc1);
            }

            acc1 = _mm256_add_ps(acc1, acc2);
            acc3 = _mm256_add_ps(acc3, acc4);
            acc5 = _mm256_add_ps(acc5, acc6);
            acc7 = _mm256_add_ps(acc7, acc8);
            acc1 = _mm256_add_ps(acc1, acc3);
            acc5 = _mm256_add_ps(acc5, acc7);
            acc1 = _mm256_add_ps(acc1, acc5);
            _mm256_store_ps(out, acc1);
            float dot_prod = out[0]+out[1]+out[2]+out[3]+out[4]+out[5]+out[6]+out[7];
            while(p < K){
                dot_prod += a[p]*b[p];
                p++;
            }
            C[static_cast<long>(i) * ldc + j] = dot_prod;
        }
    }
}