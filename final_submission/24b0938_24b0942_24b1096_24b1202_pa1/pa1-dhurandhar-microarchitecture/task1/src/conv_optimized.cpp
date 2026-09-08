// conv_optimized.cpp  STAGE 5: PUT IT ALL TOGETHER
// Hint: measure after every change. Not every "optimization" helps  let the numbers,
// not intuition, decide.

#include <immintrin.h>

#include "convolution.h"

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    // TODO(student): replace this placeholder with your best combined implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    int lim1=W-W%64;
     for (int oy = 0; oy < H; ++oy) {
        int ox=0;
        for (; ox <lim1; ox+=64) {
            __m256 acc = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps();
            __m256 acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps();
            __m256 acc7 = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    const float* idx=&in[(oy + ky) * in_stride + (ox + kx)];
                    float k_val=ker[ky * K + kx];
                    __m256 k_temp=_mm256_set1_ps(k_val);
                    __m256 vec_in =_mm256_load_ps(idx);
                    __m256 vec_in1=_mm256_load_ps(idx+8);
                    __m256 vec_in2=_mm256_load_ps(idx+16);
                    __m256 vec_in3=_mm256_load_ps(idx+24);
                    __m256 vec_in4=_mm256_load_ps(idx+32);
                    __m256 vec_in5=_mm256_load_ps(idx+40);
                    __m256 vec_in6=_mm256_load_ps(idx+48);
                    __m256 vec_in7=_mm256_load_ps(idx+56);

                    acc = _mm256_fmadd_ps(vec_in,k_temp,acc);
                    acc1 = _mm256_fmadd_ps(vec_in1,k_temp,acc1);
                    acc2 = _mm256_fmadd_ps(vec_in2,k_temp,acc2);
                    acc3 = _mm256_fmadd_ps(vec_in3,k_temp,acc3);
                    acc4 = _mm256_fmadd_ps(vec_in4,k_temp,acc4);
                    acc5 = _mm256_fmadd_ps(vec_in5,k_temp,acc5);
                    acc6 = _mm256_fmadd_ps(vec_in6,k_temp,acc6);
                    acc7 = _mm256_fmadd_ps(vec_in7,k_temp,acc7);
                }
            }
            float* out_idx=&out[oy*W+ox];
            _mm256_storeu_ps(out_idx   ,acc);
            _mm256_storeu_ps(out_idx+8 ,acc1);
            _mm256_storeu_ps(out_idx+16,acc2);
            _mm256_storeu_ps(out_idx+24,acc3);
            _mm256_storeu_ps(out_idx+32,acc4);
            _mm256_storeu_ps(out_idx+40,acc5);
            _mm256_storeu_ps(out_idx+48,acc6);
            _mm256_storeu_ps(out_idx+56,acc7);
        }

        for(;ox<W;++ox){
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                }
            }
            out[oy * W + ox] = acc;
        }
    }
}
