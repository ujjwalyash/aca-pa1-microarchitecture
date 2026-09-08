// conv_simd.cpp  STAGE 4: SIMD with AVX2 intrinsics
#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your AVX2 implementation.
     const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

     for (int oy = 0; oy < H; ++oy) {
        int ox=0;
        for (; ox <= W-8; ox+=8) {
            __m256 acc = _mm256_setzero_ps();
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    float k_val=ker[ky * K + kx];
                    __m256 k_temp=_mm256_set1_ps(k_val);
                    __m256 vec_in=_mm256_loadu_ps(&in[(oy + ky) * in_stride + (ox + kx)]);
                    acc = _mm256_fmadd_ps(vec_in,k_temp,acc);
                }
            }
            _mm256_storeu_ps(&out[oy*W+ox],acc);
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
