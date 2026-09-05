// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    for (int oy = 0; oy < H; ++oy) {
        for (int ox = 0; ox < W; ++ox) {
            float acc = 0.0f;
            int ky=0, kx=0;
            for (ky = 0; ky+2 < K; ky+=3) {
                for (kx = 0; kx+2 < K; kx+=3) {
                    acc += in[(oy + ky) * in_stride + (ox + kx  )] * ker[ky * K + kx];
                    acc += in[(oy + ky) * in_stride + (ox + kx+1)] * ker[ky * K + kx+1];
                    acc += in[(oy + ky) * in_stride + (ox + kx+2)] * ker[ky * K + kx+2];
                    
                    acc += in[(oy + ky+1) * in_stride + (ox + kx  )] * ker[(ky + 1) * K + kx];
                    acc += in[(oy + ky+1) * in_stride + (ox + kx+1)] * ker[(ky + 1) * K + kx+1];
                    acc += in[(oy + ky+1) * in_stride + (ox + kx+2)] * ker[(ky + 1) * K + kx+2];
                    
                    acc += in[(oy + ky+2) * in_stride + (ox + kx  )] * ker[(ky + 2) * K + kx];
                    acc += in[(oy + ky+2) * in_stride + (ox + kx+1)] * ker[(ky + 2) * K + kx+1];
                    acc += in[(oy + ky+2) * in_stride + (ox + kx+2)] * ker[(ky + 2) * K + kx+2];
                }
                while(kx < K){
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    acc += in[(oy + ky+1) * in_stride + (ox + kx)] * ker[(ky + 1) * K + kx];
                    acc += in[(oy + ky+2) * in_stride + (ox + kx)] * ker[(ky + 2) * K + kx];

                    kx++;
                }
            }
            
            while(ky < K){
                for (kx = 0; kx+2 < K; kx+=3) {
                    acc += in[(oy + ky) * in_stride + (ox + kx  )] * ker[ky * K + kx];
                    acc += in[(oy + ky) * in_stride + (ox + kx+1)] * ker[ky * K + kx+1];
                    acc += in[(oy + ky) * in_stride + (ox + kx+2)] * ker[ky * K + kx+2];
                }
                while(kx < K){
                    acc += in[(oy + ky) * in_stride + (ox + kx)] * ker[ky * K + kx];
                    kx++;
                }    
                
                ky++;
            }

            out[oy * W + ox] = acc;
        }
    }

}
