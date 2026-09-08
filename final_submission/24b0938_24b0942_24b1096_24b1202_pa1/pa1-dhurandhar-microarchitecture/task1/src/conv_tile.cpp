// conv_tile.cpp  STAGE 3: CACHE TILING

#include "convolution.h"

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    // TODO(student): replace this placeholder with your tiled/blocked implementation.
    const int p = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride
    const int tile_size=4096;
    for (int oy = 0; oy < H; oy+=tile_size) {
        for (int ox = 0; ox < W; ox+=tile_size) {
            int y_limit=oy+tile_size;
            if(y_limit>H){y_limit=H;}
            int x_limit=ox+tile_size;
            if(x_limit>W){x_limit=W;}
            for(int y_tile=oy;y_tile< y_limit;y_tile++){
            for(int x_tile=ox;x_tile<x_limit;x_tile++){
            float acc = 0.0f;
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    acc += in[(y_tile + ky) * in_stride + (x_tile + kx)] * ker[ky * K + kx];
                }
            }
            out[y_tile * W + x_tile] = acc;
        }
    }
}
               }
    
}
