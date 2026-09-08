#include <immintrin.h>
#include "matmul.h"

#define min(a, b) ((a) < (b) ? (a) : (b))

inline float hsum_avx(__m256 v) { __m128 lo = _mm256_castps256_ps128(v); __m128 hi = _mm256_extractf128_ps(v, 1);
     __m128 sum = _mm_add_ps(lo, hi); sum = _mm_hadd_ps(sum, sum); sum = _mm_hadd_ps(sum, sum); return _mm_cvtss_f32(sum); }

void matmul_optimized(const float* A, const float* B, float* C,
                      int M, int N, int K, int lda, int ldb, int ldc) {
    
    const int N_TILE = 128;

    for (int j_tile = 0; j_tile < N; j_tile += N_TILE) {
        int j_max = min(j_tile + N_TILE, N);
        int i = 0;
        for (; i <= M - 3; i += 3) {
            const float* a0 = A + static_cast<long>(i + 0) * lda;
            const float* a1 = A + static_cast<long>(i + 1) * lda;
            const float* a2 = A + static_cast<long>(i + 2) * lda;
            int j = j_tile;
            for (; j <= j_max - 4; j += 4) {
                const float* b0 = B + static_cast<long>(j + 0) * ldb;
                const float* b1 = B + static_cast<long>(j + 1) * ldb;
                const float* b2 = B + static_cast<long>(j + 2) * ldb;
                const float* b3 = B + static_cast<long>(j + 3) * ldb;
                __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps(), acc02 = _mm256_setzero_ps(), acc03 = _mm256_setzero_ps();
                __m256 acc10 = _mm256_setzero_ps(), acc11 = _mm256_setzero_ps(), acc12 = _mm256_setzero_ps(), acc13 = _mm256_setzero_ps();
                __m256 acc20 = _mm256_setzero_ps(), acc21 = _mm256_setzero_ps(), acc22 = _mm256_setzero_ps(), acc23 = _mm256_setzero_ps();
                int p = 0;
                for (; p <= K - 32; p += 32) {
                    __m256 va, va1, va2, vb;
                    va = _mm256_loadu_ps(a0 + p);
                    va1 = _mm256_loadu_ps(a1 + p);
                    va2 = _mm256_loadu_ps(a2 + p);
                    
                    // we load vb (a section of row in B) once and use it 3 times (for section of 3 rows of A)
                    vb = _mm256_loadu_ps(b0 + p);
                    acc00 = _mm256_fmadd_ps(va, vb, acc00); acc10 = _mm256_fmadd_ps(va1, vb, acc10); acc20 = _mm256_fmadd_ps(va2, vb, acc20);
                    vb = _mm256_loadu_ps(b1 + p);
                    acc01 = _mm256_fmadd_ps(va, vb, acc01); acc11 = _mm256_fmadd_ps(va1, vb, acc11); acc21 = _mm256_fmadd_ps(va2, vb, acc21);
                    vb = _mm256_loadu_ps(b2 + p);
                    acc02 = _mm256_fmadd_ps(va, vb, acc02); acc12 = _mm256_fmadd_ps(va1, vb, acc12); acc22 = _mm256_fmadd_ps(va2, vb, acc22);
                    vb = _mm256_loadu_ps(b3 + p);
                    acc03 = _mm256_fmadd_ps(va, vb, acc03); acc13 = _mm256_fmadd_ps(va1, vb, acc13); acc23 = _mm256_fmadd_ps(va2, vb, acc23);

                    va = _mm256_loadu_ps(a0 + p + 8);
                    va1 = _mm256_loadu_ps(a1 + p + 8);
                    va2 = _mm256_loadu_ps(a2 + p + 8);
                    
                    vb = _mm256_loadu_ps(b0 + p + 8);
                    acc00 = _mm256_fmadd_ps(va, vb, acc00); acc10 = _mm256_fmadd_ps(va1, vb, acc10); acc20 = _mm256_fmadd_ps(va2, vb, acc20);
                    vb = _mm256_loadu_ps(b1 + p + 8);
                    acc01 = _mm256_fmadd_ps(va, vb, acc01); acc11 = _mm256_fmadd_ps(va1, vb, acc11); acc21 = _mm256_fmadd_ps(va2, vb, acc21);
                    vb = _mm256_loadu_ps(b2 + p + 8);
                    acc02 = _mm256_fmadd_ps(va, vb, acc02); acc12 = _mm256_fmadd_ps(va1, vb, acc12); acc22 = _mm256_fmadd_ps(va2, vb, acc22);
                    vb = _mm256_loadu_ps(b3 + p + 8);
                    acc03 = _mm256_fmadd_ps(va, vb, acc03); acc13 = _mm256_fmadd_ps(va1, vb, acc13); acc23 = _mm256_fmadd_ps(va2, vb, acc23);

                    va = _mm256_loadu_ps(a0 + p + 16);
                    va1 = _mm256_loadu_ps(a1 + p + 16);
                    va2 = _mm256_loadu_ps(a2 + p + 16);
                    
                    vb = _mm256_loadu_ps(b0 + p + 16);
                    acc00 = _mm256_fmadd_ps(va, vb, acc00); acc10 = _mm256_fmadd_ps(va1, vb, acc10); acc20 = _mm256_fmadd_ps(va2, vb, acc20);
                    vb = _mm256_loadu_ps(b1 + p + 16);
                    acc01 = _mm256_fmadd_ps(va, vb, acc01); acc11 = _mm256_fmadd_ps(va1, vb, acc11); acc21 = _mm256_fmadd_ps(va2, vb, acc21);
                    vb = _mm256_loadu_ps(b2 + p + 16);
                    acc02 = _mm256_fmadd_ps(va, vb, acc02); acc12 = _mm256_fmadd_ps(va1, vb, acc12); acc22 = _mm256_fmadd_ps(va2, vb, acc22);
                    vb = _mm256_loadu_ps(b3 + p + 16);
                    acc03 = _mm256_fmadd_ps(va, vb, acc03); acc13 = _mm256_fmadd_ps(va1, vb, acc13); acc23 = _mm256_fmadd_ps(va2, vb, acc23);

                    va = _mm256_loadu_ps(a0 + p + 24);
                    va1 = _mm256_loadu_ps(a1 + p + 24);
                    va2 = _mm256_loadu_ps(a2 + p + 24);
                    
                    vb = _mm256_loadu_ps(b0 + p + 24);
                    acc00 = _mm256_fmadd_ps(va, vb, acc00); acc10 = _mm256_fmadd_ps(va1, vb, acc10); acc20 = _mm256_fmadd_ps(va2, vb, acc20);
                    vb = _mm256_loadu_ps(b1 + p + 24);
                    acc01 = _mm256_fmadd_ps(va, vb, acc01); acc11 = _mm256_fmadd_ps(va1, vb, acc11); acc21 = _mm256_fmadd_ps(va2, vb, acc21);
                    vb = _mm256_loadu_ps(b2 + p + 24);
                    acc02 = _mm256_fmadd_ps(va, vb, acc02); acc12 = _mm256_fmadd_ps(va1, vb, acc12); acc22 = _mm256_fmadd_ps(va2, vb, acc22);
                    vb = _mm256_loadu_ps(b3 + p + 24);
                    acc03 = _mm256_fmadd_ps(va, vb, acc03); acc13 = _mm256_fmadd_ps(va1, vb, acc13); acc23 = _mm256_fmadd_ps(va2, vb, acc23);
                }

                for (; p <= K - 8; p += 8) {
                    __m256 va = _mm256_loadu_ps(a0 + p); __m256 va1 = _mm256_loadu_ps(a1 + p); __m256 va2 = _mm256_loadu_ps(a2 + p);
                    __m256 vb = _mm256_loadu_ps(b0 + p);
                    acc00 = _mm256_fmadd_ps(va, vb, acc00); acc10 = _mm256_fmadd_ps(va1, vb, acc10); acc20 = _mm256_fmadd_ps(va2, vb, acc20);
                    vb = _mm256_loadu_ps(b1 + p);
                    acc01 = _mm256_fmadd_ps(va, vb, acc01); acc11 = _mm256_fmadd_ps(va1, vb, acc11); acc21 = _mm256_fmadd_ps(va2, vb, acc21);
                    vb = _mm256_loadu_ps(b2 + p);
                    acc02 = _mm256_fmadd_ps(va, vb, acc02); acc12 = _mm256_fmadd_ps(va1, vb, acc12); acc22 = _mm256_fmadd_ps(va2, vb, acc22);
                    vb = _mm256_loadu_ps(b3 + p);
                    acc03 = _mm256_fmadd_ps(va, vb, acc03); acc13 = _mm256_fmadd_ps(va1, vb, acc13); acc23 = _mm256_fmadd_ps(va2, vb, acc23);
                }

                float s00 = hsum_avx(acc00),s01 = hsum_avx(acc01), s02 = hsum_avx(acc02),s03 = hsum_avx(acc03);
                float s10 = hsum_avx(acc10),s11 = hsum_avx(acc11),s12 = hsum_avx(acc12), s13 = hsum_avx(acc13);
                float s20 = hsum_avx(acc20), s21 = hsum_avx(acc21),s22 = hsum_avx(acc22),s23 = hsum_avx(acc23);

                for (; p < K; ++p) {
                    float a0v = a0[p], a1v = a1[p], a2v = a2[p];
                    float b0v = b0[p], b1v = b1[p], b2v = b2[p], b3v = b3[p];

                    s00 += a0v * b0v; s01 += a0v * b1v; s02 += a0v * b2v; s03 += a0v * b3v;
                    s10 += a1v * b0v; s11 += a1v * b1v; s12 += a1v * b2v; s13 += a1v * b3v;
                    s20 += a2v * b0v; s21 += a2v * b1v; s22 += a2v * b2v; s23 += a2v * b3v;
                }

                long idx0 = static_cast<long>(i + 0) * ldc + j;
                long idx1 = static_cast<long>(i + 1) * ldc + j;
                long idx2 = static_cast<long>(i + 2) * ldc + j;

                C[idx0 + 0] = s00; C[idx0 + 1] = s01; C[idx0 + 2] = s02; C[idx0 + 3] = s03;
                C[idx1 + 0] = s10; C[idx1 + 1] = s11; C[idx1 + 2] = s12; C[idx1 + 3] = s13;
                C[idx2 + 0] = s20; C[idx2 + 1] = s21; C[idx2 + 2] = s22; C[idx2 + 3] = s23;
            }

            for (; j < j_max; ++j) {
                const float* b0 = B + static_cast<long>(j) * ldb;
                __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps(), acc2 = _mm256_setzero_ps();
                int p = 0;
                for (; p <= K - 8; p += 8) {
                    __m256 vb = _mm256_loadu_ps(b0 + p);
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), vb, acc0);
                    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a1 + p), vb, acc1);
                    acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(a2 + p), vb, acc2);
                }
                float s0 = hsum_avx(acc0), s1 = hsum_avx(acc1), s2 = hsum_avx(acc2);
                for (; p < K; ++p) {
                    float bv = b0[p];
                    s0 += a0[p] * bv; s1 += a1[p] * bv; s2 += a2[p] * bv;
                }
                C[static_cast<long>(i + 0) * ldc + j] = s0;
                C[static_cast<long>(i + 1) * ldc + j] = s1;
                C[static_cast<long>(i + 2) * ldc + j] = s2;
            }
        }

        for (; i < M; ++i) {
            const float* a0 = A + static_cast<long>(i) * lda;
            int j = j_tile;
            for (; j <= j_max - 4; j += 4) {
                const float* b0 = B + static_cast<long>(j + 0) * ldb;
                const float* b1 = B + static_cast<long>(j + 1) * ldb;
                const float* b2 = B + static_cast<long>(j + 2) * ldb;
                const float* b3 = B + static_cast<long>(j + 3) * ldb;
                __m256 acc00 = _mm256_setzero_ps(), acc01 = _mm256_setzero_ps();
                __m256 acc02 = _mm256_setzero_ps(), acc03 = _mm256_setzero_ps();

                int p = 0;
                for (; p <= K - 16; p += 16) {
                    __m256 va;
                    va = _mm256_loadu_ps(a0 + p);
                    acc00 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + p), acc00);
                    acc01 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + p), acc01);
                    acc02 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2 + p), acc02);
                    acc03 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3 + p), acc03);
                    // Chunk 8
                    va = _mm256_loadu_ps(a0 + p + 8);
                    acc00 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + p + 8), acc00);
                    acc01 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + p + 8), acc01);
                    acc02 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2 + p + 8), acc02);
                    acc03 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3 + p + 8), acc03);
                }
                for (; p <= K - 8; p += 8) {
                    __m256 va = _mm256_loadu_ps(a0 + p);
                    acc00 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b0 + p), acc00);
                    acc01 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b1 + p), acc01);
                    acc02 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b2 + p), acc02);
                    acc03 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b3 + p), acc03);
                }

                float s00 = hsum_avx(acc00), s01 = hsum_avx(acc01), s02 = hsum_avx(acc02), s03 = hsum_avx(acc03);
                for (; p < K; ++p) {
                    float a0v = a0[p];
                    s00 += a0v * b0[p]; s01 += a0v * b1[p];
                    s02 += a0v * b2[p]; s03 += a0v * b3[p];
                }
                long idx = static_cast<long>(i) * ldc + j;
                C[idx + 0] = s00; C[idx + 1] = s01; C[idx + 2] = s02; C[idx + 3] = s03;
            }

            for (; j < j_max; ++j) {
                const float* b0 = B + static_cast<long>(j) * ldb;
                __m256 acc0 = _mm256_setzero_ps();
                int p = 0;
                for (; p <= K - 8; p += 8) {
                    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a0 + p), _mm256_loadu_ps(b0 + p), acc0);
                }
                float s0 = hsum_avx(acc0);
                for (; p < K; ++p) s0 += a0[p] * b0[p];
                C[static_cast<long>(i) * ldc + j] = s0;
            }
        }
    }
}