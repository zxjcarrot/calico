#pragma once

#include <cstddef>

// Include the header for all SIMD intrinsics.
// For MSVC, you might need <intrin.h>. For GCC/Clang, <immintrin.h> is sufficient.
#include <immintrin.h>

namespace hnsw_distance {

// --- Scalar Fallback Implementation ---
// This version is used when no SIMD instructions are available.
static float
L2SqrScalar(const float *pVect1, const float *pVect2, size_t qty) {
    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = pVect1[i] - pVect2[i];
        res += t * t;
    }
    return res;
}

#if defined(__AVX512F__)
// --- AVX-512 Implementation ---
static float
L2Sqr_AVX512(const float *pVect1, const float *pVect2, size_t qty) {
    alignas(64) float TmpRes[16];
    size_t qty16 = qty / 16;
    const float *pEnd1 = pVect1 + qty16 * 16;

    __m512 diff, v1, v2;
    __m512 sum = _mm512_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm512_loadu_ps(pVect1);
        pVect1 += 16;
        v2 = _mm512_loadu_ps(pVect2);
        pVect2 += 16;
        diff = _mm512_sub_ps(v1, v2);
        sum = _mm512_fmadd_ps(diff, diff, sum);
    }

    _mm512_store_ps(TmpRes, sum);
    float res = 0.0f;
    for(int i=0; i<16; ++i) res += TmpRes[i];

    // Handle the remainder
    size_t remainder = qty % 16;
    if (remainder != 0) {
        res += L2SqrScalar(pEnd1, pVect2, remainder);
    }
    return res;
}
#endif

#if defined(__AVX__)
// --- AVX Implementation ---
static float
L2Sqr_AVX(const float *pVect1, const float *pVect2, size_t qty) {
    alignas(32) float TmpRes[8];
    size_t qty8 = qty / 8;
    const float *pEnd1 = pVect1 + qty8 * 8;

    __m256 diff, v1, v2;
    __m256 sum = _mm256_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm256_loadu_ps(pVect1);
        pVect1 += 8;
        v2 = _mm256_loadu_ps(pVect2);
        pVect2 += 8;
        diff = _mm256_sub_ps(v1, v2);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    _mm256_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    
    // Handle the remainder
    size_t remainder = qty % 8;
    if (remainder != 0) {
        res += L2SqrScalar(pEnd1, pVect2, remainder);
    }
    return res;
}
#endif

#if defined(__SSE__)
// --- SSE Implementation ---
static float
L2Sqr_SSE(const float *pVect1, const float *pVect2, size_t qty) {
    alignas(16) float TmpRes[4];
    size_t qty4 = qty / 4;
    const float *pEnd1 = pVect1 + qty4 * 4;

    __m128 diff, v1, v2;
    __m128 sum = _mm_set1_ps(0);

    while (pVect1 < pEnd1) {
        v1 = _mm_loadu_ps(pVect1);
        pVect1 += 4;
        v2 = _mm_loadu_ps(pVect2);
        pVect2 += 4;
        diff = _mm_sub_ps(v1, v2);
        sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    }

    _mm_store_ps(TmpRes, sum);
    float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    
    // Handle the remainder
    size_t remainder = qty % 4;
    if (remainder != 0) {
        res += L2SqrScalar(pEnd1, pVect2, remainder);
    }
    return res;
}
#endif

// This is the main distance metric struct that will be used as a template parameter.
// It automatically selects the best available SIMD implementation at compile time.
struct SIMDAcceleratedL2 {
    static float compare(const float* v1, const float* v2, size_t dim) {
    #if defined(__AVX512F__)
        // If AVX-512 is available, use it.
        return L2Sqr_AVX512(v1, v2, dim);
    #elif defined(__AVX__)
        // If not, but AVX is available, use it.
        return L2Sqr_AVX(v1, v2, dim);
    #elif defined(__SSE__)
        // If not, but SSE is available, use it.
        return L2Sqr_SSE(v1, v2, dim);
    #else
        // Otherwise, fall back to the simple scalar implementation.
        return L2SqrScalar(v1, v2, dim);
    #endif
    }
};

} // namespace hnsw_distance