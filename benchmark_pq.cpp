// pq_vs_l2_benchmark_fixed.cpp
// Compile (Linux/clang or g++ with AVX2/FMA support):
//   g++ -std=c++17 -O3 -march=native -mavx2 -mfma -ffast-math pq_vs_l2_benchmark_fixed.cpp -o pq_vs_l2
//   ./pq_vs_l2

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <immintrin.h>
#include <cstdlib>   // For std::aligned_alloc, std::free
#include <cstring>
#include <functional>
#include <iterator>  // Still good practice to include

#ifndef _MSC_VER
#include <stdlib.h> // For posix_memalign on non-Windows
#endif

using namespace std;
using namespace std::chrono;

// Aligned allocator for SIMD (fully standard-compliant)
template <typename T, std::size_t Align = 32>
struct aligned_allocator {
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using size_type = std::size_t;
    using difference_type = std::ptrdiff_t;

    template <class U>
    struct rebind {
        using other = aligned_allocator<U, Align>;
    };

    aligned_allocator() noexcept = default;

    template <class U>
    aligned_allocator(const aligned_allocator<U, Align>&) noexcept {}

    T* allocate(std::size_t n) {
        void* ptr = nullptr;
        #if defined(_MSC_VER)
            ptr = _aligned_malloc(n * sizeof(T), Align);
            if (!ptr) throw std::bad_alloc();
        #elif __cplusplus >= 201703L // std::aligned_alloc is C11/C++17
            ptr = std::aligned_alloc(Align, n * sizeof(T));
            if (!ptr) throw std::bad_alloc();
        #else // Fallback for older compilers / platforms
            if (posix_memalign(&ptr, Align, n * sizeof(T)) != 0) {
                throw std::bad_alloc();
            }
        #endif
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        #if defined(_MSC_VER)
            _aligned_free(p);
        #else
            std::free(p);
        #endif
    }
};

template <typename T, std::size_t Align>
bool operator==(const aligned_allocator<T, Align>&, const aligned_allocator<T, Align>&) { return true; }
template <typename T, std::size_t Align>
bool operator!=(const aligned_allocator<T, Align>&, const aligned_allocator<T, Align>&) { return false; }


// ----------------------------------------------------------
// Configuration
// ----------------------------------------------------------
constexpr int DIM     = 128;
constexpr int M       = 128;
constexpr int KS      = 256;
constexpr int N       = 1000'000;
constexpr int QUERIES = 1000;
constexpr int TOPK    = 100;

static_assert(DIM % 8 == 0, "DIM must be a multiple of 8 for AVX2");
static_assert(DIM % M == 0, "DIM must be divisible by M for PQ");

using AlignedFloatVec = std::vector<float, aligned_allocator<float>>;
using AlignedUint8Vec = std::vector<uint8_t, aligned_allocator<uint8_t>>;

// --- Distance Functions (unchanged) ---
float l2_distance_avx(const float* a, const float* b);
float pq_distance_scalar(const uint8_t* code, const float* table);
float pq_distance_unrolled(const uint8_t* code, const float* table);

// --- Main ---

int main() {
    cout << "Allocating and initializing data..." << endl;

    AlignedFloatVec vectors(static_cast<size_t>(N) * DIM);
    AlignedUint8Vec pq_codes(static_cast<size_t>(N) * M);
    AlignedFloatVec pq_tables(static_cast<size_t>(QUERIES) * M * KS);
    AlignedFloatVec query_vec(DIM);

    std::mt19937 gen(1337);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_int_distribution<uint8_t> code_dist(0, 255);

    // These loops should now work correctly
    for (auto& v : vectors) v = dist(gen);
    for (auto& c : pq_codes) c = code_dist(gen);
    for (auto& t : pq_tables) t = dist(gen) * dist(gen);
    
    std::memcpy(query_vec.data(), vectors.data(), DIM * sizeof(float));

    cout << "Warming up..." << endl;
    volatile float dummy = 0.0f;
    for(int i = 0; i < 1000; ++i) {
        dummy += l2_distance_avx(query_vec.data(), vectors.data() + (static_cast<size_t>(i) % N) * DIM);
        dummy += pq_distance_unrolled(pq_codes.data() + (static_cast<size_t>(i) % N) * M, pq_tables.data());
    }

    // ... benchmarking sections are unchanged ...
    const long long total_ops = static_cast<long long>(QUERIES) * TOPK;
    double l2_mps = 0, pq_scalar_mps = 0, pq_unrolled_mps = 0;
    
    {
        auto start = high_resolution_clock::now();
        for (int q = 0; q < QUERIES; ++q) {
            for (int i = 0; i < TOPK; ++i) {
                size_t idx = (static_cast<size_t>(q) * TOPK + i) % N;
                float d = l2_distance_avx(query_vec.data(), vectors.data() + idx * DIM);
                asm volatile("" : "+m"(d));
            }
        }
        auto end = high_resolution_clock::now();
        double l2_sec = duration_cast<nanoseconds>(end - start).count() / 1e9;
        l2_mps = total_ops / l2_sec / 1e6;
    }

    {
        auto start = high_resolution_clock::now();
        for (int q = 0; q < QUERIES; ++q) {
            const float* cur_table = pq_tables.data() + static_cast<size_t>(q) * M * KS;
            for (int i = 0; i < TOPK; ++i) {
                size_t idx = (static_cast<size_t>(q) * TOPK + i) % N;
                float d = pq_distance_scalar(pq_codes.data() + idx * M, cur_table);
                asm volatile("" : "+m"(d));
            }
        }
        auto end = high_resolution_clock::now();
        double pq_sec = duration_cast<nanoseconds>(end - start).count() / 1e9;
        pq_scalar_mps = total_ops / pq_sec / 1e6;
    }

    {
        auto start = high_resolution_clock::now();
        for (int q = 0; q < QUERIES; ++q) {
            const float* cur_table = pq_tables.data() + static_cast<size_t>(q) * M * KS;
            for (int i = 0; i < TOPK; ++i) {
                size_t idx = (static_cast<size_t>(q) * TOPK + i) % N;
                float d = pq_distance_unrolled(pq_codes.data() + idx * M, cur_table);
                asm volatile("" : "+m"(d));
            }
        }
        auto end = high_resolution_clock::now();
        double pq_sec = duration_cast<nanoseconds>(end - start).count() / 1e9;
        pq_unrolled_mps = total_ops / pq_sec / 1e6;
    }

    cout << "\n=== DiskANN PQ vs Full-precision L2 (Pure Compute) ===\n";
    cout << "Dimension: " << DIM << ", PQ: M=" << M << ", Ks=" << KS << "\n";
    cout << "Database vectors: " << N << ", Queries: " << QUERIES << ", Dists/query: " << TOPK << "\n\n";

    cout << std::fixed << std::setprecision(2);
    cout << "L2 distance (AVX2)        : " << l2_mps << " M dists/sec\n";
    cout << "PQ distance (scalar)      : " << pq_scalar_mps << " M dists/sec\n";
    cout << "PQ distance (unrolled)    : " << pq_unrolled_mps << " M dists/sec\n\n";

    double factor = (pq_unrolled_mps > 0.0) ? (l2_mps / pq_unrolled_mps) : 0.0;
    cout << "Speed ratio (L2 AVX2 / PQ Unrolled) = " << factor << "x\n";
    cout << "=> Full L2 is " << factor << "x faster than optimized PQ for pure computation.\n";

    return 0;
}


// --- Distance Function Implementations (unchanged) ---
float l2_distance_avx(const float* a, const float* b) {
    __m256 sum = _mm256_setzero_ps();
    for (int i = 0; i < DIM; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    __m128 hi = _mm256_extractf128_ps(sum, 1);
    __m128 lo = _mm256_castps256_ps128(sum);
    __m128 s  = _mm_add_ps(hi, lo);
    s         = _mm_hadd_ps(s, s);
    s         = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

float pq_distance_scalar(const uint8_t* code, const float* table) {
    float result = 0.0f;
    for (int m = 0; m < M; ++m) {
        result += table[m * KS + code[m]];
    }
    return result;
}

float pq_distance_unrolled(const uint8_t* code, const float* table) {
    float sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f, sum4 = 0.0f;
    int m = 0;
    for (; m < M - 3; m += 4) {
        sum1 += table[(m + 0) * KS + code[m + 0]];
        sum2 += table[(m + 1) * KS + code[m + 1]];
        sum3 += table[(m + 2) * KS + code[m + 2]];
        sum4 += table[(m + 3) * KS + code[m + 3]];
    }
    for (; m < M; ++m) {
        sum1 += table[m * KS + code[m]];
    }
    return sum1 + sum2 + sum3 + sum4;
}