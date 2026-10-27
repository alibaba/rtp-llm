#include <gtest/gtest.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "rtp_llm/models_py/bindings/cuda/kernels/int8_lm_head.h"

namespace rtp_llm {
namespace {

template<typename T>
T* deviceCopy(const std::vector<T>& host) {
    T* ptr = nullptr;
    EXPECT_EQ(cudaMalloc(&ptr, host.size() * sizeof(T)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(ptr, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice), cudaSuccess);
    return ptr;
}

template<typename T>
std::vector<T> hostCopy(const T* ptr, size_t n) {
    std::vector<T> host(n);
    EXPECT_EQ(cudaMemcpy(host.data(), ptr, n * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    return host;
}

std::vector<__nv_bfloat16> randomBf16(size_t n, float amplitude, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-amplitude, amplitude);
    std::vector<__nv_bfloat16>            v(n);
    for (auto& e : v) {
        e = __float2bfloat16(dist(rng));
    }
    return v;
}

// Quantize W[N, K] on the GPU, then check y = x @ (Q * S)^T against an FP32 host reference built from
// the returned Q and S. Also checks Q * S reproduces W to within half a quantization step.
void checkGemv(int64_t M, int64_t N, int64_t K) {
    ASSERT_TRUE(int8LmHeadGemvSupported(M, K)) << "M=" << M << " K=" << K;
    std::mt19937 rng(static_cast<uint32_t>(M * 131 + K));
    const auto   w = randomBf16(N * K, 0.05f, rng);
    const auto   x = randomBf16(M * K, 2.0f, rng);

    __nv_bfloat16* d_w = deviceCopy(w);
    __nv_bfloat16* d_x = deviceCopy(x);
    int8_t*        d_q = nullptr;
    float *        d_s = nullptr, *d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_q, N * K), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_s, N * sizeof(float)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, M * N * sizeof(float)), cudaSuccess);

    invokeQuantizeLmHeadInt8(d_w, d_q, d_s, N, K, nullptr);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ASSERT_EQ(invokeInt8LmHeadGemv(d_q, d_s, d_x, d_y, M, N, K, nullptr), cudaSuccess);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    const auto q = hostCopy(d_q, N * K);
    const auto s = hostCopy(d_s, N);
    const auto y = hostCopy(d_y, M * N);
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t k = 0; k < K; ++k) {
            const float wv = __bfloat162float(w[n * K + k]);
            ASSERT_LE(std::fabs(q[n * K + k] * s[n] - wv), 0.5f * s[n] * 1.001f) << "n=" << n << " k=" << k;
        }
    }
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            double ref = 0.0, mag = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                const double term = static_cast<double>(q[n * K + k]) * __bfloat162float(x[m * K + k]);
                ref += term;
                mag += std::fabs(term);
            }
            ref *= s[n];
            mag *= s[n];
            ASSERT_NEAR(y[m * N + n], ref, 1e-5 * mag + 1e-6) << "m=" << m << " n=" << n;
        }
    }
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_q);
    cudaFree(d_s);
    cudaFree(d_y);
}

// Qwen3.5-27B LM head: K = hidden = 5120. N is a slice of the per-rank vocabulary; odd so the
// grid-stride row loop has a partial tail.
TEST(Int8LmHeadTest, GemvMatchesFp32Reference) {
    checkGemv(1, 2049, 5120);
    checkGemv(8, 2049, 5120);
}

// x[M, K] is staged in shared memory: shapes above the per-block opt-in limit must be rejected so the
// caller keeps the bf16 cuBLAS path, and every accepted shape must launch.
TEST(Int8LmHeadTest, SharedMemoryLimitFallsBack) {
    int dev = 0, max_smem = 0;
    ASSERT_EQ(cudaGetDevice(&dev), cudaSuccess);
    ASSERT_EQ(cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev), cudaSuccess);
    for (int64_t m = 1; m <= 8; ++m) {
        const bool fits = m * 8192 * static_cast<int64_t>(sizeof(__nv_bfloat16)) <= max_smem;
        EXPECT_EQ(int8LmHeadGemvSupported(m, 8192), fits) << "M=" << m;
        if (fits) {
            checkGemv(m, 33, 8192);
        }
    }
    if (max_smem < 8 * 8192 * static_cast<int>(sizeof(__nv_bfloat16))) {
        EXPECT_FALSE(int8LmHeadGemvSupported(8, 8192));  // 128 KiB > 99 KiB on SM120
    }
    EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

}  // namespace
}  // namespace rtp_llm
