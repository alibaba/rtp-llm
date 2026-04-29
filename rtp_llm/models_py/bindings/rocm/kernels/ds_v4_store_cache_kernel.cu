#include "ds_v4_store_cache_kernel.h"
#include "rtp_llm/models_py/bindings/rocm/cuda_shims.h"
#include "rtp_llm/models_py/bindings/rocm/kernels/hip_float8_impl.h"
#include <hip/hip_fp16.h>
#include <cstdint>

namespace rtp_llm {

namespace {

// FP8 E4M3 max value for FNUZ format (MI300x)
// FNUZ e4m3 has max = 224.0
constexpr float kFP8E4M3MaxFNUZ = 224.0f;

// Convert float exponent to UE8M0 (unsigned 8-bit exponent-only format)
__device__ int32_t castToUE8M0(float x) {
    uint32_t u = __float_as_uint(x);
    int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF);
    uint32_t mant = u & 0x7FFFFF;
    return exp + (mant != 0 ? 1 : 0);
}

// Convert UE8M0 back to inverse scale as float
__device__ float invScaleUE8M0(int32_t exp) {
    return __uint_as_float(static_cast<uint32_t>((127 + 127 - exp) << 23));
}

// Convert float to FP8 e4m3fnuz byte
__device__ uint8_t toFP8FNUZ(float x) {
    return hip_fp8_impl::to_float8<4, 3, float, false, true>(x);
}

// ============================================================
// FlashMLA variant: [num_tokens, 512] input, 256 threads/token
// Each warp handles 64 elements (warps 0-7 handle 64 each = 512)
// Warps 0-6: FP8 e4m3fnuz values + UE8M0 per-warp scale
// Warp 7:   last 64 elements as BF16
// ============================================================

template <typename Float, typename IndicesT, uint32_t kPageBits>
__global__ void fusedStoreFlashMLACacheKernel(
    const void* input,
    void* cache,
    const void* indices,
    uint32_t num_tokens) {

    // Page layout: 584 bytes per page slot, aligned to 576-byte boundary
    // 584 = 576 (data) + 8 (scales)
    constexpr int64_t kPageBytes = ((584 << kPageBits) + 575) / 576 * 576;

    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t wid = tid / 32;  // warp id within block (0-7)
    const uint32_t lane = tid % 32; // lane within warp (0-31)

    if (bid >= num_tokens) return;

    const auto index = static_cast<const IndicesT*>(indices)[bid];
    const int32_t page = index >> kPageBits;
    const int32_t offset = index & ((1 << kPageBits) - 1);

    const auto page_ptr = static_cast<uint8_t*>(cache) + page * kPageBytes;
    auto* value_ptr = page_ptr + offset * 576;
    auto* scale_ptr = page_ptr + (576 << kPageBits) + offset * 8;

    const auto* input_ptr = static_cast<const Float*>(input) + bid * 512;

    if (wid != 7) {
        // Warps 0-6: each warp handles 64 floats -> FP8
        // Each thread handles 2 elements
        const int elem_base = wid * 64 + lane * 2;
        float x = static_cast<float>(input_ptr[elem_base]);
        float y = static_cast<float>(input_ptr[elem_base + 1]);

        float max_abs = fmaxf(fabsf(x), fabsf(y));

        // Warp-level reduce max
        for (int d = 16; d > 0; d >>= 1) {
            max_abs = fmaxf(max_abs, __shfl_xor(max_abs, d));
        }

        // Lane 0 computes scale and broadcasts inv_scale via shfl
        float inv_scale = 0.0f;
        int32_t scale_ue8m0 = 0;
        if (lane == 0) {
            max_abs = fmaxf(max_abs, 1e-4f);
            const float scale_raw = max_abs / kFP8E4M3MaxFNUZ;
            scale_ue8m0 = castToUE8M0(scale_raw);
            inv_scale = invScaleUE8M0(scale_ue8m0);
        }
        inv_scale = __shfl(inv_scale, 0);

        // Quantize each thread's 2 elements to FP8
        float qx = fmaxf(fminf(x * inv_scale, kFP8E4M3MaxFNUZ), -kFP8E4M3MaxFNUZ);
        float qy = fmaxf(fminf(y * inv_scale, kFP8E4M3MaxFNUZ), -kFP8E4M3MaxFNUZ);

        auto* fp8_ptr = reinterpret_cast<uint8_t*>(value_ptr) + wid * 64;
        fp8_ptr[lane * 2 + 0] = toFP8FNUZ(qx);
        fp8_ptr[lane * 2 + 1] = toFP8FNUZ(qy);

        // Lane 0 stores scale
        if (lane == 0) {
            scale_ptr[wid] = static_cast<uint8_t>(scale_ue8m0);
        }
    } else {
        // Warp 7: last 64 elements stored as BF16
        if (lane < 64) {
            const float val = static_cast<float>(input_ptr[448 + lane]);
            auto* bf16_ptr = reinterpret_cast<__hip_bfloat16*>(value_ptr + 448);
            bf16_ptr[lane] = __float2bfloat16(val);
        }
    }
}

// ============================================================
// Indexer variant: [num_tokens, 128] input, 1 warp/token
// Each warp handles 128 floats -> FP8 with per-warp FP32 scale
// ============================================================

template <typename Float, typename IndicesT, uint32_t kPageBits>
__global__ void fusedStoreIndexerCacheKernel(
    const void* input,
    void* cache,
    const void* indices,
    uint32_t num_tokens) {

    // Page layout: 132 bytes per page slot (128 data + 4 scale)
    constexpr int64_t kPageBytes = 132 << kPageBits;

    // Global warp ID = which token this warp processes
    const uint32_t global_wid = (blockIdx.x * blockDim.x + threadIdx.x) / 32;

    if (global_wid >= num_tokens) return;

    const auto index = static_cast<const IndicesT*>(indices)[global_wid];
    const int32_t page = index >> kPageBits;
    const int32_t offset = index & ((1 << kPageBits) - 1);

    const uint32_t lane_id = threadIdx.x % 32;

    const auto page_ptr = static_cast<uint8_t*>(cache) + page * kPageBytes;
    auto* value_ptr = page_ptr + offset * 128;
    auto* scale_ptr = page_ptr + (128 << kPageBits) + offset * 4;

    const auto* input_ptr = static_cast<const Float*>(input) + global_wid * 128;

    // Each thread handles 4 elements (32 threads * 4 = 128)
    float vals[4];
    float local_max = 0.0f;
    for (int i = 0; i < 4; ++i) {
        vals[i] = static_cast<float>(input_ptr[lane_id * 4 + i]);
        local_max = fmaxf(local_max, fabsf(vals[i]));
    }

    // Warp-level reduce max
    for (int d = 16; d > 0; d >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor(local_max, d));
    }

    // Lane 0 computes scale, stores it, and broadcasts inv_scale
    float inv_scale = 0.0f;
    if (lane_id == 0) {
        const float scale = fmaxf(local_max, 1e-4f) / kFP8E4M3MaxFNUZ;
        // Store FP32 scale
        reinterpret_cast<float*>(scale_ptr)[0] = scale;
        inv_scale = 1.0f / scale;
    }
    inv_scale = __shfl(inv_scale, 0);

    // Quantize and store FP8 FNUZ
    auto* fp8_ptr = reinterpret_cast<uint8_t*>(value_ptr);
    for (int i = 0; i < 4; ++i) {
        const float qx = fmaxf(fminf(vals[i] * inv_scale, kFP8E4M3MaxFNUZ), -kFP8E4M3MaxFNUZ);
        fp8_ptr[lane_id * 4 + i] = toFP8FNUZ(qx);
    }
}

}  // namespace

void invokeFusedStoreCacheFlashMLA(
    const void* input,
    void* cache,
    const void* indices,
    uint32_t num_tokens,
    uint32_t page_size,
    hipStream_t stream) {

    if (num_tokens == 0) return;

    // Compute log2 of page_size
    uint32_t page_bits = 0;
    while ((1u << page_bits) < page_size) page_bits++;

    constexpr uint32_t kBlockSize = 256;

    hipLaunchKernelGGL(
        (fusedStoreFlashMLACacheKernel<float, int32_t, 1>),
        dim3(num_tokens),
        dim3(kBlockSize),
        0,
        stream,
        input,
        cache,
        indices,
        num_tokens);
}

void invokeFusedStoreCacheIndexer(
    const void* input,
    void* cache,
    const void* indices,
    uint32_t num_tokens,
    uint32_t page_size,
    hipStream_t stream) {

    if (num_tokens == 0) return;

    uint32_t page_bits = 0;
    while ((1u << page_bits) < page_size) page_bits++;

    // Each warp processes one token, 32 threads per warp
    constexpr uint32_t kBlockSize = 128;  // 4 warps per block
    const uint32_t num_blocks = (num_tokens * 32 + kBlockSize - 1) / kBlockSize;

    hipLaunchKernelGGL(
        (fusedStoreIndexerCacheKernel<float, int32_t, 1>),
        dim3(num_blocks),
        dim3(kBlockSize),
        0,
        stream,
        input,
        cache,
        indices,
        num_tokens);
}

}  // namespace rtp_llm
