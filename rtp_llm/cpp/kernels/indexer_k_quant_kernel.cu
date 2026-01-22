#include "rtp_llm/cpp/kernels/indexer_k_quant_kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <stdexcept>

namespace rtp_llm {

// FP8 conversion utilities
namespace fp8 {

template<typename T, typename S>
__device__ __forceinline__ T scaled_convert(const S& val, const float& scale) {
    return static_cast<T>(val / scale);
}

// Specialization for FP8 E4M3 conversion
template<>
__device__ __forceinline__ __nv_fp8_e4m3 scaled_convert<__nv_fp8_e4m3, float>(const float& val, const float& scale) {
    return __nv_fp8_e4m3(val / scale);
}

template<>
__device__ __forceinline__ __nv_fp8_e4m3 scaled_convert<__nv_fp8_e4m3, __half>(const __half& val, const float& scale) {
    return __nv_fp8_e4m3(__half2float(val) / scale);
}

template<>
__device__ __forceinline__ __nv_fp8_e4m3 scaled_convert<__nv_fp8_e4m3, __nv_bfloat16>(const __nv_bfloat16& val,
                                                                                      const float&         scale) {
    return __nv_fp8_e4m3(__bfloat162float(val) / scale);
}

}  // namespace fp8

// Warp shuffle intrinsic wrapper
template<typename T>
__device__ __forceinline__ T warp_shfl_xor(T val, int mask, int width = 32) {
    return __shfl_xor_sync(0xffffffff, val, mask, width);
}

// Indexer K quantization and cache kernel
template<typename scalar_t, typename cache_t>
__global__ void
indexer_k_quant_and_cache_kernel(const scalar_t* __restrict__ k,            // [num_tokens, head_dim]
                                 cache_t* __restrict__ kv_cache,            // [num_blocks, block_size, cache_stride]
                                 const int64_t* __restrict__ slot_mapping,  // [num_tokens]
                                 const int  head_dim,                       // dimension of each head
                                 const int  quant_block_size,               // quantization block size
                                 const int  cache_block_size,               // cache block size
                                 const int  cache_stride,                   // stride for each token in kv_cache
                                 const bool use_ue8m0                       // use ue8m0 scale format
) {
    constexpr int VEC_SIZE  = 4;
    const int64_t token_idx = blockIdx.x;
    const int64_t head_dim_idx =
        (blockIdx.y * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
    const int64_t slot_idx     = slot_mapping[token_idx];
    const int64_t block_idx    = slot_idx / cache_block_size;
    const int64_t block_offset = slot_idx % cache_block_size;

    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0 || (head_dim_idx >= head_dim)) {
        return;
    }

    // Load values as vectorized float2 (8 bytes = 4 fp16/bf16 values)
    float2    k_val     = (reinterpret_cast<const float2*>(k))[(token_idx * head_dim + head_dim_idx) / VEC_SIZE];
    scalar_t* k_val_ptr = reinterpret_cast<scalar_t*>(&k_val);

    // Find max absolute value
    float amax = 0.0f;
    for (int i = 0; i < VEC_SIZE; i++) {
        amax = fmaxf(amax, fabsf(float(k_val_ptr[i])));
    }

    // Warp reduction to find max across warp
    for (int mask = 16; mask > 0; mask /= 2) {
        amax = fmaxf(amax, warp_shfl_xor(amax, mask));
    }

    // Compute scale based on FP8 E4M3 max value (448.0)
    float scale = fmaxf(amax, 1e-4f) / 448.0f;

    // Apply ue8m0 scale format (power of 2)
    if (use_ue8m0) {
        scale = exp2f(ceilf(log2f(scale)));
    }

    // Quantize and write to cache
    const int64_t dst_offset = block_idx * cache_block_size * cache_stride + block_offset * head_dim + head_dim_idx;

    for (int i = 0; i < VEC_SIZE; i++) {
        kv_cache[dst_offset + i] = fp8::scaled_convert<cache_t, scalar_t>(k_val_ptr[i], scale);
    }

    // Write scale (first thread in warp writes scale)
    if (threadIdx.x == 0) {
        const int64_t dst_scale_idx = block_idx * cache_block_size * cache_stride + cache_block_size * head_dim
                                      + (block_offset * head_dim + head_dim_idx) * 4 / quant_block_size;
        reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = scale;
    }
}

// Gather indexer K quantized cache kernel
template<int BLOCK_Y_SIZE>
__global__ void
cp_gather_indexer_k_quant_cache_kernel(const char* __restrict__ kv_cache,  // [num_blocks, block_size, cache_stride]
                                       char* __restrict__ dst_k,           // [num_tokens, head_dim]
                                       char* __restrict__ dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
                                       const int* __restrict__ block_table,  // [batch_size, num_blocks]
                                       const int* __restrict__ cu_seq_lens,  // [batch_size + 1]
                                       const int     batch_size,             // batch size
                                       const int64_t token_stride,           // stride for each token in dst_k
                                       const int64_t head_dim,               // dimension of each head
                                       const int64_t block_stride,           // stride for each block in kv_cache
                                       const int64_t cache_token_stride,     // stride for each token in kv_cache
                                       const int64_t cache_block_size,       // num_tokens for each block in kv_cache
                                       const int     num_blocks,             // number of blocks
                                       const int     num_tokens,             // number of tokens
                                       const int     quant_block_size        // quantization block size
) {
    constexpr int VEC_SIZE  = sizeof(float4) / sizeof(char);
    const int     token_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int     head_idx  = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;

    // Find batch index within a block
    __shared__ int batch_idx[BLOCK_Y_SIZE];
    for (int iter = 0; iter < (batch_size + blockDim.x - 1) / blockDim.x; iter++) {
        int tid = iter * blockDim.x + threadIdx.x;
        if (tid < batch_size) {
            const int seq_start = cu_seq_lens[tid];
            const int seq_end   = cu_seq_lens[tid + 1];
            if (token_idx >= seq_start && token_idx < seq_end) {
                batch_idx[threadIdx.y] = tid;
            }
        }
    }

    __syncwarp();

    if (head_idx >= head_dim || token_idx >= num_tokens) {
        return;
    }
    const int     inbatch_seq_idx = token_idx - cu_seq_lens[batch_idx[threadIdx.y]];
    const int     block_idx = block_table[batch_idx[threadIdx.y] * num_blocks + inbatch_seq_idx / cache_block_size];
    const int64_t src_block_offset     = block_idx * block_stride;
    const int64_t cache_inblock_offset = (inbatch_seq_idx % cache_block_size) * head_dim + head_idx;
    const int64_t src_inblock_offset   = src_block_offset + cache_inblock_offset;
    const int64_t dst_inblock_offset   = token_idx * token_stride + head_idx;

    reinterpret_cast<float4*>(dst_k)[dst_inblock_offset / VEC_SIZE] =
        reinterpret_cast<const float4*>(kv_cache)[src_inblock_offset / VEC_SIZE];
    ;
    if (threadIdx.x == 0) {
        const int64_t src_scale_offset =
            src_block_offset + cache_block_size * head_dim + cache_inblock_offset * 4 / quant_block_size;
        reinterpret_cast<float*>(dst_scale)[dst_inblock_offset / quant_block_size] =
            reinterpret_cast<const float*>(kv_cache)[src_scale_offset / 4];
    }
}

// Macro to dispatch kernel based on data type
#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T)                                                                  \
    indexer_k_quant_and_cache_kernel<KV_T, CACHE_T>                                                                    \
        <<<grid, block, 0, stream>>>(reinterpret_cast<const KV_T*>(k.data_ptr()),                                      \
                                     reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                                  \
                                     slot_mapping.data_ptr<int64_t>(),                                                 \
                                     head_dim,                                                                         \
                                     quant_block_size,                                                                 \
                                     cache_block_size,                                                                 \
                                     cache_stride,                                                                     \
                                     use_ue8m0);

void indexer_k_quant_and_cache(torch::Tensor&     k,                 // [num_tokens, head_dim]
                               torch::Tensor&     kv_cache,          // [num_blocks, block_size, cache_stride]
                               torch::Tensor&     slot_mapping,      // [num_tokens]
                               int64_t            quant_block_size,  // quantization block size
                               const std::string& scale_fmt) {

    int  num_tokens       = k.size(0);
    int  head_dim         = k.size(1);
    int  cache_block_size = kv_cache.size(1);
    int  cache_stride     = kv_cache.size(2);
    bool use_ue8m0        = (scale_fmt == "ue8m0");

    // Validation checks
    TORCH_CHECK(k.device() == kv_cache.device(), "k and kv_cache must be on the same device");
    TORCH_CHECK(k.device() == slot_mapping.device(), "k and slot_mapping must be on the same device");
    TORCH_CHECK(head_dim % quant_block_size == 0, "head_dim must be divisible by quant_block_size");

    constexpr int vec_size = 4;
    dim3          grid(num_tokens, (head_dim + quant_block_size * vec_size - 1) / (quant_block_size * vec_size));
    dim3          block(32, vec_size);

    const c10::cuda::CUDAGuard device_guard(k.device());
    const cudaStream_t         stream = c10::cuda::getCurrentCUDAStream();

    // Dispatch based on input data type
    if (k.scalar_type() == torch::kBFloat16) {
        CALL_INDEXER_K_QUANT_AND_CACHE(__nv_bfloat16, __nv_fp8_e4m3);
    } else {
        TORCH_CHECK(false, "Unsupported data type for indexer_k_quant_and_cache");
    }
}

// Macro to dispatch gather kernel based on block size
#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)                                                             \
    cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>                                                               \
        <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE, (head_dim + 8 * vec_size - 1) / (8 * vec_size)),       \
           dim3(8, BLOCK_Y_SIZE),                                                                                      \
           0,                                                                                                          \
           stream>>>(reinterpret_cast<const char*>(kv_cache.data_ptr()),                                               \
                     reinterpret_cast<char*>(dst_k.data_ptr()),                                                        \
                     reinterpret_cast<char*>(dst_scale.data_ptr()),                                                    \
                     block_table.data_ptr<int32_t>(),                                                                  \
                     cu_seq_lens.data_ptr<int32_t>(),                                                                  \
                     batch_size,                                                                                       \
                     dst_k.stride(0),                                                                                  \
                     dst_k.size(1),                                                                                    \
                     kv_cache.stride(0),                                                                               \
                     kv_cache.stride(1),                                                                               \
                     kv_cache.size(1),                                                                                 \
                     block_table.size(1),                                                                              \
                     num_tokens,                                                                                       \
                     quant_block_size);

void cp_gather_indexer_k_quant_cache(const torch::Tensor& kv_cache,     // [num_blocks, block_size, cache_stride]
                                     torch::Tensor&       dst_k,        // [num_tokens, head_dim]
                                     torch::Tensor&       dst_scale,    // [num_tokens, head_dim / quant_block_size * 4]
                                     const torch::Tensor& block_table,  // [batch_size, num_blocks]
                                     const torch::Tensor& cu_seq_lens   // [batch_size + 1]
) {
    int batch_size       = block_table.size(0);
    int num_tokens       = dst_k.size(0);
    int head_dim         = dst_k.size(1);
    int quant_block_size = head_dim * 4 / dst_scale.size(1);

    TORCH_CHECK(kv_cache.device() == dst_k.device(), "kv_cache and dst_k must be on the same device");
    TORCH_CHECK(kv_cache.device() == dst_scale.device(), "kv_cache and dst_scale must be on the same device");
    TORCH_CHECK(kv_cache.device() == block_table.device(), "kv_cache and block_table must be on the same device");
    TORCH_CHECK(kv_cache.device() == cu_seq_lens.device(), "kv_cache and cu_seq_lens must be on the same device");
    TORCH_CHECK(head_dim % quant_block_size == 0, "head_dim must be divisible by quant_block_size");

    constexpr int              vec_size = 16;
    const c10::cuda::CUDAGuard device_guard(kv_cache.device());
    const cudaStream_t         stream = c10::cuda::getCurrentCUDAStream();

    // Select optimal block size based on number of tokens
    if (num_tokens < 32) {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
    } else if (num_tokens < 64) {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
    } else if (num_tokens < 128) {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
    } else if (num_tokens < 256) {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
    } else if (num_tokens < 512) {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
    } else {
        CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(32);
    }
}

}  // namespace rtp_llm
