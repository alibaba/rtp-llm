// adpated from
// https://github.com/vllm-project/vllm/blob/254db42ede6beb7d3191f50084594c0ce791ce40/csrc/cache_kernels.cu#L432
#include "rtp_llm/cpp/kernels/indexer_k_quant_kernel.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <type_traits>

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

// ============================================================================
// MLA (Multi-Head Latent Attention) kernels
// ============================================================================

// Fp8KVCacheDataType enum for different cache types
enum class Fp8KVCacheDataType {
    kAuto    = 0,
    kFp8E4M3 = 1,
    kFp8E5M2 = 2,
};

// FP8 conversion with data type parameter
namespace fp8 {

template<typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__inline__ __device__ Tout scaled_convert_typed(const Tin& x, const float scale) {
    if constexpr (kv_dt == Fp8KVCacheDataType::kFp8E4M3) {
        if constexpr (std::is_same<Tout, uint8_t>::value) {
            // Convert to __nv_fp8_e4m3 first
            __nv_fp8_storage_t res = __nv_cvt_float_to_fp8(static_cast<float>(x) / scale, __NV_SATFINITE, __NV_E4M3);
            return static_cast<uint8_t>(res);
        } else {
            // For non-uint8_t types, use direct conversion
            return static_cast<Tout>(static_cast<float>(x) / scale);
        }
    } else if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        return static_cast<Tout>(x);
    }
    return Tout{};
}

}  // namespace fp8

// Concat and cache MLA kernel (non-dynamic-scaling version)
template<typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void
concat_and_cache_mla_kernel(const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
                            const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
                            cache_t* __restrict__ kv_cache,     // [num_blocks, block_size, (kv_lora_rank + pe_dim)]
                            const int64_t* __restrict__ slot_mapping,  // [num_tokens]
                            const int    block_stride,                 //
                            const int    entry_stride,                 //
                            const int    kv_c_stride,                  //
                            const int    k_pe_stride,                  //
                            const int    kv_lora_rank,                 //
                            const int    pe_dim,                       //
                            const int    block_size,                   //
                            const float* scale                         //
) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx  = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
        return;
    }
    const int64_t block_idx    = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    auto copy = [&](const scalar_t* __restrict__ src,
                    cache_t* __restrict__ dst,
                    int src_stride,
                    int dst_stride,
                    int size,
                    int offset) {
        for (int i = threadIdx.x; i < size; i += blockDim.x) {
            const int64_t src_idx = token_idx * src_stride + i;
            const int64_t dst_idx = block_idx * block_stride + block_offset * entry_stride + i + offset;
            if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
                dst[dst_idx] = src[src_idx];
            } else {
                dst[dst_idx] = fp8::scaled_convert_typed<cache_t, scalar_t, kv_dt>(src[src_idx], *scale);
            }
        }
    };

    copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
    copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

// Concat and cache MLA kernel with dynamic scaling (DeepSeek MLA)
template<typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void
concat_and_cache_ds_mla_kernel(const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
                               const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
                               cache_t* __restrict__ kv_cache,     // [num_blocks, block_size, (kv_lora_rank + pe_dim)]
                               const int64_t* __restrict__ slot_mapping,  // [num_tokens]
                               const int    block_stride,                 //
                               const int    entry_stride,                 //
                               const int    kv_c_stride,                  //
                               const int    k_pe_stride,                  //
                               const int    kv_lora_rank,                 //
                               const int    pe_dim,                       //
                               const int    block_size,                   //
                               const float* scale                         //
) {
    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx  = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
        return;
    }
    const int64_t block_idx     = slot_idx / block_size;
    const int64_t block_offset  = slot_idx % block_size;
    const int64_t dst_idx_start = block_idx * block_stride + block_offset * entry_stride;

    // For the NoPE part, each tile of 128 elements is handled by half of one warp
    // (16 threads). There are 4 total tiles, so 2 warps (64 threads).
    // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
    // The RoPE part (last 64 elements) is handled by another 1 warp (32 threads).
    // So in total, we use 3 warps (96 threads) per block.

    // Cast kv_cache to 16_bit for RoPE values
    scalar_t* kv_cache_16bit = reinterpret_cast<scalar_t*>(&kv_cache[dst_idx_start]);

    // The last warp handles the RoPE part
    if (threadIdx.x >= 64) {
        // Each thread handles two elements of RoPE
        const int8_t  pe_idx_start = (threadIdx.x - 64) * 2;
        const int64_t src_idx      = token_idx * k_pe_stride + pe_idx_start;
        // Vectorized load of two 16-bit values, performed as one 32-bit load
        const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);
        // RoPE values start after the packed 8-bit NoPE values and the
        // 32-bit scales
        const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;
        // Vectorized store of two 16-bit values, performed as one 32-bit store
        *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
        return;
    }

    // The first two warps handle the NoPE part
    const int8_t warp_idx = threadIdx.x >> 5;
    const int8_t lane_idx = threadIdx.x & 31;
    const int8_t tile_idx = warp_idx * 2 + (lane_idx >> 4);

    // Each thread handles 8 elements of NoPE
    // Load the NoPE elements for this thread into registers
    const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx.x * 8);
    // Vectorized load of eight 16-bit values, performed as an int4 load
    const int4      vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
    const scalar_t* vals    = reinterpret_cast<const scalar_t*>(&vals_i4);

    // Max absolute value of this thread's elements
    float max_abs = fmaxf(fmaxf(fmaxf(fabsf(static_cast<float>(vals[0])), fabsf(static_cast<float>(vals[1]))),
                                fmaxf(fabsf(static_cast<float>(vals[2])), fabsf(static_cast<float>(vals[3])))),
                          fmaxf(fmaxf(fabsf(static_cast<float>(vals[4])), fabsf(static_cast<float>(vals[5]))),
                                fmaxf(fabsf(static_cast<float>(vals[6])), fabsf(static_cast<float>(vals[7])))));

    // Warp-level reduction to find the max absolute value in each half-warp
#pragma unroll
    for (int offset = 8; offset > 0; offset /= 2) {
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xffffffff, max_abs, offset, 16));
    }

    // Compute the scale for the tile
    float tile_scale = max_abs / 448.f;
    tile_scale       = fmaxf(tile_scale, FLT_MIN);

    // The first lane of each half-warp writes the scale to kv_cache
    if ((lane_idx == 0) || (lane_idx == 16)) {
        float*         kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start]);
        const uint64_t dst_idx        = kv_lora_rank / 4 + tile_idx;
        kv_cache_32bit[dst_idx]       = tile_scale;
    }

    // Now all threads in the block scale and write their elements
    // NoPE data is packed in the first kv_lora_rank/2 bytes (first 256 bytes)
    const int64_t dst_idx_base = dst_idx_start + (threadIdx.x * 8);

    uint8_t result[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        result[i] = fp8::scaled_convert_typed<uint8_t, scalar_t, kv_dt>(vals[i], tile_scale);
    }

    // Store as aligned 64-bit writes
    *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) = *reinterpret_cast<const uint64_t*>(result);
}

// Concat and cache MLA kernel for MODEL1 (dynamic scaling variant)
template<typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void
concat_and_cache_ds_model1_kernel(const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
                                  const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
                                  cache_t* __restrict__ kv_cache,     // [num_blocks, block_size, bytes_per_token]
                                  const int64_t* __restrict__ slot_mapping,  // [num_tokens]
                                  const int    block_stride,                 //
                                  const int    entry_stride,                 //
                                  const int    kv_c_stride,                  //
                                  const int    k_pe_stride,                  //
                                  const int    kv_lora_rank,                 //
                                  const int    pe_dim,                       //
                                  const int    block_size,                   //
                                  const float* scale                         //
) {
    // MODEL1 Memory Layout:
    // 1. NoPE + RoPE interleaved: [NoPE: 448 bytes (FP8)] + [RoPE: 128 bytes (BF16)]
    //    Total per token: 576 bytes
    // 2. Scale Factors at end of block: 7×fp8_e8m0 + 1 byte padding = 8 bytes per token
    // Total bytes_per_token = 584 bytes

    const int64_t token_idx = blockIdx.x;
    const int64_t slot_idx  = slot_mapping[token_idx];
    // NOTE: slot_idx can be -1 if the token is padded
    if (slot_idx < 0) {
        return;
    }
    const int64_t block_idx    = slot_idx / block_size;
    const int64_t block_offset = slot_idx % block_size;

    // NoPE + RoPE section: block_idx * block_stride + block_offset * (448 + 128)
    const int64_t nope_rope_stride    = kv_lora_rank + pe_dim * sizeof(scalar_t);
    const int64_t dst_nope_rope_start = block_idx * block_stride + block_offset * nope_rope_stride;

    // Scale section: after all tokens' NoPE+RoPE, then offset by current token
    const int64_t scale_section_offset = block_size * nope_rope_stride;
    const int64_t dst_scale_start      = block_idx * block_stride + scale_section_offset + block_offset * 8;

    // For the NoPE part, each tile of 64 elements is handled by 8 threads
    // There are 7 total tiles (448 / 64 = 7), so we use 56 threads for NoPE
    // The RoPE part (64 elements) is handled by 32 threads
    // So in total, we use 3 warps (96 threads) per block to avoid warp divergence

    // Cast kv_cache to 16_bit for RoPE values
    scalar_t* kv_cache_16bit = reinterpret_cast<scalar_t*>(&kv_cache[dst_nope_rope_start]);

    // Threads 64-95 handle the RoPE part (64 BF16 values)
    if (threadIdx.x >= 64) {
        const int8_t rope_thread_idx = threadIdx.x - 64;
        // Each thread handles 2 elements of RoPE
        const int8_t  pe_idx_start = rope_thread_idx * 2;
        const int64_t src_idx      = token_idx * k_pe_stride + pe_idx_start;
        // Vectorized load of two 16-bit values, performed as one 32-bit load
        const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);
        // RoPE values start after the 448-byte NoPE section
        // Since kv_cache_16bit is scalar_t*, we need byte offset / sizeof(scalar_t)
        const int64_t dst_idx = kv_lora_rank / sizeof(scalar_t) + pe_idx_start;
        // Vectorized store of two 16-bit values, performed as one 32-bit store
        *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
        return;
    }

    // Threads 0-63 handle the NoPE part (7 tiles × 8 threads/tile)
    // Threads 56-63 are unused (only need 56 threads for 7 tiles)
    if (threadIdx.x >= 56) {
        return;
    }

    const int8_t tile_idx       = threadIdx.x / 8;  // 0-6
    const int8_t thread_in_tile = threadIdx.x % 8;  // 0-7

    // Each thread handles 8 elements of NoPE within its tile
    const int64_t src_idx_start = token_idx * kv_c_stride + tile_idx * 64 + thread_in_tile * 8;
    // Vectorized load of eight 16-bit values, performed as an int4 load
    const int4      vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
    const scalar_t* vals    = reinterpret_cast<const scalar_t*>(&vals_i4);

    // Max absolute value of this thread's elements
    float max_abs = fmaxf(fmaxf(fmaxf(fabsf(static_cast<float>(vals[0])), fabsf(static_cast<float>(vals[1]))),
                                fmaxf(fabsf(static_cast<float>(vals[2])), fabsf(static_cast<float>(vals[3])))),
                          fmaxf(fmaxf(fabsf(static_cast<float>(vals[4])), fabsf(static_cast<float>(vals[5]))),
                                fmaxf(fabsf(static_cast<float>(vals[6])), fabsf(static_cast<float>(vals[7])))));

    // Tile-level reduction to find the max absolute value in each tile (8 threads)
    // Use a mask of 0xff for the 8 threads in this tile
#pragma unroll
    for (int offset = 4; offset > 0; offset /= 2) {
        max_abs = fmaxf(max_abs, __shfl_xor_sync(0xff, max_abs, offset, 8));
    }

    // Compute the scale for the tile
    float tile_scale = max_abs / 448.f;
    tile_scale       = fmaxf(tile_scale, FLT_MIN);
    // Convert to fp8_e8m0 format (power of 2)
    tile_scale = exp2f(ceilf(log2f(tile_scale)));

    // The first thread of each tile writes the scale to the scale section
    if (thread_in_tile == 0) {
        uint8_t* scale_ptr = reinterpret_cast<uint8_t*>(&kv_cache[dst_scale_start]);
        // Convert scale to fp8_e8m0 format (stored as uint8)
        uint8_t scale_fp8   = static_cast<uint8_t>(fminf(fmaxf(log2f(tile_scale) + 127.f, 0.f), 255.f));
        scale_ptr[tile_idx] = scale_fp8;
    }

    // Now all threads write their quantized elements
    const int64_t dst_idx_base = dst_nope_rope_start + tile_idx * 64 + thread_in_tile * 8;

    uint8_t result[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        result[i] = fp8::scaled_convert_typed<uint8_t, scalar_t, kv_dt>(vals[i], tile_scale);
    }

    // Store as aligned 64-bit writes
    *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) = *reinterpret_cast<const uint64_t*>(result);
}

// Dispatch macro for MLA kernels
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)                                                             \
    concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>                                                               \
        <<<grid, block, 0, stream>>>(reinterpret_cast<const KV_T*>(kv_c.data_ptr()),                                   \
                                     reinterpret_cast<const KV_T*>(k_pe.data_ptr()),                                   \
                                     reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                                  \
                                     slot_mapping.data_ptr<int64_t>(),                                                 \
                                     block_stride,                                                                     \
                                     entry_stride,                                                                     \
                                     kv_c_stride,                                                                      \
                                     k_pe_stride,                                                                      \
                                     kv_lora_rank,                                                                     \
                                     pe_dim,                                                                           \
                                     block_size,                                                                       \
                                     reinterpret_cast<const float*>(scale.data_ptr()));

#define CALL_CONCAT_AND_CACHE_DS_MLA(KV_T, CACHE_T, KV_DTYPE)                                                          \
    concat_and_cache_ds_mla_kernel<KV_T, CACHE_T, KV_DTYPE>                                                            \
        <<<grid, block, 0, stream>>>(reinterpret_cast<const KV_T*>(kv_c.data_ptr()),                                   \
                                     reinterpret_cast<const KV_T*>(k_pe.data_ptr()),                                   \
                                     reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                                  \
                                     slot_mapping.data_ptr<int64_t>(),                                                 \
                                     block_stride,                                                                     \
                                     entry_stride,                                                                     \
                                     kv_c_stride,                                                                      \
                                     k_pe_stride,                                                                      \
                                     kv_lora_rank,                                                                     \
                                     pe_dim,                                                                           \
                                     block_size,                                                                       \
                                     reinterpret_cast<const float*>(scale.data_ptr()));

#define CALL_CONCAT_AND_CACHE_DS_MODEL1(KV_T, CACHE_T, KV_DTYPE)                                                       \
    concat_and_cache_ds_model1_kernel<KV_T, CACHE_T, KV_DTYPE>                                                         \
        <<<grid, block, 0, stream>>>(reinterpret_cast<const KV_T*>(kv_c.data_ptr()),                                   \
                                     reinterpret_cast<const KV_T*>(k_pe.data_ptr()),                                   \
                                     reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                                  \
                                     slot_mapping.data_ptr<int64_t>(),                                                 \
                                     block_stride,                                                                     \
                                     entry_stride,                                                                     \
                                     kv_c_stride,                                                                      \
                                     k_pe_stride,                                                                      \
                                     kv_lora_rank,                                                                     \
                                     pe_dim,                                                                           \
                                     block_size,                                                                       \
                                     reinterpret_cast<const float*>(scale.data_ptr()));

// Type dispatch macro based on KV cache dtype
#define DISPATCH_BY_KV_CACHE_DTYPE(SRC_DTYPE, KV_DTYPE, FN)                                                            \
    if (KV_DTYPE == "auto") {                                                                                          \
        if (SRC_DTYPE == torch::kFloat) {                                                                              \
            FN(float, float, Fp8KVCacheDataType::kAuto);                                                               \
        } else if (SRC_DTYPE == torch::kHalf) {                                                                        \
            FN(__half, __half, Fp8KVCacheDataType::kAuto);                                                             \
        } else if (SRC_DTYPE == torch::kBFloat16) {                                                                    \
            FN(__nv_bfloat16, __nv_bfloat16, Fp8KVCacheDataType::kAuto);                                               \
        } else {                                                                                                       \
            TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);                                     \
        }                                                                                                              \
    } else {                                                                                                           \
        if (KV_DTYPE == "fp8" || KV_DTYPE == "fp8_e4m3") {                                                             \
            if (SRC_DTYPE == torch::kFloat) {                                                                          \
                FN(float, uint8_t, Fp8KVCacheDataType::kFp8E4M3);                                                      \
            } else if (SRC_DTYPE == torch::kHalf) {                                                                    \
                FN(__half, uint8_t, Fp8KVCacheDataType::kFp8E4M3);                                                     \
            } else if (SRC_DTYPE == torch::kBFloat16) {                                                                \
                FN(__nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E4M3);                                              \
            } else {                                                                                                   \
                TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);                                 \
            }                                                                                                          \
        } else if (KV_DTYPE == "fp8_e5m2") {                                                                           \
            if (SRC_DTYPE == torch::kFloat) {                                                                          \
                FN(float, uint8_t, Fp8KVCacheDataType::kFp8E5M2);                                                      \
            } else if (SRC_DTYPE == torch::kHalf) {                                                                    \
                FN(__half, uint8_t, Fp8KVCacheDataType::kFp8E5M2);                                                     \
            } else if (SRC_DTYPE == torch::kBFloat16) {                                                                \
                FN(__nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E5M2);                                              \
            } else {                                                                                                   \
                TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);                                 \
            }                                                                                                          \
        } else if (KV_DTYPE == "fp8_ds_mla") {                                                                         \
            if (SRC_DTYPE == torch::kFloat) {                                                                          \
                FN(float, uint8_t, Fp8KVCacheDataType::kFp8E4M3);                                                      \
            } else if (SRC_DTYPE == torch::kHalf) {                                                                    \
                FN(__half, uint8_t, Fp8KVCacheDataType::kFp8E4M3);                                                     \
            } else if (SRC_DTYPE == torch::kBFloat16) {                                                                \
                FN(__nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E4M3);                                              \
            } else {                                                                                                   \
                TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);                                 \
            }                                                                                                          \
        } else if (KV_DTYPE == "fp8_model1_mla") {                                                                     \
            if (SRC_DTYPE == torch::kFloat) {                                                                          \
                FN(float, uint8_t, Fp8KVCacheDataType::kFp8E4M3);                                                      \
            } else if (SRC_DTYPE == torch::kHalf) {                                                                    \
                FN(__half, uint8_t, Fp8KVCacheDataType::kFp8E4M3);                                                     \
            } else if (SRC_DTYPE == torch::kBFloat16) {                                                                \
                FN(__nv_bfloat16, uint8_t, Fp8KVCacheDataType::kFp8E4M3);                                              \
            } else {                                                                                                   \
                TORCH_CHECK(false, "Unsupported input type of kv cache: ", SRC_DTYPE);                                 \
            }                                                                                                          \
        } else {                                                                                                       \
            TORCH_CHECK(false, "Unsupported data type of kv cache: ", KV_DTYPE);                                       \
        }                                                                                                              \
    }

// Main MLA concat and cache function
void concat_and_cache_mla(torch::Tensor&     kv_c,          // [num_tokens, kv_lora_rank]
                          torch::Tensor&     k_pe,          // [num_tokens, pe_dim]
                          torch::Tensor&     kv_cache,      // [num_blocks, block_size, (kv_lora_rank + pe_dim)]
                          torch::Tensor&     slot_mapping,  // [num_tokens] or [num_actual_tokens]
                          const std::string& kv_cache_dtype,
                          torch::Tensor&     scale) {
    // NOTE: In vLLM V1, key.size(0) can be different from slot_mapping.size(0) because of padding
    // for CUDA graphs. We use slot_mapping.size(0) as the number of tokens for compatibility.
    int num_tokens   = slot_mapping.size(0);
    int kv_lora_rank = kv_c.size(1);
    int pe_dim       = k_pe.size(1);
    int block_size   = kv_cache.size(1);

    if (kv_cache_dtype == "fp8_ds_mla") {
        TORCH_CHECK(kv_lora_rank == 512, "kv_lora_rank must be 512 for fp8_ds_mla");
        TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla");
        TORCH_CHECK(kv_cache.size(2) == 656 / kv_cache.element_size(),
                    "kv_cache.size(2) must be 656 bytes for fp8_ds_mla");
        TORCH_CHECK(kv_c.element_size() == 2, "kv_c.element_size() must be 2 for fp8_ds_mla");
        TORCH_CHECK(k_pe.element_size() == 2, "k_pe.element_size() must be 2 for fp8_ds_mla");
    } else if (kv_cache_dtype == "fp8_model1_mla") {
        TORCH_CHECK(kv_lora_rank == 448, "kv_lora_rank must be 448 for fp8_model1_mla");
        TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_model1_mla");
        TORCH_CHECK(kv_cache.size(2) == 584 / kv_cache.element_size(),
                    "kv_cache.size(2) must be 584 bytes for fp8_model1_mla");
        TORCH_CHECK(kv_c.element_size() == 2, "kv_c.element_size() must be 2 for fp8_model1_mla");
        TORCH_CHECK(k_pe.element_size() == 2, "k_pe.element_size() must be 2 for fp8_model1_mla");
    } else {
        TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);
    }

    int kv_c_stride  = kv_c.stride(0);
    int k_pe_stride  = k_pe.stride(0);
    int block_stride = kv_cache.stride(0);
    int entry_stride = kv_cache.stride(1);

    const c10::cuda::CUDAGuard device_guard(kv_c.device());
    const cudaStream_t         stream = c10::cuda::getCurrentCUDAStream();

    if (kv_cache_dtype == "fp8_ds_mla") {
        dim3 grid(num_tokens);
        // For the NoPE part, each tile of 128 elements is handled by half of one
        // warp (16 threads). There are 4 total tiles, so 2 warps (64 threads).
        // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
        // The RoPE part (last 64 elements) is handled by another 1 warp (32
        // threads). So in total, we use 3 warps (96 threads) per block.
        dim3 block(96);
        DISPATCH_BY_KV_CACHE_DTYPE(kv_c.scalar_type(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_DS_MLA);
    } else if (kv_cache_dtype == "fp8_model1_mla") {
        dim3 grid(num_tokens);
        // For the NoPE part, each tile of 64 elements is handled by 8 threads.
        // There are 7 total tiles (448 / 64 = 7), so 56 threads for NoPE.
        // Threads 56-63 are unused to keep full warp alignment.
        // The RoPE part (64 elements) is handled by 32 threads (threads 64-95).
        // So in total, we use 3 warps (96 threads) per block to avoid warp divergence.
        dim3 block(96);
        DISPATCH_BY_KV_CACHE_DTYPE(kv_c.scalar_type(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_DS_MODEL1);
    } else {
        dim3 grid(num_tokens);
        dim3 block(std::min(kv_lora_rank, 512));
        DISPATCH_BY_KV_CACHE_DTYPE(kv_c.scalar_type(), kv_cache_dtype, CALL_CONCAT_AND_CACHE_MLA);
    }
}

}  // namespace rtp_llm
