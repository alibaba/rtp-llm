// FP4 e2m1 per-token-group quant for indexer Q (deep_gemm layout).
// Matches deep_gemm.utils.per_token_cast_to_fp4(use_ue8m0=True, gran_k=32,
// use_packed_ue8m0=True).
#include "per_token_group_quant_fp4.h"
#include "util.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace rtp_llm {

namespace {

__device__ __forceinline__ float GroupReduceMax16(float val, const int lane_id) {
    unsigned mask = 0xffff;
    val           = fmaxf(val, __shfl_xor_sync(mask, val, 8));
    val           = fmaxf(val, __shfl_xor_sync(mask, val, 4));
    val           = fmaxf(val, __shfl_xor_sync(mask, val, 2));
    val           = fmaxf(val, __shfl_xor_sync(mask, val, 1));
    return val;
}

__device__ __forceinline__ float ceil_to_ue8m0_scale(float x) {
    uint32_t bits     = __float_as_uint(fabsf(x));
    uint32_t exp_bits = (bits >> 23) & 0xFFu;
    uint32_t mant     = bits & 0x7FFFFFu;
    uint32_t exp_up   = exp_bits + (mant != 0u ? 1u : 0u);
    exp_up            = max(1u, min(254u, exp_up));
    return __uint_as_float(exp_up << 23);
}

__device__ __forceinline__ uint8_t ue8m0_byte_from_scale(float scale) {
    return static_cast<uint8_t>((__float_as_uint(scale) >> 23) & 0xFFu);
}

__device__ __forceinline__ uint8_t quantize_fp4_e2m1_signed(float x_scaled) {
    float   ax   = fminf(fabsf(x_scaled), 6.0f);
    uint8_t code = 0;
    if (ax > 0.25f)
        code = 1;
    if (ax > 0.75f)
        code = 2;
    if (ax > 1.25f)
        code = 3;
    if (ax > 1.75f)
        code = 4;
    if (ax > 2.5f)
        code = 5;
    if (ax > 3.5f)
        code = 6;
    if (ax > 5.0f)
        code = 7;
    if (x_scaled < 0.0f && code != 0) {
        code |= 0x08u;
    }
    return code;
}

}  // namespace

template<typename scalar_t, bool USE_PACKED_UE8M0>
__global__ void per_token_group_quant_fp4_kernel(const scalar_t* __restrict__ input,
                                                 int8_t* __restrict__ output_q,
                                                 void* __restrict__ output_s,
                                                 const int   group_size,
                                                 const int   num_groups,
                                                 const int   groups_per_block,
                                                 const float eps) {
    constexpr int THREADS_PER_GROUP = 16;
    constexpr int ELEMS_PER_THREAD  = 2;
    constexpr int PACK_GROUPS       = 4;

    const int     local_group_id  = threadIdx.x / THREADS_PER_GROUP;
    const int     lane_id         = threadIdx.x % THREADS_PER_GROUP;
    const int64_t block_group_id  = static_cast<int64_t>(blockIdx.x) * groups_per_block;
    const int64_t global_group_id = block_group_id + local_group_id;

    if (global_group_id >= num_groups) {
        return;
    }

    const int64_t group_offset     = global_group_id * group_size;
    const int     thread_elem_base = lane_id * ELEMS_PER_THREAD;

    float vals[ELEMS_PER_THREAD];
    float local_absmax = eps;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        vals[i]      = static_cast<float>(input[group_offset + thread_elem_base + i]);
        local_absmax = fmaxf(local_absmax, fabsf(vals[i]));
    }

    local_absmax = GroupReduceMax16(local_absmax, lane_id);

    const float scale     = ceil_to_ue8m0_scale(fmaxf(local_absmax, eps) / 6.0f);
    const float inv_scale = 1.0f / scale;

    uint8_t codes[ELEMS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        codes[i] = quantize_fp4_e2m1_signed(vals[i] * inv_scale);
    }

    const int64_t out_byte_idx = (group_offset + thread_elem_base) / 2;
    output_q[out_byte_idx]     = static_cast<int8_t>((codes[0] & 0x0F) | ((codes[1] & 0x0F) << 4));

    if constexpr (USE_PACKED_UE8M0) {
        __shared__ uint8_t sm_ue8m0[PACK_GROUPS];
        if (lane_id == 0) {
            sm_ue8m0[local_group_id] = ue8m0_byte_from_scale(scale);
        }
        __syncthreads();
        if (local_group_id == 0 && lane_id == 0) {
            int32_t packed = 0;
#pragma unroll
            for (int i = 0; i < PACK_GROUPS; ++i) {
                packed |= static_cast<int32_t>(sm_ue8m0[i]) << (8 * i);
            }
            reinterpret_cast<int32_t*>(output_s)[block_group_id / PACK_GROUPS] = packed;
        }
    } else if (lane_id == 0) {
        reinterpret_cast<float*>(output_s)[global_group_id] = scale;
    }
}

void per_token_group_quant_fp4(torch::Tensor& input,
                               torch::Tensor& output_q,
                               torch::Tensor& output_s,
                               int64_t        group_size,
                               double         eps,
                               bool           use_packed_ue8m0) {
    CHECK_INPUT(input);
    CHECK_INPUT(output_q);
    CHECK_INPUT(output_s);

    TORCH_CHECK(group_size == 32, "per_token_group_quant_fp4: v1 supports group_size=32 only, got ", group_size);
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16 || input.scalar_type() == torch::kHalf,
                "per_token_group_quant_fp4: input must be bf16 or fp16");
    TORCH_CHECK(output_q.scalar_type() == torch::kChar, "per_token_group_quant_fp4: output_q must be int8");
    TORCH_CHECK(input.numel() % group_size == 0,
                "per_token_group_quant_fp4: input.numel must be divisible by group_size");
    TORCH_CHECK(output_q.numel() == input.numel() / 2, "per_token_group_quant_fp4: output_q size mismatch");

    const int num_groups = static_cast<int>(input.numel() / group_size);

    if (use_packed_ue8m0) {
        TORCH_CHECK(output_s.scalar_type() == torch::kInt,
                    "per_token_group_quant_fp4: packed UE8M0 output_s must be int32");
        TORCH_CHECK(num_groups % 4 == 0,
                    "per_token_group_quant_fp4: num_groups must be divisible by 4 for packed UE8M0");
        TORCH_CHECK(output_s.numel() == num_groups / 4,
                    "per_token_group_quant_fp4: output_s size mismatch for packed UE8M0");
    } else {
        TORCH_CHECK(output_s.scalar_type() == torch::kFloat,
                    "per_token_group_quant_fp4: unpacked output_s must be float32");
        TORCH_CHECK(output_s.numel() == num_groups,
                    "per_token_group_quant_fp4: output_s size mismatch for float32 scales");
    }

    if (num_groups == 0) {
        return;
    }

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int THREADS_PER_GROUP = 16;
    int           groups_per_block  = use_packed_ue8m0 ? 4 : 1;
    if (!use_packed_ue8m0) {
        if (num_groups % 16 == 0) {
            groups_per_block = 16;
        } else if (num_groups % 8 == 0) {
            groups_per_block = 8;
        } else if (num_groups % 4 == 0) {
            groups_per_block = 4;
        } else if (num_groups % 2 == 0) {
            groups_per_block = 2;
        }
    }

    const int num_blocks  = (num_groups + groups_per_block - 1) / groups_per_block;
    const int num_threads = groups_per_block * THREADS_PER_GROUP;

#define LAUNCH_KERNEL(T, PACKED)                                                                                       \
    do {                                                                                                               \
        dim3 grid(num_blocks);                                                                                         \
        dim3 block(num_threads);                                                                                       \
        per_token_group_quant_fp4_kernel<T, PACKED>                                                                    \
            <<<grid, block, 0, stream>>>(reinterpret_cast<const T*>(input.data_ptr()),                                 \
                                         output_q.data_ptr<int8_t>(),                                                  \
                                         output_s.data_ptr(),                                                          \
                                         static_cast<int>(group_size),                                                 \
                                         num_groups,                                                                   \
                                         groups_per_block,                                                             \
                                         static_cast<float>(eps));                                                     \
    } while (0)

    DISPATCH_PYTORCH_DTYPE_TO_CTYPE_FP16(input.scalar_type(), scalar_t, [&] {
        if (use_packed_ue8m0) {
            LAUNCH_KERNEL(scalar_t, true);
        } else {
            LAUNCH_KERNEL(scalar_t, false);
        }
        return true;
    });

#undef LAUNCH_KERNEL
}

}  // namespace rtp_llm
