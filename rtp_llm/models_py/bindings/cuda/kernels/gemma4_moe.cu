#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_moe.h"

#include <ATen/cuda/ScanUtils.cuh>
#include <ATen/native/cuda/SortingCommon.cuh>
#include <ATen/native/cuda/SortingRadixSelect.cuh>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

constexpr int kBf16PerVector  = sizeof(uint4) / sizeof(__nv_bfloat16);
constexpr int kGemma4Experts  = 128;
constexpr int kGemma4TopK     = 8;
constexpr int kTopKSortSize   = 32;
constexpr int kTopKSortBlockX = kTopKSortSize / 2;
constexpr int kTopKSortBlockY = 16;

template<typename T>
struct AddOp {
    __device__ __forceinline__ T operator()(const T& lhs, const T& rhs) const {
        return lhs + rhs;
    }
};

template<typename T>
__device__ inline void swapVars(T& lhs, T& rhs) {
    T tmp = lhs;
    lhs   = rhs;
    rhs   = tmp;
}

template<typename Comparator, typename K, typename V>
__device__ inline void bitonicSwap(K&                key_a,
                                   V&                value_a,
                                   bool&             valid_a,
                                   K&                key_b,
                                   V&                value_b,
                                   bool&             valid_b,
                                   bool              direction,
                                   const Comparator& comparator) {
    const bool swap = (comparator(key_a, key_b) && valid_a) || !valid_b;
    if (swap == direction) {
        swapVars(key_a, key_b);
        swapVars(value_a, value_b);
        swapVars(valid_a, valid_b);
    }
}

union alignas(16) Bf16x8 {
    uint4         packed;
    __nv_bfloat16 values[kBf16PerVector];
};

// Fixed-shape specialization of PyTorch 2.11 CUDA topk selection and small bitonic sort.
__global__ void gemma4GatherTopK8Bf16Kernel(const at::BFloat16* __restrict__ input,
                                            at::BFloat16* __restrict__ output_values,
                                            int64_t* __restrict__ output_indices,
                                            int64_t tokens) {
    __shared__ int scan_smem[32];
    const int64_t  token = blockIdx.x;
    if (token >= tokens) {
        return;
    }
    const at::BFloat16* input_row  = input + token * kGemma4Experts;
    at::BFloat16*       output_row = output_values + token * kGemma4TopK;
    int64_t*            index_row  = output_indices + token * kGemma4TopK;

    at::BFloat16 threshold;
    at::native::radixSelect<at::BFloat16, uint32_t, uint32_t>(
        input_row, kGemma4TopK, true, kGemma4Experts, 1, scan_smem, &threshold);
    const auto         threshold_bits = at::native::TopKTypeConfig<at::BFloat16>::convert(threshold);
    const int          index          = threadIdx.x;
    const at::BFloat16 value          = doLdg(input_row + index);
    const auto         value_bits     = at::native::TopKTypeConfig<at::BFloat16>::convert(value);

    int write_index;
    int carry;
    at::cuda::exclusiveBinaryPrefixScan<int, true>(
        scan_smem, value_bits > threshold_bits, &write_index, &carry, AddOp<int>());
    if (value_bits > threshold_bits) {
        output_row[write_index] = value;
        index_row[write_index]  = index;
    }
    const int first_equal_index = carry;
    const int remaining         = kGemma4TopK - first_equal_index;

    at::cuda::exclusiveBinaryPrefixScan<int, true>(
        scan_smem, value_bits == threshold_bits, &write_index, &carry, AddOp<int>());
    if (value_bits == threshold_bits && write_index < remaining) {
        output_row[first_equal_index + write_index] = value;
        index_row[first_equal_index + write_index]  = index;
    }
}

__global__ void gemma4SortTopK8Bf16Kernel(at::BFloat16* output_values, int64_t* output_indices, int64_t tokens) {
    __shared__ at::BFloat16 shared_values[kTopKSortBlockY][kTopKSortSize];
    __shared__ int64_t      shared_indices[kTopKSortBlockY][kTopKSortSize];
    __shared__ bool         shared_valid[kTopKSortBlockY][kTopKSortSize];
    const int64_t           token     = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    const bool              row_valid = token < tokens;
    auto                    values    = shared_values[threadIdx.y];
    auto                    indices   = shared_indices[threadIdx.y];
    auto                    valid     = shared_valid[threadIdx.y];

#pragma unroll
    for (int item = 0; item < 2; ++item) {
        const int  index      = threadIdx.x + item * blockDim.x;
        const bool item_valid = row_valid && index < kGemma4TopK;
        values[index]         = item_valid ? output_values[token * kGemma4TopK + index] : at::BFloat16{};
        indices[index]        = item_valid ? output_indices[token * kGemma4TopK + index] : int64_t{};
        valid[index]          = item_valid;
    }

    const at::native::GTOp<at::BFloat16, true> comparator;
#pragma unroll
    for (unsigned int size = 2; size < kTopKSortSize; size *= 2) {
        const bool direction = (threadIdx.x & (size / 2)) != 0;
#pragma unroll
        for (unsigned int stride = size / 2; stride > 0; stride /= 2) {
            __syncthreads();
            const unsigned int position = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
            bitonicSwap(values[position],
                        indices[position],
                        valid[position],
                        values[position + stride],
                        indices[position + stride],
                        valid[position + stride],
                        direction,
                        comparator);
        }
    }
#pragma unroll
    for (unsigned int stride = kTopKSortSize / 2; stride > 0; stride /= 2) {
        __syncthreads();
        const unsigned int position = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
        bitonicSwap(values[position],
                    indices[position],
                    valid[position],
                    values[position + stride],
                    indices[position + stride],
                    valid[position + stride],
                    false,
                    comparator);
    }
    __syncthreads();

    if (row_valid) {
#pragma unroll
        for (int item = 0; item < 2; ++item) {
            const int index = threadIdx.x + item * blockDim.x;
            if (index < kGemma4TopK) {
                output_values[token * kGemma4TopK + index]  = values[index];
                output_indices[token * kGemma4TopK + index] = indices[index];
            }
        }
    }
}

__global__ void gemma4WeightedReorderBf16Kernel(const uint4* __restrict__ expert_output,
                                                const __nv_bfloat16* __restrict__ sorted_weight,
                                                const int64_t* __restrict__ inverse_permutation,
                                                uint4* __restrict__ output,
                                                int64_t vector_count,
                                                int32_t vectors_per_row) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= vector_count) {
        return;
    }
    const int64_t output_row    = output_index / vectors_per_row;
    const int32_t vector_in_row = static_cast<int32_t>(output_index % vectors_per_row);
    const int64_t input_row     = inverse_permutation[output_row];
    Bf16x8        values;
    Bf16x8        weighted;
    values.packed      = expert_output[input_row * vectors_per_row + vector_in_row];
    const float weight = __bfloat162float(sorted_weight[input_row]);
#pragma unroll
    for (int i = 0; i < kBf16PerVector; ++i) {
        weighted.values[i] = __float2bfloat16_rn(__bfloat162float(values.values[i]) * weight);
    }
    output[output_index] = weighted.packed;
}

__global__ void gemma4GatherSortedExpertInputBf16Kernel(const uint4* __restrict__ input,
                                                        const int64_t* __restrict__ permutation,
                                                        uint4* __restrict__ output,
                                                        int64_t vector_count,
                                                        int32_t vectors_per_row,
                                                        int32_t top_k) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= vector_count) {
        return;
    }
    const int64_t output_row    = output_index / vectors_per_row;
    const int32_t vector_in_row = static_cast<int32_t>(output_index % vectors_per_row);
    const int64_t input_row     = permutation[output_row] / top_k;
    output[output_index]        = input[input_row * vectors_per_row + vector_in_row];
}

__global__ void gemma4Top8SumBf16Kernel(const uint4* __restrict__ input,
                                        uint4* __restrict__ output,
                                        int64_t vector_count,
                                        int32_t vectors_per_row) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= vector_count) {
        return;
    }
    const int64_t token         = output_index / vectors_per_row;
    const int32_t vector_in_row = static_cast<int32_t>(output_index % vectors_per_row);
    float         pair_sums[4][kBf16PerVector];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        Bf16x8 first;
        Bf16x8 second;
        first.packed  = input[(token * 8 + pair) * vectors_per_row + vector_in_row];
        second.packed = input[(token * 8 + pair + 4) * vectors_per_row + vector_in_row];
#pragma unroll
        for (int i = 0; i < kBf16PerVector; ++i) {
            pair_sums[pair][i] = __bfloat162float(first.values[i]) + __bfloat162float(second.values[i]);
        }
    }
    Bf16x8 reduced;
#pragma unroll
    for (int i = 0; i < kBf16PerVector; ++i) {
        const float sum   = ((pair_sums[0][i] + pair_sums[1][i]) + pair_sums[2][i]) + pair_sums[3][i];
        reduced.values[i] = __float2bfloat16_rn(sum);
    }
    output[output_index] = reduced.packed;
}

__global__ void gemma4FinalizeRouterWeightsBf16Kernel(const __nv_bfloat16* __restrict__ top_weights,
                                                      const int64_t* __restrict__ top_indices,
                                                      const __nv_bfloat16* __restrict__ expert_scales,
                                                      __nv_bfloat16* __restrict__ output,
                                                      int64_t tokens) {
    const int64_t token = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (token >= tokens) {
        return;
    }
    const int64_t offset = token * 8;
    float         pair_sums[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
        pair_sums[pair] =
            __bfloat162float(top_weights[offset + pair]) + __bfloat162float(top_weights[offset + pair + 4]);
    }
    const float         sum               = ((pair_sums[0] + pair_sums[1]) + pair_sums[2]) + pair_sums[3];
    const __nv_bfloat16 denominator       = __float2bfloat16_rn(sum);
    const float         denominator_float = __bfloat162float(denominator);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __nv_bfloat16 normalized =
            __float2bfloat16_rn(__fdiv_rn(__bfloat162float(top_weights[offset + i]), denominator_float));
        const float expert_scale = __bfloat162float(expert_scales[top_indices[offset + i]]);
        output[offset + i]       = __float2bfloat16_rn(__fmul_rn(__bfloat162float(normalized), expert_scale));
    }
}

__global__ void
gemma4ExpertHistogramKernel(const int64_t* expert_ids, int32_t* counts, int64_t rows, int32_t num_experts) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows) {
        const int64_t expert = expert_ids[index];
        if (expert >= 0 && expert < num_experts) {
            atomicAdd(counts + expert, 1);
        }
    }
}

__global__ void
gemma4ExpertPrefixKernel(const int32_t* counts, int32_t* cursors, int32_t* offsets, int32_t num_experts) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        int32_t running = 0;
        for (int32_t expert = 0; expert < num_experts; ++expert) {
            cursors[expert] = running;
            running += counts[expert];
            offsets[expert] = running;
        }
    }
}

__global__ void gemma4ExpertScatterKernel(const int64_t*       expert_ids,
                                          const __nv_bfloat16* weights,
                                          int32_t*             cursors,
                                          int64_t*             permutation,
                                          int64_t*             inverse_permutation,
                                          __nv_bfloat16*       sorted_weights,
                                          int64_t              rows) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < rows) {
        const int64_t expert         = expert_ids[index];
        const int32_t sorted_index   = atomicAdd(cursors + expert, 1);
        permutation[sorted_index]    = index;
        inverse_permutation[index]   = sorted_index;
        sorted_weights[sorted_index] = weights[index];
    }
}

}  // namespace

void invokeGemma4TopK8Bf16(const __nv_bfloat16* input,
                           __nv_bfloat16*       output_values,
                           int64_t*             output_indices,
                           int64_t              tokens,
                           cudaStream_t         stream) {
    if (tokens == 0) {
        return;
    }
    gemma4GatherTopK8Bf16Kernel<<<static_cast<uint32_t>(tokens), kGemma4Experts, 0, stream>>>(
        reinterpret_cast<const at::BFloat16*>(input),
        reinterpret_cast<at::BFloat16*>(output_values),
        output_indices,
        tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    const dim3 block(kTopKSortBlockX, kTopKSortBlockY);
    const dim3 grid(static_cast<uint32_t>((tokens + kTopKSortBlockY - 1) / kTopKSortBlockY));
    gemma4SortTopK8Bf16Kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<at::BFloat16*>(output_values), output_indices, tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4WeightedReorderBf16(const __nv_bfloat16* expert_output,
                                     const __nv_bfloat16* sorted_weight,
                                     const int64_t*       inverse_permutation,
                                     __nv_bfloat16*       output,
                                     int64_t              rows,
                                     int32_t              hidden_size,
                                     cudaStream_t         stream) {
    if (rows == 0) {
        return;
    }
    constexpr int threads         = 256;
    const int32_t vectors_per_row = hidden_size / kBf16PerVector;
    const int64_t vector_count    = rows * vectors_per_row;
    const int     blocks          = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4WeightedReorderBf16Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(expert_output),
                                                                    sorted_weight,
                                                                    inverse_permutation,
                                                                    reinterpret_cast<uint4*>(output),
                                                                    vector_count,
                                                                    vectors_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4GatherSortedExpertInputBf16(const __nv_bfloat16* input,
                                             const int64_t*       permutation,
                                             __nv_bfloat16*       output,
                                             int64_t              rows,
                                             int32_t              top_k,
                                             int32_t              hidden_size,
                                             cudaStream_t         stream) {
    if (rows == 0) {
        return;
    }
    constexpr int threads         = 256;
    const int32_t vectors_per_row = hidden_size / kBf16PerVector;
    const int64_t vector_count    = rows * vectors_per_row;
    const int     blocks          = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4GatherSortedExpertInputBf16Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(input),
                                                                            permutation,
                                                                            reinterpret_cast<uint4*>(output),
                                                                            vector_count,
                                                                            vectors_per_row,
                                                                            top_k);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4Top8SumBf16(
    const __nv_bfloat16* input, __nv_bfloat16* output, int64_t tokens, int32_t hidden_size, cudaStream_t stream) {
    if (tokens == 0) {
        return;
    }
    constexpr int threads         = 256;
    const int32_t vectors_per_row = hidden_size / kBf16PerVector;
    const int64_t vector_count    = tokens * vectors_per_row;
    const int     blocks          = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4Top8SumBf16Kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input), reinterpret_cast<uint4*>(output), vector_count, vectors_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4FinalizeRouterWeightsBf16(const __nv_bfloat16* top_weights,
                                           const int64_t*       top_indices,
                                           const __nv_bfloat16* expert_scales,
                                           __nv_bfloat16*       output,
                                           int64_t              tokens,
                                           cudaStream_t         stream) {
    if (tokens == 0) {
        return;
    }
    constexpr int threads = 256;
    const int     blocks  = static_cast<int>((tokens + threads - 1) / threads);
    gemma4FinalizeRouterWeightsBf16Kernel<<<blocks, threads, 0, stream>>>(
        top_weights, top_indices, expert_scales, output, tokens);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4PrepareGroupedMoe(const int64_t*       expert_ids,
                                   const __nv_bfloat16* weights,
                                   int64_t*             permutation,
                                   int64_t*             inverse_permutation,
                                   __nv_bfloat16*       sorted_weights,
                                   int32_t*             offsets,
                                   int32_t*             scratch,
                                   int64_t              rows,
                                   int32_t              num_experts,
                                   cudaStream_t         stream) {
    if (rows == 0) {
        return;
    }
    constexpr int threads = 256;
    const int     blocks  = static_cast<int>((rows + threads - 1) / threads);
    C10_CUDA_CHECK(cudaMemsetAsync(scratch, 0, static_cast<size_t>(num_experts * 2) * sizeof(int32_t), stream));
    int32_t* counts  = scratch;
    int32_t* cursors = scratch + num_experts;
    gemma4ExpertHistogramKernel<<<blocks, threads, 0, stream>>>(expert_ids, counts, rows, num_experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gemma4ExpertPrefixKernel<<<1, 1, 0, stream>>>(counts, cursors, offsets, num_experts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    gemma4ExpertScatterKernel<<<blocks, threads, 0, stream>>>(
        expert_ids, weights, cursors, permutation, inverse_permutation, sorted_weights, rows);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
