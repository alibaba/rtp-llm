#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_rmsnorm_parts.h"

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include <algorithm>

namespace rtp_llm {
namespace {

template<bool Strided>
__device__ __forceinline__ int64_t
inputOffset(int64_t index, int32_t head_count, int64_t token_stride, int64_t head_stride, int32_t hidden_size) {
    if constexpr (!Strided) {
        return index;
    }
    const int64_t row = index / hidden_size;
    const int64_t dim = index % hidden_size;
    return (row / head_count) * token_stride + (row % head_count) * head_stride + dim;
}

template<bool Strided>
__global__ void gemma4RmsSquareBf16Kernel(const __nv_bfloat16* input,
                                          float*               output,
                                          int64_t              numel,
                                          int32_t              head_count,
                                          int64_t              token_stride,
                                          int64_t              head_stride,
                                          int32_t              hidden_size) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < numel) {
        const int64_t source_index = inputOffset<Strided>(index, head_count, token_stride, head_stride, hidden_size);
        const float   value        = __bfloat162float(input[source_index]);
        output[index]              = value * value;
    }
}

template<bool Strided>
__global__ void gemma4RmsApplyBf16Kernel(const __nv_bfloat16* input,
                                         const float*         inv_rms,
                                         const float*         weight,
                                         __nv_bfloat16*       output,
                                         int64_t              numel,
                                         int32_t              head_count,
                                         int64_t              token_stride,
                                         int64_t              head_stride,
                                         int32_t              hidden_size) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < numel) {
        const int64_t source_index = inputOffset<Strided>(index, head_count, token_stride, head_stride, hidden_size);
        float         value        = __bfloat162float(input[source_index]) * inv_rms[index / hidden_size];
        if (weight != nullptr) {
            value *= weight[index % hidden_size];
        }
        output[index] = __float2bfloat16(value);
    }
}

template<bool ComputeInvRms>
__global__ void gemma4RmsMeanFp32Kernel(
    const float4* input, float* output, int64_t rows, int32_t hidden_size, float factor, float eps) {
    const int64_t row             = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    float         accumulators[4] = {};
    if (row < rows) {
        const int64_t vectors_per_row = hidden_size / 4;
        const float4* row_input       = input + row * vectors_per_row;
        for (int64_t vector = threadIdx.x; vector < vectors_per_row; vector += blockDim.x) {
            const float4 value = row_input[vector];
            accumulators[0] += value.x;
            accumulators[1] += value.y;
            accumulators[2] += value.z;
            accumulators[3] += value.w;
        }
    }
    float                   sum = ((accumulators[0] + accumulators[1]) + accumulators[2]) + accumulators[3];
    extern __shared__ float shared[];
    const int               shared_index = threadIdx.y * blockDim.x + threadIdx.x;
    if (blockDim.x > 32) {
        shared[shared_index] = sum;
        for (int offset = blockDim.x / 2; offset >= 32; offset >>= 1) {
            __syncthreads();
            if (threadIdx.x < offset) {
                sum += shared[shared_index + offset];
                shared[shared_index] = sum;
            }
        }
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (threadIdx.x == 0 && row < rows) {
        if constexpr (ComputeInvRms) {
            const float mean = __fmul_rn(sum, factor);
            output[row]      = rsqrtf(__fadd_rn(mean, eps));
        } else {
            output[row] = sum * factor;
        }
    }
}

// ---------------------------------------------------------------------------
// Single-pass fused RMSNorm: computes inv_rms with the EXACT lane/tree
// reduction order of gemma4RmsMeanFp32Kernel applied to the squares the
// square kernel would have produced (value*value per element, fp32), then
// applies (x*inv)*weight in the apply kernel's order. Output is bit-equal
// to the square->inv->apply chain while reading x once (the second read
// in the apply phase hits L1/L2) instead of writing and re-reading an
// fp32 square copy.
// ---------------------------------------------------------------------------
template<bool Strided, bool Weighted>
__global__ void gemma4RmsFusedBf16Kernel(const __nv_bfloat16* input,
                                         const float*         weight,
                                         __nv_bfloat16*       output,
                                         int64_t              rows,
                                         int32_t              head_count,
                                         int64_t              token_stride,
                                         int64_t              head_stride,
                                         int32_t              hidden_size,
                                         float                factor,
                                         float                eps) {
    const int64_t row             = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    const int     width           = blockDim.x;
    const int64_t vectors_per_row = hidden_size / 4;
    float         accumulators[4] = {};
    const bool    active          = row < rows;
    if (active) {
        for (int64_t vector = threadIdx.x; vector < vectors_per_row; vector += width) {
            const int64_t base = vector * 4;
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                const int64_t logical = row * hidden_size + base + component;
                const int64_t source =
                    inputOffset<Strided>(logical, head_count, token_stride, head_stride, hidden_size);
                const float value = __bfloat162float(input[source]);
                // round each op separately: an FMA contraction of acc + v*v
                // would break bit-equality with the square->mean chain (the
                // square kernel rounds the product before the mean kernel
                // adds it), and nvcc defaults to --fmad=true
                accumulators[component] = __fadd_rn(accumulators[component], __fmul_rn(value, value));
            }
        }
    }
    float                   sum = ((accumulators[0] + accumulators[1]) + accumulators[2]) + accumulators[3];
    extern __shared__ float shared[];
    const int               shared_index = threadIdx.y * width + threadIdx.x;
    if (width > 32) {
        shared[shared_index] = sum;
        for (int offset = width / 2; offset >= 32; offset >>= 1) {
            __syncthreads();
            if (threadIdx.x < offset) {
                sum += shared[shared_index + offset];
                shared[shared_index] = sum;
            }
        }
        __syncthreads();
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    // inv_rms slots live past the reduction scratch (width*height floats).
    float* row_inv = shared + width * blockDim.y;
    if (threadIdx.x == 0 && active) {
        const float mean     = __fmul_rn(sum, factor);
        row_inv[threadIdx.y] = rsqrtf(__fadd_rn(mean, eps));
    }
    __syncthreads();
    if (active) {
        const float inv = row_inv[threadIdx.y];
        for (int64_t vector = threadIdx.x; vector < vectors_per_row; vector += width) {
            const int64_t base = vector * 4;
#pragma unroll
            for (int component = 0; component < 4; ++component) {
                const int64_t logical = row * hidden_size + base + component;
                const int64_t source =
                    inputOffset<Strided>(logical, head_count, token_stride, head_stride, hidden_size);
                float value = __bfloat162float(input[source]) * inv;
                if constexpr (Weighted) {
                    value *= weight[base + component];
                }
                // Write at the dense logical index (the strided source
                // offset only governs reads, matching the apply kernel).
                output[logical] = __float2bfloat16(value);
            }
        }
    }
}

template<bool Weighted>
void launchGemma4RmsFusedBf16(const __nv_bfloat16* input,
                              const float*         weight,
                              __nv_bfloat16*       output,
                              int64_t              rows,
                              int32_t              head_count,
                              int64_t              token_stride,
                              int64_t              head_stride,
                              int32_t              hidden_size,
                              float                eps,
                              cudaStream_t         stream) {
    if (rows == 0) {
        return;
    }
    int height = 1;
    while (height < 16 && static_cast<int64_t>(height * 2) <= rows) {
        height *= 2;
    }
    const int    input_vectors = hidden_size / 4;
    const int    width         = std::min(input_vectors, 512 / height);
    const float  factor        = static_cast<float>(rows) / static_cast<float>(rows * hidden_size);
    const dim3   block(width, height);
    const int    blocks       = static_cast<int>((rows + height - 1) / height);
    const size_t shared_bytes = (static_cast<size_t>(width) * height + height) * sizeof(float);
    const bool   strided =
        !(token_stride == static_cast<int64_t>(head_count) * hidden_size && head_stride == hidden_size);
    if (strided) {
        gemma4RmsFusedBf16Kernel<true, Weighted><<<blocks, block, shared_bytes, stream>>>(
            input, weight, output, rows, head_count, token_stride, head_stride, hidden_size, factor, eps);
    } else {
        gemma4RmsFusedBf16Kernel<false, Weighted><<<blocks, block, shared_bytes, stream>>>(
            input, weight, output, rows, head_count, token_stride, head_stride, hidden_size, factor, eps);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

void invokeGemma4RmsSquareBf16(const __nv_bfloat16* input,
                               float*               output,
                               int64_t              numel,
                               int32_t              head_count,
                               int64_t              token_stride,
                               int64_t              head_stride,
                               int32_t              hidden_size,
                               cudaStream_t         stream) {
    if (numel == 0) {
        return;
    }
    constexpr int threads = 256;
    const int     blocks  = static_cast<int>((numel + threads - 1) / threads);
    if (token_stride == static_cast<int64_t>(head_count) * hidden_size && head_stride == hidden_size) {
        gemma4RmsSquareBf16Kernel<false>
            <<<blocks, threads, 0, stream>>>(input, output, numel, head_count, token_stride, head_stride, hidden_size);
    } else {
        gemma4RmsSquareBf16Kernel<true>
            <<<blocks, threads, 0, stream>>>(input, output, numel, head_count, token_stride, head_stride, hidden_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4RmsApplyBf16(const __nv_bfloat16* input,
                              const float*         inv_rms,
                              const float*         weight,
                              __nv_bfloat16*       output,
                              int64_t              numel,
                              int32_t              head_count,
                              int64_t              token_stride,
                              int64_t              head_stride,
                              int32_t              hidden_size,
                              cudaStream_t         stream) {
    if (numel == 0) {
        return;
    }
    constexpr int threads = 256;
    const int     blocks  = static_cast<int>((numel + threads - 1) / threads);
    if (token_stride == static_cast<int64_t>(head_count) * hidden_size && head_stride == hidden_size) {
        gemma4RmsApplyBf16Kernel<false><<<blocks, threads, 0, stream>>>(
            input, inv_rms, weight, output, numel, head_count, token_stride, head_stride, hidden_size);
    } else {
        gemma4RmsApplyBf16Kernel<true><<<blocks, threads, 0, stream>>>(
            input, inv_rms, weight, output, numel, head_count, token_stride, head_stride, hidden_size);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4RmsMeanFp32(
    const float* input, float* output, int64_t rows, int32_t hidden_size, cudaStream_t stream) {
    if (rows == 0) {
        return;
    }
    int height = 1;
    while (height < 16 && static_cast<int64_t>(height * 2) <= rows) {
        height *= 2;
    }
    const int    input_vectors = hidden_size / 4;
    const int    width         = std::min(input_vectors, 512 / height);
    const float  factor        = static_cast<float>(rows) / static_cast<float>(rows * hidden_size);
    const dim3   block(width, height);
    const int    blocks       = static_cast<int>((rows + height - 1) / height);
    const size_t shared_bytes = width > 32 ? static_cast<size_t>(width * height) * sizeof(float) : 0;
    gemma4RmsMeanFp32Kernel<false><<<blocks, block, shared_bytes, stream>>>(
        reinterpret_cast<const float4*>(input), output, rows, hidden_size, factor, 0.0f);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4RmsInvFp32(
    const float* input, float* output, int64_t rows, int32_t hidden_size, float eps, cudaStream_t stream) {
    if (rows == 0) {
        return;
    }
    int height = 1;
    while (height < 16 && static_cast<int64_t>(height * 2) <= rows) {
        height *= 2;
    }
    const int    input_vectors = hidden_size / 4;
    const int    width         = std::min(input_vectors, 512 / height);
    const float  factor        = static_cast<float>(rows) / static_cast<float>(rows * hidden_size);
    const dim3   block(width, height);
    const int    blocks       = static_cast<int>((rows + height - 1) / height);
    const size_t shared_bytes = width > 32 ? static_cast<size_t>(width * height) * sizeof(float) : 0;
    gemma4RmsMeanFp32Kernel<true><<<blocks, block, shared_bytes, stream>>>(
        reinterpret_cast<const float4*>(input), output, rows, hidden_size, factor, eps);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4RmsFusedBf16(const __nv_bfloat16* input,
                              const float*         weight,
                              __nv_bfloat16*       output,
                              int64_t              rows,
                              int32_t              head_count,
                              int64_t              token_stride,
                              int64_t              head_stride,
                              int32_t              hidden_size,
                              float                eps,
                              cudaStream_t         stream) {
    if (weight != nullptr) {
        launchGemma4RmsFusedBf16<true>(
            input, weight, output, rows, head_count, token_stride, head_stride, hidden_size, eps, stream);
    } else {
        launchGemma4RmsFusedBf16<false>(
            input, weight, output, rows, head_count, token_stride, head_stride, hidden_size, eps, stream);
    }
}

}  // namespace rtp_llm
