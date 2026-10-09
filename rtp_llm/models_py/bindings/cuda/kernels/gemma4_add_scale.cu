#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_add_scale.h"

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace rtp_llm {
namespace {

constexpr int kBf16PerVector = sizeof(uint4) / sizeof(__nv_bfloat16);

union alignas(16) Bf16x8 {
    uint4         packed;
    __nv_bfloat16 values[kBf16PerVector];
};

__global__ void gemma4AddBf16Kernel(const uint4* __restrict__ lhs,
                                    const uint4* __restrict__ rhs,
                                    uint4* __restrict__ output,
                                    int64_t vector_count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= vector_count) {
        return;
    }
    Bf16x8 lhs_values;
    Bf16x8 rhs_values;
    Bf16x8 output_values;
    lhs_values.packed = lhs[index];
    rhs_values.packed = rhs[index];
#pragma unroll
    for (int i = 0; i < kBf16PerVector; ++i) {
        output_values.values[i] =
            __float2bfloat16_rn(__bfloat162float(lhs_values.values[i]) + __bfloat162float(rhs_values.values[i]));
    }
    output[index] = output_values.packed;
}

__global__ void
gemma4ScaleBf16Kernel(const uint4* __restrict__ input, float scale, uint4* __restrict__ output, int64_t vector_count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= vector_count) {
        return;
    }
    Bf16x8 input_values;
    Bf16x8 output_values;
    input_values.packed = input[index];
#pragma unroll
    for (int i = 0; i < kBf16PerVector; ++i) {
        output_values.values[i] = __float2bfloat16_rn(__bfloat162float(input_values.values[i]) * scale);
    }
    output[index] = output_values.packed;
}

__global__ void gemma4GatherRowsBf16Kernel(const uint4* __restrict__ input,
                                           const int32_t* __restrict__ indices,
                                           uint4* __restrict__ output,
                                           int64_t vector_count,
                                           int32_t vectors_per_row) {
    const int64_t output_index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (output_index >= vector_count) {
        return;
    }
    const int64_t output_row    = output_index / vectors_per_row;
    const int32_t vector_in_row = static_cast<int32_t>(output_index % vectors_per_row);
    const int64_t input_row     = indices[output_row];
    output[output_index]        = input[input_row * vectors_per_row + vector_in_row];
}

__global__ void gemma4LogitSoftcapFp32Kernel(
    const float* __restrict__ input, float* __restrict__ output, int64_t numel, float reciprocal_cap, float cap) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < numel) {
        output[index] = __fmul_rn(tanhf(__fmul_rn(input[index], reciprocal_cap)), cap);
    }
}

__global__ void gemma4AddScaleBf16Kernel(const uint4* __restrict__ residual,
                                         const uint4* __restrict__ hidden,
                                         const __nv_bfloat16* __restrict__ scale,
                                         uint4* __restrict__ output,
                                         int64_t vector_count) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= vector_count) {
        return;
    }
    Bf16x8 residual_values;
    Bf16x8 hidden_values;
    Bf16x8 output_values;
    residual_values.packed  = residual[index];
    hidden_values.packed    = hidden[index];
    const float scale_value = __bfloat162float(scale[0]);
#pragma unroll
    for (int i = 0; i < kBf16PerVector; ++i) {
        const __nv_bfloat16 sum = __float2bfloat16_rn(__bfloat162float(residual_values.values[i])
                                                      + __bfloat162float(hidden_values.values[i]));
        output_values.values[i] = __float2bfloat16_rn(__bfloat162float(sum) * scale_value);
    }
    output[index] = output_values.packed;
}

__global__ void gemma4RouterScaleBf16Kernel(const uint4* __restrict__ input,
                                            const uint4* __restrict__ channel_scale,
                                            float scalar_scale,
                                            uint4* __restrict__ output,
                                            int64_t vector_count,
                                            int32_t vectors_per_row) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= vector_count) {
        return;
    }
    Bf16x8 input_values;
    Bf16x8 scale_values;
    Bf16x8 output_values;
    input_values.packed = input[index];
    scale_values.packed = channel_scale[index % vectors_per_row];
#pragma unroll
    for (int i = 0; i < kBf16PerVector; ++i) {
        const __nv_bfloat16 channel_scaled =
            __float2bfloat16_rn(__bfloat162float(input_values.values[i]) * __bfloat162float(scale_values.values[i]));
        output_values.values[i] = __float2bfloat16_rn(__bfloat162float(channel_scaled) * scalar_scale);
    }
    output[index] = output_values.packed;
}

}  // namespace

void invokeGemma4AddBf16(
    const __nv_bfloat16* lhs, const __nv_bfloat16* rhs, __nv_bfloat16* output, int64_t numel, cudaStream_t stream) {
    if (numel == 0) {
        return;
    }
    constexpr int threads      = 256;
    const int64_t vector_count = numel / kBf16PerVector;
    const int     blocks       = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4AddBf16Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(lhs),
                                                        reinterpret_cast<const uint4*>(rhs),
                                                        reinterpret_cast<uint4*>(output),
                                                        vector_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4ScaleBf16(
    const __nv_bfloat16* input, float scale, __nv_bfloat16* output, int64_t numel, cudaStream_t stream) {
    if (numel == 0) {
        return;
    }
    constexpr int threads      = 256;
    const int64_t vector_count = numel / kBf16PerVector;
    const int     blocks       = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4ScaleBf16Kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint4*>(input), scale, reinterpret_cast<uint4*>(output), vector_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4GatherRowsBf16(const __nv_bfloat16* input,
                                const int32_t*       indices,
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
    gemma4GatherRowsBf16Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(input),
                                                               indices,
                                                               reinterpret_cast<uint4*>(output),
                                                               vector_count,
                                                               vectors_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4LogitSoftcapFp32(const float* input, float* output, int64_t numel, float cap, cudaStream_t stream) {
    if (numel == 0) {
        return;
    }
    constexpr int threads        = 256;
    const int     blocks         = static_cast<int>((numel + threads - 1) / threads);
    const float   reciprocal_cap = 1.0f / cap;
    gemma4LogitSoftcapFp32Kernel<<<blocks, threads, 0, stream>>>(input, output, numel, reciprocal_cap, cap);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4AddScaleBf16(const __nv_bfloat16* residual,
                              const __nv_bfloat16* hidden,
                              const __nv_bfloat16* scale,
                              __nv_bfloat16*       output,
                              int64_t              numel,
                              cudaStream_t         stream) {
    if (numel == 0) {
        return;
    }
    constexpr int threads      = 256;
    const int64_t vector_count = numel / kBf16PerVector;
    const int     blocks       = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4AddScaleBf16Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(residual),
                                                             reinterpret_cast<const uint4*>(hidden),
                                                             scale,
                                                             reinterpret_cast<uint4*>(output),
                                                             vector_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void invokeGemma4RouterScaleBf16(const __nv_bfloat16* input,
                                 const __nv_bfloat16* channel_scale,
                                 float                scalar_scale,
                                 __nv_bfloat16*       output,
                                 int64_t              numel,
                                 int32_t              hidden_size,
                                 cudaStream_t         stream) {
    if (numel == 0) {
        return;
    }
    constexpr int threads         = 256;
    const int32_t vectors_per_row = hidden_size / kBf16PerVector;
    const int64_t vector_count    = numel / kBf16PerVector;
    const int     blocks          = static_cast<int>((vector_count + threads - 1) / threads);
    gemma4RouterScaleBf16Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const uint4*>(input),
                                                                reinterpret_cast<const uint4*>(channel_scale),
                                                                scalar_scale,
                                                                reinterpret_cast<uint4*>(output),
                                                                vector_count,
                                                                vectors_per_row);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
