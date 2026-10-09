#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_geglu.h"

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAMathCompat.h>
#include <cuda_runtime.h>

#include <cmath>

namespace rtp_llm {
namespace {

__global__ void gemma4GegluTanhBf16Kernel(const __nv_bfloat16* gate_up,
                                          __nv_bfloat16*       output,
                                          int64_t              numel,
                                          int32_t              intermediate_size) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= numel) {
        return;
    }
    const int64_t       row       = index / intermediate_size;
    const int32_t       column    = static_cast<int32_t>(index % intermediate_size);
    const int64_t       row_start = row * intermediate_size * 2;
    const float         x         = __bfloat162float(gate_up[row_start + column]);
    constexpr float     kBeta     = M_SQRT2 * M_2_SQRTPI * 0.5f;
    constexpr float     kKappa    = 0.044715f;
    const float         x_cube    = x * x * x;
    const float         inner     = kBeta * (x + kKappa * x_cube);
    const float         gelu      = 0.5f * x * (1.0f + c10::cuda::compat::tanh(inner));
    const __nv_bfloat16 gelu_bf16 = __float2bfloat16_rn(gelu);
    const float         up        = __bfloat162float(gate_up[row_start + intermediate_size + column]);
    output[index]                 = __float2bfloat16_rn(__bfloat162float(gelu_bf16) * up);
}

}  // namespace

void invokeGemma4GegluTanhBf16(
    const __nv_bfloat16* gate_up, __nv_bfloat16* output, int64_t rows, int32_t intermediate_size, cudaStream_t stream) {
    if (rows == 0) {
        return;
    }
    const int64_t numel   = rows * intermediate_size;
    constexpr int threads = 256;
    const int     blocks  = static_cast<int>((numel + threads - 1) / threads);
    gemma4GegluTanhBf16Kernel<<<blocks, threads, 0, stream>>>(gate_up, output, numel, intermediate_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace rtp_llm
