#include "rtp_llm/cpp/kernels/atex/ops/f16/gate_silu.h"
#include "rtp_llm/cpp/kernels/atex/block/reduce.cuh"
#include "rtp_llm/cpp/kernels/atex/copy/vec.cuh"

namespace atex {
namespace impl {

__device__ inline fp32_t sigmoid(const fp32_t x) {
    return 1.0f / (1.0f + expf(-x));
}

template<typename f16x2_t, uint32_t TPB, uint32_t VPT>
__global__ void device_gate_silu_f16(const f16x2_t* const __restrict__ x,  // [m, n]
                                     f16x2_t* __restrict__ o,              // [m, n // 2]
                                     const uint32_t N,
                                     const uint32_t numel) {
    const uint32_t bid = blockIdx.x;
    const uint32_t tid = threadIdx.x;

    f16x2_t local_value[VPT];
    f16x2_t local_gate[VPT];
    f16x2_t local_y[VPT];

    for (uint32_t index = (bid * TPB * VPT + tid * VPT) * 2; index < numel, index += gridDim.x * TPB * VPT * 2) {
        atex::copy<sizeof(f16x2_t) * VPT>(x + index + 0, local_value);
        atex::copy<sizeof(f16x2_t) * VPT>(x + index + N, local_gate);

#pragma unroll
        for (uint32_t i = 0; i < VPT; i++) {
            fp32x2_t value = atex::cvt_f16x2_to_f32x2(local_value[i]);
            fp32x2_t gate  = atex::cvt_f16x2_to_f32x2(local_gate[i]);

            value.x = sigmoid(gate.x) * value.x;
            value.y = sigmoid(gate.y) * value.y;

            local_y[i] = cvt_f32x2_to_f16x2<f16x2_t>(value);
        }

        atex::copy<sizeof(f16x2_t) * VPT>(local_y, o + index / 2);
    }
}

Tensor launch_gate_silu_bf16(const Tensor& x) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous.");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor.");
    TORCH_CHECK(x.scalar_type() == c10::ScalarType::BFloat16, "x must be BF16.");

    auto          sizes = x.sizes();
    auto          ndim  = x.ndimension();
    const int64_t n     = sizes[ndim - 1];
    TORCH_CHECK(n % 16 == 0, "n must be a multiple of 16");

    std::vector<int64_t> output_sizes = {m, n / 2};
    Tensor               output       = at::empty(output_sizes, x.options());

    constexpr uint32_t TPB = 256;
    constexpr uint32_t VPT = 16 / sizeof(bf16x2_t);

    // Launch configuration
    const uint32_t N     = n / 2;               // half size
    const uint32_t numel = output.numel() / 2;  // Number of half2 in output
    const uint32_t grid  = (numel + TPB * VPT - 1) / (TPB * VPT);

    // Launch kernel
    device_gate_silu_f16<bf16x2_t, TPB, VPT><<<grid, TPB>>>(PTR<bf16x2_t>(x), PTR<bf16x2_t>(output), N, numel);

    // Error check
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

Tensor launch_gate_silu_fp16(const Tensor& x) {
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous.");
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor.");
    TORCH_CHECK(x.scalar_type() == c10::ScalarType::Half, "x must be FP16.");

    auto          sizes = x.sizes();
    auto          ndim  = x.ndimension();
    const int64_t n     = sizes[ndim - 1];
    TORCH_CHECK(n % 16 == 0, "n must be a multiple of 16");

    std::vector<int64_t> output_sizes = {m, n / 2};
    Tensor               output       = at::empty(output_sizes, x.options());

    constexpr uint32_t TPB = 256;
    constexpr uint32_t VPT = 16 / sizeof(fp16x2_t);

    // Launch configuration
    const uint32_t N     = n / 2;               // half size
    const uint32_t numel = output.numel() / 2;  // Number of half2 in output
    const uint32_t grid  = (numel + TPB * VPT - 1) / (TPB * VPT);

    // Launch kernel
    device_gate_silu_f16<fp16x2_t, TPB, VPT><<<grid, TPB>>>(PTR<fp16x2_t>(x), PTR<fp16x2_t>(output), N, numel);

    // Error check
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA kernel launch failed: ", cudaGetErrorString(err));

    return output;
}

}  // namespace impl
}  // namespace atex