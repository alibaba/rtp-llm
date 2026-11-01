// Test-only binding. The two compiled kernel translation units are unchanged main sources.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "per_token_group_quant_8bit.h"
#include "per_token_group_quant_8bit_v2.h"

void quant(torch::Tensor input, torch::Tensor q, torch::Tensor scales, bool v2) {
    TORCH_CHECK(input.is_cuda() && input.dim() == 2 && input.is_contiguous(), "expected contiguous CUDA matrix");
    TORCH_CHECK(input.scalar_type() == torch::kBFloat16, "probe accepts BF16 only");
    TORCH_CHECK(input.size(0) > 0 && input.size(1) % 32 == 0, "invalid input shape");
    TORCH_CHECK(q.device() == input.device() && scales.device() == input.device(), "device mismatch");
    TORCH_CHECK(q.sizes() == input.sizes() && q.is_contiguous() && q.scalar_type() == torch::kFloat8_e4m3fn, "bad quant output");
    const auto aligned_m = (input.size(0) + 3) / 4 * 4;
    TORCH_CHECK(scales.dim() == 2 && scales.scalar_type() == torch::kInt32
                && scales.size(0) == input.size(0) && scales.size(1) == (input.size(1) / 32 + 3) / 4
                && scales.stride(0) == 1 && scales.stride(1) == aligned_m, "bad scale layout");
    const c10::cuda::CUDAGuard guard(input.device());
    const double eps = std::numeric_limits<float>::min();
    if (v2) {
        rtp_llm::sgl_per_token_group_quant_8bit_v2(input, q, scales, 32, eps, -448., 448., true, false, std::nullopt);
    } else {
        rtp_llm::per_token_group_quant_8bit(input, q, scales, 32, eps, -448., 448., true);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("quant", &quant); }
