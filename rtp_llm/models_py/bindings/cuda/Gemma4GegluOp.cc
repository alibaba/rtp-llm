#include "rtp_llm/models_py/bindings/cuda/Gemma4GegluOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_geglu.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <limits>

namespace torch_ext {

at::Tensor gemma4_geglu_tanh_bf16(const at::Tensor& gate_up) {
    TORCH_CHECK(gate_up.is_cuda(), "gemma4_geglu_tanh_bf16: input must be a CUDA tensor");
    TORCH_CHECK(gate_up.scalar_type() == at::kBFloat16, "gemma4_geglu_tanh_bf16: input must be bfloat16");
    TORCH_CHECK(gate_up.is_contiguous(), "gemma4_geglu_tanh_bf16: input must be contiguous");
    TORCH_CHECK(gate_up.dim() == 2 && gate_up.size(1) > 0 && gate_up.size(1) % 2 == 0,
                "gemma4_geglu_tanh_bf16: input must be [rows,2*intermediate_size]");
    TORCH_CHECK(gate_up.size(1) / 2 <= std::numeric_limits<int32_t>::max(),
                "gemma4_geglu_tanh_bf16: intermediate size exceeds int32 limit");

    const c10::cuda::CUDAGuard device_guard(gate_up.device());
    const int64_t              intermediate_size = gate_up.size(1) / 2;
    auto                       output            = at::empty({gate_up.size(0), intermediate_size}, gate_up.options());
    rtp_llm::invokeGemma4GegluTanhBf16(reinterpret_cast<const __nv_bfloat16*>(gate_up.const_data_ptr<at::BFloat16>()),
                                       reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
                                       gate_up.size(0),
                                       static_cast<int32_t>(intermediate_size),
                                       at::cuda::getCurrentCUDAStream(gate_up.get_device()).stream());
    return output;
}

}  // namespace torch_ext
