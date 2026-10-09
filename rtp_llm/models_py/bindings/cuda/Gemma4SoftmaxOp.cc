#include "rtp_llm/models_py/bindings/cuda/Gemma4SoftmaxOp.h"
#include "rtp_llm/models_py/bindings/cuda/kernels/gemma4_softmax_8192.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cstdint>
#include <limits>

namespace torch_ext {

at::Tensor gemma4_softmax_8192_bf16(const at::Tensor& input, int64_t query_start, int64_t window_left) {
    TORCH_CHECK(input.is_cuda(), "gemma4_softmax_8192_bf16: input must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == at::kBFloat16, "gemma4_softmax_8192_bf16: input must be bfloat16");
    TORCH_CHECK(input.is_contiguous(), "gemma4_softmax_8192_bf16: input must be contiguous");
    TORCH_CHECK(input.dim() >= 2 && input.size(-1) == 8192, "gemma4_softmax_8192_bf16: final dimension must be 8192");

    const int64_t rows         = input.numel() / 8192;
    const int64_t query_length = input.size(-2);
    TORCH_CHECK(rows <= std::numeric_limits<uint32_t>::max(), "gemma4_softmax_8192_bf16: row count exceeds grid limit");
    TORCH_CHECK(query_length <= std::numeric_limits<int32_t>::max(),
                "gemma4_softmax_8192_bf16: query length exceeds int32 limit");
    TORCH_CHECK(query_start >= -1 && query_start <= std::numeric_limits<int32_t>::max(),
                "gemma4_softmax_8192_bf16: query_start is outside int32 range");
    TORCH_CHECK(window_left >= -1 && window_left <= 8192,
                "gemma4_softmax_8192_bf16: window_left must be -1 or within the key length");
    TORCH_CHECK(query_start < 0 || query_start + query_length <= 8192,
                "gemma4_softmax_8192_bf16: causal query range exceeds key length");
    TORCH_CHECK(query_start >= 0 || window_left < 0,
                "gemma4_softmax_8192_bf16: window_left requires a causal query_start");

    const c10::cuda::CUDAGuard device_guard(input.device());
    auto                       output = at::empty_like(input);
    rtp_llm::invokeGemma4Softmax8192Bf16(reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr<at::BFloat16>()),
                                         reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr<at::BFloat16>()),
                                         rows,
                                         static_cast<int32_t>(query_length),
                                         static_cast<int32_t>(query_start),
                                         static_cast<int32_t>(window_left),
                                         at::cuda::getCurrentCUDAStream(input.get_device()).stream());
    return output;
}

}  // namespace torch_ext
