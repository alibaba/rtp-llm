#include "rtp_llm/models_py/bindings/cuda/ops/StandaloneOps.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/common/kernels/activation_kernels.h"
#include "rtp_llm/models_py/bindings/common/kernels/mask_logits.h"

namespace rtp_llm {

void cudaSoftmaxInplace(torch::Tensor& input, cudaStream_t stream) {
    RTP_LLM_CHECK(input.dim() == 2);
    RTP_LLM_CHECK(input.is_contiguous());
    int  m     = input.size(0);
    int  n     = input.size(1);
    auto dtype = input.scalar_type();
    if (dtype == torch::kFloat32) {
        invokeAddBiasSoftMax<float>(input.data_ptr<float>(), nullptr, nullptr, nullptr, m, n, n, stream);
    } else if (dtype == torch::kFloat16) {
        invokeAddBiasSoftMax<half>(
            reinterpret_cast<half*>(input.data_ptr<at::Half>()), nullptr, nullptr, nullptr, m, n, n, stream);
    } else if (dtype == torch::kBFloat16) {
        invokeAddBiasSoftMax<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(input.data_ptr<at::BFloat16>()),
                                            nullptr,
                                            nullptr,
                                            nullptr,
                                            m,
                                            n,
                                            n,
                                            stream);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "cudaSoftmaxInplace: unsupported dtype");
    }
}

void cudaMaskLogits(torch::Tensor& logits, const torch::Tensor& mask, cudaStream_t stream) {
    RTP_LLM_CHECK(logits.dim() == 2);
    RTP_LLM_CHECK(mask.dim() == 2);
    RTP_LLM_CHECK(logits.size(0) == mask.size(0));
    RTP_LLM_CHECK(logits.size(1) == mask.size(1));
    size_t batch_size = logits.size(0);
    size_t vocab_size = logits.size(1);
    auto   dtype      = logits.scalar_type();
    if (dtype == torch::kFloat32) {
        invokeMaskLogits<float>(logits.data_ptr<float>(), mask.data_ptr<uint8_t>(), batch_size, vocab_size, stream);
    } else if (dtype == torch::kFloat16) {
        invokeMaskLogits<half>(reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
                               mask.data_ptr<uint8_t>(),
                               batch_size,
                               vocab_size,
                               stream);
    } else if (dtype == torch::kBFloat16) {
        invokeMaskLogits<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
                                        mask.data_ptr<uint8_t>(),
                                        batch_size,
                                        vocab_size,
                                        stream);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "cudaMaskLogits: unsupported dtype");
    }
}

}  // namespace rtp_llm
