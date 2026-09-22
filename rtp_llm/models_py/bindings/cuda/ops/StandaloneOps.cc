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

void cudaApplyPackedMaskLogits(const torch::Tensor& logits,
                               const torch::Tensor& packed_allow_mask,
                               const torch::Tensor& row_indices,
                               size_t               vocab_size,
                               cudaStream_t         stream) {
    RTP_LLM_CHECK(logits.dim() == 2 && logits.is_cuda());
    RTP_LLM_CHECK(logits.stride(1) == 1);
    RTP_LLM_CHECK(packed_allow_mask.dim() == 2 && packed_allow_mask.is_cuda());
    RTP_LLM_CHECK(packed_allow_mask.scalar_type() == torch::kInt32 && packed_allow_mask.stride(1) == 1);
    RTP_LLM_CHECK(row_indices.dim() == 1 && row_indices.is_cuda() && row_indices.is_contiguous());
    RTP_LLM_CHECK(row_indices.scalar_type() == torch::kInt32);
    RTP_LLM_CHECK(row_indices.numel() == packed_allow_mask.size(0));
    RTP_LLM_CHECK(vocab_size > 0 && vocab_size <= static_cast<size_t>(logits.size(1)));
    RTP_LLM_CHECK(static_cast<size_t>(packed_allow_mask.size(1)) >= (vocab_size + 31) / 32);

    const int mask_rows          = static_cast<int>(packed_allow_mask.size(0));
    const int logits_rows        = static_cast<int>(logits.size(0));
    const int logits_row_stride  = static_cast<int>(logits.stride(0));
    const int bitmask_row_stride = static_cast<int>(packed_allow_mask.stride(0));
    const int bitmask_words      = static_cast<int>(packed_allow_mask.size(1));
    if (mask_rows == 0) {
        return;
    }

    if (logits.scalar_type() == torch::kFloat32) {
        invokePackedMaskLogits<float>(logits.data_ptr<float>(),
                                      packed_allow_mask.data_ptr<int32_t>(),
                                      row_indices.data_ptr<int32_t>(),
                                      mask_rows,
                                      logits_rows,
                                      logits_row_stride,
                                      static_cast<int>(vocab_size),
                                      bitmask_row_stride,
                                      bitmask_words,
                                      stream);
    } else if (logits.scalar_type() == torch::kFloat16) {
        invokePackedMaskLogits<half>(reinterpret_cast<half*>(logits.data_ptr<at::Half>()),
                                     packed_allow_mask.data_ptr<int32_t>(),
                                     row_indices.data_ptr<int32_t>(),
                                     mask_rows,
                                     logits_rows,
                                     logits_row_stride,
                                     static_cast<int>(vocab_size),
                                     bitmask_row_stride,
                                     bitmask_words,
                                     stream);
    } else if (logits.scalar_type() == torch::kBFloat16) {
        invokePackedMaskLogits<__nv_bfloat16>(reinterpret_cast<__nv_bfloat16*>(logits.data_ptr<at::BFloat16>()),
                                              packed_allow_mask.data_ptr<int32_t>(),
                                              row_indices.data_ptr<int32_t>(),
                                              mask_rows,
                                              logits_rows,
                                              logits_row_stride,
                                              static_cast<int>(vocab_size),
                                              bitmask_row_stride,
                                              bitmask_words,
                                              stream);
    } else {
        RTP_LLM_CHECK_WITH_INFO(false, "cudaApplyPackedMaskLogits: unsupported dtype");
    }
}

}  // namespace rtp_llm
