#pragma once

#include "torch/extension.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

inline torch::Tensor reshapeMultimodalExtraInput(const torch::Tensor& extra_input, const torch::Tensor& feature) {
    RTP_LLM_CHECK_WITH_INFO(feature.dim() == 2, "multimodal feature must be 2-D when mm_extra_input is present");
    const int64_t feature_len = feature.size(0);
    const int64_t hidden_size = feature.size(1);
    RTP_LLM_CHECK_WITH_INFO(feature_len > 0 && hidden_size > 0,
                            "multimodal feature tokens and hidden size must be positive");
    const int64_t layer_numel = feature_len * hidden_size;
    RTP_LLM_CHECK_WITH_INFO(extra_input.numel() % layer_numel == 0,
                            "mm_extra_input numel (%ld) is not divisible by tokens * hidden (%ld)",
                            extra_input.numel(),
                            layer_numel);
    return extra_input.reshape({extra_input.numel() / layer_numel, feature_len, hidden_size});
}

inline torch::Tensor sliceDeepstackExtraInput(const torch::Tensor& deepstack, int64_t token_start, int64_t token_end) {
    RTP_LLM_CHECK_WITH_INFO(deepstack.dim() == 3, "reshaped mm_extra_input must be 3-D");
    const int64_t token_count = deepstack.size(1);
    RTP_LLM_CHECK_WITH_INFO(token_start >= 0 && token_start <= token_end && token_end <= token_count,
                            "mm_extra_input token slice [%ld, %ld) is outside [0, %ld)",
                            token_start,
                            token_end,
                            token_count);
    return deepstack.slice(1, token_start, token_end).contiguous().reshape({-1});
}

inline torch::Tensor sliceMultimodalExtraInput(const torch::Tensor& extra_input,
                                               const torch::Tensor& feature,
                                               int64_t              token_start,
                                               int64_t              token_end) {
    return sliceDeepstackExtraInput(reshapeMultimodalExtraInput(extra_input, feature), token_start, token_end);
}

}  // namespace rtp_llm
