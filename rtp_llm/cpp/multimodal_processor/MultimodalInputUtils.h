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
    // Token-aligned vocabulary IDs follow the same crop as the media tensor.
    // Larger opaque sidecars retain the existing DeepStack layout.
    if (extra_input.dim() == 1 && extra_input.scalar_type() == torch::kInt32 && feature.dim() == 2
        && extra_input.numel() == feature.size(0)) {
        RTP_LLM_CHECK_WITH_INFO(feature.dim() == 2 && feature.size(0) > 0 && feature.size(1) > 0,
                                "semantic IDs require positive 2-D features");
        RTP_LLM_CHECK_WITH_INFO(token_start >= 0 && token_start <= token_end && token_end <= feature.size(0),
                                "semantic ID slice is outside feature tokens");
        return extra_input.slice(0, token_start, token_end).contiguous();
    }
    return sliceDeepstackExtraInput(reshapeMultimodalExtraInput(extra_input, feature), token_start, token_end);
}

}  // namespace rtp_llm
