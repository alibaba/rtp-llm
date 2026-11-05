#pragma once

#include <cstdint>

namespace rtp_llm {

// Matches the existing temperature kernel's scale. Folding a positive scale
// into selected candidates preserves the sampling distribution up to FP rounding.
inline float sampleTemperatureInvScale(float temperature) {
    return 1.0f / (temperature + 1e-6f);
}

// Sampler-only policy. Penalties currently follow temperature and cannot be
// reordered; do_sample=false restores untempered rows. An all-unit batch skips
// temperature entirely, while zero temperature stays on the greedy path.
inline bool greedyTemperatureFoldedForRequest(int64_t      batch_size,
                                              const float* temperatures_host,
                                              int          force_topk,
                                              bool         has_do_sample_false_row,
                                              bool         has_cum_log_probs,
                                              bool         has_return_original_all_probs,
                                              bool         has_active_logits_penalty) {
    if (force_topk <= 0 || has_do_sample_false_row || has_cum_log_probs || has_return_original_all_probs
        || has_active_logits_penalty || batch_size <= 0 || temperatures_host == nullptr) {
        return false;
    }
    bool any_scaled = false;
    for (int64_t i = 0; i < batch_size; ++i) {
        const float temperature = temperatures_host[i];
        if (temperature == 0.0f) {
            return false;
        }
        any_scaled = any_scaled || temperature != 1.0f;
    }
    return any_scaled;
}

}  // namespace rtp_llm
