#pragma once

#include <cstdint>

namespace rtp_llm {

// The *_tps fields retain the existing collector semantics: token counts for the last step.
struct ExecutorStepMetrics {
    int64_t last_step_timestamp_us = 0;
    int64_t context_batch_size     = 0;
    int64_t generate_batch_size    = 0;
    int64_t execute_token_size     = 0;
    int64_t context_tps            = 0;
    int64_t generate_tps           = 0;
    int64_t model_forward_us       = 0;
};

}  // namespace rtp_llm
