#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"

#include <cstdint>

namespace rtp_llm {

// Scan CPU tokens once to produce the single-pass mask and both pooling rows.
// Missing profiles use CLS_QI for warm-up; malformed profiles are rejected.
BertUqiBatchMetadata buildBertUqiInputs(const torch::Tensor& combo_tokens,
                                        const torch::Tensor& input_lengths,
                                        int32_t              segment_token_id,
                                        int32_t              separator_token_id);

}  // namespace rtp_llm
