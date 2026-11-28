#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

// Normalize CPU row metadata, summarize this single-beam batch and reserve
// its RNG offsets before uploading. Must run before sampling or row packing.
GreedyParams prepareSamplingParams(const GreedyParams& params, int64_t rows);
// Bind device token storage and prepare the ngram pointer table, if needed.
void prepareSamplingTokenIds(GreedyParams& params, const torch::Tensor& token_ids);

}  // namespace rtp_llm
