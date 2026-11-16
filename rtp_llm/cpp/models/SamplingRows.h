#pragma once

#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm {
// Select prepared row parameters and optional per-logical-row history using an
// int64 row map. Ngram pointers are bound to the selected history storage.
GreedyParams gatherSamplingRows(const GreedyParams& params, const torch::Tensor& row_map, const torch::Tensor& history);

// Mapped sampling returns the sampled column [logical_rows, 1]. Optional outputs
// remain undefined; omitted rows have token/probability 0, index -1 and success=false.
SamplerOutput scatterSamplingRows(const SamplerOutput& output, const torch::Tensor& inverse);
}  // namespace rtp_llm
