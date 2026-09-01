#pragma once

#include "rtp_llm/cpp/models/SamplingTypes.h"

namespace rtp_llm {

// Stable engine-facing entry points. Platform kernels stay private to the
// CUDA / ROCm binding target linked into librtp_compute_ops.so.
GreedyOutput     execSampleGreedy(const GreedyParams& params);
torch::Tensor    execSampleFromProbs(const torch::Tensor& probabilities);
BeamSearchOutput execSampleBeamSearch(BeamSearchParams params);
void             execRejectionSampling(const RejectionSamplingParams& params);
void             execMappingDraft2Target(const MappingDraft2TargetParams& params);

}  // namespace rtp_llm
