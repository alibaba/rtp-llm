#pragma once

#include "rtp_llm/models_py/bindings/core/OpData.h"

namespace rtp_llm {

// Forwarders to platform sampling kernels (CUDA / ROCm).
GreedyOutput     execSampleGreedy(const GreedyParams& params);
BeamSearchOutput execSampleBeamSearch(BeamSearchParams params);
void             execChainSpeculativeSampling(const SpeculativeSamplingParams& params);
void             execRejectionSampling(const RejectionSamplingParams& params);

}  // namespace rtp_llm
