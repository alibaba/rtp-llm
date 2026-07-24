#include "rtp_llm/cpp/models/SamplingOps.h"

#include <utility>

namespace rtp_llm {

// Platform implementations live in CudaSampleOp.cc / CudaBeamSearchOp.cc /
// the ROCm equivalents. We simply expose them under exec* names.
GreedyOutput     sampleGreedy(const GreedyParams& params);
BeamSearchOutput sampleBeamSearch(BeamSearchParams params);
void             chainSpeculativeSampling(const SpeculativeSamplingParams& params);
void             rejectionSampling(const RejectionSamplingParams& params);

GreedyOutput execSampleGreedy(const GreedyParams& params) {
    return sampleGreedy(params);
}

BeamSearchOutput execSampleBeamSearch(BeamSearchParams params) {
    return sampleBeamSearch(std::move(params));
}

void execChainSpeculativeSampling(const SpeculativeSamplingParams& params) {
    chainSpeculativeSampling(params);
}

void execRejectionSampling(const RejectionSamplingParams& params) {
    rejectionSampling(params);
}

}  // namespace rtp_llm
