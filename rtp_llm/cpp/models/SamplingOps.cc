#include "rtp_llm/cpp/models/SamplingOps.h"

#include <utility>

namespace rtp_llm {

GreedyOutput     sampleGreedy(const GreedyParams& params);
torch::Tensor    sampleFromProbs(const torch::Tensor& probabilities);
BeamSearchOutput sampleBeamSearch(BeamSearchParams params);
void             rejectionSampling(const RejectionSamplingParams& params);
void             mappingDraft2Target(const MappingDraft2TargetParams& params);

GreedyOutput execSampleGreedy(const GreedyParams& params) {
    return sampleGreedy(params);
}

torch::Tensor execSampleFromProbs(const torch::Tensor& probabilities) {
    return sampleFromProbs(probabilities);
}

BeamSearchOutput execSampleBeamSearch(BeamSearchParams params) {
    return sampleBeamSearch(std::move(params));
}

void execRejectionSampling(const RejectionSamplingParams& params) {
    rejectionSampling(params);
}

void execMappingDraft2Target(const MappingDraft2TargetParams& params) {
    mappingDraft2Target(params);
}

}  // namespace rtp_llm
