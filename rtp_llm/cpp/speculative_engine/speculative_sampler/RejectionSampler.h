#pragma once

#include "rtp_llm/cpp/speculative_engine/speculative_sampler/SpeculativeSampler.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

class RejectionSampler: public SpeculativeSampler {
public:
    RejectionSampler(rtp_llm::DeviceBase* device): SpeculativeSampler(device) {}
    absl::StatusOr<SpeculativeSamplerOutput> sample(const std::list<GenerateStreamPtr>& streams,
                                                    const ProposeOutput&                proposer_output,
                                                    const ScoreOutput&                  scorer_output) const override;

private:
    absl::StatusOr<size_t> top1Sample(size_t                                    propose_step,
                      const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                      const SpeculativeExecutorStreamOutputPtr& scorer_stream_output) const;
    absl::StatusOr<size_t> stochasticSample(size_t                                    propose_step,
                            const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                            const SpeculativeExecutorStreamOutputPtr& scorer_stream_output) const;
};

}  // namespace rtp_llm