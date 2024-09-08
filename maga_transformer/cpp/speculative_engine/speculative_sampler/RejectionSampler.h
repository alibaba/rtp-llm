#pragma once

#include "maga_transformer/cpp/speculative_engine/speculative_sampler/SpeculativeSampler.h"
#include "src/fastertransformer/utils/logger.h"

namespace rtp_llm {

class RejectionSampler: public SpeculativeSampler {
public:
    RejectionSampler(ft::DeviceBase* device): SpeculativeSampler(device) {}
    absl::StatusOr<SpeculativeSamplerOutput> sample(const std::list<GenerateStreamPtr>& streams,
                                                    const ProposeOutput&                proposer_output,
                                                    const ScoreOutput&                  scorer_output) const override;

private:
    size_t top1Sample(size_t                                    propose_step,
                      const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                      const SpeculativeExecutorStreamOutputPtr& scorer_stream_output) const;
    size_t stochasticSample(size_t                                    propose_step,
                            const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                            const SpeculativeExecutorStreamOutputPtr& scorer_stream_output) const;
};

}  // namespace rtp_llm