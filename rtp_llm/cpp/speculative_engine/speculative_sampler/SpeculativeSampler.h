#pragma once

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/dataclass/EngineInitParameter.h"
#include "rtp_llm/cpp/stream/GenerateStream.h"
#include "rtp_llm/cpp/speculative_engine/propose_executor/ProposeOutput.h"
#include "rtp_llm/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "rtp_llm/cpp/speculative_engine/speculative_sampler/SpeculativeSamplerOutput.h"

namespace rtp_llm {

class SpeculativeSampler {
public:
    SpeculativeSampler(rtp_llm::DeviceBase* device): device_(device) {}
    virtual absl::StatusOr<SpeculativeSamplerOutput> sample(const std::list<GenerateStreamPtr>& streams,
                                                            const ProposeOutput&                proposer_output,
                                                            const ScoreOutput& scorer_output) const = 0;

protected:
    rtp_llm::DeviceBase* device_;
};

std::unique_ptr<SpeculativeSampler>
createSpeculativeSampler(const std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_param,
                         rtp_llm::DeviceBase*                                      device);

}  // namespace rtp_llm