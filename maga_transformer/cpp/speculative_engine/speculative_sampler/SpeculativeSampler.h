#pragma once

#include "absl/status/statusor.h"
#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "maga_transformer/cpp/stream/GenerateStream.h"
#include "maga_transformer/cpp/speculative_engine/propose_executor/ProposeOutput.h"
#include "maga_transformer/cpp/speculative_engine/score_executor/ScoreOutput.h"
#include "maga_transformer/cpp/speculative_engine/speculative_sampler/SpeculativeSamplerOutput.h"

namespace rtp_llm {

class SpeculativeSampler {
public:
    SpeculativeSampler(ft::DeviceBase* device): device_(device) {}
    virtual absl::StatusOr<SpeculativeSamplerOutput> sample(const std::list<GenerateStreamPtr>& streams,
                                                            const ProposeOutput&                proposer_output,
                                                            const ScoreOutput& scorer_output) const = 0;

protected:
    ft::DeviceBase* device_;
};

std::unique_ptr<SpeculativeSampler>
createSpeculativeSampler(const std::unique_ptr<ProposeModelEngineInitParams>& propose_model_engine_init_param,
                         ft::DeviceBase*                                      device);

}  // namespace rtp_llm