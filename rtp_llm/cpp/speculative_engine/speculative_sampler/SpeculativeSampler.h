#pragma once

#include "absl/status/statusor.h"
#include "rtp_llm/cpp/dataclass/EngineInitParameter.h"
#include "rtp_llm/cpp/stream/GenerateStream.h"

namespace rtp_llm {

struct SpeculativeSamplerOutput {
public:
    std::string debugString() const {
        std::stringstream debug_string;
        debug_string << "SpeculativeSamplerOutput { ";
        debug_string << "propose_token_num: " << propose_token_num << ", accept_token_num: " << accept_token_num << "}";
        return debug_string.str();
    }

public:
    size_t propose_token_num = 0;
    size_t accept_token_num  = 0;
};

class SpeculativeSampler {
public:
    SpeculativeSampler(rtp_llm::DeviceBase* device): device_(device) {}
    absl::StatusOr<SpeculativeSamplerOutput> sample(const std::list<GenerateStreamPtr>& streams) const;

private:
    absl::StatusOr<size_t> top1Sample(size_t                                    propose_step,
                                      const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                                      const SpeculativeExecutorStreamOutputPtr& scorer_stream_output,
                                      bool                                      force_accept = false) const;
    absl::StatusOr<size_t> stochasticSample(size_t                                    propose_step,
                                            const SpeculativeExecutorStreamOutputPtr& propose_stream_output,
                                            const SpeculativeExecutorStreamOutputPtr& scorer_stream_output,
                                            bool                                      force_accept = false) const;

protected:
    rtp_llm::DeviceBase* device_;
};


}  // namespace rtp_llm