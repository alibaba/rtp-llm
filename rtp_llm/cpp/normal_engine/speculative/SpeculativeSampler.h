#pragma once

#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/EngineInitParams.h"
#include "rtp_llm/cpp/engine_base/ProposeModelEngineInitParams.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"

namespace rtp_llm {

namespace speculative {

struct SpeculativeSamplerOutput {
public:
    std::vector<BufferPtr> accept_tokens;
    std::vector<int>       accept_len;
};

class SpeculativeSampler {
public:
    SpeculativeSampler(rtp_llm::DeviceBase* device, size_t propose_step):
        device_(device), propose_step_(propose_step) {}

    SpeculativeSamplerOutput forward(const std::list<GenerateStreamPtr>& streams,
                                     SamplerOutput&                      draft_sampler_output,
                                     SamplerOutput&                      target_sampler_output) const;

private:
    void batchSample(SpeculativeSamplerOutput&           sample_output,
                     const std::list<GenerateStreamPtr>& streams,
                     SamplerOutput&                      draft_sampler_output,
                     SamplerOutput&                      target_sampler_output) const;
    void streamSample(SpeculativeSamplerOutput&           sample_output,
                      const std::list<GenerateStreamPtr>& streams,
                      SamplerOutput&                      draft_sampler_output,
                      SamplerOutput&                      target_sampler_output) const;

protected:
    rtp_llm::DeviceBase* device_;
    size_t               propose_step_;
};

}  // namespace speculative
}  // namespace rtp_llm