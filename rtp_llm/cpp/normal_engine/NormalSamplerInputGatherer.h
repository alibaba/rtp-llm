#pragma once

#include <list>

#include <torch/all.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"

namespace rtp_llm {

class NormalSamplerInputGatherer {
public:
    explicit NormalSamplerInputGatherer(bool allow_history_elision = false):
        allow_history_elision_(allow_history_elision) {}

    absl::StatusOr<SamplerInputs> gather(const StreamGroups&    stream_groups,
                                         const GptModelInputs&  model_inputs,
                                         const GptModelOutputs& model_output) const;

    SamplerInputs allocateSamplerInputs(const StreamGroups& stream_groups,
                                        size_t              total_batch_size_in,
                                        size_t              total_batch_size_out,
                                        size_t              propose_step = 0,
                                        bool                materialize_host_token_ids = true) const;

    void fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                 std::list<GenerateStreamPtr>& all_streams,
                                 bool                          score_batch  = false,
                                 size_t                        propose_step = 0) const;

    void setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                  std::list<GenerateStreamPtr>& all_streams,
                                  bool                          score_batch = false) const;
private:
    bool allow_history_elision_;
};

}  // namespace rtp_llm
