#pragma once

#include <list>
#include <memory>

#include <torch/all.h>
#include "absl/status/statusor.h"
#include "rtp_llm/cpp/engine_base/stream/StreamGroups.h"
#include "rtp_llm/cpp/models/SampleInfos.h"
#include "rtp_llm/cpp/models/logits_processor/BaseLogitsProcessor.h"

namespace rtp_llm {

class BaseLogitsProcessor;

class NormalSamplerInputGatherer {
public:
    NormalSamplerInputGatherer() = default;

    // True when at least one stream asks for a repetition/presence/frequency
    // penalty or a no-repeat-ngram ban, i.e. when this batch can actually run
    // one of those kernels. An all-default request (repetition_penalty == 1,
    // presence_penalty == 0, frequency_penalty == 0, no_repeat_ngram_size unset)
    // returns false: the four per-row penalty arrays then stay undefined, which
    // Sampler::forward narrows to std::nullopt, so no pinned allocation, host
    // fill or activity scan happens for them (and the sampler, which uploads them
    // only when a penalty is active, never sees them at all).
    static bool anyStreamPenaltyEnabled(const std::list<GenerateStreamPtr>& streams);

    // True when any row of the four per-row penalty arrays holds a non-default
    // value (repetition != 1, presence/frequency != 0, ngram != 0). Undefined
    // arrays contribute nothing: they mean "every row uses the default", the
    // state anyStreamPenaltyEnabled leaves them in. Conservative for a defined
    // array that is not on the host: both call sites treat "active" as the safe
    // side (materialize the token history / keep the private logits copy).
    static bool anyPenaltyRowActive(const SamplerInputs& inputs);

    absl::StatusOr<SamplerInputs> gather(const StreamGroups&    stream_groups,
                                         const GptModelInputs&  model_inputs,
                                         const GptModelOutputs& model_output) const;

    // `materialize_host_token_ids=false` skips the full-width pinned token
    // snapshot. The MTP verify path builds its token matrix on the device from
    // one history row per request, so that snapshot would only be allocated and
    // never read.
    SamplerInputs allocateSamplerInputs(const StreamGroups& stream_groups,
                                        size_t              total_batch_size_in,
                                        size_t              total_batch_size_out,
                                        size_t              propose_step               = 0,
                                        bool                materialize_host_token_ids = true) const;

    void fillSamplerCommonInputs(SamplerInputs&                sampler_inputs,
                                 std::list<GenerateStreamPtr>& all_streams,
                                 bool                          score_batch  = false,
                                 size_t                        propose_step = 0) const;

    void setLogitsProcessorInputs(SamplerInputs&                sampler_inputs,
                                  std::list<GenerateStreamPtr>& all_streams,
                                  bool                          score_batch = false) const;
};

}  // namespace rtp_llm
