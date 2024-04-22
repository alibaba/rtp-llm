#pragma once

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "maga_transformer/cpp/dataclass/MergedQuery.h"
#include "maga_transformer/cpp/dataclass/StreamGroups.h"
#include <list>

namespace rtp_llm {
class BatchStreamProcessor {
public:
    virtual ~BatchStreamProcessor() {}
    virtual absl::StatusOr<GptModelInputs> gatherModelInput(const StreamGroups& stream_groups) const     = 0;
    virtual absl::StatusOr<SamplerInputs>  gatherSamplerInput(const StreamGroups&    stream_groups,
                                                              const GptModelOutputs& model_output) const = 0;

    virtual absl::Status dispatch(const StreamGroups&                  stream_groups,
                                  const std::unique_ptr<MergedOutput>& merge_outputs) const = 0;

    virtual absl::Status createAttentionMask(GptModelInputs& input) const {
        // default causal mask, don't need init
        return absl::OkStatus();
    };
};

}  // namespace rtp_llm
