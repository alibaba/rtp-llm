#include "maga_transformer/cpp/speculative_engine/SpeculativeBatchStreamProcessor.h"

namespace rtp_llm {

absl::StatusOr<SpeculativeSamplerInput>
SpeculativeBatchStreamProcessor::gatherSpeculativeSamplerInput(const StreamGroups&    stream_groups,
                                                               const GptModelOutputs& model_output) const {
    return absl::UnimplementedError("not support yet");
}

absl::Status SpeculativeBatchStreamProcessor::dispatch(const StreamGroups&             stream_groups,
                                                       const GptModelOutputs&          model_outputs,
                                                       const SpeculativeSamplerOutput& sampler_outputs) const {
    return absl::UnimplementedError("not support yet");
}

}  // namespace rtp_llm
