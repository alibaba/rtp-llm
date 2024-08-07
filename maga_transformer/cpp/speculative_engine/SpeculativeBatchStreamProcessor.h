#pragma once

#include "maga_transformer/cpp/normal_engine/NormalBatchStreamProcessor.h"
#include "maga_transformer/cpp/speculative_engine/SpeculativeSampler.h"

namespace rtp_llm {

class SpeculativeBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    SpeculativeBatchStreamProcessor(const ft::GptInitParameter& params, bool need_attention_mask): NormalBatchStreamProcessor(params, need_attention_mask) {}

    absl::StatusOr<SpeculativeSamplerInput> gatherSpeculativeSamplerInput(const StreamGroups&    stream_groups,
                                                                          const GptModelOutputs& model_output) const;

    absl::Status dispatch(const StreamGroups&             stream_groups,
                          const GptModelOutputs&          model_outputs,
                          const SpeculativeSamplerOutput& sampler_outputs) const;
};

}  // namespace rtp_llm
