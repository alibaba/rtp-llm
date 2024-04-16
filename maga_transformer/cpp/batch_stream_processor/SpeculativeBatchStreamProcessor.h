#pragma once

#include "maga_transformer/cpp/batch_stream_processor/NormalBatchStreamProcessor.h"

namespace rtp_llm {

class SpeculativeBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    SpeculativeBatchStreamProcessor(const GptInitParameter& params): NormalBatchStreamProcessor(params) {}

    static absl::Status updateSPInput(GptModelInputs& input, const SamplerOutput& sampler_output);
    static absl::Status createValidateInput(GptModelInputs& input);
};

}  // namespace rtp_llm
