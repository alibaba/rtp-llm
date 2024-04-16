#include "maga_transformer/cpp/batch_stream_processor/SpeculativeBatchStreamProcessor.h"

namespace rtp_llm {

absl::Status SpeculativeBatchStreamProcessor::createValidateInput(GptModelInputs& input) {
    return absl::UnimplementedError("not impl yet!");
}

absl::Status SpeculativeBatchStreamProcessor::updateSPInput(GptModelInputs&      input,
                                                            const SamplerOutput& sampler_output) {
    return absl::UnimplementedError("not impl yet!");
}

}  // namespace rtp_llm
