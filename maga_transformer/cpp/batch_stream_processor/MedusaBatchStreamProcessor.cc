#include "maga_transformer/cpp/batch_stream_processor/MedusaBatchStreamProcessor.h"

namespace rtp_llm {

absl::Status MedusaBatchStreamProcessor::createAttentionMask(GptModelInputs& input) const {
    return absl::UnimplementedError("not impl yet!");
}

}  // namespace rtp_llm
