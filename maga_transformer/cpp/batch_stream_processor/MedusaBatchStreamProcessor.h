#pragma once

#include "maga_transformer/cpp/batch_stream_processor/NormalBatchStreamProcessor.h"

namespace rtp_llm {

class MedusaBatchStreamProcessor: public NormalBatchStreamProcessor {
public:
    absl::Status createAttentionMask(GptModelInputs& input) const override;
};

}  // namespace rtp_llm
