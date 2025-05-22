#pragma once

#include <torch/extension.h>
#include "rtp_llm/cpp/embedding_engine/EmbeddingStream.h"

namespace th = torch;

namespace rtp_llm {
class EmbeddingQueryConverter {
public:
    static EmbeddingStreamPtr convertEmbeddingInputs(
        const torch::Tensor& token_ids,
        const torch::Tensor& token_type_ids,
        const torch::Tensor& input_lengths,
        int request_id,
        std::optional<MultimodalFeature> multimodal_features = std::nullopt);
};

} // namespace rtp_llm
