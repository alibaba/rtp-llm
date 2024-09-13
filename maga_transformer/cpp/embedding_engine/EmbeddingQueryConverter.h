#pragma once

#include <torch/extension.h>
#include "maga_transformer/cpp/embedding_engine/EmbeddingStream.h"

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
    static th::Tensor convertEmbeddingOutputs(EmbeddingStreamPtr stream);
};

} // namespace rtp_llm
