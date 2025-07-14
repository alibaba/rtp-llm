#pragma once
#include "rtp_llm/cpp/models/GptModel.h"

namespace rtp_llm {

class Eagle3Model: public GptModel {
public:
    Eagle3Model(const GptModelInitParams& params): GptModel(params) {};

    EmbeddingPostOutput embeddingPost(const rtp_llm::BufferPtr& hidden_states, const GptModelInputs& inputs) override;
};

}  // namespace rtp_llm
