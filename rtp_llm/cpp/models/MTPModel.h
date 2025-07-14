#pragma once
#include "rtp_llm/cpp/models/GptModel.h"

namespace rtp_llm {

class MTPModel: public GptModel {
public:
    MTPModel(const GptModelInitParams& params):
        GptModel(params), reverse_e_h_norm_(params.description.reverse_e_h_norm) {
        RTP_LLM_LOG_INFO("MTP Model reverse e_h_norm : %d", reverse_e_h_norm_);
    };

    EmbeddingPostOutput embeddingPost(const rtp_llm::BufferPtr& hidden_states, const GptModelInputs& inputs) override;

private:
    bool reverse_e_h_norm_ = false;
};

}  // namespace rtp_llm
