#pragma once
#include "maga_transformer/cpp/models/GptModel.h"

namespace ft = fastertransformer;

namespace rtp_llm {


class MTPModel : public GptModel {
public:
    MTPModel(const GptModelInitParams& params) :
        GptModel(params), reverse_e_h_norm_(params.description.reverse_e_h_norm) {
            FT_LOG_INFO("MTP Model reverse e_h_norm : %d", reverse_e_h_norm_);
        };

    ft::BufferPtr embeddingPost(const ft::BufferPtr& hidden_states, const GptModelInputs& inputs) override;

private:
    bool reverse_e_h_norm_ = false;
};

}  // namespace rtp_llm
