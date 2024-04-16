#pragma once

#include "src/fastertransformer/core/Buffer.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/FfnWeight.h"
#include "src/fastertransformer/layers/attention_layers/AttentionWeight.h"
#include "src/fastertransformer/models/W.h"
#include "src/fastertransformer/th_op/GptInitParameter.h"
#include <unordered_map>

namespace rtp_llm {
namespace ft = fastertransformer;
namespace W  = ft::W;

template<typename T>
T* maybe_get(const std::unordered_map<std::string, ft::ConstBufferPtr>& m, const std::string& name) {
    auto it = m.find(name);
    if (it == m.end()) {
        return nullptr;
    }
    return reinterpret_cast<T*>(it->second->data());
}

template<typename T>
struct GptGlobalWeights {
public:
    ft::DenseWeight<T>     embedding_table;
    ft::DenseWeight<T>     prefix_encoder_embedding;
    ft::DenseWeight<T>     lm_head;
    ft::DenseWeight<T>     position_encoding_table;
    ft::LayerNormWeight<T> pre_layernorm_weights;
    ft::LayerNormWeight<T> post_layernorm_weights;
    GptGlobalWeights(const std::unordered_map<std::string, ft::ConstBufferPtr>& global_weights) {
        embedding_table.kernel          = maybe_get<T>(global_weights, W::embedding);
        lm_head.kernel                  = maybe_get<T>(global_weights, W::lm_head);
        position_encoding_table.kernel  = maybe_get<T>(global_weights, W::wpe);
        pre_layernorm_weights.gamma     = maybe_get<T>(global_weights, W::pre_attn_ln_gamma);
        pre_layernorm_weights.beta      = maybe_get<T>(global_weights, W::pre_attn_ln_beta);
        post_layernorm_weights.gamma    = maybe_get<T>(global_weights, W::final_ln_gamma);
        post_layernorm_weights.beta     = maybe_get<T>(global_weights, W::final_ln_beta);
        prefix_encoder_embedding.kernel = maybe_get<T>(global_weights, W::prefix_w);
    }
};

}  // namespace rtp_llm
