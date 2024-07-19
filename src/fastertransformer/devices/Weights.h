#pragma once

#include "src/fastertransformer/core/Buffer.h"

#include <optional>
#include <memory>
#include <unordered_map>

namespace fastertransformer {

struct LayerNormWeights {
    ConstBufferPtr gamma = nullptr;
    ConstBufferPtr beta = nullptr;
    LayerNormWeights() = default;

    LayerNormWeights(ConstBufferPtr& gamma,
                     ConstBufferPtr& beta) :
        gamma(std::move(gamma)),
        beta(std::move(beta)) {}

    LayerNormWeights(BufferPtr& gamma,
                     BufferPtr& beta) :
        gamma(std::move(gamma)),
        beta(std::move(beta)) {}
};

typedef std::shared_ptr<const LayerNormWeights> LayerNormWeightsPtr;

struct DenseWeights {
    ConstBufferPtr kernel = nullptr;
    ConstBufferPtr bias = nullptr;

    DenseWeights() = default;

    DenseWeights(BufferPtr& kernel) : kernel(std::move(kernel)) {};

    DenseWeights(ConstBufferPtr& kernel) : kernel(std::move(kernel)) {};

    DenseWeights(ConstBufferPtr& kernel,
                 ConstBufferPtr& bias) :
                 kernel(std::move(kernel)),
                 bias(std::move(bias)) {};

    DenseWeights(BufferPtr& kernel,
                 BufferPtr& bias) :
                 kernel(std::move(kernel)),
                 bias(std::move(bias)) {};
};

typedef std::shared_ptr<const DenseWeights> DenseWeightsPtr;


struct LoraWeights {
    ConstBufferPtr A;
    ConstBufferPtr B;
};
typedef std::shared_ptr<const LoraWeights>  LoraWeightsPtr;

struct LoraWeightsMap {
    std::unordered_map<int64_t, LoraWeights> lora_map_;

    bool hasLoraWeight(int64_t lora_id) const {
        auto it = lora_map_.find(lora_id);
        return it != lora_map_.end();
    }

    LoraWeights getLoraWeight(int64_t lora_id) const {
        FT_CHECK(hasLoraWeight(lora_id));
        auto it = lora_map_.find(lora_id);
        return it->second;
    }


    void setLoRAWeight(int64_t lora_id,
                       ConstBufferPtr lora_a,
                       ConstBufferPtr lora_b)
    {
        lora_map_[lora_id] = LoraWeights({lora_a, lora_b});
    }

    void removeLoRAWeight(int64_t lora_id) {
        if (lora_map_.find(lora_id) == lora_map_.end()) {
            return;
        }
        lora_map_.erase(lora_id);
    }
};

struct AttentionLayerWeights {
    std::shared_ptr<const LayerNormWeights> pre_attention_layernorm;
    std::shared_ptr<const DenseWeights>     qkv_weight;
    std::shared_ptr<LoraWeightsMap>         qkv_lora_weights;
    std::shared_ptr<const LayerNormWeights> attention_layernorm;

    std::shared_ptr<const LayerNormWeights> q_norm_weight;
    std::shared_ptr<const LayerNormWeights> k_norm_weight;

    std::shared_ptr<const DenseWeights>     output_weight;
    std::shared_ptr<LoraWeightsMap>         output_lora_weights;

    std::shared_ptr<const DenseWeights>     smoother_weight;
    std::shared_ptr<const DenseWeights>     shift_weight;

    std::shared_ptr<const DenseWeights>     linear_bias_slopes_weight;
};

struct FfnLayerWeights {
    std::shared_ptr<const DenseWeights>     up_weight;
    std::shared_ptr<const DenseWeights>     moe_up_weight;
    std::shared_ptr<LoraWeightsMap>         up_lora_weights;

    std::shared_ptr<const DenseWeights>     gate_weight;
    std::shared_ptr<const DenseWeights>     moe_gate_weight;
    std::shared_ptr<LoraWeightsMap>         gate_lora_weights;

    std::shared_ptr<const DenseWeights>     down_weight;
    std::shared_ptr<const DenseWeights>     moe_down_weight;
    std::shared_ptr<LoraWeightsMap>         down_lora_weights;

    std::shared_ptr<const DenseWeights>     moe_gating_weight;

    std::shared_ptr<const DenseWeights>     smoother_weight;
    ConstBufferPtr                          act_scale;

    // these fields are for Qwen Mode model.
    // See https://github.com/huggingface/transformers/blob/0f67ba1d741d65b07d549daf4ee157609ce4f9c1/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L803
    std::shared_ptr<FfnLayerWeights>        shared_expert;
    std::shared_ptr<const DenseWeights>     shared_expert_gate;
};

struct LayerWeights {
    std::shared_ptr<const LayerNormWeights> pre_layernorm;
    AttentionLayerWeights                   self_attention_weights;
    std::shared_ptr<const DenseWeights>     pre_attention_smoother_weight;
    std::shared_ptr<const LayerNormWeights> post_layernorm;
    std::shared_ptr<const LayerNormWeights> post_layernorm_2;
    FfnLayerWeights                         ffn_weights;
    std::shared_ptr<const LayerNormWeights> post_ffn_layernorm;
};

// TODO: This Weights class might be refactor into a complete model description
// which includes more info like norm type, activation type, etc.
struct Weights {
    std::shared_ptr<const DenseWeights>     embedding;
    std::shared_ptr<const DenseWeights>     prefix_encoder_embedding;
    std::shared_ptr<const LayerNormWeights> pre_decoder_layernorm;
    std::shared_ptr<const DenseWeights>     position_encoding;
    std::shared_ptr<const DenseWeights>     token_type_embedding;
    std::vector<LayerWeights>               layers;
    std::shared_ptr<const LayerNormWeights> final_layernorm;
    std::shared_ptr<const DenseWeights>     linear_bias_slopes;
    std::shared_ptr<const DenseWeights>     lm_head;
    std::shared_ptr<const DenseWeights>     medusa_head;
};

using WeightsPtr = std::shared_ptr<const Weights>;

}  // namespace fastertransformer
