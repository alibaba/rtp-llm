#pragma once

#include "src/fastertransformer/core/Buffer.h"

#include <optional>
#include <memory>
#include <unordered_map>

namespace fastertransformer {

// These weights should correspond to `maga_transformer/utils/model_weight.py`

struct LayerNormWeights {
    ConstBufferPtr gamma;
    ConstBufferPtr beta;
};

typedef std::unique_ptr<const LayerNormWeights> LayerNormWeightsPtr;

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

};

typedef std::unique_ptr<const DenseWeights> DenseWeightsPtr;

struct LoraWeights {
    ConstBufferPtr A;
    ConstBufferPtr B;
    ConstBufferPtr A_scale;
    ConstBufferPtr B_scale;
};

typedef std::unordered_map<std::string, LoraWeights> LoraWeightsMap;

struct AttentionLayerWeights {
    std::unique_ptr<const LayerNormWeights> pre_layernorm;
    std::unique_ptr<const LayerNormWeights> pre_attention_layernorm;
    std::unique_ptr<const DenseWeights>     qkv_weight;
    std::unique_ptr<const LoraWeightsMap>   query_lora_weights;
    std::unique_ptr<const LayerNormWeights> attention_layernorm;

    std::unique_ptr<const DenseWeights>     output_weight;
    std::unique_ptr<const LoraWeightsMap>   output_lora_weights;
    std::unique_ptr<const LayerNormWeights> post_layernorm;
};

struct FfnLayerWeights {
    std::unique_ptr<const DenseWeights>     up_weight;
    std::unique_ptr<const LoraWeightsMap>   up_lora_weights;

    std::unique_ptr<const DenseWeights>     gate_weight;
    std::unique_ptr<const LoraWeightsMap>   gate_lora_weights;

    std::unique_ptr<const DenseWeights>     down_weight;
    std::unique_ptr<const LoraWeightsMap>   down_lora_weights;
    std::unique_ptr<const LayerNormWeights> dense_layernorm;

    std::unique_ptr<const DenseWeights>     moe_gating_weight;

    FfnLayerWeights() = default;

    FfnLayerWeights(std::unique_ptr<const DenseWeights> up_weight,
                    std::unique_ptr<const DenseWeights> gate_weight,
                    std::unique_ptr<const DenseWeights> down_weight) :
                    up_weight(std::move(up_weight)),
                    gate_weight(std::move(gate_weight)),
                    down_weight(std::move(down_weight)) {}
};

struct LayerWeights {
    AttentionLayerWeights self_attention_weights;
    FfnLayerWeights       ffn_weights;
};

// TODO: This Weights class might be refactor into a complete model description
// which includes more info like norm type, activation type, etc.
struct Weights {
    std::unique_ptr<const DenseWeights>     embedding;
    std::unique_ptr<const DenseWeights>     prefix_encoder_embedding;
    std::unique_ptr<const LayerNormWeights> pre_decoder_layernorm;
    std::unique_ptr<const DenseWeights>     position_encoding;
    std::vector<LayerWeights>               layers;
    std::unique_ptr<const LayerNormWeights> final_layernorm;
    std::unique_ptr<const DenseWeights>     lm_head;
    std::unique_ptr<const DenseWeights>     medusa_head;
};

using WeightsPtr = std::unique_ptr<const Weights>;

}  // namespace fastertransformer
