#pragma once

#include "src/fastertransformer/core/Tensor.h"

#include <optional>
#include <unordered_map>

namespace fastertransformer {

// These weights should correspond to `maga_transformer/utils/model_weight.py`

struct LayerNormWeights {
    Tensor gamma;
    Tensor beta;
};

struct DenseWeights {
    Tensor                kernel;
    std::optional<Tensor> bias;
};

struct LoraWeights {
    Tensor A;
    Tensor B;
    std::optional<Tensor> A_scale;
    std::optional<Tensor> B_scale;
};

typedef std::unordered_map<std::string, LoraWeights> LoraWeightsMap;

struct AttentionLayerWeights {
    std::optional<LayerNormWeights> pre_layernorm;
    std::optional<LayerNormWeights> pre_attention_layernorm;
    DenseWeights                    query_weight;
    std::optional<LoraWeightsMap>   query_lora_weights;
    std::optional<LayerNormWeights> attention_layernorm;

    DenseWeights                    attention_output_weight;
    std::optional<LoraWeightsMap>   attention_output_lora_weights;
    std::optional<LayerNormWeights> post_layernorm;
};

struct FfnLayerWeights {
    DenseWeights                  intermediate_weight;
    std::optional<LoraWeightsMap> intermediate_lora_weights;

    std::optional<DenseWeights>   intermediate_weight2;
    std::optional<LoraWeightsMap> intermediate_lora_weights2;

    std::optional<DenseWeights>     intermediate_weight3;
    std::optional<LoraWeightsMap>   intermediate_lora_weights3;
    std::optional<LayerNormWeights> dense_layernorm;

    std::optional<DenseWeights> gating_weights;
};

struct LayerWeights {
    AttentionLayerWeights self_attention_weights;
    FfnLayerWeights       ffn_weights;
};

struct Weights {
    DenseWeights                    embedding;
    std::optional<DenseWeights>     prefix_encoder_embedding;
    std::optional<LayerNormWeights> pre_decoder_layernorm;
    std::optional<DenseWeights>     position_encoding;
    std::vector<LayerWeights>       layers;
    std::optional<LayerNormWeights> final_layernorm;
    std::optional<DenseWeights>     lm_head;
    std::optional<DenseWeights>     medusa_head;
};

}  // namespace fastertransformer
