#pragma once

#include <torch/extension.h>

#include <optional>
#include <memory>
#include <unordered_map>
#include <thread>
#include <shared_mutex>

namespace rtp_llm {

struct LayerNormWeights {
    torch::Tensor gamma;
    torch::Tensor beta;
    torch::Tensor static_scale;
    torch::Tensor static_scale_reciprocal;
    LayerNormWeights() = default;
};

typedef std::shared_ptr<const LayerNormWeights> LayerNormWeightsPtr;

struct DenseWeights {
    torch::Tensor kernel;
    torch::Tensor bias;
    torch::Tensor scales;  // for quantized weights (FP8, INT4, INT8)
    torch::Tensor zeros;   // for quantized weights
    DenseWeights() = default;
};

typedef std::shared_ptr<DenseWeights> DenseWeightsPtr;

struct AttentionLayerWeights {
    std::shared_ptr<const LayerNormWeights> pre_attention_layernorm;
    std::shared_ptr<const DenseWeights>     qkv_weight;
    std::shared_ptr<const LayerNormWeights> attention_layernorm;

    std::shared_ptr<const LayerNormWeights> q_norm_weight;
    std::shared_ptr<const LayerNormWeights> k_norm_weight;

    std::shared_ptr<const DenseWeights> output_weight;

    std::shared_ptr<const DenseWeights> static_quant_weight;
    std::shared_ptr<const DenseWeights> static_scale_reciprocal_weight;
    std::shared_ptr<const DenseWeights> smoother_weight;
    std::shared_ptr<const DenseWeights> shift_weight;

    std::shared_ptr<const DenseWeights> linear_bias_slopes_weight;

    // mla weights
    std::shared_ptr<const DenseWeights>     fusedqkrope_weight;
    std::shared_ptr<const DenseWeights>     fusedqkrope_no_lora_weight;
    std::shared_ptr<const DenseWeights>     q_b_weight;
    std::shared_ptr<const DenseWeights>     kv_a_weight;
    std::shared_ptr<const DenseWeights>     k_nope_weight;
    std::shared_ptr<const DenseWeights>     k_rope_weight;
    std::shared_ptr<const DenseWeights>     v_weight;
    std::shared_ptr<const LayerNormWeights> q_a_norm_weight;
    std::shared_ptr<const LayerNormWeights> kv_a_norm_weight;

    // mla decode weights
    std::shared_ptr<const DenseWeights> kc_weight;
    std::shared_ptr<const DenseWeights> vc_weight;

    // rope cos sin cache
    torch::Tensor rope_cos_sin_cache;
};

struct FfnLayerWeights {
    std::shared_ptr<const DenseWeights> up_weight;

    std::shared_ptr<const DenseWeights> gate_weight;
    std::shared_ptr<DenseWeights>       moe_gate_weight;

    std::shared_ptr<const DenseWeights> down_weight;
    std::shared_ptr<DenseWeights>       moe_down_weight;
    std::shared_ptr<const DenseWeights> gate_up_weight;

    std::shared_ptr<const DenseWeights> moe_gating_weight;

    std::shared_ptr<const DenseWeights> smoother_weight;
    torch::Tensor                       act_scale;
    std::shared_ptr<const DenseWeights> intermediate_weight2_static_scale_weight;
    std::shared_ptr<const DenseWeights> intermediate_weight2_static_scale_reciprocal_weight;
    std::shared_ptr<const DenseWeights> intermediate_weight3_static_scale_weight;
    std::shared_ptr<const DenseWeights> intermediate_weight3_static_scale_reciprocal_weight;
    // these fields are for Qwen Mode model.
    // See https://github.com/huggingface/transformers/blo
    // dingb/0f67ba1d741d65b07d549daf4ee157609ce4f9c1/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L803
    std::shared_ptr<FfnLayerWeights>    shared_expert;
    std::shared_ptr<const DenseWeights> shared_expert_gate;
    torch::Tensor                       e_score_correction_bias;  // noaux_tc

    torch::Tensor log2phy;
    torch::Tensor logic_expert_cnt;
};

struct LayerWeights {
    std::shared_ptr<const LayerNormWeights> pre_layernorm;
    AttentionLayerWeights                   self_attention_weights;
    std::shared_ptr<const LayerNormWeights> post_layernorm;
    std::shared_ptr<const DenseWeights>     post_layernorm_quant_scale;
    FfnLayerWeights                         ffn_weights;
    std::shared_ptr<const LayerNormWeights> post_ffn_layernorm;

    // mtp
    std::shared_ptr<const LayerNormWeights> enorm;
    std::shared_ptr<const LayerNormWeights> hnorm;
    std::shared_ptr<const DenseWeights>     eh_proj;
    std::shared_ptr<const LayerNormWeights> mtp_final_layernorm;

    // eagle3
    std::shared_ptr<const LayerNormWeights> eagle3_input_norm;
    std::shared_ptr<const LayerNormWeights> eagle3_fc_norm;
    std::shared_ptr<const DenseWeights>     eagle3_fc_proj;
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
};

using WeightsPtr = std::shared_ptr<const Weights>;

}  // namespace rtp_llm
