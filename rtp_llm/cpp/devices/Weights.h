#pragma once

#include "rtp_llm/cpp/core/Buffer.h"

#include <optional>
#include <memory>
#include <unordered_map>
#include <thread>
#include <shared_mutex>

namespace rtp_llm {

struct LayerNormWeights {
    ConstBufferPtr gamma                   = nullptr;
    ConstBufferPtr beta                    = nullptr;
    ConstBufferPtr static_scale            = nullptr;
    ConstBufferPtr static_scale_reciprocal = nullptr;
    LayerNormWeights()                     = default;

    LayerNormWeights(ConstBufferPtr& gamma, ConstBufferPtr& beta): gamma(std::move(gamma)), beta(std::move(beta)) {}

    LayerNormWeights(BufferPtr& gamma, BufferPtr& beta): gamma(std::move(gamma)), beta(std::move(beta)) {}

    LayerNormWeights(ConstBufferPtr& gamma,
                     ConstBufferPtr& beta,
                     ConstBufferPtr& static_scale,
                     ConstBufferPtr& static_scale_reciprocal):
        gamma(std::move(gamma)),
        beta(std::move(beta)),
        static_scale(std::move(static_scale)),
        static_scale_reciprocal(std::move(static_scale_reciprocal)) {}

    LayerNormWeights(BufferPtr& gamma, BufferPtr& beta, BufferPtr& static_scale, BufferPtr& static_scale_reciprocal):
        gamma(std::move(gamma)),
        beta(std::move(beta)),
        static_scale(std::move(static_scale)),
        static_scale_reciprocal(std::move(static_scale_reciprocal)) {}
};

typedef std::shared_ptr<const LayerNormWeights> LayerNormWeightsPtr;

struct DenseWeights {
    ConstBufferPtr kernel = nullptr;
    ConstBufferPtr bias   = nullptr;
    DenseWeights()        = default;

    DenseWeights(BufferPtr& kernel): kernel(std::move(kernel)) {};

    DenseWeights(ConstBufferPtr& kernel): kernel(std::move(kernel)) {};

    DenseWeights(ConstBufferPtr& kernel, ConstBufferPtr& bias): kernel(std::move(kernel)), bias(std::move(bias)) {};

    DenseWeights(BufferPtr& kernel, BufferPtr& bias): kernel(std::move(kernel)), bias(std::move(bias)) {};
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
    ConstBufferPtr rope_cos_sin_cache;
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
    ConstBufferPtr                      act_scale;
    std::shared_ptr<const DenseWeights> intermediate_weight2_static_scale_weight;
    std::shared_ptr<const DenseWeights> intermediate_weight2_static_scale_reciprocal_weight;
    std::shared_ptr<const DenseWeights> intermediate_weight3_static_scale_weight;
    std::shared_ptr<const DenseWeights> intermediate_weight3_static_scale_reciprocal_weight;
    // these fields are for Qwen Mode model.
    // See https://github.com/huggingface/transformers/blo
    // dingb/0f67ba1d741d65b07d549daf4ee157609ce4f9c1/src/transformers/models/qwen2_moe/modeling_qwen2_moe.py#L803
    std::shared_ptr<FfnLayerWeights>    shared_expert;
    std::shared_ptr<const DenseWeights> shared_expert_gate;
    ConstBufferPtr                      e_score_correction_bias;  // noaux_tc

    ConstBufferPtr log2phy;
    ConstBufferPtr logic_expert_cnt;
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

    // for sp decode prune vocab
    ConstBufferPtr d2t_map;
    ConstBufferPtr t2d_map;
};

using WeightsPtr = std::shared_ptr<const Weights>;

}  // namespace rtp_llm
