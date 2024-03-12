#pragma once

#include <string>

namespace fastertransformer {
namespace W {

// global
static const std::string embedding            = "embedding";
static const std::string lm_head              = "lm_head";
static const std::string prefix_w             = "transformer.prefix_encoder.embedding.weight";
static const std::string pre_decoder_ln_beta  = "pre_decoder_layernorm.bias";
static const std::string pre_decoder_ln_gamma = "pre_decoder_layernorm.gamma";
static const std::string wpe                  = "position_encoding.weight";
static const std::string final_ln_gamma       = "final_layernorm.gamma";
static const std::string final_ln_beta        = "final_layernorm.beta";

// layer
static const std::string pre_ln_gamma      = "pre_layernorm_weights.gamma";
static const std::string pre_ln_beta       = "pre_layernorm_weights.beta";
static const std::string pre_attn_ln_gamma = "pre_attn_layernorm_weights.gamma";
static const std::string pre_attn_ln_beta  = "pre_attn_layernorm_weights.beta";
static const std::string attn_qkv_w        = "self_attention_weights.query_weight.kernel";
static const std::string attn_qkv_b        = "self_attention_weights.query_weight.bias";
static const std::string attn_ln_gamma     = "self_attention_weights.attention_layernorm.gamma";
static const std::string attn_ln_beta      = "self_attention_weights.attention_layernorm.beta";
static const std::string attn_o_w          = "self_attention_weights.attention_output_weight.kernel";
static const std::string attn_o_b          = "self_attention_weights.attention_output_weight.bias";
static const std::string post_ln_gamma     = "post_layernorm_weights.gamma";
static const std::string post_ln_beta      = "post_layernorm_weights.beta";

static const std::string ffn_w1       = "ffn_weights.intermediate_weight.kernel";
static const std::string ffn_b1       = "ffn_weights.intermediate_weight.bias";
static const std::string ffn_w3       = "ffn_weights.intermediate_weight3.kernel";
static const std::string ffn_b3       = "ffn_weights.intermediate_weight3.bias";
static const std::string ffn_ln_gamma = "ffn_weights.dense_layernorm.gamma";
static const std::string ffn_ln_beta  = "ffn_weights.dense_layernorm.beta";
static const std::string ffn_w2       = "ffn_weights.intermediate_weight2.kernel";
static const std::string ffn_b2       = "ffn_weights.intermediate_weight2.bias";
static const std::string ffn_gate     = "ffn_weights.gate.kernel";
static const std::string post_ffn_ln_gamma     = "post_ffn_layernorm_weights.gamma";
static const std::string post_ffn_ln_beta      = "post_ffn_layernorm_weights.beta";

}  // namespace W
}  // namespace fastertransformer
