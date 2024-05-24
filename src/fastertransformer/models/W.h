#pragma once

#include <string>

namespace fastertransformer {
namespace W {

// global
static const std::string embedding            = "embedding";
static const std::string token_type_embedding = "token_type_embedding.weight";
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
static const std::string qk_ln_gamma       = "self_attention_weights.qk_layernorm.gamma";
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
static const std::string post_ffn_ln_gamma     = "post_ffn_layernorm_weights.gamma";
static const std::string post_ffn_ln_beta      = "post_ffn_layernorm_weights.beta";

static const std::string shared_expert_gate_w = "ffn_weights.shared_expert_gate.kernel";
static const std::string moe_w1   = "partial_moe_weights.intermediate_weight.kernel";
static const std::string moe_b1   = "partial_moe_weights.intermediate_weight.bias";
static const std::string moe_w3   = "partial_moe_weights.intermediate_weight3.kernel";
static const std::string moe_b3   = "partial_moe_weights.intermediate_weight3.bias";
static const std::string moe_w2   = "partial_moe_weights.intermediate_weight2.kernel";
static const std::string moe_b2   = "partial_moe_weights.intermediate_weight2.bias";
static const std::string moe_gate = "partial_moe_weights.gate.kernel";

static const std::string attn_qkv_z = "self_attention_weights.query_weight.zero";
static const std::string attn_qkv_s = "self_attention_weights.query_weight.weight_only_quant_scale";
static const std::string attn_o_z = "self_attention_weights.attention_output_weight.zero";
static const std::string attn_o_s = "self_attention_weights.attention_output_weight.weight_only_quant_scale";
static const std::string ffn_z1 = "ffn_weights.intermediate_weight.zero";
static const std::string ffn_s1 = "ffn_weights.intermediate_weight.weight_only_quant_scale";
static const std::string ffn_z3 = "ffn_weights.intermediate_weight3.zero";
static const std::string ffn_s3 = "ffn_weights.intermediate_weight3.weight_only_quant_scale";
static const std::string ffn_act_s = "ffn_weights.intermediate_weight2.act_quant_scale";

static const std::string ffn_z2 = "ffn_weights.intermediate_weight2.zero";
static const std::string ffn_s2 = "ffn_weights.intermediate_weight2.weight_only_quant_scale";
static const std::string moe_z1 = "partial_moe_weights.intermediate_weight.zero";
static const std::string moe_s1 = "partial_moe_weights.intermediate_weight.weight_only_quant_scale";
static const std::string moe_z3 = "partial_moe_weights.intermediate_weight3.zero";
static const std::string moe_s3 = "partial_moe_weights.intermediate_weight3.weight_only_quant_scale";
static const std::string moe_z2 = "partial_moe_weights.intermediate_weight2.zero";
static const std::string moe_s2 = "partial_moe_weights.intermediate_weight2.weight_only_quant_scale";

static const std::string attn_o_smoother = "self_attention_weights.attention_output_weight.smoother";
static const std::string attn_o_shift = "self_attention_weights.attention_output_weight.shift";
static const std::string ffn_smoother = "ffn_weights.intermediate_weight2.smoother";

}
}
