#pragma once

#include <string>

namespace rtp_llm {
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
static const std::string linear_bias_slopes   = "linear_bias_slopes";

// mtp
static const std::string multi_tokens_predict_enorm          = "multi_tokens_predict_enorm.weight";
static const std::string multi_tokens_predict_hnorm          = "multi_tokens_predict_hnorm.weight";
static const std::string multi_tokens_predict_eh_proj        = "multi_tokens_predict_eh_proj.weight";
static const std::string multi_tokens_predict_final_ln_gamma = "multi_tokens_predict_final_layernorm.gamma";
static const std::string multi_tokens_predict_final_ln_beta  = "multi_tokens_predict_final_layernorm.beta";
static const std::string multi_tokens_predict_d2t_map        = "multi_tokens_predict_d2t_map";
static const std::string multi_tokens_predict_t2d_map        = "multi_tokens_predict_t2d_map";

// eagle3
static const std::string eagle3_fc_proj          = "eagle3_fc.weight";
static const std::string eagle3_fc_norm_gamma    = "eagle3_fc.gamma";
static const std::string eagle3_input_norm_gamma = "eagle3_input.gamma";

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

// attention layer for bert jina
static const std::string post_ln_2_gamma = "post_layernorm_weights_2.gamma";
static const std::string post_ln_2_beta  = "post_layernorm_weights_2.beta";
static const std::string q_ln_gamma      = "self_attention_weights.q_layernorm.gamma";
static const std::string q_ln_beta       = "self_attention_weights.q_layernorm.beta";
static const std::string k_ln_gamma      = "self_attention_weights.k_layernorm.gamma";
static const std::string k_ln_beta       = "self_attention_weights.k_layernorm.beta";

// mla
static const std::string mla_fusedqkrope = "self_attention_weights.mla.fusedqkrope.kernel";
// for lite
static const std::string mla_fusedqkrope_no_lora = "self_attention_weights.mla.fusedqkrope_no_lora.kernel";
static const std::string attn_q_b                = "self_attention_weights.mla.query_b_weight.kernel";
static const std::string attn_k_nope             = "self_attention_weights.mla.key_nope_weight.kernel";
static const std::string attn_v                  = "self_attention_weights.mla.value_weight.kernel";
static const std::string q_a_ln_gamma            = "self_attention_weights.mla.query_a_layernorm_weight.gamma";
static const std::string q_a_ln_beta             = "self_attention_weights.mla.query_a_layernorm_weight.beta";
static const std::string kv_a_ln_gamma           = "self_attention_weights.mla.key_value_a_layernorm_weight.gamma";
static const std::string kv_a_ln_beta            = "self_attention_weights.mla.key_value_a_layernorm_weight.beta";

// mla_absorb
static const std::string mla_kc   = "self_attention_weights.mla.kc.kernel";
static const std::string mla_vc   = "self_attention_weights.mla.vc.kernel";
static const std::string mla_kc_s = "self_attention_weights.mla.kc.weight_only_quant_scale";
static const std::string mla_vc_s = "self_attention_weights.mla.vc.weight_only_quant_scale";
// rotary embedding cos sin cache
static const std::string rope_cos_sin_cache = "rotary_embedding.cos_sin_cache";

static const std::string mla_fusedqkrope_s = "self_attention_weights.mla.fusedqkrope.weight_only_quant_scale";
static const std::string mla_fusedqkrope_no_lora_s =
    "self_attention_weights.mla.fusedqkrope_no_lora.weight_only_quant_scale";
static const std::string attn_q_b_s    = "self_attention_weights.mla.query_b_weight.weight_only_quant_scale";
static const std::string attn_k_nope_s = "self_attention_weights.mla.key_nope_weight.weight_only_quant_scale";
static const std::string attn_v_s      = "self_attention_weights.mla.value_weight.weight_only_quant_scale";

static const std::string ffn_w1            = "ffn_weights.intermediate_weight.kernel";
static const std::string ffn_b1            = "ffn_weights.intermediate_weight.bias";
static const std::string ffn_w3            = "ffn_weights.intermediate_weight3.kernel";
static const std::string ffn_b3            = "ffn_weights.intermediate_weight3.bias";
static const std::string ffn_w13           = "ffn_weights.intermediate_weight13.kernel";
static const std::string ffn_b13           = "ffn_weights.intermediate_weight13.bias";
static const std::string ffn_ln_gamma      = "ffn_weights.dense_layernorm.gamma";
static const std::string ffn_ln_beta       = "ffn_weights.dense_layernorm.beta";
static const std::string ffn_w2            = "ffn_weights.intermediate_weight2.kernel";
static const std::string ffn_b2            = "ffn_weights.intermediate_weight2.bias";
static const std::string post_ffn_ln_gamma = "post_ffn_layernorm_weights.gamma";
static const std::string post_ffn_ln_beta  = "post_ffn_layernorm_weights.beta";

static const std::string cross_attn_pre_ln_gamma = "cross_attention_weights_pre_layernorm.gamma";
static const std::string cross_attn_pre_ln_beta  = "cross_attention_weights_pre_layernorm.beta";
static const std::string cross_attn_qkv_w        = "cross_attention_weights.query_weight.weight";
static const std::string cross_attn_qkv_b        = "cross_attention_weights.query_weight.bias";
static const std::string cross_attn_o_w          = "cross_attention_weights.output_weight.weight";
static const std::string cross_attn_o_b          = "cross_attention_weights.output_weight.bias";

static const std::string shared_expert_gate_w     = "ffn_weights.shared_expert_gate.kernel";
static const std::string moe_w1                   = "partial_moe_weights.intermediate_weight.kernel";
static const std::string moe_b1                   = "partial_moe_weights.intermediate_weight.bias";
static const std::string moe_w2                   = "partial_moe_weights.intermediate_weight2.kernel";
static const std::string moe_b2                   = "partial_moe_weights.intermediate_weight2.bias";
static const std::string moe_gate                 = "partial_moe_weights.gate.kernel";
static const std::string moe_e_score_correction_b = "partial_moe_weights.e_score_correction_bias";

// eplb
static const std::string logic_expert_cnt = "moe_eplb.logic_expert_cnt";
static const std::string log2phy          = "moe_eplb.log2phy";

static const std::string attn_qkv_z = "self_attention_weights.query_weight.zero";
static const std::string attn_qkv_s = "self_attention_weights.query_weight.weight_only_quant_scale";
static const std::string attn_o_z   = "self_attention_weights.attention_output_weight.zero";
static const std::string attn_o_s   = "self_attention_weights.attention_output_weight.weight_only_quant_scale";
static const std::string ffn_z1     = "ffn_weights.intermediate_weight.zero";
static const std::string ffn_s1     = "ffn_weights.intermediate_weight.weight_only_quant_scale";
static const std::string ffn_z3     = "ffn_weights.intermediate_weight3.zero";
static const std::string ffn_s3     = "ffn_weights.intermediate_weight3.weight_only_quant_scale";
static const std::string ffn_z13    = "ffn_weights.intermediate_weight13.zero";
static const std::string ffn_s13    = "ffn_weights.intermediate_weight13.weight_only_quant_scale";
static const std::string ffn_act_s  = "ffn_weights.intermediate_weight2.act_quant_scale";
static const std::string ffn_z2     = "ffn_weights.intermediate_weight2.zero";
static const std::string ffn_s2     = "ffn_weights.intermediate_weight2.weight_only_quant_scale";
static const std::string moe_z1     = "partial_moe_weights.intermediate_weight.zero";
static const std::string moe_s1     = "partial_moe_weights.intermediate_weight.weight_only_quant_scale";
static const std::string moe_z2     = "partial_moe_weights.intermediate_weight2.zero";
static const std::string moe_s2     = "partial_moe_weights.intermediate_weight2.weight_only_quant_scale";

static const std::string attn_i_smoother = "self_attention_weights.query_weight.smoother";
static const std::string attn_i_shift    = "self_attention_weights.query_weight.shift";
static const std::string attn_o_smoother = "self_attention_weights.attention_output_weight.smoother";
static const std::string attn_o_shift    = "self_attention_weights.attention_output_weight.shift";
static const std::string ffn_smoother    = "ffn_weights.intermediate_weight2.smoother";

// static quant
static const std::string pre_decoder_ln_s         = "pre_decoder_layernorm.static_quant";
static const std::string pre_decoder_ln_static_sr = "pre_decoder_layernorm.static_quant_reciprocal";
static const std::string pre_ln_s                 = "pre_layernorm_weights.static_quant";
static const std::string pre_ln_sr                = "pre_layernorm_weights.static_quant_reciprocal";
static const std::string attention_output_s       = "self_attention_weights.attention_output_weight.static_quant";
static const std::string attention_output_sr = "self_attention_weights.attention_output_weight.static_quant_reciprocal";
static const std::string post_ln_s           = "post_layernorm_weights.static_quant";
static const std::string post_ln_sr          = "post_layernorm_weights.static_quant_reciprocal";
static const std::string ffn_intermediate_weight2_s  = "ffn_weights.intermediate_weight2.static_quant";
static const std::string ffn_intermediate_weight2_sr = "ffn_weights.intermediate_weight2.static_quant_reciprocal";
static const std::string ffn_intermediate_weight3_s  = "ffn_weights.intermediate_weight3.static_quant";
static const std::string ffn_intermediate_weight3_sr = "ffn_weights.intermediate_weight3.static_quant_reciprocal";
static const std::string post_ffn_ln_s               = "post_ffn_layernorm_weights.static_quant";
static const std::string post_ffn_ln_sr              = "post_ffn_layernorm_weights.static_quant_reciprocal";

// fp8 extra W
static const std::string attn_qkv_act_scale = "self_attention_weights.query_weight.act_quant_scale";
static const std::string attn_o_act_scale   = "self_attention_weights.attention_output.act_quant_scale";
static const std::string ffn_w1_act_scale   = "ffn_weights.intermediate_weight.act_quant_scale";
static const std::string ffn_w2_act_scale   = "ffn_weights.intermediate_weight2.act_quant_scale";  // same to ffn_act_s
static const std::string ffn_w3_act_scale   = "ffn_weights.intermediate_weight3.act_quant_scale";
static const std::string attn_qkv_act_scale_inv = "self_attention_weights.query_weight.act_quant_scale_inv";
static const std::string attn_o_act_scale_inv   = "self_attention_weights.attention_output.act_quant_scale_inv";
static const std::string ffn_w1_act_scale_inv   = "ffn_weights.intermediate_weight.act_quant_scale_inv";
static const std::string ffn_w2_act_scale_inv   = "ffn_weights.intermediate_weight2.act_quant_scale_inv";
static const std::string ffn_w3_act_scale_inv   = "ffn_weights.intermediate_weight3.act_quant_scale_inv";
}  // namespace W
}  // namespace rtp_llm
