#include "rtp_llm/cpp/engine_base/WeightsConverter.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include <memory>
using namespace std;

namespace rtp_llm {

WeightsConverter::WeightsConverter(bool need_copy, rtp_llm::QuantAlgo quant_alog):
    need_copy_(need_copy), quant_algo_(quant_alog) {}

torch::Tensor WeightsConverter::CopyTensorToGPU(const torch::Tensor& tensor) {
    if (need_copy_) {
        auto gpu_tensor = torch::empty_like(tensor, torch::TensorOptions().device(torch::kCUDA));
        gpu_tensor.copy_(tensor, /*non_blocking=*/true);
        return gpu_tensor;
    } else {
        return tensor;
    }
}

torch::Tensor WeightsConverter::mayFindTensor(const TensorMap& map, const std::string& key) {
    auto it = map.find(key);
    if (it != map.end()) {
        return CopyTensorToGPU(it->second);
    }
    return torch::Tensor();
}

rtp_llm::LayerNormWeightsPtr WeightsConverter::mayCreateLayerNormWeights(const TensorMap&   map,
                                                                         const std::string& gamma_key,
                                                                         const std::string& beta_key,
                                                                         const std::string& scale_key,
                                                                         const std::string& scale_reciprocal_key) {
    if (map.count(gamma_key) > 0) {
        const auto layer_norm_weights               = new LayerNormWeights();
        layer_norm_weights->gamma                   = mayFindTensor(map, gamma_key);
        layer_norm_weights->beta                    = mayFindTensor(map, beta_key);
        layer_norm_weights->static_scale            = mayFindTensor(map, scale_key);
        layer_norm_weights->static_scale_reciprocal = mayFindTensor(map, scale_reciprocal_key);
        return unique_ptr<const LayerNormWeights>(layer_norm_weights);
    }
    return nullptr;
}

rtp_llm::DenseWeightsPtr WeightsConverter::mayCreateDenseWeights(const TensorMap&   map,
                                                                 const std::string& kernel_key,
                                                                 const std::string& bias_key,
                                                                 const std::string& scales_key,
                                                                 const std::string& zeros_key) {
    if (map.count(kernel_key) <= 0) {
        return nullptr;
    }

    const auto dense_weights = new DenseWeights();
    if (!bias_key.empty()) {
        dense_weights->bias = mayFindTensor(map, bias_key);
    }
    auto scales_tensor = mayFindTensor(map, scales_key);
    auto zeros_tensor  = mayFindTensor(map, zeros_key);

    if (!scales_tensor.defined()) {
        // No quantization - just store kernel directly
        dense_weights->kernel = mayFindTensor(map, kernel_key);
    } else {
        auto kernel       = mayFindTensor(map, kernel_key);
        auto kernel_sizes = kernel.sizes().vec();
        auto dtype        = kernel.scalar_type();
        if (quant_algo_.isFp8() && scales_tensor.defined()) {
            dtype = torch::kFloat8_e4m3fn;
        } else if (quant_algo_.isQuant() && scales_tensor.defined()) {
            RTP_LLM_LOG_DEBUG(
                "load weight_only qbuffer weight [%s] scale [%s]", kernel_key.c_str(), scales_key.c_str());
            if (quant_algo_.getWeightBits() == 4) {
                // TYPE_INT4X2 doesn't have a torch equivalent, use kInt8 with doubled last dim
                dtype = torch::kInt8;
#if USING_CK_INT4  // for composable kernel specific
                kernel_sizes[kernel.dim() - 2] = kernel_sizes[kernel.dim() - 2] * 2;
#else
                kernel_sizes[kernel.dim() - 1] = kernel_sizes[kernel.dim() - 1] * 2;
#endif
            }
        }
        // Create a view with adjusted shape/dtype
        auto adjusted_kernel = torch::from_blob(
            kernel.data_ptr(),
            kernel_sizes,
            [kernel](void*) { /* prevent free, original tensor is captured */ },
            torch::TensorOptions().dtype(dtype).device(kernel.device()));
        dense_weights->kernel = adjusted_kernel;
        dense_weights->scales = scales_tensor;
        if (zeros_tensor.defined()) {
            dense_weights->zeros = zeros_tensor;
        }
        RTP_LLM_LOG_DEBUG("quant_method:%d, kernel_key:%s have scale, kernel shape: [%s]",
                          quant_algo_.getQuantMethod(),
                          kernel_key.c_str(),
                          adjusted_kernel.sizes().vec().empty() ?
                              "empty" :
                              ([&]() -> std::string {
                                  std::string s;
                                  for (size_t i = 0; i < adjusted_kernel.sizes().size(); ++i) {
                                      if (i > 0)
                                          s += ", ";
                                      s += std::to_string(adjusted_kernel.size(i));
                                  }
                                  return s;
                              })()
                                  .c_str());
    }

    return unique_ptr<DenseWeights>(dense_weights);
}

rtp_llm::FfnLayerWeights WeightsConverter::createFfnWeights(const TensorMap& map) {
    rtp_llm::FfnLayerWeights ffn_weights;

    ffn_weights.up_weight      = mayCreateDenseWeights(map, W::ffn_w3, W::ffn_b3, W::ffn_s3, W::ffn_z3);
    ffn_weights.gate_weight    = mayCreateDenseWeights(map, W::ffn_w1, W::ffn_b1, W::ffn_s1, W::ffn_z1);
    ffn_weights.down_weight    = mayCreateDenseWeights(map, W::ffn_w2, W::ffn_b2, W::ffn_s2, W::ffn_z2);
    ffn_weights.gate_up_weight = mayCreateDenseWeights(map, W::ffn_w13, W::ffn_b13, W::ffn_s13, W::ffn_z13);

    ffn_weights.moe_gating_weight       = mayCreateDenseWeights(map, W::moe_gate);
    ffn_weights.moe_gate_weight         = mayCreateDenseWeights(map, W::moe_w1, W::moe_b1, W::moe_s1, W::moe_z1);
    ffn_weights.moe_down_weight         = mayCreateDenseWeights(map, W::moe_w2, W::moe_b2, W::moe_s2, W::moe_z2);
    ffn_weights.e_score_correction_bias = mayFindTensor(map, W::moe_e_score_correction_b);

    ffn_weights.smoother_weight = mayCreateDenseWeights(map, W::ffn_smoother);
    ffn_weights.act_scale       = mayFindTensor(map, W::ffn_act_s);

    ffn_weights.intermediate_weight2_static_scale_weight = mayCreateDenseWeights(map, W::ffn_intermediate_weight2_s);
    ffn_weights.intermediate_weight2_static_scale_reciprocal_weight =
        mayCreateDenseWeights(map, W::ffn_intermediate_weight2_sr);

    // for qwen moe
    if (ffn_weights.moe_gating_weight) {
        // this moe layer has a parallel dense ffn layer as shared expert.
        if (ffn_weights.gate_up_weight) {
            ffn_weights.shared_expert                 = make_shared<rtp_llm::FfnLayerWeights>();
            ffn_weights.shared_expert->gate_up_weight = move(ffn_weights.gate_up_weight);
            ffn_weights.shared_expert->down_weight    = move(ffn_weights.down_weight);
        } else if (ffn_weights.up_weight) {
            ffn_weights.shared_expert              = make_shared<rtp_llm::FfnLayerWeights>();
            ffn_weights.shared_expert->up_weight   = move(ffn_weights.up_weight);
            ffn_weights.shared_expert->gate_weight = move(ffn_weights.gate_weight);
            ffn_weights.shared_expert->down_weight = move(ffn_weights.down_weight);
        }

        // for qwen moe
        ffn_weights.shared_expert_gate = mayCreateDenseWeights(map, W::shared_expert_gate_w);
    }

    // eplb stats
    ffn_weights.logic_expert_cnt = mayFindTensor(map, W::logic_expert_cnt);
    ffn_weights.log2phy          = mayFindTensor(map, W::log2phy);

    return ffn_weights;
}

rtp_llm::AttentionLayerWeights WeightsConverter::createAttentionWeights(const TensorMap& map) {
    rtp_llm::AttentionLayerWeights attention_weights;
    attention_weights.pre_attention_layernorm =
        mayCreateLayerNormWeights(map, W::pre_attn_ln_gamma, W::pre_attn_ln_beta);

    attention_weights.qkv_weight =
        mayCreateDenseWeights(map, W::attn_qkv_w, W::attn_qkv_b, W::attn_qkv_s, W::attn_qkv_z);

    // some model doesn't have qkv weight, so we need to create a empty weight.
    if (!attention_weights.qkv_weight) {
        attention_weights.qkv_weight = std::unique_ptr<const DenseWeights>(new DenseWeights());
    }

    attention_weights.q_norm_weight = mayCreateLayerNormWeights(map, W::q_ln_gamma, W::q_ln_beta);
    attention_weights.k_norm_weight = mayCreateLayerNormWeights(map, W::k_ln_gamma, W::k_ln_beta);

    attention_weights.attention_layernorm = mayCreateLayerNormWeights(map, W::attn_ln_gamma, W::attn_ln_beta);
    attention_weights.output_weight = mayCreateDenseWeights(map, W::attn_o_w, W::attn_o_b, W::attn_o_s, W::attn_o_z);

    attention_weights.shift_weight = mayCreateDenseWeights(map, W::attn_o_shift);

    attention_weights.smoother_weight = mayCreateDenseWeights(map, W::attn_o_smoother);

    // mla weights
    attention_weights.fusedqkrope_weight = mayCreateDenseWeights(map, W::mla_fusedqkrope, "", W::mla_fusedqkrope_s);
    attention_weights.fusedqkrope_no_lora_weight =
        mayCreateDenseWeights(map, W::mla_fusedqkrope_no_lora, "", W::mla_fusedqkrope_no_lora_s);
    attention_weights.q_b_weight       = mayCreateDenseWeights(map, W::attn_q_b, "", W::attn_q_b_s);
    attention_weights.k_nope_weight    = mayCreateDenseWeights(map, W::attn_k_nope, "", W::attn_k_nope_s);
    attention_weights.v_weight         = mayCreateDenseWeights(map, W::attn_v, "", W::attn_v_s);
    attention_weights.q_a_norm_weight  = mayCreateLayerNormWeights(map, W::q_a_ln_gamma, W::q_a_ln_beta);
    attention_weights.kv_a_norm_weight = mayCreateLayerNormWeights(map, W::kv_a_ln_gamma, W::kv_a_ln_beta);

    attention_weights.kc_weight = mayCreateDenseWeights(map, W::mla_kc, "", W::mla_kc_s);
    attention_weights.vc_weight = mayCreateDenseWeights(map, W::mla_vc, "", W::mla_vc_s);

    attention_weights.static_quant_weight            = mayCreateDenseWeights(map, W::attention_output_s);
    attention_weights.static_scale_reciprocal_weight = mayCreateDenseWeights(map, W::attention_output_sr);

    return attention_weights;
}

std::unique_ptr<TensorMaps> WeightsConverter::convertLayerWeights(py::object py_layer_weights) {
    TensorMaps tensor_layer_weights;
    auto       layers_weights_vec = convertPyObjectToVec(py_layer_weights);
    for (auto& layer_weights : layers_weights_vec) {
        TensorMap weights;
        for (auto& it : convertPyObjectToDict(layer_weights)) {
            weights.emplace(it.first, convertPyObjectToTensor(it.second));
        }
        tensor_layer_weights.emplace_back(std::move(weights));
    }
    return std::make_unique<TensorMaps>(std::move(tensor_layer_weights));
}

std::unique_ptr<TensorMap> WeightsConverter::convertGlobalWeight(py::object py_global_weight) {
    TensorMap global_weights;
    auto      global_weights_dict = convertPyObjectToDict(py_global_weight);
    for (auto& it : global_weights_dict) {
        global_weights.emplace(it.first, convertPyObjectToTensor(it.second));
    }
    return std::make_unique<TensorMap>(std::move(global_weights));
}

std::unique_ptr<rtp_llm::Weights> WeightsConverter::createGptWeights(py::object layer_weights,
                                                                     py::object global_weight) {
    return std::move(
        createGptWeights(std::move(convertLayerWeights(layer_weights)), std::move(convertGlobalWeight(global_weight))));
}

std::unique_ptr<rtp_llm::Weights> WeightsConverter::createGptWeights(std::unique_ptr<TensorMaps> layer_weights,
                                                                     std::unique_ptr<TensorMap>  global_weight) {
    auto             layers_weights = *layer_weights;
    rtp_llm::Weights gpt_weights;

    // make global weight
    gpt_weights.embedding                = mayCreateDenseWeights(*global_weight, W::embedding);
    gpt_weights.prefix_encoder_embedding = mayCreateDenseWeights(*global_weight, W::prefix_w);
    gpt_weights.pre_decoder_layernorm    = mayCreateLayerNormWeights(*global_weight,
                                                                  W::pre_decoder_ln_gamma,
                                                                  W::pre_decoder_ln_beta,
                                                                  W::pre_decoder_ln_s,
                                                                  W::pre_decoder_ln_static_sr);
    gpt_weights.position_encoding        = mayCreateDenseWeights(*global_weight, W::wpe);
    gpt_weights.token_type_embedding     = mayCreateDenseWeights(*global_weight, W::token_type_embedding);
    gpt_weights.final_layernorm = mayCreateLayerNormWeights(*global_weight, W::final_ln_gamma, W::final_ln_beta);
    gpt_weights.lm_head         = mayCreateDenseWeights(*global_weight, W::lm_head);

    gpt_weights.linear_bias_slopes = mayCreateDenseWeights(*global_weight, W::linear_bias_slopes);

    for (auto& layer_weights : layers_weights) {
        rtp_llm::LayerWeights layer_ws;
        layer_ws.pre_layernorm =
            mayCreateLayerNormWeights(layer_weights, W::pre_ln_gamma, W::pre_ln_beta, W::pre_ln_s, W::pre_ln_sr);

        layer_ws.post_ffn_layernorm = mayCreateLayerNormWeights(
            layer_weights, W::post_ffn_ln_gamma, W::post_ffn_ln_beta, W::post_ffn_ln_s, W::post_ffn_ln_sr);

        layer_ws.post_layernorm =
            mayCreateLayerNormWeights(layer_weights, W::post_ln_gamma, W::post_ln_beta, W::post_ln_s, W::post_ln_sr);

        layer_ws.self_attention_weights = createAttentionWeights(layer_weights);

        layer_ws.self_attention_weights.rope_cos_sin_cache = mayFindTensor(*global_weight, W::rope_cos_sin_cache);
        if (layer_ws.self_attention_weights.rope_cos_sin_cache.defined()) {
            RTP_LLM_CHECK_WITH_INFO(layer_ws.self_attention_weights.rope_cos_sin_cache.scalar_type() == torch::kFloat32,
                                    "rope_cos_sin_cache must be fp32");
        }

        layer_ws.ffn_weights = createFfnWeights(layer_weights);

        // mtp
        layer_ws.enorm               = mayCreateLayerNormWeights(layer_weights, W::multi_tokens_predict_enorm);
        layer_ws.hnorm               = mayCreateLayerNormWeights(layer_weights, W::multi_tokens_predict_hnorm);
        layer_ws.eh_proj             = mayCreateDenseWeights(layer_weights, W::multi_tokens_predict_eh_proj);
        layer_ws.mtp_final_layernorm = mayCreateLayerNormWeights(
            layer_weights, W::multi_tokens_predict_final_ln_gamma, W::multi_tokens_predict_final_ln_beta);
        if (layer_ws.mtp_final_layernorm != nullptr) {
            gpt_weights.final_layernorm = layer_ws.mtp_final_layernorm;
        }

        // eagle3
        layer_ws.eagle3_fc_norm    = mayCreateLayerNormWeights(layer_weights, W::eagle3_fc_norm_gamma);
        layer_ws.eagle3_input_norm = mayCreateLayerNormWeights(layer_weights, W::eagle3_input_norm_gamma);
        layer_ws.eagle3_fc_proj    = mayCreateDenseWeights(layer_weights, W::eagle3_fc_proj);

        gpt_weights.layers.emplace_back(std::move(layer_ws));
    }
    return std::make_unique<rtp_llm::Weights>(gpt_weights);
}

}  // namespace rtp_llm
