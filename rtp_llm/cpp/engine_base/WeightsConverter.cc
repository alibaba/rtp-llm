#include "rtp_llm/cpp/engine_base/WeightsConverter.h"
#include "rtp_llm/cpp/pybind/PyUtils.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/BufferHelper.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include <memory>
using namespace std;

namespace rtp_llm {

rtp_llm::ConstBufferPtr WeightsConverter::CopyTensorToBufferPtr(const torch::Tensor& tensor) {
    auto buffer = torchTensor2Buffer(tensor);
    if (need_copy_) {
        auto new_buffer =
            device_->allocateBuffer(BufferParams(buffer->type(), buffer->shape(), AllocationType::DEVICE));

        device_->noBlockCopy(CopyParams{*new_buffer, *buffer});
        return new_buffer;
    } else {
        return buffer;
    }
}

rtp_llm::ConstBufferPtr WeightsConverter::mayFindBuffer(const ConstBufferPtrMap& map, const std::string& key) {
    auto it = map.find(key);
    if (it != map.end()) {
        return it->second;
    }
    return nullptr;
}

rtp_llm::LayerNormWeightsPtr WeightsConverter::mayCreateLayerNormWeights(const ConstBufferPtrMap& map,
                                                                         const std::string&       gamma_key,
                                                                         const std::string&       beta_key,
                                                                         const std::string&       scale_key,
                                                                         const std::string& scale_reciprocal_key) {
    if (map.count(gamma_key) > 0) {
        const auto layer_norm_weights               = new LayerNormWeights();
        layer_norm_weights->gamma                   = mayFindBuffer(map, gamma_key);
        layer_norm_weights->beta                    = mayFindBuffer(map, beta_key);
        layer_norm_weights->static_scale            = mayFindBuffer(map, scale_key);
        layer_norm_weights->static_scale_reciprocal = mayFindBuffer(map, scale_reciprocal_key);
        return unique_ptr<const LayerNormWeights>(layer_norm_weights);
    }
    return nullptr;
}

rtp_llm::DenseWeightsPtr WeightsConverter::mayCreateDenseWeights(const ConstBufferPtrMap& map,
                                                                 const std::string&       kernel_key,
                                                                 const std::string&       bias_key,
                                                                 const std::string&       scales_key,
                                                                 const std::string&       zeros_key) {
    if (map.count(kernel_key) <= 0) {
        return nullptr;
    }

    const auto dense_weights = new DenseWeights();
    if (!bias_key.empty()) {
        dense_weights->bias = mayFindBuffer(map, bias_key);
    }
    auto      scales = map.count(scales_key) > 0 ? mayFindBuffer(map, scales_key) : nullptr;
    BufferPtr scalesBuffer;
    if (scales) {
        scalesBuffer = BufferPtr(new Buffer(scales->where(), scales->type(), scales->shape(), scales->data()));
    }
    auto      zeros = map.count(zeros_key) > 0 ? mayFindBuffer(map, zeros_key) : nullptr;
    BufferPtr zerosBuffer;
    if (zeros) {
        zerosBuffer = BufferPtr(new Buffer(zeros->where(), zeros->type(), zeros->shape(), zeros->data()));
    } else if (scales) {
        zerosBuffer = BufferPtr(new Buffer(scales->where(), scales->type(), {0}, nullptr));
    }
    if (map.count(scales_key) == 0) {
        dense_weights->kernel = mayFindBuffer(map, kernel_key);
    } else {
        auto kernel = mayFindBuffer(map, kernel_key);
        auto shape  = kernel->shape();
        auto dtype  = kernel->type();
        if (quant_algo_.isFp8() && scales) {
            dtype = DataType::TYPE_FP8_E4M3;
        } else if (quant_algo_.isQuant() && scales) {
            RTP_LLM_LOG_DEBUG(
                "load weight_only qbuffer weight [%s] scale [%s]", kernel_key.c_str(), scales_key.c_str());
            if (quant_algo_.getWeightBits() == 4) {
                dtype = DataType::TYPE_INT4X2;

#if USING_CK_INT4  // for composable kernel specific
                shape[kernel->dim() - 2] = shape[kernel->dim() - 2] * 2;
#else
                shape[kernel->dim() - 1] = shape[kernel->dim() - 1] * 2;
#endif
            }
        }
        dense_weights->kernel =
            ConstBufferPtr(new rtp_llm::QBuffer(BufferPtr(new Buffer(kernel->where(), dtype, shape, kernel->data())),
                                                std::move(scalesBuffer),
                                                std::move(zerosBuffer)));
        RTP_LLM_LOG_DEBUG("quant_method:%d, kernel_key:%s have scale use Qbuffer, kernel:%s",
                          quant_algo_.getQuantMethod(),
                          kernel_key.c_str(),
                          kernel->debugString().c_str());
    }

    return unique_ptr<DenseWeights>(dense_weights);
}

rtp_llm::FfnLayerWeights WeightsConverter::createFfnWeights(const ConstBufferPtrMap& map) {
    rtp_llm::FfnLayerWeights ffn_weights;

    ffn_weights.up_weight      = mayCreateDenseWeights(map, W::ffn_w3, W::ffn_b3, W::ffn_s3, W::ffn_z3);
    ffn_weights.gate_weight    = mayCreateDenseWeights(map, W::ffn_w1, W::ffn_b1, W::ffn_s1, W::ffn_z1);
    ffn_weights.down_weight    = mayCreateDenseWeights(map, W::ffn_w2, W::ffn_b2, W::ffn_s2, W::ffn_z2);
    ffn_weights.gate_up_weight = mayCreateDenseWeights(map, W::ffn_w13, W::ffn_b13, W::ffn_s13, W::ffn_z13);

    ffn_weights.moe_gating_weight       = mayCreateDenseWeights(map, W::moe_gate);
    ffn_weights.moe_gate_weight         = mayCreateDenseWeights(map, W::moe_w1, W::moe_b1, W::moe_s1, W::moe_z1);
    ffn_weights.moe_down_weight         = mayCreateDenseWeights(map, W::moe_w2, W::moe_b2, W::moe_s2, W::moe_z2);
    ffn_weights.e_score_correction_bias = mayFindBuffer(map, W::moe_e_score_correction_b);

    ffn_weights.smoother_weight = mayCreateDenseWeights(map, W::ffn_smoother);
    ffn_weights.act_scale       = mayFindBuffer(map, W::ffn_act_s);

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
    ffn_weights.logic_expert_cnt = mayFindBuffer(map, W::logic_expert_cnt);
    ffn_weights.log2phy          = mayFindBuffer(map, W::log2phy);

    return ffn_weights;
}

rtp_llm::AttentionLayerWeights WeightsConverter::createAttentionWeights(const ConstBufferPtrMap& map) {
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

std::unique_ptr<ConstBufferPtrMaps>
WeightsConverter::convertLayerWeights(std::unique_ptr<TensorMaps> tensor_layer_weights) {
    ConstBufferPtrMaps layer_weights;
    for (auto& layer_weight : *tensor_layer_weights) {
        ConstBufferPtrMap weights;
        for (auto& it : layer_weight) {
            weights.emplace(it.first, CopyTensorToBufferPtr(it.second));
        }
        layer_weights.emplace_back(std::move(weights));
    }
    return std::make_unique<ConstBufferPtrMaps>(std::move(layer_weights));
}

std::unique_ptr<ConstBufferPtrMap>
WeightsConverter::convertGlobalWeight(std::unique_ptr<TensorMap> tensor_global_weight) {
    ConstBufferPtrMap global_weights;
    for (auto& it : *tensor_global_weight) {
        global_weights.emplace(it.first, CopyTensorToBufferPtr(it.second));
    }
    return std::make_unique<ConstBufferPtrMap>(std::move(global_weights));
}

std::unique_ptr<rtp_llm::Weights> WeightsConverter::createGptWeights(py::object layer_weights,
                                                                     py::object global_weight) {
    return std::move(
        createGptWeights(std::move(convertLayerWeights(layer_weights)), std::move(convertGlobalWeight(global_weight))));
}

std::unique_ptr<rtp_llm::Weights> WeightsConverter::createGptWeights(std::unique_ptr<TensorMaps> layer_weights,
                                                                     std::unique_ptr<TensorMap>  global_weight) {
    return std::move(createGptWeights(std::move(convertLayerWeights(std::move(layer_weights))),
                                      std::move(convertGlobalWeight(std::move(global_weight)))));
}

std::unique_ptr<rtp_llm::Weights> WeightsConverter::createGptWeights(std::unique_ptr<ConstBufferPtrMaps> layer_weights,
                                                                     std::unique_ptr<ConstBufferPtrMap> global_weight) {
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
    gpt_weights.d2t_map            = mayFindBuffer(*global_weight, W::multi_tokens_predict_d2t_map);
    gpt_weights.t2d_map            = mayFindBuffer(*global_weight, W::multi_tokens_predict_t2d_map);

    for (auto& layer_weights : layers_weights) {
        rtp_llm::LayerWeights layer_ws;
        layer_ws.pre_layernorm =
            mayCreateLayerNormWeights(layer_weights, W::pre_ln_gamma, W::pre_ln_beta, W::pre_ln_s, W::pre_ln_sr);

        layer_ws.post_ffn_layernorm = mayCreateLayerNormWeights(
            layer_weights, W::post_ffn_ln_gamma, W::post_ffn_ln_beta, W::post_ffn_ln_s, W::post_ffn_ln_sr);

        layer_ws.post_layernorm =
            mayCreateLayerNormWeights(layer_weights, W::post_ln_gamma, W::post_ln_beta, W::post_ln_s, W::post_ln_sr);

        layer_ws.self_attention_weights = createAttentionWeights(layer_weights);

        layer_ws.self_attention_weights.rope_cos_sin_cache = mayFindBuffer(*global_weight, W::rope_cos_sin_cache);
        if (layer_ws.self_attention_weights.rope_cos_sin_cache) {
            RTP_LLM_CHECK_WITH_INFO(layer_ws.self_attention_weights.rope_cos_sin_cache->type() == DataType::TYPE_FP32,
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

/////////////////////////////////deprected///////////////////////////

std::unique_ptr<ConstBufferPtrMaps> WeightsConverter::convertLayerWeights_(py::object py_layer_weights) {
    return convertLayerWeights(std::move(convertLayerWeights(py_layer_weights)));
}

std::unique_ptr<ConstBufferPtrMap> WeightsConverter::convertGlobalWeight_(py::object py_global_weight) {
    return convertGlobalWeight(std::move(convertGlobalWeight(py_global_weight)));
}

}  // namespace rtp_llm
