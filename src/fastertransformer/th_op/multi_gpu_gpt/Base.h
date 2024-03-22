#pragma once

#include <string>
#include <unordered_map>
#include "src/fastertransformer/models/W.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLoRALayerWeight.h"
#include "src/fastertransformer/th_op/th_utils.h"

namespace th = torch;
namespace ft = fastertransformer;
namespace W = fastertransformer::W;

namespace torch_ext {

inline at::ScalarType getScalarType(const std::string& data_type) {
    at::ScalarType scalar_type;
    if (data_type == "fp16") {
        scalar_type = at::ScalarType::Half;

    } else if (data_type == "bf16") {
        scalar_type = at::ScalarType::BFloat16;

    } else if (data_type == "fp32") {
        scalar_type = at::ScalarType::Float;

    } else {
        FT_LOG_ERROR("datatype not implemented %s", data_type.c_str());
    }
    return scalar_type;
}

template<typename T>
T *maybe_get(const std::unordered_map<std::string, th::Tensor> &m, const std::string &name) {
    auto it = m.find(name);
    if (it == m.end()) {
        return nullptr;
    }

    return get_ptr<T>(it->second);
}

template<typename T>
std::vector<ft::ParallelGptDecoderLayerWeight<T>*>
loadWeights(int                                                             pp_size,
            size_t                                                          pp_rank,
            size_t                                                          num_layers,
            c10::intrusive_ptr<QuantAlgo>                                   quant_algo,
            const std::vector<std::unordered_map<std::string, th::Tensor>>& weights,
            const std::vector<ft::ParallelGptDecoderLoRALayerWeight<T>*>*   lora_weights     = nullptr)
{
    size_t local_num_layers = (num_layers + pp_size - 1) / pp_size;

    std::vector<ft::ParallelGptDecoderLayerWeight<T>*> gpt_layer_weights;
    for (size_t i = 0; i < (size_t)num_layers; i++) {
        gpt_layer_weights.push_back(new ft::ParallelGptDecoderLayerWeight<T>(quant_algo->int8_mode_));

        if (i / local_num_layers != pp_rank) {
            // Layer i is not in the current pipeline parallel group.
            continue;
        }

        // set lora
        if (lora_weights) {
            auto layer_lora_weights = (*lora_weights)[i];
            gpt_layer_weights[i]->self_attention_weights.query_weight.lora_weights = &layer_lora_weights->q_weights;
            gpt_layer_weights[i]->self_attention_weights.attention_output_weight.lora_weights =
                &layer_lora_weights->attention_output_weights;
            gpt_layer_weights[i]->ffn_weights.intermediate_weight.lora_weights =
                &layer_lora_weights->ffn_intermediate_weights;
            gpt_layer_weights[i]->ffn_weights.intermediate_weight2.lora_weights =
                &layer_lora_weights->ffn_intermediate_weights2;
            gpt_layer_weights[i]->ffn_weights.output_weight.lora_weights = &layer_lora_weights->ffn_output_weights;
        }

        gpt_layer_weights[i]->pre_layernorm_weights.gamma                           = maybe_get<T>(weights[i], W::pre_ln_gamma);
        gpt_layer_weights[i]->pre_layernorm_weights.beta                            = maybe_get<T>(weights[i], W::pre_ln_beta);
        gpt_layer_weights[i]->pre_attn_layernorm_weights.gamma                      = maybe_get<T>(weights[i], W::pre_attn_ln_gamma);
        gpt_layer_weights[i]->pre_attn_layernorm_weights.beta                       = maybe_get<T>(weights[i], W::pre_attn_ln_beta);
        gpt_layer_weights[i]->self_attention_weights.query_weight.bias              = maybe_get<T>(weights[i], W::attn_qkv_b);
        gpt_layer_weights[i]->self_attention_weights.attention_layernorm.gamma      = maybe_get<T>(weights[i], W::attn_ln_gamma);
        gpt_layer_weights[i]->self_attention_weights.attention_layernorm.beta       = maybe_get<T>(weights[i], W::attn_ln_beta);
        gpt_layer_weights[i]->self_attention_weights.attention_output_weight.bias   = maybe_get<T>(weights[i], W::attn_o_b);
        gpt_layer_weights[i]->self_attn_layernorm_weights.gamma                     = maybe_get<T>(weights[i], W::post_ln_gamma);
        gpt_layer_weights[i]->self_attn_layernorm_weights.beta                      = maybe_get<T>(weights[i], W::post_ln_beta);
        gpt_layer_weights[i]->ffn_weights.intermediate_weight.bias                  = maybe_get<T>(weights[i], W::ffn_b1);
        gpt_layer_weights[i]->ffn_weights.intermediate_weight2.bias                 = maybe_get<T>(weights[i], W::ffn_b3);
        gpt_layer_weights[i]->ffn_weights.dense_layernorm.gamma                     = maybe_get<T>(weights[i], W::ffn_ln_gamma);
        gpt_layer_weights[i]->ffn_weights.dense_layernorm.beta                      = maybe_get<T>(weights[i], W::ffn_ln_beta);
        gpt_layer_weights[i]->ffn_weights.output_weight.bias                        = maybe_get<T>(weights[i], W::ffn_b2);
        gpt_layer_weights[i]->ffn_weights.gating_weight.kernel                     = maybe_get<T>(weights[i], W::ffn_gate);
        gpt_layer_weights[i]->posf_ffn_layernorm_weights.gamma                     = maybe_get<T>(weights[i], W::post_ffn_ln_gamma);
        gpt_layer_weights[i]->posf_ffn_layernorm_weights.beta                      = maybe_get<T>(weights[i], W::post_ffn_ln_beta);

        if (quant_algo->int8_mode_) {
            gpt_layer_weights[i]->self_attention_weights.query_weight.int8_kernel            = maybe_get<int8_t>(weights[i], W::attn_qkv_w);
            gpt_layer_weights[i]->self_attention_weights.attention_output_weight.int8_kernel = maybe_get<int8_t>(weights[i], W::attn_o_w);
            gpt_layer_weights[i]->ffn_weights.intermediate_weight.int8_kernel                = get_ptr<int8_t>(weights[i].at(W::ffn_w1));
            gpt_layer_weights[i]->ffn_weights.intermediate_weight2.int8_kernel               = maybe_get<int8_t>(weights[i], W::ffn_w3);
            gpt_layer_weights[i]->ffn_weights.output_weight.int8_kernel                      = get_ptr<int8_t>(weights[i].at(W::ffn_w2));

            gpt_layer_weights[i]->self_attention_weights.query_weight.weight_only_quant_scale            = maybe_get<T>(weights[i], W::attn_qkv_s);
            gpt_layer_weights[i]->self_attention_weights.attention_output_weight.weight_only_quant_scale = maybe_get<T>(weights[i], W::attn_o_s);
            gpt_layer_weights[i]->ffn_weights.intermediate_weight.weight_only_quant_scale                = get_ptr<T>(weights[i].at(W::ffn_s1));
            gpt_layer_weights[i]->ffn_weights.intermediate_weight2.weight_only_quant_scale               = maybe_get<T>(weights[i], W::ffn_s3);
            gpt_layer_weights[i]->ffn_weights.output_weight.weight_only_quant_scale                      = get_ptr<T>(weights[i].at(W::ffn_s2));
        }
        if(quant_algo->int4_mode_){
            gpt_layer_weights[i]->self_attention_weights.query_weight.int4_kernel            = maybe_get<int8_t>(weights[i], W::attn_qkv_w);
            gpt_layer_weights[i]->self_attention_weights.attention_output_weight.int4_kernel = maybe_get<int8_t>(weights[i], W::attn_o_w);
            gpt_layer_weights[i]->ffn_weights.intermediate_weight.int4_kernel                = get_ptr<int8_t>(weights[i].at(W::ffn_w1));
            gpt_layer_weights[i]->ffn_weights.intermediate_weight2.int4_kernel               = maybe_get<int8_t>(weights[i], W::ffn_w3);
            gpt_layer_weights[i]->ffn_weights.output_weight.int4_kernel                      = get_ptr<int8_t>(weights[i].at(W::ffn_w2));

            gpt_layer_weights[i]->self_attention_weights.query_weight.weight_only_quant_scale            = maybe_get<T>(weights[i], W::attn_qkv_s);
            gpt_layer_weights[i]->self_attention_weights.attention_output_weight.weight_only_quant_scale = maybe_get<T>(weights[i], W::attn_o_s);
            gpt_layer_weights[i]->ffn_weights.intermediate_weight.weight_only_quant_scale                = get_ptr<T>(weights[i].at(W::ffn_s1));
            gpt_layer_weights[i]->ffn_weights.intermediate_weight2.weight_only_quant_scale               = maybe_get<T>(weights[i], W::ffn_s3);
            gpt_layer_weights[i]->ffn_weights.output_weight.weight_only_quant_scale                      = get_ptr<T>(weights[i].at(W::ffn_s2));

            gpt_layer_weights[i]->self_attention_weights.query_weight.int4_zeros            = maybe_get<int8_t>(weights[i], W::attn_qkv_z);
            gpt_layer_weights[i]->self_attention_weights.attention_output_weight.int4_zeros = maybe_get<int8_t>(weights[i], W::attn_o_z);
            gpt_layer_weights[i]->ffn_weights.intermediate_weight.int4_zeros                = maybe_get<int8_t>(weights[i], W::ffn_z1);
            gpt_layer_weights[i]->ffn_weights.intermediate_weight2.int4_zeros               = maybe_get<int8_t>(weights[i], W::ffn_z3);
            gpt_layer_weights[i]->ffn_weights.output_weight.int4_zeros                      = maybe_get<int8_t>(weights[i], W::ffn_z2);
        }
        else {
            gpt_layer_weights[i]->self_attention_weights.query_weight.kernel            = maybe_get<T>(weights[i], W::attn_qkv_w);
            gpt_layer_weights[i]->self_attention_weights.attention_output_weight.kernel = maybe_get<T>(weights[i], W::attn_o_w);
            gpt_layer_weights[i]->ffn_weights.intermediate_weight.kernel                = get_ptr<T>(weights[i].at(W::ffn_w1));
            gpt_layer_weights[i]->ffn_weights.intermediate_weight2.kernel               = maybe_get<T>(weights[i], W::ffn_w3);
            gpt_layer_weights[i]->ffn_weights.output_weight.kernel                      = get_ptr<T>(weights[i].at(W::ffn_w2));
        }
    }
    return gpt_layer_weights;
}

template<typename T>
void loadLoRAWeights(int                                                             num_layers,
                     int                                                             lora_id,
                     const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_a_weights,
                     const std::vector<std::unordered_map<std::string, th::Tensor>>& lora_b_weights,
                     std::vector<ft::ParallelGptDecoderLoRALayerWeight<T>*>&         gpt_lora_layer_weights)
{
    // lora_a: [m, r]
    // lora_b: [r, n]
    auto get_lora_rank = [](const std::unordered_map<std::string, th::Tensor>& lora_b_weights,
                            const std::string&                                 name) -> int {
        auto it = lora_b_weights.find(name);
        if (it == lora_b_weights.end()) {
            return 0;
        }
        return it->second.size(0);
    };
    constexpr int lora_num = 5;
    std::string lora_name_list[5] = {W::attn_qkv_w, W::attn_o_w, W::ffn_w1, W::ffn_w2, W::ffn_w3};
    for (size_t i = 0; i < num_layers; i++) {
        T *lora_a, *lora_b;
        int lora_rank = 0;
        for (int j = 0; j < lora_num; j++)
        {
            const std::string& name = lora_name_list[j];
            lora_a = maybe_get<T>(lora_a_weights[i], name);
            lora_b = maybe_get<T>(lora_b_weights[i], name);
            lora_rank = get_lora_rank(lora_b_weights[i], name);
            gpt_lora_layer_weights[i]->setLoRAWeight(name, lora_id, lora_a, lora_b, lora_rank);
        }
    }
}

template<typename T>
void removeLoRAWeights(int lora_id, std::vector<ft::ParallelGptDecoderLoRALayerWeight<T>*>& gpt_lora_layer_weights)
{
    for (auto &layer : gpt_lora_layer_weights) {
        layer->q_weights.removeLoRAWeight(lora_id);
        layer->attention_output_weights.removeLoRAWeight(lora_id);
        layer->ffn_intermediate_weights.removeLoRAWeight(lora_id);
        layer->ffn_intermediate_weights2.removeLoRAWeight(lora_id);
        layer->ffn_output_weights.removeLoRAWeight(lora_id);
    }
}


}
