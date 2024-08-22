#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/models/W.h"
#include "src/fastertransformer/utils/py_utils/pybind_utils.h"
#include <memory>
#if defined(__aarch64__)
#include "src/fastertransformer/devices/arm_impl/gemm_opt/ArmGemmKernel.h"
#endif
using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

ft::ConstBufferPtr WeightsConverter::CopyTensorToBufferPtr(const torch::Tensor& tensor) {
    auto buffer = torchTensor2Buffer(tensor);
    if (need_copy_) {
        auto new_buffer = device_->allocateBuffer({buffer->type(),
                                                   buffer->shape(),
                                                   AllocationType::HOST});
        device_->copy({*new_buffer, *buffer});
        return new_buffer;
    } else {
        return buffer;
    }
}

ft::ConstBufferPtr
WeightsConverter::mayFindBuffer(const ConstBufferPtrMap& map,
                                const std::string& key)
{
    auto it = map.find(key);
    if (it != map.end()) {
#if defined(__aarch64__)
        return prepareGemmWeight(key, it->second);
#endif
        return it->second;
    }
    return nullptr;
}

ft::LayerNormWeightsPtr
WeightsConverter::mayCreateLayerNormWeights(const ConstBufferPtrMap& map,
                                            const std::string& gamma_key,
                                            const std::string& beta_key,
                                            const std::string& scale_key,
                                            const std::string& scale_reciprocal_key)
{
    if (map.count(gamma_key) > 0) {
        const auto layer_norm_weights = new LayerNormWeights();
        layer_norm_weights->gamma     = mayFindBuffer(map, gamma_key);
        layer_norm_weights->beta      = mayFindBuffer(map, beta_key);
        layer_norm_weights->static_scale = mayFindBuffer(map, scale_key);
        layer_norm_weights->static_scale_reciprocal = mayFindBuffer(map, scale_reciprocal_key);
        return unique_ptr<const LayerNormWeights>(layer_norm_weights);
    }
    return nullptr;
}

ft::DenseWeightsPtr
WeightsConverter::mayCreateDenseWeights(const ConstBufferPtrMap& map,
                                        const std::string& kernel_key,
                                        const std::string& bias_key,
                                        const std::string& scales_key,
                                        const std::string& zeros_key)
{
    if (map.count(kernel_key) > 0) {
        const auto dense_weights = new DenseWeights();
        if (!bias_key.empty()) {
            dense_weights->bias = mayFindBuffer(map, bias_key);
        }
        if (map.count(scales_key) <= 0) {
            dense_weights->kernel = mayFindBuffer(map, kernel_key);
        } else if (map.count(zeros_key) <= 0) {
            auto kernel = mayFindBuffer(map, kernel_key);
            auto scales = mayFindBuffer(map, scales_key);
            // construct qbuffer need kernel and scales has no ref.
            FT_LOG_DEBUG("load qbuffer weight [%s] ", scales_key.c_str());
            dense_weights->kernel = ConstBufferPtr(
                new ft::QBuffer(BufferPtr(new Buffer(kernel->where(),
                                                     kernel->type(),
                                                     kernel->shape(),
                                                     kernel->data())),
                                BufferPtr(new Buffer(scales->where(),
                                                     scales->type(),
                                                     scales->shape(),
                                                     scales->data())),
                                BufferPtr(new Buffer(scales->where(),
                                                     scales->type(),
                                                     {0},
                                                     nullptr))));
        } else {
            auto kernel = mayFindBuffer(map, kernel_key);
            auto scales = mayFindBuffer(map, scales_key);
            auto zeros  = mayFindBuffer(map, zeros_key);
            FT_LOG_DEBUG("load qbuffer weight [%s] ", zeros_key.c_str());
            auto dtype_ = kernel->type();
            auto shape_ = kernel->shape();
            if (quant_algo_.getWeightBits() == 4) {
                dtype_ = DataType::TYPE_INT4X2;
                shape_[kernel->dim()-1] = shape_[kernel->dim()-1] * 2;
            }
            dense_weights->kernel = ConstBufferPtr(
                new ft::QBuffer(BufferPtr(new Buffer(kernel->where(),
                                                     dtype_,
                                                     shape_,
                                                     kernel->data())),
                                BufferPtr(new Buffer(scales->where(),
                                                     scales->type(),
                                                     scales->shape(),
                                                     scales->data())),
                                BufferPtr(new Buffer(zeros->where(),
                                                     zeros->type(),
                                                     zeros->shape(),
                                                     zeros->data()))));
            FT_LOG_DEBUG("weight shape is [%s] ", autil::StringUtil::toString(dense_weights->kernel->shape()).c_str());
        }
        return unique_ptr<const DenseWeights>(dense_weights);

    }
    return nullptr;
}

ft::FfnLayerWeights
WeightsConverter::createFfnWeights(const ConstBufferPtrMap& map) {
    ft::FfnLayerWeights ffn_weights;

    ffn_weights.up_weight   = mayCreateDenseWeights(map, W::ffn_w3, W::ffn_b3, W::ffn_s3, W::ffn_z3);
    ffn_weights.gate_weight = mayCreateDenseWeights(map, W::ffn_w1, W::ffn_b1, W::ffn_s1, W::ffn_z1);
    ffn_weights.down_weight = mayCreateDenseWeights(map, W::ffn_w2, W::ffn_b2, W::ffn_s2, W::ffn_z2);

    ffn_weights.moe_gating_weight = mayCreateDenseWeights(map, W::moe_gate);
    ffn_weights.moe_up_weight     = mayCreateDenseWeights(map, W::moe_w3, W::moe_b3, W::moe_s3, W::moe_z3);
    ffn_weights.moe_gate_weight   = mayCreateDenseWeights(map, W::moe_w1, W::moe_b1, W::moe_s1, W::moe_z1);
    ffn_weights.moe_down_weight   = mayCreateDenseWeights(map, W::moe_w2, W::moe_b2, W::moe_s2, W::moe_z2);

    ffn_weights.smoother_weight = mayCreateDenseWeights(map, W::ffn_smoother);
    ffn_weights.act_scale       = mayFindBuffer(map, W::ffn_act_s);

    ffn_weights.intermediate_weight2_static_scale_weight = mayCreateDenseWeights(map, W::ffn_intermediate_weight2_s);
    ffn_weights.intermediate_weight2_static_scale_reciprocal_weight = mayCreateDenseWeights(map, W::ffn_intermediate_weight2_sr);

    // for qwen moe
    if (ffn_weights.moe_gating_weight) {
        // this moe layer has a parallel dense ffn layer as shared expert.
        if (ffn_weights.up_weight) {
            ffn_weights.shared_expert = make_shared<ft::FfnLayerWeights>();
            ffn_weights.shared_expert->up_weight = move(ffn_weights.up_weight);
            ffn_weights.shared_expert->gate_weight = move(ffn_weights.gate_weight);
            ffn_weights.shared_expert->down_weight = move(ffn_weights.down_weight);

            // for qwen moe
            ffn_weights.shared_expert_gate = mayCreateDenseWeights(map, W::shared_expert_gate_w);
        }
    }

    return ffn_weights;
}

ft::AttentionLayerWeights
WeightsConverter::createAttentionWeights(const ConstBufferPtrMap& map) {
    ft::AttentionLayerWeights attention_weights;
    attention_weights.pre_attention_layernorm = mayCreateLayerNormWeights(map,
                                                                          W::pre_attn_ln_gamma,
                                                                          W::pre_attn_ln_beta);

    attention_weights.qkv_weight = mayCreateDenseWeights(map,
                                                         W::attn_qkv_w,
                                                         W::attn_qkv_b,
                                                         W::attn_qkv_s,
                                                         W::attn_qkv_z);

    attention_weights.q_norm_weight = mayCreateLayerNormWeights(map, W::q_ln_gamma, W::q_ln_beta);
    attention_weights.k_norm_weight = mayCreateLayerNormWeights(map, W::k_ln_gamma, W::k_ln_beta);

    attention_weights.attention_layernorm = mayCreateLayerNormWeights(map,
                                                                      W::attn_ln_gamma,
                                                                      W::attn_ln_beta);
    attention_weights.output_weight = mayCreateDenseWeights(map,
                                                            W::attn_o_w,
                                                            W::attn_o_b,
                                                            W::attn_o_s,
                                                            W::attn_o_z);

    attention_weights.shift_weight = mayCreateDenseWeights(map,
                                                           W::attn_o_shift);

    attention_weights.smoother_weight = mayCreateDenseWeights(map,
                                                              W::attn_o_smoother);

    attention_weights.static_quant_weight = mayCreateDenseWeights(map, W::attention_output_s);
    attention_weights.static_scale_reciprocal_weight = mayCreateDenseWeights(map, W::attention_output_sr);

    return attention_weights;
}

std::unique_ptr<TensorMaps>
WeightsConverter::convertLayerWeights(py::object py_layer_weights) {
    TensorMaps tensor_layer_weights;
    auto layers_weights_vec = ft::convertPyObjectToVec(py_layer_weights);
    for (auto& layer_weights : layers_weights_vec) {
        TensorMap weights;
        for (auto& it : convertPyObjectToDict(layer_weights)) {
            weights.emplace(it.first, ft::convertPyObjectToTensor(it.second));
        }
        tensor_layer_weights.emplace_back(std::move(weights));
    }
    return std::make_unique<TensorMaps>(std::move(tensor_layer_weights));
}

std::unique_ptr<TensorMap>
WeightsConverter::convertGlobalWeight(py::object py_global_weight) {
    TensorMap global_weights;
    auto global_weights_dict = ft::convertPyObjectToDict(py_global_weight);
    for (auto& it : global_weights_dict) {
        global_weights.emplace(it.first, ft::convertPyObjectToTensor(it.second));
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

std::unique_ptr<ft::Weights>
WeightsConverter::createGptWeights(py::object layer_weights,
                                   py::object global_weight)
{
    return std::move(createGptWeights(std::move(convertLayerWeights(layer_weights)),
                                      std::move(convertGlobalWeight(global_weight))));
}

std::unique_ptr<ft::Weights>
WeightsConverter::createGptWeights(std::unique_ptr<TensorMaps> layer_weights,
                                   std::unique_ptr<TensorMap>  global_weight)
{
    return std::move(createGptWeights(std::move(convertLayerWeights(std::move(layer_weights))),
                                      std::move(convertGlobalWeight(std::move(global_weight)))));
}

std::unique_ptr<ft::Weights>
WeightsConverter::createGptWeights(std::unique_ptr<ConstBufferPtrMaps> layer_weights,
                                   std::unique_ptr<ConstBufferPtrMap>  global_weight)
{
    auto        layers_weights = *layer_weights;
    ft::Weights gpt_weights;
    // make global weight
    gpt_weights.embedding = mayCreateDenseWeights(*global_weight,
                                                   W::embedding);
    gpt_weights.prefix_encoder_embedding = mayCreateDenseWeights(*global_weight,
                                                                  W::prefix_w);
    gpt_weights.pre_decoder_layernorm = mayCreateLayerNormWeights(*global_weight,
                                                                   W::pre_decoder_ln_gamma,
                                                                   W::pre_decoder_ln_beta,
                                                                   W::pre_decoder_ln_s,
                                                                   W::pre_decoder_ln_static_sr);
    gpt_weights.position_encoding = mayCreateDenseWeights(*global_weight,
                                                           W::wpe);
    gpt_weights.token_type_embedding = mayCreateDenseWeights(*global_weight,
                                                              W::token_type_embedding);
    gpt_weights.final_layernorm = mayCreateLayerNormWeights(*global_weight,
                                                             W::final_ln_gamma,
                                                             W::final_ln_beta);
    gpt_weights.lm_head = mayCreateDenseWeights(*global_weight,
                                                 W::lm_head);

    gpt_weights.linear_bias_slopes = mayCreateDenseWeights(*global_weight, W::linear_bias_slopes);

    for (auto& layer_weights : layers_weights) {
        ft::LayerWeights layer_ws;
        layer_ws.pre_attention_smoother_weight = mayCreateDenseWeights(layer_weights, W::attn_i_smoother);
        layer_ws.pre_layernorm = mayCreateLayerNormWeights(layer_weights,
                                                               W::pre_ln_gamma,
                                                               W::pre_ln_beta,
                                                               W::pre_ln_s,
                                                               W::pre_ln_sr);

        layer_ws.post_ffn_layernorm = mayCreateLayerNormWeights(layer_weights,
                                                                    W::post_ffn_ln_gamma,
                                                                    W::post_ffn_ln_beta,
                                                                    W::post_ffn_ln_s,
                                                                    W::post_ffn_ln_sr);

        layer_ws.post_layernorm = mayCreateLayerNormWeights(layer_weights,
                                                                W::post_ln_gamma,
                                                                W::post_ln_beta,
                                                                W::post_ln_s,
                                                                W::post_ln_sr);

        layer_ws.post_layernorm_2 = mayCreateLayerNormWeights(layer_weights, W::post_ln_2_gamma, W::post_ln_2_beta);

        layer_ws.self_attention_weights = createAttentionWeights(layer_weights);
        layer_ws.ffn_weights = createFfnWeights(layer_weights);
        gpt_weights.layers.emplace_back(std::move(layer_ws));
    }
    return std::make_unique<ft::Weights>(gpt_weights);
}




/////////////////////////////////deprected///////////////////////////


std::unique_ptr<ConstBufferPtrMaps> WeightsConverter::convertLayerWeights_(py::object py_layer_weights) {
    return convertLayerWeights(std::move(convertLayerWeights(py_layer_weights)));
}

std::unique_ptr<ConstBufferPtrMap>  WeightsConverter::convertGlobalWeight_(py::object py_global_weight) {
    return convertGlobalWeight(std::move(convertGlobalWeight(py_global_weight)));
}

std::tuple<ft::GptInitParameter, std::unique_ptr<ft::Weights>> prepareEngineInitParams(py::object model, bool sp_model) {
    if (sp_model) {
        model = model.attr("model");
    }
     const ft::GptInitParameter& gpt_init_params = model.attr("config").attr("gpt_init_params").cast<ft::GptInitParameter>();
    py::object                  py_layers_weights = model.attr("weight").attr("weights");
    py::object                  py_global_weights = model.attr("weight").attr("global_weights");

    auto convert = rtp_llm::WeightsConverter(false, gpt_init_params.quant_algo_);
    auto gpt_weight = convert.createGptWeights(py_layers_weights, py_global_weights);

    return {gpt_init_params, std::move(gpt_weight)};
}


}  // namespace rtp_llm
