#include "maga_transformer/cpp/dataclass/EngineInitParameter.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/core/BufferHelper.h"
#include "src/fastertransformer/core/torch_utils/BufferTorchUtils.h"
#include "src/fastertransformer/models/W.h"
#include "src/fastertransformer/utils/pybind_utils.h"
#include <memory>

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

EngineInitParams::EngineInitParams(
    const ft::GptInitParameter&                                             gpt_init_parameter,
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layers_weights,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              global_weights):
    gpt_init_parameter(gpt_init_parameter),
    gpt_weights(std::move(WeightsConverter::createGptWeights(layers_weights, global_weights))),
    layers_weights(layers_weights),
    global_weights(global_weights) {}

ConstBufferPtr WeightsConverter::mayFindTensor2Buffer(unordered_map<string, th::Tensor> tensor_map, const string& key) {
    if (tensor_map.count(key) > 0) {
        auto buffer = torchTensor2Buffer(tensor_map.at(key));
        if (need_copy_) {
            auto new_buffer = device_->allocateBuffer({buffer->type(), buffer->shape(), AllocationType::DEVICE});
            device_->copy({*new_buffer, *buffer});
            return new_buffer;
        } else {
            return buffer;
        }
    }
    return nullptr;
}

LayerNormWeightsPtr WeightsConverter::mayCreateLayerNormWeights(unordered_map<string, th::Tensor> tensor_map,
                                                                const string&                     gamma_key,
                                                                const string&                     beta_key) {
    if (tensor_map.count(gamma_key) > 0) {
        const auto layer_norm_weights = new LayerNormWeights();
        layer_norm_weights->gamma     = mayFindTensor2Buffer(tensor_map, gamma_key);
        layer_norm_weights->beta      = mayFindTensor2Buffer(tensor_map, beta_key);
        return unique_ptr<const LayerNormWeights>(layer_norm_weights);
    }
    return nullptr;
}

DenseWeightsPtr WeightsConverter::mayCreateDenseWeights(unordered_map<string, th::Tensor> tensor_map,
                                                        const string&                     kernel_key,
                                                        const string&                     bias_key) {
    if (tensor_map.count(kernel_key) > 0) {
        const auto dense_weights = new DenseWeights();
        dense_weights->kernel    = mayFindTensor2Buffer(tensor_map, kernel_key);
        if (!bias_key.empty()) {
            dense_weights->bias = mayFindTensor2Buffer(tensor_map, bias_key);
        }
        return unique_ptr<const DenseWeights>(dense_weights);
    }
    return nullptr;
}

DenseWeightsPtr WeightsConverter::createDenseWeights(unordered_map<string, th::Tensor> tensor_map,
                                                     const string&                     kernel_key,
                                                     const string&                     bias_key) {
    auto weights = mayCreateDenseWeights(tensor_map, kernel_key, bias_key);
    // assert(weights);
    return std::move(weights);
}

unique_ptr<const Weights> WeightsConverter::convertPythonWeights(const PyModelWeights& py_weights) {
    const auto weights = new Weights();

    const auto global_weights = py_weights.model_global_weights_;
    weights->embedding        = createDenseWeights(global_weights, W::embedding);

    weights->pre_decoder_layernorm =
        mayCreateLayerNormWeights(global_weights, W::pre_decoder_ln_gamma, W::pre_decoder_ln_beta);
    weights->final_layernorm = mayCreateLayerNormWeights(global_weights, W::final_ln_gamma, W::final_ln_beta);

    const auto layer_num = py_weights.layer_weights_.size();
    weights->layers.reserve(layer_num);
    for (size_t layer_id = 0; layer_id < layer_num; layer_id++) {
        const auto   py_layer_weights = py_weights.layer_weights_[layer_id];
        LayerWeights layer_weights;

        layer_weights.pre_layernorm = mayCreateLayerNormWeights(py_layer_weights, W::pre_ln_gamma, W::pre_ln_beta);

        auto& attention_weights = layer_weights.self_attention_weights;
        attention_weights.pre_attention_layernorm =
            mayCreateLayerNormWeights(py_layer_weights, W::pre_attn_ln_gamma, W::pre_attn_ln_beta);
        attention_weights.qkv_weight = createDenseWeights(py_layer_weights, W::attn_qkv_w, W::attn_qkv_b);
        attention_weights.attention_layernorm =
            mayCreateLayerNormWeights(py_layer_weights, W::attn_ln_gamma, W::attn_ln_beta);
        attention_weights.output_weight = createDenseWeights(py_layer_weights, W::attn_o_w, W::attn_o_b);

        layer_weights.post_layernorm = mayCreateLayerNormWeights(py_layer_weights, W::post_ln_gamma, W::post_ln_beta);

        auto& ffn_weights             = layer_weights.ffn_weights;
        ffn_weights.gate_weight       = createDenseWeights(py_layer_weights, W::ffn_w1, W::ffn_b1);
        ffn_weights.down_weight       = createDenseWeights(py_layer_weights, W::ffn_w2, W::ffn_b2);
        ffn_weights.up_weight         = createDenseWeights(py_layer_weights, W::ffn_w3, W::ffn_b3);
        ffn_weights.dense_layernorm   = mayCreateLayerNormWeights(py_layer_weights, W::ffn_ln_gamma, W::ffn_ln_beta);
        ffn_weights.moe_gating_weight = createDenseWeights(py_layer_weights, W::moe_gate);

        weights->layers.push_back(std::move(layer_weights));
    }

    weights->final_layernorm = mayCreateLayerNormWeights(global_weights, W::final_ln_gamma, W::final_ln_beta);
    weights->lm_head         = createDenseWeights(global_weights, W::lm_head);

    return unique_ptr<const Weights>(weights);
}

std::unordered_map<std::string, ft::ConstBufferPtr>
WeightsConverter::convertPyWeightsMap(py::object py_global_weights) {
    std::unordered_map<std::string, ft::ConstBufferPtr> global_weights;
    auto global_weights_cc = fastertransformer::convertPyObjectToDict(py_global_weights);
    for (auto& it : global_weights_cc) {
        global_weights.emplace(it.first, ft::torchTensor2Buffer(it.second));
    }
    return global_weights;
}

std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>
WeightsConverter::convertPyWeightsMapVec(py::object py_layers_weights) {
    std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>> layers_weights;
    auto layers_weights_cc = fastertransformer::convertPyobjectToVectorDict(py_layers_weights);
    for (auto& layer_weights : layers_weights_cc) {
        std::unordered_map<std::string, ft::ConstBufferPtr> weights;
        for (auto& it : layer_weights) {
            weights.emplace(it.first, ft::torchTensor2Buffer(it.second));
        }
        layers_weights.emplace_back(std::move(weights));
    }
    return layers_weights;
}

ft::Weights WeightsConverter::createGptWeights(
    const std::vector<std::unordered_map<std::string, ft::ConstBufferPtr>>& layers_weights_,
    const std::unordered_map<std::string, ft::ConstBufferPtr>&              weights_) {
    auto        layers_weights = layers_weights_;
    auto        weights        = weights_;
    ft::Weights gpt_weights;
    gpt_weights.embedding = make_unique<const ft::DenseWeights>(weights[W::embedding]);
    if (weights[W::prefix_w]) {
        gpt_weights.prefix_encoder_embedding = make_unique<const ft::DenseWeights>(weights[W::prefix_w]);
    }
    if (weights[W::pre_decoder_ln_gamma]) {
        gpt_weights.pre_decoder_layernorm =
            make_unique<const ft::LayerNormWeights>(weights[W::pre_decoder_ln_gamma], weights[W::pre_decoder_ln_beta]);
    }
    if (weights[W::wpe]) {
        gpt_weights.position_encoding = make_unique<const ft::DenseWeights>(weights[W::wpe]);
    }
    if (weights[W::token_type_embedding]) {
        gpt_weights.token_type_embedding = make_unique<const ft::DenseWeights>(weights[W::token_type_embedding]);
    }
    if (weights[W::final_ln_gamma]) {
        gpt_weights.final_layernorm =
            make_unique<const ft::LayerNormWeights>(weights[W::final_ln_gamma], weights[W::final_ln_beta]);
    }
    if (weights[W::lm_head]) {
        gpt_weights.lm_head = make_unique<const ft::DenseWeights>(weights[W::lm_head]);
    }
    for (auto& layer_weights : layers_weights) {
        ft::LayerWeights layer_ws;
        if (layer_weights[W::pre_ln_gamma]) {
            layer_ws.pre_layernorm =
                make_unique<const ft::LayerNormWeights>(layer_weights[W::pre_ln_gamma], layer_weights[W::pre_ln_beta]);
        }
        if (layer_weights[W::post_ffn_ln_gamma]) {
            layer_ws.post_ffn_layernorm = make_unique<const ft::LayerNormWeights>(layer_weights[W::post_ffn_ln_gamma],
                                                                                  layer_weights[W::post_ffn_ln_beta]);
        }
        if (layer_weights[W::post_ln_gamma]) {
            layer_ws.post_layernorm = make_unique<const ft::LayerNormWeights>(layer_weights[W::post_ln_gamma],
                                                                              layer_weights[W::post_ln_beta]);
        }

        auto& attention_weights = layer_ws.self_attention_weights;
        if (layer_weights[W::pre_attn_ln_gamma]) {
            attention_weights.pre_attention_layernorm = make_unique<const ft::LayerNormWeights>(
                layer_weights[W::pre_attn_ln_gamma], layer_weights[W::pre_attn_ln_beta]);
        }
        auto qkv_weights_ori       = layer_weights[W::attn_qkv_w]->view(0, layer_weights[W::attn_qkv_w]->shape()[0]);
        ConstBufferPtr qkv_weights = make_unique<const ft::Buffer>(
            qkv_weights_ori.where(),
            qkv_weights_ori.type(),
            vector<size_t>{qkv_weights_ori.shape()[0], qkv_weights_ori.size() / qkv_weights_ori.shape()[0]},
            qkv_weights_ori.data());
        ConstBufferPtr qkv_weights_bias;
        if (layer_weights[W::attn_qkv_b]) {
            auto qkv_weights_bias_ori = layer_weights[W::attn_qkv_b]->view(0, layer_weights[W::attn_qkv_b]->shape()[0]);
            qkv_weights_bias          = make_unique<const ft::Buffer>(qkv_weights_bias_ori.where(),
                                                             qkv_weights_bias_ori.type(),
                                                             vector<size_t>{qkv_weights_bias_ori.size()},
                                                             qkv_weights_bias_ori.data());
        }
        attention_weights.qkv_weight = make_unique<const ft::DenseWeights>(qkv_weights, layer_weights[W::attn_qkv_b]);
        if (layer_weights[W::attn_ln_gamma]) {
            attention_weights.attention_layernorm = make_unique<const ft::LayerNormWeights>(
                layer_weights[W::attn_ln_gamma], layer_weights[W::attn_ln_beta]);
        }
        attention_weights.output_weight =
            make_unique<const ft::DenseWeights>(layer_weights[W::attn_o_w], layer_weights[W::attn_o_b]);

        auto& ffn_weights     = layer_ws.ffn_weights;
        ffn_weights.up_weight = make_unique<const ft::DenseWeights>(layer_weights[W::ffn_w3], layer_weights[W::ffn_b3]);
        ffn_weights.gate_weight =
            make_unique<const ft::DenseWeights>(layer_weights[W::ffn_w1], layer_weights[W::ffn_b1]);
        ffn_weights.down_weight =
            make_unique<const ft::DenseWeights>(layer_weights[W::ffn_w2], layer_weights[W::ffn_b2]);
        if (layer_weights[W::ffn_ln_gamma]) {
            ffn_weights.dense_layernorm =
                make_unique<const ft::LayerNormWeights>(layer_weights[W::ffn_ln_gamma], layer_weights[W::ffn_ln_beta]);
        }

        gpt_weights.layers.emplace_back(std::move(layer_ws));
    }
    return std::move(gpt_weights);
}

}  // namespace rtp_llm
