#include "maga_transformer/cpp/utils/WeightsUtils.h"
#include "src/fastertransformer/devices/utils/BufferUtils.h"
#include "src/fastertransformer/devices/DeviceFactory.h"
#include "src/fastertransformer/models/W.h"

using namespace std;
using namespace fastertransformer;

namespace rtp_llm {

ConstBufferPtr WeightsConverter::mayFindTensor2Buffer(
    unordered_map<string, th::Tensor> tensor_map, const string& key)
{
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

LayerNormWeightsPtr WeightsConverter::mayCreateLayerNormWeights(
    unordered_map<string, th::Tensor> tensor_map, const string& gamma_key, const string& beta_key)
{
    if (tensor_map.count(gamma_key) > 0) {
        const auto layer_norm_weights = new LayerNormWeights();
        layer_norm_weights->gamma = mayFindTensor2Buffer(tensor_map, gamma_key);
        layer_norm_weights->beta = mayFindTensor2Buffer(tensor_map, beta_key);
        return unique_ptr<const LayerNormWeights>(layer_norm_weights);
    }
    return nullptr;
}

DenseWeightsPtr WeightsConverter::mayCreateDenseWeights(
    unordered_map<string, th::Tensor> tensor_map, const string& kernel_key, const string& bias_key) {
    if (tensor_map.count(kernel_key) > 0) {
        const auto dense_weights = new DenseWeights();
        dense_weights->kernel = mayFindTensor2Buffer(tensor_map, kernel_key);
        if (!bias_key.empty()) {
            dense_weights->bias = mayFindTensor2Buffer(tensor_map, bias_key);
        }
        return unique_ptr<const DenseWeights>(dense_weights);
    }
    return nullptr;
}

DenseWeightsPtr WeightsConverter::createDenseWeights(unordered_map<string, th::Tensor> tensor_map,
                                   const string& kernel_key, const string& bias_key) {
    auto weights = mayCreateDenseWeights(tensor_map, kernel_key, bias_key);
    assert(weights);
    return move(weights);
}

unique_ptr<const Weights> WeightsConverter::convertPythonWeights(const PyModelWeights& py_weights) {
    const auto weights = new Weights();

    const auto global_weights = py_weights.model_global_weights_;
    weights->embedding = createDenseWeights(global_weights, W::embedding);

    weights->pre_decoder_layernorm = mayCreateLayerNormWeights(
        global_weights, W::pre_decoder_ln_gamma, W::pre_decoder_ln_beta);
    weights->final_layernorm = mayCreateLayerNormWeights(
        global_weights, W::final_ln_gamma, W::final_ln_beta);

    const auto layer_num = py_weights.layer_weights_.size();
    weights->layers.reserve(layer_num);
    for (size_t layer_id = 0; layer_id < layer_num; layer_id++) {
        const auto py_layer_weights = py_weights.layer_weights_[layer_id];
        LayerWeights layer_weights;

        auto& attention_weights = layer_weights.self_attention_weights;
        attention_weights.pre_layernorm = mayCreateLayerNormWeights(
            py_layer_weights, W::pre_ln_gamma, W::pre_ln_beta);
        attention_weights.pre_attention_layernorm = mayCreateLayerNormWeights(
            py_layer_weights, W::pre_attn_ln_gamma, W::pre_attn_ln_beta);
        attention_weights.qkv_weight = createDenseWeights(
            py_layer_weights, W::attn_qkv_w, W::attn_qkv_b);
        attention_weights.attention_layernorm = mayCreateLayerNormWeights(
            py_layer_weights, W::attn_ln_gamma, W::attn_ln_beta);
        attention_weights.output_weight = createDenseWeights(
            py_layer_weights, W::attn_o_w, W::attn_o_b);
        attention_weights.post_layernorm = mayCreateLayerNormWeights(
            py_layer_weights, W::post_ln_gamma, W::post_ln_beta);

        auto& ffn_weights = layer_weights.ffn_weights;
        ffn_weights.gate_weight = createDenseWeights(
            py_layer_weights, W::ffn_w1, W::ffn_b1);
        ffn_weights.down_weight = createDenseWeights(
            py_layer_weights, W::ffn_w2, W::ffn_b2);
        ffn_weights.up_weight = createDenseWeights(
            py_layer_weights, W::ffn_w3, W::ffn_b3);
        ffn_weights.dense_layernorm = mayCreateLayerNormWeights(
            py_layer_weights, W::ffn_ln_gamma, W::ffn_ln_beta);
        ffn_weights.moe_gating_weight = createDenseWeights(py_layer_weights, W::ffn_gate);

        weights->layers.push_back(move(layer_weights));
    }

    weights->final_layernorm = mayCreateLayerNormWeights(
        global_weights, W::final_ln_gamma, W::final_ln_beta);
    weights->lm_head = createDenseWeights(global_weights, W::lm_head);

    return unique_ptr<const Weights>(weights);
}

}

