#pragma once

#include "rtp_llm/cpp/core/Types.h"

#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/WeightsConverter.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include <memory>

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

EngineInitParams createMockEngineInitParams(DeviceBase* device) {
    auto params                     = GptInitParameter();
    params.head_num_                = 2;
    params.size_per_head_           = 64;
    params.num_layers_              = 2;
    params.max_seq_len_             = 20;
    params.vocab_size_              = 20;
    params.hidden_size_             = 128;
    params.head_num_kv_             = 2;
    params.block_nums_              = 100;
    params.reuse_cache_             = false;
    params.max_generate_batch_size_ = 128;
    params.max_context_batch_size_  = 128;
    params.kv_cache_data_type_      = DataType::TYPE_FP16;
    const size_t inter_size         = 512;
    params.inter_size_              = inter_size;
    params.inter_padding_size_      = inter_size;
    params.seq_size_per_block_      = 2;
    params.reserve_runtime_mem_mb_  = 1024;
    typedef half            T;
    const at::ScalarType    scalar_type  = at::ScalarType::Half;
    const rtp_llm::DataType data_type    = getTensorType<T>();
    auto                    mem_type     = rtp_llm::MemoryType::MEMORY_GPU;
    const size_t            hidden_units = 128;
    auto data = device->allocateBuffer({data_type, {inter_size, inter_size}, AllocationType::DEVICE}, {});

    auto word_embeddings =
        make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{(size_t)20, hidden_units}, data->data());
    auto lm_head =
        make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{(size_t)20, hidden_units}, data->data());
    std::unordered_map<std::string, rtp_llm::ConstBufferPtr> global_weights;
    global_weights.emplace(W::embedding, std::move(word_embeddings));
    global_weights.emplace(W::lm_head, std::move(lm_head));

    std::vector<std::unordered_map<std::string, rtp_llm::ConstBufferPtr>> layer_weights;
    for (int i = 0; i < params.num_layers_; ++i) {
        auto pre_layernorm_weights =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto pre_layernorm_beta =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto post_layernorm_weights =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto post_layernorm_beta =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto qkv_weights = make_unique<const rtp_llm::Buffer>(
            mem_type, data_type, vector<size_t>{hidden_units, 3 * hidden_units}, data->data());
        auto qkv_weights_b = make_unique<const rtp_llm::Buffer>(
            mem_type, data_type, vector<size_t>{hidden_units, 3, hidden_units}, data->data());
        auto attention_layernorm =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto attention_layernorm_beta =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{hidden_units}, data->data());
        auto attention_output_weight = make_unique<const rtp_llm::Buffer>(
            mem_type, data_type, vector<size_t>{hidden_units, hidden_units}, data->data());
        auto attention_output_weight_beta = make_unique<const rtp_llm::Buffer>(
            mem_type, data_type, vector<size_t>{hidden_units, hidden_units}, data->data());
        auto ffn_weight = make_unique<const rtp_llm::Buffer>(
            mem_type, data_type, vector<size_t>{hidden_units, inter_size}, data->data());
        auto ffn_weight_beta = make_unique<const rtp_llm::Buffer>(
            mem_type, data_type, vector<size_t>{hidden_units, inter_size}, data->data());
        auto ffn_output_weight = make_unique<const rtp_llm::Buffer>(
            mem_type, data_type, vector<size_t>{inter_size, hidden_units}, data->data());
        auto ffn_output_weight_beta = make_unique<const rtp_llm::Buffer>(
            mem_type, data_type, vector<size_t>{inter_size, hidden_units}, data->data());
        auto ffn_layer_norm =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{inter_size}, data->data());
        auto ffn_layer_norm_beta =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{inter_size}, data->data());
        std::unordered_map<std::string, rtp_llm::ConstBufferPtr> weights;
        weights.emplace(W::pre_ln_gamma, std::move(pre_layernorm_weights));
        weights.emplace(W::pre_ln_beta, std::move(pre_layernorm_beta));
        weights.emplace(W::attn_qkv_w, std::move(qkv_weights));
        weights.emplace(W::attn_qkv_b, std::move(qkv_weights_b));
        weights.emplace(W::attn_ln_gamma, std::move(attention_layernorm));
        weights.emplace(W::attn_ln_beta, std::move(attention_layernorm_beta));
        weights.emplace(W::attn_o_w, std::move(attention_output_weight));
        weights.emplace(W::attn_o_b, std::move(attention_output_weight_beta));
        weights.emplace(W::post_ln_gamma, std::move(post_layernorm_weights));
        weights.emplace(W::post_ln_beta, std::move(post_layernorm_beta));
        weights.emplace(W::ffn_w3, std::move(ffn_weight));
        weights.emplace(W::ffn_b3, std::move(ffn_weight_beta));
        weights.emplace(W::ffn_w2, std::move(ffn_output_weight));
        weights.emplace(W::ffn_b2, std::move(ffn_output_weight_beta));
        weights.emplace(W::ffn_ln_gamma, std::move(ffn_layer_norm));
        weights.emplace(W::ffn_ln_beta, std::move(ffn_layer_norm_beta));
        layer_weights.push_back(std::move(weights));
    }
    auto                      convert = rtp_llm::WeightsConverter(false);
    rtp_llm::EngineInitParams rtp_llm_params(
        0,
        params,
        std::move(*convert.createGptWeights(std::make_unique<ConstBufferPtrMaps>(layer_weights),
                                            std::make_unique<ConstBufferPtrMap>(global_weights))));
    return rtp_llm_params;
}

using loraLayerMap = std::vector<std::unordered_map<std::string, rtp_llm::ConstBufferPtr>>;

std::array<loraLayerMap, 2> createMockLoraWeights(DeviceBase* device) {
    size_t                  rank                = 8;
    size_t                  head_num_           = 2;
    size_t                  size_per_head_      = 64;
    size_t                  num_layers_         = 2;
    size_t                  max_seq_len_        = 20;
    size_t                  vocab_size_         = 20;
    size_t                  hidden_size_        = 128;
    size_t                  head_num_kv_        = 2;
    size_t                  block_nums_         = 100;
    const size_t            inter_size          = 512;
    size_t                  inter_size_         = inter_size;
    size_t                  inter_padding_size_ = inter_size;
    typedef half            T;
    const at::ScalarType    scalar_type  = at::ScalarType::Half;
    const rtp_llm::DataType data_type    = getTensorType<T>();
    auto                    mem_type     = rtp_llm::MemoryType::MEMORY_GPU;
    const size_t            hidden_units = 128;
    auto data = device->allocateBuffer({data_type, {inter_size, inter_size}, AllocationType::DEVICE}, {});

    loraLayerMap lora_a_weights;
    loraLayerMap lora_b_weights;
    for (int i = 0; i < num_layers_; ++i) {
        auto qkv_weights_lora_a =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{hidden_units, rank}, data->data());
        auto qkv_weights_lora_b = make_unique<const rtp_llm::Buffer>(
            mem_type, data_type, vector<size_t>{rank, 3 * hidden_units}, data->data());
        auto attention_output_weight_lora_a =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{hidden_units, rank}, data->data());
        auto attention_output_weight_lora_b =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{rank, hidden_units}, data->data());
        auto ffn_weight_lora_a =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{hidden_units, rank}, data->data());
        auto ffn_weight_lora_b =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{rank, inter_size}, data->data());
        auto ffn_output_weight_lora_a =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{inter_size, rank}, data->data());
        auto ffn_output_weight_lora_b =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{rank, hidden_units}, data->data());
        std::unordered_map<std::string, rtp_llm::ConstBufferPtr> weights_lora_a;
        std::unordered_map<std::string, rtp_llm::ConstBufferPtr> weights_lora_b;
        weights_lora_a.emplace(W::attn_qkv_w, std::move(qkv_weights_lora_a));
        weights_lora_a.emplace(W::attn_o_w, std::move(attention_output_weight_lora_a));
        weights_lora_a.emplace(W::ffn_w3, std::move(ffn_weight_lora_a));
        weights_lora_a.emplace(W::ffn_w2, std::move(ffn_output_weight_lora_a));
        weights_lora_b.emplace(W::attn_qkv_w, std::move(qkv_weights_lora_b));
        weights_lora_b.emplace(W::attn_o_w, std::move(attention_output_weight_lora_b));
        weights_lora_b.emplace(W::ffn_w3, std::move(ffn_weight_lora_b));
        weights_lora_b.emplace(W::ffn_w2, std::move(ffn_output_weight_lora_b));
        lora_a_weights.push_back(std::move(weights_lora_a));
        lora_b_weights.push_back(std::move(weights_lora_b));
    }
    std::array<loraLayerMap, 2> result{lora_a_weights, lora_b_weights};
    return result;
}

}  // namespace rtp_llm
