#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <cstring>
#include <memory>
#include <cuda_fp16.h>
#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"

#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/WeightsConverter.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/devices/testing/TestBase.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

struct CustomConfig {
    bool                                    reuse_cache        = false;
    DataType                                kv_cache_data_type = DataType::TYPE_FP16;
    std::map<std::string, std::vector<int>> multi_task_prompt_tokens;
};

rtp_llm::EngineInitParams
createEngineInitParams(DeviceBase* device, const CustomConfig& config,
                      rtp_llm::ModelConfig& model_config,
                      rtp_llm::RuntimeConfig& runtime_config,
                      rtp_llm::KVCacheConfig& kv_cache_config) {
    model_config.attn_config.head_num = 2;
    model_config.attn_config.size_per_head = 64;
    model_config.num_layers                   = 2;
    model_config.max_seq_len                  = 20;
    model_config.vocab_size                   = 100;
    model_config.hidden_size                  = 128;
    model_config.attn_config.kv_head_num = 2;
    model_config.activation_type = ActivationType::Silu;
    kv_cache_config.test_block_num                   = 100;
    kv_cache_config.reuse_cache                  = config.reuse_cache;
    kv_cache_config.multi_task_prompt_tokens     = config.multi_task_prompt_tokens;
    runtime_config.max_generate_batch_size      = 128;
    runtime_config.fifo_scheduler_config.max_context_batch_size       = 128;
    model_config.attn_config.kv_cache_dtype = config.kv_cache_data_type == DataType::TYPE_INT8 ? KvCacheDataType::INT8 : (config.kv_cache_data_type == DataType::TYPE_FP8_E4M3 ? KvCacheDataType::FP8 : KvCacheDataType::BASE);
    model_config.special_tokens.eos_token_id = -1;  // never eos

    const size_t inter_size        = 512;
    // inter_size is now calculated in ModelDeployWeightInfo, not in ModelConfig
    model_config.attn_config.tokens_per_block = 2;
    runtime_config.reserve_runtime_mem_mb = 1024;
    typedef half            T;
    const rtp_llm::DataType data_type    = getTensorType<T>();
    auto                    mem_type     = rtp_llm::MemoryType::MEMORY_GPU;
    const size_t            hidden_units = 128;

    const auto tensor   = torch::ones({inter_size * inter_size}, torch::kHalf) * 0.001;
    auto       buf_host = torchTensor2Buffer(tensor);
    auto       data     = device->allocateBuffer({data_type, {inter_size, inter_size}, AllocationType::DEVICE}, {});
    device->copy({*data, *buf_host});

    auto word_embeddings = make_unique<const Buffer>(
        mem_type, data_type, vector<size_t>{(size_t)model_config.vocab_size, hidden_units}, data->data(), [data](Buffer*) {
        });
    auto lm_head = make_unique<const rtp_llm::Buffer>(
        mem_type, data_type, vector<size_t>{(size_t)model_config.vocab_size, hidden_units}, data->data());
    std::unordered_map<std::string, rtp_llm::ConstBufferPtr> global_weights;
    global_weights.emplace(W::embedding, std::move(word_embeddings));
    global_weights.emplace(W::lm_head, std::move(lm_head));

    std::vector<std::unordered_map<std::string, rtp_llm::ConstBufferPtr>> layer_weights;
    for (int i = 0; i < model_config.num_layers; ++i) {
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
        auto qkv_weights_b =
            make_unique<const rtp_llm::Buffer>(mem_type, data_type, vector<size_t>{3 * hidden_units}, data->data());
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
    auto                      weights = convert.createGptWeights(std::make_unique<ConstBufferPtrMaps>(layer_weights),
                                            std::make_unique<ConstBufferPtrMap>(global_weights));
    
    // Create all config objects with defaults
    rtp_llm::MMModelConfig mm_model_config;
    model_config.mm_model_config = mm_model_config;
    rtp_llm::ParallelismConfig parallelism_config;
    rtp_llm::PDSepConfig pd_sep_config;
    rtp_llm::ConcurrencyConfig concurrency_config;
    rtp_llm::FMHAConfig fmha_config;
    rtp_llm::ProfilingDebugLoggingConfig profiling_debug_logging_config;
    rtp_llm::HWKernelConfig hw_kernel_config;
    rtp_llm::DeviceResourceConfig device_resource_config;
    rtp_llm::MoeConfig moe_config;
    rtp_llm::ModelSpecificConfig model_specific_config;
    rtp_llm::SpeculativeExecutionConfig sp_config;
    rtp_llm::CacheStoreConfig cache_store_config;
    rtp_llm::MiscellaneousConfig misc_config;
    rtp_llm::ArpcConfig arpc_config;
    rtp_llm::GrpcConfig grpc_config;
    rtp_llm::FfnDisAggregateConfig ffn_disaggregate_config;
    rtp_llm::VitConfig vit_config;
    
    rtp_llm::EngineInitParams rtp_llm_params(0, 
                                             model_config,
                                             parallelism_config,
                                             runtime_config,
                                             pd_sep_config,
                                             concurrency_config,
                                             fmha_config,
                                             kv_cache_config,
                                             profiling_debug_logging_config,
                                             hw_kernel_config,
                                             device_resource_config,
                                             moe_config,
                                             model_specific_config,
                                             sp_config,
                                             cache_store_config,
                                             misc_config,
                                             arpc_config,
                                             grpc_config,
                                             ffn_disaggregate_config,
                                             vit_config,
                                             std::move(*weights));
    return rtp_llm_params;
}

std::shared_ptr<NormalEngine>
createMockEngine(DeviceBase* device, const CustomConfig& config) {
    rtp_llm::ModelConfig model_config;
    rtp_llm::RuntimeConfig runtime_config;
    rtp_llm::KVCacheConfig kv_cache_config;
    EngineInitParams              rtp_llm_params = createEngineInitParams(device, config, model_config, runtime_config, kv_cache_config);
    std::shared_ptr<NormalEngine> engine         = make_shared<NormalEngine>(rtp_llm_params);
    return engine;
}

}  // namespace rtp_llm
