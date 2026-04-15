#pragma once

#include <c10/core/ScalarType.h>
#include <c10/util/Half.h>
#include <cstring>
#include <memory>
#include <cuda_fp16.h>
#include "c10/util/intrusive_ptr.h"
#include "torch/all.h"

#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/normal_engine/NormalExecutor.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/WeightsConverter.h"
#include "rtp_llm/cpp/core/Types.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/config/ConfigModules.h"

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

// Mock model that returns random logits for testing NormalEngine without Python
class MockModel: public ModelBase {
public:
    MockModel(size_t vocab_size): vocab_size_(vocab_size) {}

    GptModelOutputs forward(const GptModelInputs& inputs) override {
        GptModelOutputs outputs;
        // lm_output_indexes tells us how many logits rows to produce
        int64_t num_tokens = inputs.lm_output_indexes.defined() ? inputs.lm_output_indexes.size(0) : 1;
        outputs.logits     = torch::randn({num_tokens, (int64_t)vocab_size_},
                                      torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
        return outputs;
    }

private:
    size_t vocab_size_;
};

struct CustomConfig {
    bool                                    reuse_cache        = false;
    DataType                                kv_cache_data_type = DataType::TYPE_FP16;
    std::map<std::string, std::vector<int>> multi_task_prompt_tokens;
};

rtp_llm::EngineInitParams createEngineInitParams(const CustomConfig&     config,
                                                 rtp_llm::ModelConfig&   model_config,
                                                 rtp_llm::RuntimeConfig& runtime_config,
                                                 rtp_llm::KVCacheConfig& kv_cache_config) {
    model_config.attn_config.head_num        = 2;
    model_config.attn_config.size_per_head   = 64;
    model_config.num_layers                  = model_config.num_layers != 0 ? model_config.num_layers : 2;
    model_config.max_seq_len                 = 20;
    model_config.vocab_size                  = 100;
    model_config.hidden_size                 = 128;
    model_config.attn_config.kv_head_num     = 2;
    model_config.activation_type             = ActivationType::Silu;
    kv_cache_config.test_block_num           = 100;
    kv_cache_config.reuse_cache              = config.reuse_cache;
    kv_cache_config.multi_task_prompt_tokens = config.multi_task_prompt_tokens;
    runtime_config.max_generate_batch_size   = 128;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 128;
    model_config.attn_config.kv_cache_dtype =
        config.kv_cache_data_type == DataType::TYPE_INT8 ?
            KvCacheDataType::INT8 :
            (config.kv_cache_data_type == DataType::TYPE_FP8_E4M3 ? KvCacheDataType::FP8 : KvCacheDataType::BASE);
    model_config.special_tokens.eos_token_id = -1;  // never eos

    const size_t inter_size = 512;
    // inter_size is now calculated in ModelDeployWeightInfo, not in ModelConfig
    model_config.attn_config.tokens_per_block = 2;
    runtime_config.reserve_runtime_mem_mb     = 1024;
    const size_t hidden_units                 = 128;

    auto opts = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);

    // Create a CUDA tensor filled with 0.001 as the backing data for all weights
    auto data = (torch::ones({(long)(inter_size * inter_size)}, torch::TensorOptions().dtype(torch::kHalf)) * 0.001)
                    .to(torch::kCUDA);
    auto data_ptr = data.data_ptr();

    auto make_weight = [&](std::vector<int64_t> shape) -> torch::Tensor {
        return torch::from_blob(data_ptr, shape, opts);
    };

    TensorMap global_weights;
    global_weights.emplace(W::embedding, make_weight({(long)model_config.vocab_size, (long)hidden_units}));
    global_weights.emplace(W::lm_head, make_weight({(long)model_config.vocab_size, (long)hidden_units}));

    TensorMaps layer_weights;
    for (int i = 0; i < model_config.num_layers; ++i) {
        TensorMap weights;
        weights.emplace(W::pre_ln_gamma, make_weight({(long)hidden_units}));
        weights.emplace(W::pre_ln_beta, make_weight({(long)hidden_units}));
        weights.emplace(W::attn_qkv_w, make_weight({(long)hidden_units, (long)(3 * hidden_units)}));
        weights.emplace(W::attn_qkv_b, make_weight({(long)(3 * hidden_units)}));
        weights.emplace(W::attn_ln_gamma, make_weight({(long)hidden_units}));
        weights.emplace(W::attn_ln_beta, make_weight({(long)hidden_units}));
        weights.emplace(W::attn_o_w, make_weight({(long)hidden_units, (long)hidden_units}));
        weights.emplace(W::attn_o_b, make_weight({(long)hidden_units, (long)hidden_units}));
        weights.emplace(W::post_ln_gamma, make_weight({(long)hidden_units}));
        weights.emplace(W::post_ln_beta, make_weight({(long)hidden_units}));
        weights.emplace(W::ffn_w3, make_weight({(long)hidden_units, (long)inter_size}));
        weights.emplace(W::ffn_b3, make_weight({(long)hidden_units, (long)inter_size}));
        weights.emplace(W::ffn_w2, make_weight({(long)inter_size, (long)hidden_units}));
        weights.emplace(W::ffn_b2, make_weight({(long)inter_size, (long)hidden_units}));
        weights.emplace(W::ffn_ln_gamma, make_weight({(long)inter_size}));
        weights.emplace(W::ffn_ln_beta, make_weight({(long)inter_size}));
        layer_weights.push_back(std::move(weights));
    }
    auto convert = rtp_llm::WeightsConverter(false);
    auto weights = convert.createGptWeights(std::make_unique<TensorMaps>(std::move(layer_weights)),
                                            std::make_unique<TensorMap>(std::move(global_weights)));

    // Create all config objects with defaults
    rtp_llm::MMModelConfig mm_model_config;
    model_config.mm_model_config = mm_model_config;
    rtp_llm::ParallelismConfig           parallelism_config;
    rtp_llm::PDSepConfig                 pd_sep_config;
    rtp_llm::ConcurrencyConfig           concurrency_config;
    rtp_llm::FMHAConfig                  fmha_config;
    rtp_llm::ProfilingDebugLoggingConfig profiling_debug_logging_config;
    rtp_llm::HWKernelConfig              hw_kernel_config;
    rtp_llm::DeviceResourceConfig        device_resource_config;
    rtp_llm::MoeConfig                   moe_config;
    rtp_llm::ModelSpecificConfig         model_specific_config;
    rtp_llm::SpeculativeExecutionConfig  sp_config;
    rtp_llm::CacheStoreConfig            cache_store_config;
    rtp_llm::MiscellaneousConfig         misc_config;
    rtp_llm::ArpcConfig                  arpc_config;
    rtp_llm::GrpcConfig                  grpc_config;
    rtp_llm::FfnDisAggregateConfig       ffn_disaggregate_config;
    rtp_llm::VitConfig                   vit_config;

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

std::shared_ptr<NormalEngine> createMockEngine(const CustomConfig& config) {
    rtp_llm::ModelConfig   model_config;
    rtp_llm::RuntimeConfig runtime_config;
    rtp_llm::KVCacheConfig kv_cache_config;
    EngineInitParams rtp_llm_params = createEngineInitParams(config, model_config, runtime_config, kv_cache_config);
    // Set test model factory before engine construction so the model is available during startLoop()
    size_t vocab                       = model_config.vocab_size;
    NormalExecutor::test_model_factory = [vocab](const GptModelInitParams&) {
        return std::make_unique<MockModel>(vocab);
    };
    std::shared_ptr<NormalEngine> engine = make_shared<NormalEngine>(rtp_llm_params, nullptr);
    NormalExecutor::test_model_factory   = nullptr;
    return engine;
}

}  // namespace rtp_llm
