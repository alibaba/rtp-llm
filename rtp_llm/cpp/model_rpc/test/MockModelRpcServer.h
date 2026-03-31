#pragma once

#include "rtp_llm/cpp/core/Types.h"

#include "rtp_llm/cpp/normal_engine/NormalEngine.h"
#include "rtp_llm/cpp/engine_base/schedulers/FIFOScheduler.h"
#include "rtp_llm/cpp/engine_base/WeightsConverter.h"
#include "rtp_llm/cpp/testing/TestBase.h"
#include "rtp_llm/cpp/models/models_weight/W.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include <memory>
#include <torch/all.h>

using namespace std;
namespace W = rtp_llm::W;

namespace rtp_llm {

EngineInitParams createMockEngineInitParams() {
    ModelConfig model_config;
    model_config.attn_config.head_num       = 2;
    model_config.attn_config.size_per_head  = 64;
    model_config.num_layers                 = 2;
    model_config.max_seq_len                = 20;
    model_config.vocab_size                 = 20;
    model_config.hidden_size                = 128;
    model_config.attn_config.kv_head_num    = 2;
    model_config.attn_config.kv_cache_dtype = KvCacheDataType::BASE;
    const size_t inter_size                 = 512;
    model_config.inter_size                 = inter_size;
    // inter_padding_size is now calculated in ModelDeployWeightInfo, not in ModelConfig
    model_config.attn_config.tokens_per_block = 2;

    RuntimeConfig runtime_config;
    runtime_config.max_generate_batch_size                      = 128;
    runtime_config.fifo_scheduler_config.max_context_batch_size = 128;
    runtime_config.reserve_runtime_mem_mb                       = 1024;

    MMModelConfig     mm_model_config;
    ParallelismConfig parallelism_config;
    EPLBConfig        eplb_config;
    PDSepConfig       pd_sep_config;
    ConcurrencyConfig concurrency_config;
    FMHAConfig        fmha_config;
    KVCacheConfig     kv_cache_config;
    kv_cache_config.test_block_num = 100;
    kv_cache_config.reuse_cache    = false;
    ProfilingDebugLoggingConfig profiling_debug_logging_config;
    HWKernelConfig              hw_kernel_config;
    DeviceResourceConfig        device_resource_config;
    MoeConfig                   moe_config;
    ModelSpecificConfig         model_specific_config;
    SpeculativeExecutionConfig  sp_config;
    CacheStoreConfig            cache_store_config;
    MiscellaneousConfig         misc_config;
    ArpcConfig                  arpc_config;
    FfnDisAggregateConfig       ffn_disaggregate_config;
    typedef half                T;
    const at::ScalarType        scalar_type  = at::ScalarType::Half;
    const size_t                hidden_units = 128;
    auto                        opts         = torch::TensorOptions().dtype(scalar_type).device(torch::kCUDA);
    auto                        data         = torch::empty({(long)inter_size, (long)inter_size}, opts);

    auto make_weight = [&](std::vector<int64_t> shape) -> torch::Tensor {
        return torch::from_blob(data.data_ptr(), shape, opts);
    };

    TensorMap global_weights;
    global_weights.emplace(W::embedding, make_weight({20, (long)hidden_units}));
    global_weights.emplace(W::lm_head, make_weight({20, (long)hidden_units}));

    TensorMaps layer_weights;
    for (int i = 0; i < model_config.num_layers; ++i) {
        TensorMap weights;
        weights.emplace(W::pre_ln_gamma, make_weight({(long)hidden_units}));
        weights.emplace(W::pre_ln_beta, make_weight({(long)hidden_units}));
        weights.emplace(W::attn_qkv_w, make_weight({(long)hidden_units, 3 * (long)hidden_units}));
        weights.emplace(W::attn_qkv_b, make_weight({(long)hidden_units, 3, (long)hidden_units}));
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
    auto convert                 = rtp_llm::WeightsConverter(false);
    model_config.mm_model_config = mm_model_config;
    rtp_llm::EngineInitParams rtp_llm_params(
        0,
        model_config,
        parallelism_config,
        runtime_config,
        eplb_config,
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
        ffn_disaggregate_config,
        std::move(*convert.createGptWeights(std::make_unique<TensorMaps>(layer_weights),
                                            std::make_unique<TensorMap>(global_weights))));
    return rtp_llm_params;
}

}  // namespace rtp_llm
