#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <numeric>

#include "rtp_llm/cpp/cache/HybridConfigCreator.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/SingleConfigCreator.h"
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  bool                     is_mtp) {
    if (model_config.hybrid_attention_config.enable_hybrid_attention) {
        return HybridConfigCreator::createHybridConfig(model_config, parallelism_config, is_mtp);
    } else {
        return SingleConfigCreator::createSingleConfig(model_config, parallelism_config, is_mtp);
    }
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig config    = CacheConfigCreator::createBasicConfig(model_config, parallelism_config);
    uint32_t    block_num = 0;

    config.linear_step = kv_cache_config.linear_step;
    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
        block_num = kv_cache_mem_size / config.block_size_bytes;
    }
    RTP_LLM_CHECK_WITH_INFO(block_num > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_num,
                            static_cast<long>(config.block_size_bytes / 1024 / 1024));

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
    config.block_num            = static_cast<int>(block_num);
    RTP_LLM_LOG_INFO("kv cache block nums is %u, allows storing %ld tokens", block_num, kv_cache_seq_len);
    if (kv_cache_seq_len < model_config.max_seq_len) {
        RTP_LLM_LOG_WARNING("kv cache block nums %u can only store %ld tokens, less than max_seq_len %ld, "
                            "this is dangerous, consider decrease max_seq_len",
                            block_num,
                            kv_cache_seq_len,
                            model_config.max_seq_len);
    }
    return config;
}

CacheConfig CacheConfigCreator::createSpConfig(const ModelConfig&                 score_model_config,
                                               const ModelConfig&                 propose_model_config,
                                               const ParallelismConfig&           parallelism_config,
                                               const RuntimeConfig&               runtime_config,
                                               const KVCacheConfig&               kv_cache_config,
                                               const SpeculativeExecutionConfig&  sp_config,
                                               const std::optional<WarmUpResult>& warm_up_result,
                                               bool                               is_mtp,
                                               bool                               is_eagle) {
    CacheConfig score_config = CacheConfigCreator::createBasicConfig(score_model_config, parallelism_config, false);
    CacheConfig propose_config =
        CacheConfigCreator::createBasicConfig(propose_model_config, parallelism_config, is_mtp);

    int num_mtp_modules = 1;
    if (is_mtp) {
        num_mtp_modules = sp_config.gen_num_per_cycle;
        if (is_eagle) {
            num_mtp_modules = 1;
        }
    }

    uint32_t total_layer_num = score_config.layer_num;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_layer_num += propose_config.layer_num;
    }

    size_t total_block_size_bytes = score_config.block_size_bytes;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_block_size_bytes += propose_config.block_size_bytes;
    }

    size_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);

        block_num = kv_cache_mem_size
                    / (static_cast<size_t>(score_config.block_size_bytes)
                       + static_cast<size_t>(propose_config.block_size_bytes) * static_cast<size_t>(num_mtp_modules));
    }

    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least 1 block but %zu", block_num);

    CacheConfig config      = score_config;
    config.linear_step      = std::max(1, kv_cache_config.linear_step);
    config.layer_all_num    = total_layer_num;
    config.block_size_bytes = total_block_size_bytes;
    // config.block_size       = config.block_size_bytes / rtp_llm::getTypeSize(config.dtype);
    config.block_num = block_num;

    // Record global layer ids for BlockPool address lookup.
    // - Main model global_layer_ids[0] covers all layers across main + mtp modules: [0 .. total_layer_num-1].
    // - Each mtp_sub_config has its own global_layer_ids[0] range for its local layers.
    config.global_layer_ids.clear();
    config.global_layer_ids.resize(1);
    config.global_layer_ids[0].resize(total_layer_num);
    for (uint32_t i = 0; i < total_layer_num; ++i) {
        config.global_layer_ids[0][i] = static_cast<int>(i);
    }

    const uint32_t main_layer_num = score_config.layer_num;
    const uint32_t mtp_layer_num  = propose_config.layer_num;

    size_t full_gid = 0;
    if (config.group_types.size() > 1) {
        for (size_t gid = 0; gid < config.group_types.size(); ++gid) {
            if (config.group_types[gid] == CacheGroupType::FULL) {
                full_gid = gid;
                break;
            }
        }
    }

    // Each sub-model needs an independent CacheConfig because global_layer_ids differs per module.
    config.mtp_sub_configs.clear();
    config.mtp_sub_configs.reserve(num_mtp_modules);
    config.layer_to_group_id.resize(total_layer_num, 0);

    for (int m = 0; m < num_mtp_modules; ++m) {
        auto sub_cfg           = std::make_shared<CacheConfig>(propose_config);
        sub_cfg->block_num     = block_num;
        sub_cfg->layer_all_num = sub_cfg->layer_num;

        sub_cfg->global_layer_ids.clear();
        sub_cfg->global_layer_ids.resize(1);
        sub_cfg->global_layer_ids[0].resize(mtp_layer_num);
        for (uint32_t l = 0; l < mtp_layer_num; ++l) {
            int global_layer_id                       = main_layer_num + m * mtp_layer_num + l;
            sub_cfg->global_layer_ids[0][l]           = global_layer_id;
            config.layer_to_group_id[global_layer_id] = static_cast<int>(full_gid);
        }

        sub_cfg->layer_to_group_id.assign(static_cast<size_t>(sub_cfg->layer_num), static_cast<int>(full_gid));
        config.mtp_sub_configs.push_back(sub_cfg);
    }

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
    RTP_LLM_LOG_INFO("CacheConfig created: is_mtp=%d, total_layers=%u, num_mtp_modules=%d, block_num=%zu, "
                     "allows storing %zu tokens, total_block_size=%zu bytes (main=%zu + %d*propose=%zu)",
                     is_mtp,
                     total_layer_num,
                     num_mtp_modules,
                     block_num,
                     kv_cache_seq_len,
                     total_block_size_bytes,
                     score_config.block_size_bytes,
                     num_mtp_modules,
                     propose_config.block_size_bytes);

    RTP_LLM_LOG_INFO("CacheConfig debugString(main_score_model):\n%s", score_config.debugString().c_str());
    for (size_t i = 0; i < config.mtp_sub_configs.size(); ++i) {
        const auto& sub = config.mtp_sub_configs[i];
        RTP_LLM_LOG_INFO("CacheConfig debugString(sub_propose_model[%zu]):\n%s", i, sub->debugString().c_str());
    }

    return config;
}

}  // namespace rtp_llm
