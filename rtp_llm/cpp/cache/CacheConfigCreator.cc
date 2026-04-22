#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <numeric>

#include "rtp_llm/cpp/cache/HybridConfigCreator.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/SingleConfigCreator.h"
#include "rtp_llm/cpp/core/DeviceData.h"
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

    const int linear_step = kv_cache_config.linear_step;
    for (auto& alloc_config : config.allocator_configs) {
        alloc_config.linear_step = linear_step;
    }
    const size_t main_block_size = config.getAllocatorConfig(0).block_size_bytes;
    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
        block_num = kv_cache_mem_size / main_block_size;
    }
    RTP_LLM_CHECK_WITH_INFO(block_num > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_num,
                            static_cast<long>(main_block_size / 1024 / 1024));

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;

    // Sync block_num into allocator_configs (created by createBasicConfig with block_num=0).
    for (auto& alloc_config : config.allocator_configs) {
        alloc_config.block_num = static_cast<uint32_t>(block_num);
    }

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

    const auto& score_ac   = score_config.getAllocatorConfig(0);
    const auto& propose_ac = propose_config.getAllocatorConfig(0);

    const uint32_t main_layer_num = score_ac.layer_num;
    const uint32_t mtp_layer_num  = propose_ac.layer_num;

    uint32_t total_layer_num = main_layer_num;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_layer_num += mtp_layer_num;
    }

    const size_t main_block_size        = score_config.getAllocatorConfig(0).block_size_bytes;
    const size_t sub_block_size         = propose_config.getAllocatorConfig(0).block_size_bytes;
    size_t       total_block_size_bytes = main_block_size;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_block_size_bytes += sub_block_size;
    }

    // -----------------------------------------------------------------------
    // Compute per-model block_num.
    //
    // MAIN_MODEL_KVCACHE_RATIO (integer, 0–99): percentage of available KV
    // cache memory allocated to the main model.  The remaining memory is split
    // equally among all MTP sub-models.  Example: "80" → main gets 80 %,
    // each of N sub-models gets (100-80)/N %.
    //
    // Default (env var absent or empty): proportional to block sizes, which
    // is equivalent to giving all models the same block_num (backward compat).
    // -----------------------------------------------------------------------
    size_t main_block_num = 0;
    size_t sub_block_num  = 0;

    if (kv_cache_config.test_block_num > 0) {
        // Unit-test override: every model gets the same fixed block_num.
        main_block_num = kv_cache_config.test_block_num;
        sub_block_num  = kv_cache_config.test_block_num;
    } else {
        const size_t kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);

        const int ratio_pct = kv_cache_config.main_model_kvcache_ratio;

        if (ratio_pct != 0) {
            RTP_LLM_CHECK_WITH_INFO(
                ratio_pct > 0 && ratio_pct < 100, "main_model_kvcache_ratio must be in (0, 100), got %d", ratio_pct);
            const double main_ratio = ratio_pct / 100.0;
            const double sub_ratio_each =
                (num_mtp_modules > 0) ? (1.0 - main_ratio) / static_cast<double>(num_mtp_modules) : 0.0;
            main_block_num = static_cast<size_t>(main_ratio * static_cast<double>(kv_cache_mem_size)
                                                 / static_cast<double>(main_block_size));
            sub_block_num  = (num_mtp_modules > 0 && sub_block_size > 0) ?
                                 static_cast<size_t>(sub_ratio_each * static_cast<double>(kv_cache_mem_size)
                                                    / static_cast<double>(sub_block_size)) :
                                 0;
            RTP_LLM_LOG_INFO("main_model_kvcache_ratio=%d: main_block_num=%zu, sub_block_num=%zu",
                             ratio_pct,
                             main_block_num,
                             sub_block_num);
        } else {
            // Default: proportional allocation → equal block_num for all models.
            // total_block_num = kv_cache_mem_size / total_block_size_bytes.
            const size_t total_block_num =
                kv_cache_mem_size / (main_block_size + sub_block_size * static_cast<size_t>(num_mtp_modules));
            main_block_num = total_block_num;
            sub_block_num  = total_block_num;
        }
    }

    RTP_LLM_CHECK_WITH_INFO(
        main_block_num > 0, "kv cache needs at least 1 block for main model but got %zu", main_block_num);
    RTP_LLM_CHECK_WITH_INFO(num_mtp_modules == 0 || sub_block_num > 0,
                            "kv cache needs at least 1 block for each MTP sub-model but got %zu",
                            sub_block_num);

    // Build result CacheConfig (shared/global fields + per-model allocator_configs).
    CacheConfig config   = score_config;
    config.layer_all_num = total_layer_num;

    const int linear_step = std::max(1, kv_cache_config.linear_step);

    // Determine which group in the MAIN model is "FULL" (used for global layer mapping).
    size_t full_gid = 0;
    if (score_ac.group_types.size() > 1) {
        for (size_t gid = 0; gid < score_ac.group_types.size(); ++gid) {
            if (score_ac.group_types[gid] == CacheGroupType::FULL) {
                full_gid = gid;
                break;
            }
        }
    }

    // Build global layer_to_group_id and layer_to_block_stride_bytes
    // (global layer_id space: main layers [0, main_layer_num) then MTP layers).
    // These are used by PD-separation connectors that transfer KV data across all models.
    config.layer_to_group_id.resize(total_layer_num, 0);
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(total_layer_num), 0);

    const size_t score_layers = static_cast<size_t>(main_layer_num);
    RTP_LLM_CHECK_WITH_INFO(score_config.layer_to_block_stride_bytes.size() == score_layers,
                            "score_config.layer_to_block_stride_bytes size mismatch, got=%zu need=%zu",
                            score_config.layer_to_block_stride_bytes.size(),
                            score_layers);
    for (size_t l = 0; l < score_layers; ++l) {
        config.layer_to_group_id[l]           = score_ac.layer_to_group_id[l];
        config.layer_to_block_stride_bytes[l] = score_config.layer_to_block_stride_bytes[l];
    }

    RTP_LLM_CHECK_WITH_INFO(propose_ac.layer_to_block_stride_bytes.size() == static_cast<size_t>(mtp_layer_num),
                            "propose_ac.layer_to_block_stride_bytes size mismatch, got=%zu need=%u",
                            propose_ac.layer_to_block_stride_bytes.size(),
                            mtp_layer_num);
    for (int m = 0; m < num_mtp_modules; ++m) {
        for (uint32_t l = 0; l < mtp_layer_num; ++l) {
            const int global_layer_id                 = main_layer_num + m * mtp_layer_num + l;
            config.layer_to_group_id[global_layer_id] = static_cast<int>(full_gid);
            config.layer_to_block_stride_bytes[global_layer_id] =
                propose_ac.layer_to_block_stride_bytes[static_cast<size_t>(l)];
        }
    }

    // Build allocator_configs:
    //   allocator_configs[0]   = main model (score)
    //   allocator_configs[1..] = MTP sub-models (propose), one entry per module
    config.allocator_configs.clear();
    config.allocator_configs.reserve(1 + static_cast<size_t>(num_mtp_modules));

    {
        KVCacheAllocatorConfig main_alloc = score_ac;
        main_alloc.model_id               = 0;
        main_alloc.block_num              = static_cast<uint32_t>(main_block_num);
        main_alloc.linear_step            = linear_step;
        config.allocator_configs.push_back(std::move(main_alloc));
    }

    for (int m = 0; m < num_mtp_modules; ++m) {
        KVCacheAllocatorConfig mtp_alloc = propose_ac;
        mtp_alloc.model_id               = static_cast<size_t>(m + 1);
        mtp_alloc.block_num              = static_cast<uint32_t>(sub_block_num);
        mtp_alloc.linear_step            = 1;
        // MTP sub-models always route all layers to a single FULL group (group 0 locally).
        mtp_alloc.layer_to_group_id.assign(static_cast<size_t>(mtp_layer_num), 0);
        config.allocator_configs.push_back(std::move(mtp_alloc));
    }

    const auto kv_cache_seq_len = static_cast<size_t>(main_block_num) * config.seq_size_per_block;
    RTP_LLM_LOG_INFO("createSpConfig: is_mtp=%d, total_layers=%u, num_mtp_modules=%d, "
                     "main_block_num=%zu (%.1f MB), sub_block_num=%zu (%.1f MB), "
                     "kv_cache_seq_len=%zu, total_block_size=%zu bytes (main=%zu + %d*sub=%zu)",
                     is_mtp,
                     total_layer_num,
                     num_mtp_modules,
                     main_block_num,
                     static_cast<double>(main_block_num) * main_block_size / 1024.0 / 1024.0,
                     sub_block_num,
                     static_cast<double>(sub_block_num) * sub_block_size / 1024.0 / 1024.0,
                     kv_cache_seq_len,
                     total_block_size_bytes,
                     main_block_size,
                     num_mtp_modules,
                     sub_block_size);

    RTP_LLM_LOG_INFO("CacheConfig debugString:\n%s", config.debugString().c_str());

    return config;
}

}  // namespace rtp_llm
