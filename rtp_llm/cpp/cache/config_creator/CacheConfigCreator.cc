#include "rtp_llm/cpp/cache/config_creator/CacheConfigCreator.h"

#include <numeric>
#include <algorithm>

#include "rtp_llm/cpp/cache/config_creator/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/config_creator/HybridConfigCreator.h"
#include "rtp_llm/cpp/cache/config_creator/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/config_creator/SingleConfigCreator.h"
#include "rtp_llm/cpp/cache/spec/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

size_t steppedBytes(size_t bytes, int step) {
    return (bytes > 0 && step > 1) ? bytes / static_cast<size_t>(step) : bytes;
}

size_t nonExplicitFixedPoolHbmBytes(const CacheConfig& config) {
    // Only independent-pool configs use per-group HBM accounting; SingleConfig
    // and HybridConfig leave use_independent_block_pools false.
    if (!config.use_independent_block_pools) {
        return 0;
    }

    size_t bytes = 0;
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        const auto& spec = config.specForGroup(gid);
        if (spec == nullptr || !spec->isFixedCache()) {
            continue;
        }
        if (!config.usesExplicitIndependentBlocks(gid)) {
            bytes += config.blockSizeBytesForGroup(gid);
        }
    }
    return bytes;
}

size_t effectivePagedBlockBytes(const CacheConfig& config, int step) {
    return config.block_size_bytes + steppedBytes(nonExplicitFixedPoolHbmBytes(config), step);
}

void setupKernelSeqSize(CacheConfig& config, const KVCacheConfig& kv_cache_config, const char* config_name) {
    if (kv_cache_config.kernel_seq_size_per_block > 0) {
        const auto kernel_seq_size_per_block = static_cast<size_t>(kv_cache_config.kernel_seq_size_per_block);
        // Generic divisibility check. Desc-based hybrid pool layouts validate
        // their own stricter alignment during createBasicConfig().
        RTP_LLM_CHECK_WITH_INFO(config.seq_size_per_block % kernel_seq_size_per_block == 0,
                                "%s seq_size_per_block(%zu) must be divisible by kernel_seq_size_per_block(%zu)",
                                config_name,
                                config.seq_size_per_block,
                                kernel_seq_size_per_block);
        config.kernel_seq_size_per_block = kernel_seq_size_per_block;
    } else if (config.kernel_seq_size_per_block == 0 || config.kernel_seq_size_per_block == config.seq_size_per_block) {
        config.kernel_seq_size_per_block = config.seq_size_per_block;
    }
}

uint32_t computeBlockNum(CacheConfig&                                     config,
                         const ModelConfig&                               model_config,
                         const RuntimeConfig&                             runtime_config,
                         const KVCacheConfig&                             kv_cache_config,
                         const ParallelismConfig&                         parallelism_config,
                         const std::optional<WarmUpResult>&               warm_up_result,
                         const std::optional<SpeculativeExecutionConfig>& sp_config) {
    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        config.finalizeBlockNums(kv_cache_config.test_block_num, runtime_config);
        return static_cast<uint32_t>(kv_cache_config.test_block_num);
    }

    const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
        runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
    // Explicitly-sized pool reservation depends on runtime scheduler limits,
    // so finalize it here after RuntimeConfig is available.
    config.finalizeBlockNums(0, runtime_config);

    size_t paged_budget = kv_cache_mem_size;
    if (config.explicitly_sized_pool_reserve_bytes > 0) {
        RTP_LLM_CHECK_WITH_INFO(kv_cache_mem_size > config.explicitly_sized_pool_reserve_bytes,
                                "kv cache budget %zu MiB is smaller than explicitly-sized pool reservation %zu MiB "
                                "(reduce explicitly sized pool blocks if needed)",
                                kv_cache_mem_size / 1024 / 1024,
                                config.explicitly_sized_pool_reserve_bytes / 1024 / 1024);
        paged_budget = kv_cache_mem_size - config.explicitly_sized_pool_reserve_bytes;
        RTP_LLM_LOG_INFO("kv cache: total budget %zu MiB, explicitly-sized pool reserve %zu MiB, paged budget %zu MiB",
                         kv_cache_mem_size / 1024 / 1024,
                         config.explicitly_sized_pool_reserve_bytes / 1024 / 1024,
                         paged_budget / 1024 / 1024);
    }
    const int joint_step = std::max(1, config.linear_step);
    return static_cast<uint32_t>(paged_budget / effectivePagedBlockBytes(config, joint_step));
}

}  // namespace

LayerKVCacheSpecs CacheConfigCreator::buildLayerSpecsFromDescs(const LayerKVCacheSpecDescs& layer_descs,
                                                               const SpecBuildContext&      ctx,
                                                               int64_t                      expected_layer_num) {
    RTP_LLM_CHECK_WITH_INFO(layer_descs.size() == static_cast<size_t>(expected_layer_num),
                            "kv_cache_spec_descs size %zu != num_layers %ld",
                            layer_descs.size(),
                            expected_layer_num);
    LayerKVCacheSpecs layer_specs(layer_descs.size());
    for (size_t layer_id = 0; layer_id < layer_descs.size(); ++layer_id) {
        const auto& descs = layer_descs[layer_id];
        RTP_LLM_CHECK_WITH_INFO(!descs.empty(), "kv_cache_spec_descs layer %zu has no descs", layer_id);
        auto& specs = layer_specs[layer_id];
        specs.reserve(descs.size());
        for (const auto& desc : descs) {
            specs.push_back(SpecBuilder::build(desc, ctx));
        }
    }
    return layer_specs;
}

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  bool                     is_mtp,
                                                  int                      gen_num_per_cycle) {
    // Routing priority:
    //   1. enable_independent_kv_cache_pools=true  → HybridPool (independent BlockPool per group)
    //   2. enable_hybrid_attention=true             → HybridType  (shared BlockPool across groups)
    //   3. else                                     → Single       (standard MHA/MLA path)
    if (model_config.hybrid_attention_config.enable_independent_kv_cache_pools) {
        return HybridPoolConfigCreator::createConfig(
            model_config, parallelism_config, kv_cache_config, is_mtp, gen_num_per_cycle);
    } else if (model_config.hybrid_attention_config.enable_hybrid_attention) {
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
    CacheConfig config =
        CacheConfigCreator::createBasicConfig(model_config, parallelism_config, kv_cache_config, false, 0);

    config.linear_step = kv_cache_config.linear_step;
    setupKernelSeqSize(config, kv_cache_config, "cache");

    uint32_t block_num = computeBlockNum(config, model_config, runtime_config, kv_cache_config,
                                         parallelism_config, warm_up_result, sp_config);
    RTP_LLM_CHECK_WITH_INFO(block_num > 0,
                            "kv cache needs at least 1 block but %ld, each block needs %ld MiB memory",
                            block_num,
                            static_cast<long>(config.block_size_bytes / 1024 / 1024));

    const auto kv_cache_seq_len = static_cast<size_t>(block_num) * config.seq_size_per_block;
    config.block_num            = static_cast<int>(block_num);
    config.finalizeBlockNums(block_num, runtime_config);
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
    CacheConfig score_config = CacheConfigCreator::createBasicConfig(
        score_model_config, parallelism_config, kv_cache_config, false, sp_config.gen_num_per_cycle);
    CacheConfig propose_config = CacheConfigCreator::createBasicConfig(
        propose_model_config, parallelism_config, kv_cache_config, is_mtp, sp_config.gen_num_per_cycle);

    setupKernelSeqSize(score_config, kv_cache_config, "score");
    setupKernelSeqSize(propose_config, kv_cache_config, "propose");

    int num_mtp_modules = 1;
    if (is_mtp) {
        num_mtp_modules = sp_config.gen_num_per_cycle;
        if (is_eagle) {
            num_mtp_modules = 1;
        }
    }

    // Fixed-pool block counts depend on runtime scheduler limits. Finalize the
    // score and propose configs before sizing the shared paged budget so fixed
    // state pools are accounted outside the paged KV-cache block budget.
    score_config.finalizeBlockNums(0, runtime_config);
    propose_config.finalizeBlockNums(0, runtime_config);

    uint32_t total_layer_num = score_config.layer_num;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_layer_num += propose_config.layer_num;
    }

    size_t total_block_size_bytes = score_config.block_size_bytes;
    for (int i = 0; i < num_mtp_modules; ++i) {
        total_block_size_bytes += propose_config.block_size_bytes;
    }

    const size_t explicit_pool_reserve =
        score_config.explicitly_sized_pool_reserve_bytes
        + propose_config.explicitly_sized_pool_reserve_bytes * static_cast<size_t>(num_mtp_modules);

    size_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);

        size_t paged_budget = kv_cache_mem_size;
        if (explicit_pool_reserve > 0) {
            RTP_LLM_CHECK_WITH_INFO(kv_cache_mem_size > explicit_pool_reserve,
                                    "sp kv cache budget %zu MiB is smaller than explicitly-sized pool reservation %zu MiB "
                                    "(reduce explicitly sized pool blocks if needed)",
                                    kv_cache_mem_size / 1024 / 1024,
                                    explicit_pool_reserve / 1024 / 1024);
            paged_budget = kv_cache_mem_size - explicit_pool_reserve;
            RTP_LLM_LOG_INFO(
                "sp kv cache: total budget %zu MiB, explicitly-sized pool reserve %zu MiB (score=%zu MiB + propose=%zu MiB x %d), paged budget %zu MiB",
                kv_cache_mem_size / 1024 / 1024,
                explicit_pool_reserve / 1024 / 1024,
                score_config.explicitly_sized_pool_reserve_bytes / 1024 / 1024,
                propose_config.explicitly_sized_pool_reserve_bytes / 1024 / 1024,
                num_mtp_modules,
                paged_budget / 1024 / 1024);
        }

        const int joint_step     = std::max(1, kv_cache_config.linear_step);
        auto      effective_size = [&](const CacheConfig& cfg) -> size_t {
            return effectivePagedBlockBytes(cfg, joint_step);
        };
        block_num =
            paged_budget
            / (effective_size(score_config) + effective_size(propose_config) * static_cast<size_t>(num_mtp_modules));
    }

    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least 1 block but %zu", block_num);

    CacheConfig config      = score_config;
    config.linear_step      = std::max(1, kv_cache_config.linear_step);
    config.layer_all_num    = total_layer_num;
    config.block_size_bytes = total_block_size_bytes;
    // config.block_size       = config.block_size_bytes / rtp_llm::getTypeSize(config.dtype);
    config.block_num                = block_num;
    config.explicitly_sized_pool_reserve_bytes = explicit_pool_reserve;

    const uint32_t main_layer_num = score_config.layer_num;
    const uint32_t mtp_layer_num  = propose_config.layer_num;

    // Each sub-model needs an independent CacheConfig because global_layer_ids differs per module.
    config.mtp_sub_configs.clear();
    config.mtp_sub_configs.reserve(num_mtp_modules);
    config.resizeLayerRoutes(static_cast<size_t>(total_layer_num));
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(total_layer_num), 0);

    // Main(score) model per-layer stride (kv + scale).
    // This is expected to be fully populated by createBasicConfig() (Single/Hybrid creators).
    const size_t score_layers = static_cast<size_t>(main_layer_num);
    RTP_LLM_CHECK_WITH_INFO(score_config.layer_to_block_stride_bytes.size() == score_layers,
                            "score_config.layer_to_block_stride_bytes size mismatch, got=%zu need=%zu",
                            score_config.layer_to_block_stride_bytes.size(),
                            score_layers);
    for (size_t l = 0; l < score_layers; ++l) {
        config.layer_to_block_stride_bytes[l] = score_config.layer_to_block_stride_bytes[l];
    }

    for (int m = 0; m < num_mtp_modules; ++m) {
        RTP_LLM_CHECK_WITH_INFO(propose_config.layer_to_block_stride_bytes.size() == static_cast<size_t>(mtp_layer_num),
                                "sub_cfg.layer_to_block_stride_bytes size mismatch, got=%zu need=%u",
                                propose_config.layer_to_block_stride_bytes.size(),
                                mtp_layer_num);
        auto sub_cfg = config.mergeMTPModule(propose_config, m, main_layer_num);
        sub_cfg->finalizeBlockNums(static_cast<uint32_t>(block_num), runtime_config);
        config.mtp_sub_configs.push_back(sub_cfg);
    }

    config.finalizeBlockNums(static_cast<uint32_t>(block_num), runtime_config);
    config.explicitly_sized_pool_reserve_bytes = explicit_pool_reserve;

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
