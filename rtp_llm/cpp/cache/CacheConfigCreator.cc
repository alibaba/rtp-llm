#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/SingleConfigCreator.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

bool blockNumFitsBudget(uint32_t block_num, size_t total_budget_bytes, const KVCacheBlockBudget& budget, int step) {
    if (budget.explicit_pool_reserve_bytes > total_budget_bytes) {
        return false;
    }

    size_t remaining = total_budget_bytes - budget.explicit_pool_reserve_bytes;
    if (budget.paged_block_bytes > 0) {
        if (static_cast<size_t>(block_num) > remaining / budget.paged_block_bytes) {
            return false;
        }
        remaining -= static_cast<size_t>(block_num) * budget.paged_block_bytes;
    }

    const auto safe_step  = static_cast<uint32_t>(std::max(1, step));
    const auto swa_blocks = block_num / safe_step + (block_num % safe_step != 0 ? 1u : 0u);
    return budget.swa_block_bytes == 0 || static_cast<size_t>(swa_blocks) <= remaining / budget.swa_block_bytes;
}

KVCacheBlockBudget blockBudgetForConfig(const CacheConfig& config) {
    KVCacheBlockBudget budget;
    if (!config.use_independent_block_pools) {
        budget.paged_block_bytes = config.block_size_bytes;
        return budget;
    }

    budget.explicit_pool_reserve_bytes = config.explicitly_sized_pool_reserve_bytes;
    for (size_t gid = 0; gid < static_cast<size_t>(config.groupNums()); ++gid) {
        if (config.usesExplicitIndependentBlocks(gid)) {
            continue;
        }
        const auto group_bytes = config.blockSizeBytesForGroup(gid);
        switch (config.typeForGroup(gid)) {
            case CacheGroupType::FULL:
            case CacheGroupType::LINEAR:
                budget.paged_block_bytes += group_bytes;
                break;
            case CacheGroupType::SWA:
                budget.swa_block_bytes += group_bytes;
                break;
        }
    }
    return budget;
}

void addBlockBudget(KVCacheBlockBudget& total, const KVCacheBlockBudget& addition, size_t multiplier = 1) {
    const auto add = [multiplier](size_t& dst, size_t value, const char* name) {
        RTP_LLM_CHECK_WITH_INFO(multiplier == 0 || value <= (std::numeric_limits<size_t>::max() - dst) / multiplier,
                                "kv cache %s budget overflow: current=%zu addition=%zu multiplier=%zu",
                                name,
                                dst,
                                value,
                                multiplier);
        dst += value * multiplier;
    };
    add(total.explicit_pool_reserve_bytes, addition.explicit_pool_reserve_bytes, "explicit reserve");
    add(total.paged_block_bytes, addition.paged_block_bytes, "paged block bytes");
    add(total.swa_block_bytes, addition.swa_block_bytes, "SWA block bytes");
}

void setupKernelSeqSize(CacheConfig& config, const KVCacheConfig& kv_cache_config, const char* config_name) {
    if (kv_cache_config.kernel_seq_size_per_block > 0) {
        const auto kernel_seq_size_per_block = static_cast<size_t>(kv_cache_config.kernel_seq_size_per_block);
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
    config.finalizeBlockNums(0, runtime_config);

    const auto block_budget = blockBudgetForConfig(config);
    if (block_budget.explicit_pool_reserve_bytes > 0) {
        RTP_LLM_CHECK_WITH_INFO(kv_cache_mem_size > block_budget.explicit_pool_reserve_bytes,
                                "kv cache budget %zu MiB is smaller than explicitly-sized pool reservation %zu MiB "
                                "(reduce explicitly sized pool blocks if needed)",
                                kv_cache_mem_size / 1024 / 1024,
                                block_budget.explicit_pool_reserve_bytes / 1024 / 1024);
        RTP_LLM_LOG_INFO("kv cache: total budget %zu MiB, explicitly-sized pool reserve %zu MiB",
                         kv_cache_mem_size / 1024 / 1024,
                         block_budget.explicit_pool_reserve_bytes / 1024 / 1024);
    }
    return maxKVCacheBlockNumForBudget(kv_cache_mem_size, block_budget, config.linear_step);
}

}  // namespace

uint32_t maxKVCacheBlockNumForBudget(size_t total_budget_bytes, const KVCacheBlockBudget& budget, int linear_step) {
    RTP_LLM_CHECK_WITH_INFO(budget.paged_block_bytes > 0 || budget.swa_block_bytes > 0,
                            "kv cache block budget has zero marginal block bytes");

    uint32_t low  = 0;
    uint32_t high = std::numeric_limits<uint32_t>::max();
    while (low < high) {
        const uint32_t mid = low + static_cast<uint32_t>((static_cast<uint64_t>(high) - low + 1) / 2);
        if (blockNumFitsBudget(mid, total_budget_bytes, budget, linear_step)) {
            low = mid;
        } else {
            high = mid - 1;
        }
    }
    return low;
}

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
                                                  bool                     is_mtp,
                                                  int                      gen_num_per_cycle) {
    CacheConfig config;
    if (model_config.hybrid_attention_config.enable_independent_kv_cache_pools) {
        KVCacheConfig no_override_config;
        no_override_config.seq_size_per_block        = 0;
        no_override_config.kernel_seq_size_per_block = 0;
        config                                       = HybridPoolConfigCreator::createConfig(
            model_config, parallelism_config, no_override_config, is_mtp, gen_num_per_cycle);
    } else if (model_config.hybrid_attention_config.enable_hybrid_attention) {
        config = HybridConfigCreator::createHybridConfig(model_config, parallelism_config, is_mtp, gen_num_per_cycle);
    } else {
        config = SingleConfigCreator::createSingleConfig(model_config, parallelism_config, is_mtp, gen_num_per_cycle);
    }

    if (!model_config.hybrid_attention_config.enable_independent_kv_cache_pools) {
        const auto full_group_num = std::count_if(
            config.topology().groups().begin(), config.topology().groups().end(), [](const GroupBase& group) {
                return group.policy.group_type == CacheGroupType::FULL && group.spec
                       && (group.spec->type == KVCacheSpecType::MultiHeadAttention
                           || group.spec->type == KVCacheSpecType::MultiHeadLatentAttention);
            });
        RTP_LLM_CHECK_WITH_INFO(full_group_num == 1,
                                "cache config requires exactly one FULL MHA/MLA cache group, got %zu",
                                static_cast<size_t>(full_group_num));
    }
    return config;
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig config =
        model_config.hybrid_attention_config.enable_independent_kv_cache_pools ?
            HybridPoolConfigCreator::createConfig(model_config, parallelism_config, kv_cache_config, false, 0) :
            CacheConfigCreator::createBasicConfig(model_config, parallelism_config, false, 0);

    config.linear_step = kv_cache_config.linear_step;
    setupKernelSeqSize(config, kv_cache_config, "cache");

    uint32_t block_num = computeBlockNum(
        config, model_config, runtime_config, kv_cache_config, parallelism_config, warm_up_result, sp_config);
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
    CacheConfig score_config =
        score_model_config.hybrid_attention_config.enable_independent_kv_cache_pools ?
            HybridPoolConfigCreator::createConfig(
                score_model_config, parallelism_config, kv_cache_config, false, sp_config.gen_num_per_cycle) :
            CacheConfigCreator::createBasicConfig(
                score_model_config, parallelism_config, false, sp_config.gen_num_per_cycle);
    CacheConfig propose_config =
        propose_model_config.hybrid_attention_config.enable_independent_kv_cache_pools ?
            HybridPoolConfigCreator::createConfig(
                propose_model_config, parallelism_config, kv_cache_config, is_mtp, sp_config.gen_num_per_cycle) :
            CacheConfigCreator::createBasicConfig(
                propose_model_config, parallelism_config, is_mtp, sp_config.gen_num_per_cycle);

    const int joint_step       = std::max(1, kv_cache_config.linear_step);
    score_config.linear_step   = joint_step;
    propose_config.linear_step = joint_step;

    setupKernelSeqSize(score_config, kv_cache_config, "score");
    setupKernelSeqSize(propose_config, kv_cache_config, "propose");

    int num_mtp_modules = 1;
    if (is_mtp) {
        num_mtp_modules = sp_config.gen_num_per_cycle;
        if (is_eagle) {
            num_mtp_modules = 1;
        }
    }

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

    KVCacheBlockBudget joint_budget = blockBudgetForConfig(score_config);
    addBlockBudget(joint_budget, blockBudgetForConfig(propose_config), static_cast<size_t>(num_mtp_modules));
    const size_t explicit_pool_reserve = joint_budget.explicit_pool_reserve_bytes;

    size_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);

        if (explicit_pool_reserve > 0) {
            RTP_LLM_CHECK_WITH_INFO(
                kv_cache_mem_size > explicit_pool_reserve,
                "sp kv cache budget %zu MiB is smaller than explicitly-sized pool reservation %zu MiB "
                "(reduce explicitly sized pool blocks if needed)",
                kv_cache_mem_size / 1024 / 1024,
                explicit_pool_reserve / 1024 / 1024);
            RTP_LLM_LOG_INFO(
                "sp kv cache: total budget %zu MiB, explicitly-sized pool reserve %zu MiB (score=%zu MiB + propose=%zu MiB x %d)",
                kv_cache_mem_size / 1024 / 1024,
                explicit_pool_reserve / 1024 / 1024,
                score_config.explicitly_sized_pool_reserve_bytes / 1024 / 1024,
                propose_config.explicitly_sized_pool_reserve_bytes / 1024 / 1024,
                num_mtp_modules);
        }
        block_num = maxKVCacheBlockNumForBudget(kv_cache_mem_size, joint_budget, joint_step);
    }

    RTP_LLM_CHECK_WITH_INFO(block_num > 0, "kv cache needs at least 1 block but %zu", block_num);

    CacheConfig config                         = score_config;
    config.linear_step                         = joint_step;
    config.layer_all_num                       = score_config.layer_num;
    config.block_size_bytes                    = total_block_size_bytes;
    config.block_num                           = block_num;
    config.explicitly_sized_pool_reserve_bytes = explicit_pool_reserve;

    const uint32_t main_layer_num = score_config.layer_num;
    const uint32_t mtp_layer_num  = propose_config.layer_num;

    config.mtp_sub_configs.clear();
    config.mtp_sub_configs.reserve(num_mtp_modules);
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(total_layer_num), 0);

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
