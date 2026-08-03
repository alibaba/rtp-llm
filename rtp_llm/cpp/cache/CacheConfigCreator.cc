#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/SingleConfigCreator.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

namespace {

void validatePublishedCacheConfig(const CacheConfig& config) {
    const auto full_attention_group_num =
        std::count_if(config.topology().groups().begin(), config.topology().groups().end(), [](const GroupBase& group) {
            return group.policy.group_type == CacheGroupType::FULL && group.spec
                   && (group.spec->type == KVCacheSpecType::MultiHeadAttention
                       || group.spec->type == KVCacheSpecType::MultiHeadLatentAttention);
        });
    RTP_LLM_CHECK_WITH_INFO(
        full_attention_group_num <= 1,
        "multiple FULL MHA/MLA cache groups (%zu) are not supported: FMHA parameters bind one block table before "
        "the layer loop",
        static_cast<size_t>(full_attention_group_num));
    RTP_LLM_CHECK_WITH_INFO(config.use_typed_cache_regions || full_attention_group_num == 1,
                            "cache config requires exactly one FULL MHA/MLA cache group, got %zu",
                            static_cast<size_t>(full_attention_group_num));
}

size_t checkedAdd(size_t lhs, size_t rhs, const char* what) {
    RTP_LLM_CHECK_WITH_INFO(
        rhs <= std::numeric_limits<size_t>::max() - lhs, "kv cache %s overflow: %zu + %zu", what, lhs, rhs);
    return lhs + rhs;
}

size_t checkedMul(size_t lhs, size_t rhs, const char* what) {
    RTP_LLM_CHECK_WITH_INFO(
        lhs == 0 || rhs <= std::numeric_limits<size_t>::max() / lhs, "kv cache %s overflow: %zu * %zu", what, lhs, rhs);
    return lhs * rhs;
}

uint64_t checkedLcm(uint64_t lhs, uint64_t rhs) {
    RTP_LLM_CHECK_WITH_INFO(lhs > 0 && rhs > 0, "kv cache capacity LCM requires positive spans");
    const uint64_t gcd = std::gcd(lhs, rhs);
    RTP_LLM_CHECK_WITH_INFO(lhs / gcd <= std::numeric_limits<uint64_t>::max() / rhs,
                            "kv cache physical span LCM overflow: %lu and %lu",
                            lhs,
                            rhs);
    return lhs / gcd * rhs;
}

uint64_t explicitTestCapacity(const ModelConfig& model_config, const KVCacheConfig& kv_cache_config) {
    const auto blocks = static_cast<uint64_t>(kv_cache_config.test_block_num);
    const auto span   = static_cast<uint64_t>(model_config.attn_config.tokens_per_block);
    RTP_LLM_CHECK_WITH_INFO(span > 0 && blocks <= std::numeric_limits<uint64_t>::max() / span,
                            "test block capacity overflow: blocks=%lu span=%lu",
                            blocks,
                            span);
    return blocks * span;
}

uint64_t computeTokenCapacity(const CacheConfig&                               config,
                              const ModelConfig&                               model_config,
                              const RuntimeConfig&                             runtime_config,
                              const KVCacheConfig&                             kv_cache_config,
                              const ParallelismConfig&                         parallelism_config,
                              const std::optional<WarmUpResult>&               warm_up_result,
                              const std::optional<SpeculativeExecutionConfig>& sp_config) {
    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        return explicitTestCapacity(model_config, kv_cache_config);
    }

    const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
        runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
    return maxKVCacheTokenCapacityForBudget(kv_cache_mem_size, config);
}

}  // namespace

uint64_t maxKVCacheTokenCapacityForBudget(size_t total_budget_bytes, const CacheConfig& config) {
    struct LogicalGroup {
        uint64_t span;
        size_t   block_bytes;
    };
    std::vector<LogicalGroup> logical_groups;
    uint64_t                  period             = 1;
    size_t                    fixed_budget_bytes = 0;

    for (const auto& group : config.topology().groups()) {
        RTP_LLM_CHECK_WITH_INFO(
            group.seq_size_per_block > 0, "kv cache tag=%s has zero physical seq_size_per_block", group.tag.c_str());
        const size_t block_bytes = config.blockSizeBytesForGroup(group.tag);
        if (group.policy.fixed_block_num > 0) {
            if (group.policy.charge_to_paged_budget) {
                fixed_budget_bytes =
                    checkedAdd(fixed_budget_bytes,
                               checkedMul(group.policy.fixed_block_num, block_bytes, "fixed group bytes"),
                               "fixed group budget");
            }
            continue;
        }
        RTP_LLM_CHECK_WITH_INFO(
            block_bytes > 0, "kv cache logical tag=%s has zero marginal block bytes", group.tag.c_str());
        logical_groups.push_back({group.seq_size_per_block, block_bytes});
        period = checkedLcm(period, group.seq_size_per_block);
    }
    RTP_LLM_CHECK_WITH_INFO(!logical_groups.empty(), "kv cache has no LOGICAL capacity group");
    RTP_LLM_CHECK_WITH_INFO(fixed_budget_bytes <= total_budget_bytes,
                            "kv cache budget %zu is smaller than fixed pool reservation %zu",
                            total_budget_bytes,
                            fixed_budget_bytes);

    size_t period_bytes = 0;
    for (const auto& group : logical_groups) {
        RTP_LLM_CHECK_WITH_INFO(period % group.span == 0, "invalid kv cache capacity period");
        period_bytes =
            checkedAdd(period_bytes,
                       checkedMul(static_cast<size_t>(period / group.span), group.block_bytes, "period group bytes"),
                       "period bytes");
    }
    RTP_LLM_CHECK_WITH_INFO(period_bytes > 0, "kv cache capacity period has zero marginal bytes");

    const size_t   available_budget = total_budget_bytes - fixed_budget_bytes;
    const uint64_t budget_periods   = available_budget / period_bytes;
    const uint64_t full_periods     = budget_periods;
    RTP_LLM_CHECK_WITH_INFO(full_periods <= std::numeric_limits<uint64_t>::max() / period,
                            "kv cache token capacity overflow");
    const uint64_t base_tokens      = full_periods * period;
    const size_t   base_bytes       = checkedMul(static_cast<size_t>(full_periods), period_bytes, "base period bytes");
    const size_t   remainder_budget = available_budget - base_bytes;

    constexpr size_t kMaxCapacityBoundaries = 1 << 20;
    size_t           boundary_count         = 1;
    for (const auto& group : logical_groups) {
        const uint64_t group_boundary_count = period / group.span;
        RTP_LLM_CHECK_WITH_INFO(group_boundary_count <= kMaxCapacityBoundaries - boundary_count,
                                "kv cache capacity period has too many boundaries: period=%lu span=%lu limit=%zu",
                                period,
                                group.span,
                                kMaxCapacityBoundaries);
        boundary_count += static_cast<size_t>(group_boundary_count);
    }
    std::vector<uint64_t> boundaries;
    boundaries.reserve(boundary_count + 1);
    boundaries.push_back(0);
    for (const auto& group : logical_groups) {
        for (uint64_t boundary = group.span;; boundary += group.span) {
            boundaries.push_back(boundary);
            if (boundary == period) {
                break;
            }
            RTP_LLM_CHECK_WITH_INFO(boundary < period && group.span <= period - boundary,
                                    "kv cache capacity boundary increment overflow");
        }
    }
    std::sort(boundaries.begin(), boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(), boundaries.end()), boundaries.end());
    uint64_t best = base_tokens;
    for (const uint64_t offset : boundaries) {
        RTP_LLM_CHECK_WITH_INFO(offset <= std::numeric_limits<uint64_t>::max() - base_tokens,
                                "kv cache boundary token overflow");
        const uint64_t candidate    = base_tokens + offset;
        size_t         offset_bytes = 0;
        for (const auto& group : logical_groups) {
            const uint64_t blocks = offset / group.span + (offset % group.span != 0 ? 1 : 0);
            offset_bytes =
                checkedAdd(offset_bytes,
                           checkedMul(static_cast<size_t>(blocks), group.block_bytes, "boundary group bytes"),
                           "boundary bytes");
        }
        if (offset_bytes <= remainder_budget) {
            best = candidate;
        }
    }
    RTP_LLM_CHECK_WITH_INFO(best > 0, "kv cache budget %zu cannot provide capacity for one token", total_budget_bytes);
    return best;
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
    const auto physical_seq_size_per_block = static_cast<uint32_t>(model_config.attn_config.tokens_per_block);
    const auto kernel_seq_size_per_block   = model_config.attn_config.kernel_tokens_per_block > 0 ?
                                                 static_cast<uint32_t>(model_config.attn_config.kernel_tokens_per_block) :
                                                 physical_seq_size_per_block;
    RTP_LLM_CHECK_WITH_INFO(physical_seq_size_per_block > 0, "basic cache seq_size_per_block must be > 0");
    RTP_LLM_CHECK_WITH_INFO(kernel_seq_size_per_block > 0
                                && physical_seq_size_per_block % kernel_seq_size_per_block == 0,
                            "basic cache seq_size_per_block(%u) must be divisible by kernel_seq_size_per_block(%u)",
                            physical_seq_size_per_block,
                            kernel_seq_size_per_block);

    CacheConfig config;
    if (model_config.hybrid_attention_config.enable_hybrid_attention) {
        config = HybridPoolConfigCreator::createConfig(model_config, parallelism_config, is_mtp, gen_num_per_cycle);
    } else {
        config = SingleConfigCreator::createSingleConfig(model_config, parallelism_config, is_mtp, gen_num_per_cycle);
    }
    validatePublishedCacheConfig(config);
    return config;
}

CacheConfig CacheConfigCreator::createConfig(const ModelConfig&                               model_config,
                                             const ParallelismConfig&                         parallelism_config,
                                             const RuntimeConfig&                             runtime_config,
                                             const KVCacheConfig&                             kv_cache_config,
                                             const std::optional<WarmUpResult>&               warm_up_result,
                                             const std::optional<SpeculativeExecutionConfig>& sp_config) {
    CacheConfig config = createBasicConfig(model_config, parallelism_config, false, 0);

    config.linear_step = kv_cache_config.linear_step;

    const uint64_t capacity_tokens = computeTokenCapacity(
        config, model_config, runtime_config, kv_cache_config, parallelism_config, warm_up_result, sp_config);
    RTP_LLM_CHECK_WITH_INFO(capacity_tokens > 0, "kv cache needs capacity for at least 1 token");
    config.applyTokenCapacity(capacity_tokens);
    RTP_LLM_LOG_INFO("kv cache joint capacity is %lu tokens", capacity_tokens);
    if (capacity_tokens < static_cast<uint64_t>(model_config.max_seq_len)) {
        RTP_LLM_LOG_WARNING("kv cache can only store %lu tokens, less than max_seq_len %ld, "
                            "this is dangerous, consider decrease max_seq_len",
                            capacity_tokens,
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
        createBasicConfig(score_model_config, parallelism_config, false, sp_config.gen_num_per_cycle);
    CacheConfig propose_config =
        createBasicConfig(propose_model_config, parallelism_config, is_mtp, sp_config.gen_num_per_cycle);

    score_config.linear_step   = kv_cache_config.linear_step;
    propose_config.linear_step = kv_cache_config.linear_step;

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

    CacheConfig config   = score_config;
    config.layer_all_num = score_config.layer_num;

    const uint32_t main_layer_num = score_config.layer_num;

    config.mtp_sub_configs.clear();
    config.mtp_sub_configs.reserve(num_mtp_modules);

    for (int m = 0; m < num_mtp_modules; ++m) {
        auto sub_cfg = config.mergeMTPModule(propose_config, m, main_layer_num);
        config.mtp_sub_configs.push_back(sub_cfg);
    }

    uint64_t capacity_tokens = 0;
    if (kv_cache_config.test_block_num > 0) {
        capacity_tokens = explicitTestCapacity(score_model_config, kv_cache_config);
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);
        capacity_tokens = maxKVCacheTokenCapacityForBudget(kv_cache_mem_size, config);
    }
    RTP_LLM_CHECK_WITH_INFO(capacity_tokens > 0, "SP kv cache needs capacity for at least 1 token");
    config.applyTokenCapacity(capacity_tokens);

    RTP_LLM_LOG_INFO("CacheConfig created: is_mtp=%d, total_layers=%u, num_mtp_modules=%d, "
                     "allows storing %lu tokens",
                     is_mtp,
                     total_layer_num,
                     num_mtp_modules,
                     capacity_tokens);

    RTP_LLM_LOG_INFO("CacheConfig debugString(main_score_model):\n%s", score_config.debugString().c_str());
    for (size_t i = 0; i < config.mtp_sub_configs.size(); ++i) {
        const auto& sub = config.mtp_sub_configs[i];
        RTP_LLM_LOG_INFO("CacheConfig debugString(sub_propose_model[%zu]):\n%s", i, sub->debugString().c_str());
    }

    return config;
}

}  // namespace rtp_llm
