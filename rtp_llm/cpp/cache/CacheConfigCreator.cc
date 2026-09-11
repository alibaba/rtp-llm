#include "rtp_llm/cpp/cache/CacheConfigCreator.h"

#include <numeric>
#include <algorithm>
#include <cstdlib>
#include <limits>

#include "rtp_llm/cpp/cache/HybridPoolConfigCreator.h"
#include "rtp_llm/cpp/cache/HybridConfigCreator.h"
#include "rtp_llm/cpp/cache/MemoryEvaluationHelper.h"
#include "rtp_llm/cpp/cache/SingleConfigCreator.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

namespace {

bool hasTypedHybridPoolLayout(const ModelConfig& model_config) {
    return !model_config.attn_config.layer_compress_ratios.empty();
}

bool shouldUseHybridPoolLayout(const ModelConfig& model_config) {
    return hasTypedHybridPoolLayout(model_config)
           || (model_config.hybrid_attention_config.enable_hybrid_attention
               && model_config.hybrid_attention_config.enable_independent_kv_cache_pools);
}

size_t steppedBytes(size_t bytes, int step) {
    return (bytes > 0 && step > 1) ? bytes / static_cast<size_t>(step) : bytes;
}

size_t fallbackFixedPoolHbmBytes(const CacheConfig& config) {
    if (config.fixed_pool_uses_pinned_cpu) {
        return 0u;
    }
    if (config.use_typed_cache_regions && !config.group_region_names.empty()
        && config.group_region_names.size() == config.group_block_size_bytes.size()) {
        size_t bytes = 0;
        for (size_t gid = 0; gid < config.group_region_names.size(); ++gid) {
            const auto region = config.group_region_names[gid];
            if (!isDsv4FixedRegion(region)) {
                continue;
            }
            const bool explicit_hca = region == KVCacheRegionName::HCA_STATE && config.dsv4_hca_state_pool_blocks > 0;
            const bool explicit_fixed = config.dsv4_fixed_pool_blocks > 0;
            if (!explicit_hca && !explicit_fixed) {
                bytes += config.group_block_size_bytes[gid];
            }
        }
        return bytes;
    }
    return config.swa_block_size_bytes + config.state_block_size_bytes;
}

size_t effectivePagedBlockBytes(const CacheConfig& config, int step) {
    return config.block_size_bytes + steppedBytes(fallbackFixedPoolHbmBytes(config), step);
}

bool hasDsv4FixedPoolBytes(const CacheConfig& config) {
    return config.state_block_size_bytes > 0 || config.swa_block_size_bytes > 0;
}

void validateDsv4KernelSeqSize(size_t seq_size_per_block, size_t kernel_seq_size_per_block, const char* config_name) {
    constexpr size_t kDsv4KernelSeqSizeAlignment = 128;
    RTP_LLM_CHECK_WITH_INFO(kernel_seq_size_per_block >= kDsv4KernelSeqSizeAlignment
                                && kernel_seq_size_per_block % kDsv4KernelSeqSizeAlignment == 0,
                            "%s DSV4 kernel_seq_size_per_block(%zu) must be >= %zu and a multiple of %zu",
                            config_name,
                            kernel_seq_size_per_block,
                            kDsv4KernelSeqSizeAlignment,
                            kDsv4KernelSeqSizeAlignment);
    RTP_LLM_CHECK_WITH_INFO(seq_size_per_block >= kernel_seq_size_per_block
                                && seq_size_per_block % kernel_seq_size_per_block == 0,
                            "%s DSV4 seq_size_per_block(%zu) must be >= kernel_seq_size_per_block(%zu) "
                            "and divisible by it",
                            config_name,
                            seq_size_per_block,
                            kernel_seq_size_per_block);
}

size_t nonnegativeEnv(const char* name) {
    const char* value = std::getenv(name);
    if (!value || !*value) {
        return 0;
    }
    const std::string text(value);
    RTP_LLM_CHECK_WITH_INFO(std::all_of(text.begin(), text.end(), [](char c) { return c >= '0' && c <= '9'; }),
                            "%s must be a nonnegative integer", name);
    return std::stoull(text);
}

// Keep wire/block strides unchanged. Only residency and logical capacity
// change; connectors still address each layer's KV and indexer tensors.
void configurePinnedMla(CacheConfig& config, const RuntimeConfig& runtime, size_t topk, size_t query_tokens) {
    const size_t host_mb = nonnegativeEnv("RTP_LLM_DSA_MLA_HOST_CACHE_MB");
    if (!host_mb) {
        return;
    }
    RTP_LLM_CHECK_WITH_INFO(config.use_mla && config.is_sparse && config.groupNums() == 1
                                && !config.use_independent_block_pools && topk > 0,
                            "pinned MLA working set requires a non-hybrid DSA MLA model");
    RTP_LLM_CHECK_WITH_INFO(host_mb <= std::numeric_limits<size_t>::max() / (1024 * 1024),
                            "pinned MLA host budget overflows size_t");
    size_t scale_bytes = config.kv_scale_size_bytes;
    auto group_count = [](const CacheConfig& cfg) -> size_t {
        return cfg.layer_to_indexer_kv_slot.empty() ? cfg.layer_num :
            1 + *std::max_element(cfg.layer_to_indexer_kv_slot.begin(), cfg.layer_to_indexer_kv_slot.end());
    };
    size_t groups = group_count(config);
    for (const auto& sub : config.mtp_sub_configs) {
        scale_bytes += sub->kv_scale_size_bytes;
        groups += group_count(*sub);
    }
    const size_t tokens_per_block = config.seq_size_per_block;
    const size_t mla_bytes = (config.block_size_bytes - scale_bytes) / tokens_per_block;
    const size_t resident_bytes = mla_bytes + groups * 20;
    // Optional GPU snapshots of the allocator generations cost one int64 per
    // logical block and shared-index group in addition to token maps.
    const size_t generation_block_bytes =
        nonnegativeEnv("RTP_LLM_DSA_MLA_GENERATION_SNAPSHOT") == 1 ? groups * sizeof(int64_t) : 0;
    const size_t logical_block_hbm = scale_bytes + groups * tokens_per_block * 8 + generation_block_bytes;
    const size_t budget = static_cast<size_t>(config.block_num) * config.block_size_bytes;
    const size_t requested = nonnegativeEnv("RTP_LLM_DSA_MLA_RESIDENT_TOKENS");
    const size_t selected = static_cast<size_t>(runtime.max_generate_batch_size) * topk * query_tokens;
    RTP_LLM_CHECK_WITH_INFO(!requested || requested >= selected,
                            "pinned MLA resident tokens must cover the maximum decode/verify topk batch (%zu)", selected);
    RTP_LLM_CHECK_WITH_INFO(requested % tokens_per_block == 0,
                            "pinned MLA resident tokens must be a multiple of %zu", tokens_per_block);
    const size_t minimum = requested ? requested :
        std::max<size_t>(1, (selected + tokens_per_block - 1) / tokens_per_block) * tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(budget > groups * 4096
                                && minimum < (budget - groups * 4096) / resident_bytes,
                            "pinned MLA HBM budget cannot hold the configured decode batch working set");
    // Reserve the worst-case working set and the overflow Indexer's HBM first.
    // Spend the remaining HBM on complete MLA blocks, never a host mirror.
    const size_t mla_block_bytes = mla_bytes * tokens_per_block;
    const size_t remaining = budget - minimum * resident_bytes - groups * 4096;
    const size_t hbm_block_bytes = mla_block_bytes + logical_block_hbm;
    // Block zero is reserved; keep at least one usable complete HBM block.
    RTP_LLM_CHECK_WITH_INFO(remaining > 2 * hbm_block_bytes, "no HBM space for complete MLA blocks");
    const size_t pin_blocks = std::min(host_mb * 1024 * 1024 / mla_block_bytes,
                                       (remaining - 2 * hbm_block_bytes) / logical_block_hbm);
    const size_t hbm_blocks = (remaining - pin_blocks * logical_block_hbm) / hbm_block_bytes;
    const size_t blocks = hbm_blocks + pin_blocks;
    constexpr size_t max_tokens = std::numeric_limits<int32_t>::max();
    RTP_LLM_CHECK_WITH_INFO(pin_blocks > 0 && hbm_blocks > 1
                                && blocks <= max_tokens / tokens_per_block,
                            "invalid tiered MLA capacity: HBM=%zu pin=%zu", hbm_blocks, pin_blocks);
    // Physical IDs also include the working set after the complete HBM blocks.
    // Check before multiplication/addition so the validation cannot overflow.
    RTP_LLM_CHECK_WITH_INFO(minimum <= max_tokens
                                && hbm_blocks <= (max_tokens - minimum) / tokens_per_block,
                            "tiered MLA physical token capacity must fit int32: HBM blocks=%zu resident tokens=%zu",
                            hbm_blocks, minimum);
    config.dsa_mla_resident_tokens = minimum;
    config.dsa_mla_hbm_blocks = hbm_blocks;
    config.block_num = static_cast<uint32_t>(blocks);
    config.finalizeBlockNums(config.block_num, runtime);
    for (auto& sub : config.mtp_sub_configs) {
        sub->block_num = config.block_num;
        sub->dsa_mla_resident_tokens = minimum;
        sub->dsa_mla_hbm_blocks = hbm_blocks;
        sub->finalizeBlockNums(config.block_num, runtime);
    }
    RTP_LLM_LOG_INFO("DSA MLA tiered cache: hbm_blocks=%zu pinned_blocks=%zu logical_tokens=%zu "
                     "resident_tokens=%zu pinned_bytes=%zu indexer_hbm_bytes=%zu "
                     "working_set_hbm_bytes=%zu full_block_hbm_bytes=%zu budget=%zu",
                     hbm_blocks, pin_blocks, blocks * tokens_per_block, minimum,
                     pin_blocks * mla_block_bytes, blocks * scale_bytes,
                     minimum * resident_bytes, hbm_blocks * mla_block_bytes, budget);
}

}  // namespace

CacheConfig CacheConfigCreator::createBasicConfig(const ModelConfig&       model_config,
                                                  const ParallelismConfig& parallelism_config,
                                                  const KVCacheConfig&     kv_cache_config,
                                                  bool                     is_mtp,
                                                  int                      gen_num_per_cycle) {
    if (shouldUseHybridPoolLayout(model_config)) {
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
    uint32_t block_num = 0;

    config.linear_step = kv_cache_config.linear_step;
    if (kv_cache_config.kernel_seq_size_per_block > 0) {
        const auto kernel_seq_size_per_block = static_cast<size_t>(kv_cache_config.kernel_seq_size_per_block);
        if (hasTypedHybridPoolLayout(model_config)) {
            validateDsv4KernelSeqSize(config.seq_size_per_block, kernel_seq_size_per_block, "cache");
        } else {
            RTP_LLM_CHECK_WITH_INFO(kv_cache_config.seq_size_per_block % kv_cache_config.kernel_seq_size_per_block == 0,
                                    "seq_size_per_block(%d) must be divisible by kernel_seq_size_per_block(%d)",
                                    kv_cache_config.seq_size_per_block,
                                    kv_cache_config.kernel_seq_size_per_block);
        }
        config.kernel_seq_size_per_block = kernel_seq_size_per_block;
    } else if (config.kernel_seq_size_per_block == 0 || config.kernel_seq_size_per_block == config.seq_size_per_block) {
        // Default: kernel block size == physical block size (no split). Keep
        // any explicit value already set by createBasicConfig (e.g. DSV4 forces
        // kernel_seq_size_per_block = 256 even when physical seq_size > 256).
        config.kernel_seq_size_per_block = config.seq_size_per_block;
    }

    // DSV4 fixed-pool residency toggle must be set before the pre-pass
    // finalizeBlockNums so CPU-backed STATE/SWA_KV bytes are excluded from HBM.
    config.fixed_pool_uses_pinned_cpu = kv_cache_config.dsv4_fixed_pool_use_memory && hasDsv4FixedPoolBytes(config);

    if (kv_cache_config.test_block_num > 0) {
        RTP_LLM_LOG_INFO("KVCacheConfig explicitly specified kv cache block num %d", kv_cache_config.test_block_num);
        block_num = kv_cache_config.test_block_num;
        config.finalizeBlockNums(block_num, runtime_config);
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, model_config, parallelism_config, warm_up_result, sp_config);
        // Fixed-pool sizing depends on runtime scheduler limits, so finalize it
        // here after RuntimeConfig is available. The temporary global block
        // number is only used for groups that are not fixed pools.
        config.finalizeBlockNums(0, runtime_config);
        // Deduct fixed-pool reservation from the budget so paged pools don't overcommit HBM.
        size_t paged_budget = kv_cache_mem_size;
        if (config.fixed_pool_reserve_bytes > 0) {
            RTP_LLM_CHECK_WITH_INFO(kv_cache_mem_size > config.fixed_pool_reserve_bytes,
                                    "kv cache budget %zu MiB is smaller than fixed-pool reservation %zu MiB "
                                    "(DSV4_FIXED_POOL_BLOCKS=%u, DSV4_HCA_STATE_POOL_BLOCKS=%u; reduce it if needed)",
                                    kv_cache_mem_size / 1024 / 1024,
                                    config.fixed_pool_reserve_bytes / 1024 / 1024,
                                    config.dsv4_fixed_pool_blocks,
                                    config.dsv4_hca_state_pool_blocks);
            paged_budget = kv_cache_mem_size - config.fixed_pool_reserve_bytes;
            RTP_LLM_LOG_INFO("kv cache: total budget %zu MiB, fixed-pool reserve %zu MiB, paged budget %zu MiB",
                             kv_cache_mem_size / 1024 / 1024,
                             config.fixed_pool_reserve_bytes / 1024 / 1024,
                             paged_budget / 1024 / 1024);
        }
        const int  joint_step       = std::max(1, config.linear_step);
        block_num = paged_budget / effectivePagedBlockBytes(config, joint_step);
    }
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
    configurePinnedMla(config, runtime_config, model_config.attn_config.indexer_topk, 1);
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

    if (kv_cache_config.kernel_seq_size_per_block > 0) {
        const size_t kernel_seq_size_per_block = static_cast<size_t>(kv_cache_config.kernel_seq_size_per_block);
        if (hasTypedHybridPoolLayout(score_model_config)) {
            validateDsv4KernelSeqSize(score_config.seq_size_per_block, kernel_seq_size_per_block, "score");
        } else {
            RTP_LLM_CHECK_WITH_INFO(score_config.seq_size_per_block % kernel_seq_size_per_block == 0,
                                    "score seq_size_per_block(%zu) must be divisible by kernel_seq_size_per_block(%zu)",
                                    score_config.seq_size_per_block,
                                    kernel_seq_size_per_block);
        }
        if (hasTypedHybridPoolLayout(propose_model_config)) {
            validateDsv4KernelSeqSize(propose_config.seq_size_per_block, kernel_seq_size_per_block, "propose");
        } else {
            RTP_LLM_CHECK_WITH_INFO(
                propose_config.seq_size_per_block % kernel_seq_size_per_block == 0,
                "propose seq_size_per_block(%zu) must be divisible by kernel_seq_size_per_block(%zu)",
                propose_config.seq_size_per_block,
                kernel_seq_size_per_block);
        }
        score_config.kernel_seq_size_per_block   = kernel_seq_size_per_block;
        propose_config.kernel_seq_size_per_block = kernel_seq_size_per_block;
    } else {
        // Default: kernel block size == physical block size (no split). Keep
        // any explicit value already set by createBasicConfig (e.g. DSV4
        // forces kernel_seq_size_per_block = 256 even when physical
        // seq_size_per_block > 256); only fill in when unset or already
        // matches the physical block.
        if (score_config.kernel_seq_size_per_block == 0
            || score_config.kernel_seq_size_per_block == score_config.seq_size_per_block) {
            score_config.kernel_seq_size_per_block = score_config.seq_size_per_block;
        }
        if (propose_config.kernel_seq_size_per_block == 0
            || propose_config.kernel_seq_size_per_block == propose_config.seq_size_per_block) {
            propose_config.kernel_seq_size_per_block = propose_config.seq_size_per_block;
        }
    }

    int num_mtp_modules = 1;
    if (is_mtp) {
        num_mtp_modules = sp_config.gen_num_per_cycle;
        if (is_eagle) {
            num_mtp_modules = 1;
        }
    }

    // DSV4 fixed-pool residency toggle (mirror createConfig) must precede
    // pre-pass so fixed_pool_reserve_bytes accounts for CPU-backed regions.
    score_config.fixed_pool_uses_pinned_cpu =
        kv_cache_config.dsv4_fixed_pool_use_memory && hasDsv4FixedPoolBytes(score_config);
    propose_config.fixed_pool_uses_pinned_cpu =
        kv_cache_config.dsv4_fixed_pool_use_memory && hasDsv4FixedPoolBytes(propose_config);

    // Fixed-pool block counts depend on runtime scheduler limits. Finalize the
    // score and propose configs before sizing the shared paged budget so DSV4
    // state/SWA pools are accounted outside the paged KV-cache block budget.
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

    const size_t fixed_reserve = score_config.fixed_pool_reserve_bytes
                                 + propose_config.fixed_pool_reserve_bytes * static_cast<size_t>(num_mtp_modules);

    size_t block_num = 0;
    if (kv_cache_config.test_block_num > 0) {
        block_num = kv_cache_config.test_block_num;
    } else {
        const auto kv_cache_mem_size = MemoryEvaluationHelper::getKVCacheMemorySize(
            runtime_config, kv_cache_config, score_model_config, parallelism_config, warm_up_result, sp_config);

        size_t paged_budget = kv_cache_mem_size;
        if (fixed_reserve > 0) {
            RTP_LLM_CHECK_WITH_INFO(kv_cache_mem_size > fixed_reserve,
                                    "sp kv cache budget %zu MiB is smaller than fixed-pool reservation %zu MiB "
                                    "(DSV4_FIXED_POOL_BLOCKS; reduce it if needed)",
                                    kv_cache_mem_size / 1024 / 1024,
                                    fixed_reserve / 1024 / 1024);
            paged_budget = kv_cache_mem_size - fixed_reserve;
            RTP_LLM_LOG_INFO(
                "sp kv cache: total budget %zu MiB, fixed-pool reserve %zu MiB (score=%zu MiB + propose=%zu MiB x %d), paged budget %zu MiB",
                kv_cache_mem_size / 1024 / 1024,
                fixed_reserve / 1024 / 1024,
                score_config.fixed_pool_reserve_bytes / 1024 / 1024,
                propose_config.fixed_pool_reserve_bytes / 1024 / 1024,
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
    config.fixed_pool_reserve_bytes = fixed_reserve;

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
    config.layer_group_types.resize(total_layer_num, CacheGroupType::FULL);
    config.layer_to_block_stride_bytes.assign(static_cast<size_t>(total_layer_num), 0);
    const size_t region_name_count = static_cast<size_t>(KVCacheRegionName::REGION_COUNT);
    if (!config.layer_region_to_group_id.empty()) {
        const size_t prev = config.layer_region_to_group_id.size();
        config.layer_region_to_group_id.resize(static_cast<size_t>(total_layer_num));
        for (size_t l = prev; l < static_cast<size_t>(total_layer_num); ++l) {
            config.layer_region_to_group_id[l].assign(region_name_count, -1);
        }
    }
    if (!config.layer_to_group_ids.empty()) {
        config.layer_to_group_ids.resize(static_cast<size_t>(total_layer_num));
    }

    // Main(score) model per-layer stride (kv + scale).
    // This is expected to be fully populated by createBasicConfig() (Single/Hybrid creators).
    const size_t score_layers = static_cast<size_t>(main_layer_num);
    RTP_LLM_CHECK_WITH_INFO(score_config.layer_to_block_stride_bytes.size() == score_layers,
                            "score_config.layer_to_block_stride_bytes size mismatch, got=%zu need=%zu",
                            score_config.layer_to_block_stride_bytes.size(),
                            score_layers);
    for (size_t l = 0; l < score_layers; ++l) {
        config.layer_to_block_stride_bytes[l] = score_config.layer_to_block_stride_bytes[l];
        if (l < score_config.layer_group_types.size()) {
            config.layer_group_types[l] = score_config.layer_group_types[l];
        }
    }

    for (int m = 0; m < num_mtp_modules; ++m) {
        auto sub_cfg           = std::make_shared<CacheConfig>(propose_config);
        sub_cfg->block_num     = block_num;
        sub_cfg->layer_all_num = sub_cfg->layer_num;

        const std::vector<std::vector<int>> propose_per_group = propose_config.global_layer_ids;
        sub_cfg->global_layer_ids.assign(propose_per_group.size(), {});
        RTP_LLM_CHECK_WITH_INFO(sub_cfg->layer_to_block_stride_bytes.size() == static_cast<size_t>(mtp_layer_num),
                                "sub_cfg.layer_to_block_stride_bytes size mismatch, got=%zu need=%u",
                                sub_cfg->layer_to_block_stride_bytes.size(),
                                mtp_layer_num);
        for (size_t g = 0; g < propose_per_group.size(); ++g) {
            for (int local_lid : propose_per_group[g]) {
                if (local_lid < 0 || local_lid >= static_cast<int>(mtp_layer_num)) {
                    continue;
                }
                const int global_layer_id = main_layer_num + m * mtp_layer_num + local_lid;
                sub_cfg->global_layer_ids[g].push_back(global_layer_id);

                // Keep the propose model's group placement. DSV4 MTP is
                // SWA-only and lives in the SWA typed pool, not the first FULL
                // pool. Non-typed hybrid configs fall back to the full group.
                const int target_gid =
                    (g < config.global_layer_ids.size()) ? static_cast<int>(g) : static_cast<int>(full_gid);
                config.layer_to_group_id[global_layer_id] = target_gid;
                if (target_gid >= 0 && target_gid < static_cast<int>(config.global_layer_ids.size())) {
                    config.global_layer_ids[static_cast<size_t>(target_gid)].push_back(global_layer_id);
                }
                if (!config.layer_to_group_ids.empty()
                    && static_cast<size_t>(global_layer_id) < config.layer_to_group_ids.size()) {
                    config.layer_to_group_ids[static_cast<size_t>(global_layer_id)].push_back(target_gid);
                }
                if (!config.layer_region_to_group_id.empty()
                    && static_cast<size_t>(global_layer_id) < config.layer_region_to_group_id.size()
                    && g < propose_config.group_region_names.size()) {
                    const auto region = static_cast<size_t>(propose_config.group_region_names[g]);
                    if (region < config.layer_region_to_group_id[static_cast<size_t>(global_layer_id)].size()) {
                        config.layer_region_to_group_id[static_cast<size_t>(global_layer_id)][region] = target_gid;
                    }
                }
                if (target_gid >= 0 && static_cast<size_t>(target_gid) < config.group_block_size_bytes.size()) {
                    size_t stride_bytes = 0;
                    if (g < propose_config.group_kv_block_stride_bytes.size()) {
                        stride_bytes += propose_config.group_kv_block_stride_bytes[g];
                    }
                    if (g < propose_config.group_kv_scale_stride_bytes.size()) {
                        stride_bytes += propose_config.group_kv_scale_stride_bytes[g];
                    }
                    config.group_block_size_bytes[static_cast<size_t>(target_gid)] += stride_bytes;
                }

                const int stride_bytes = sub_cfg->layer_to_block_stride_bytes[static_cast<size_t>(local_lid)];
                config.layer_to_block_stride_bytes[static_cast<size_t>(global_layer_id)] = stride_bytes;
                if (static_cast<size_t>(local_lid) < sub_cfg->layer_group_types.size()) {
                    config.layer_group_types[static_cast<size_t>(global_layer_id)] =
                        sub_cfg->layer_group_types[static_cast<size_t>(local_lid)];
                }
            }
        }

        sub_cfg->layer_to_group_id.assign(static_cast<size_t>(sub_cfg->layer_num), -1);
        for (size_t g = 0; g < propose_per_group.size(); ++g) {
            for (int local_lid : propose_per_group[g]) {
                if (local_lid >= 0 && static_cast<size_t>(local_lid) < sub_cfg->layer_to_group_id.size()) {
                    sub_cfg->layer_to_group_id[static_cast<size_t>(local_lid)] = static_cast<int>(g);
                }
            }
        }
        sub_cfg->finalizeBlockNums(static_cast<uint32_t>(block_num), runtime_config);
        config.mtp_sub_configs.push_back(sub_cfg);
    }

    config.finalizeBlockNums(static_cast<uint32_t>(block_num), runtime_config);
    config.fixed_pool_reserve_bytes = fixed_reserve;

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

    configurePinnedMla(config, runtime_config, score_model_config.attn_config.indexer_topk,
                       static_cast<size_t>(sp_config.gen_num_per_cycle) + 1);
    return config;
}

}  // namespace rtp_llm
