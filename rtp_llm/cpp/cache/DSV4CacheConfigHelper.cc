#include "rtp_llm/cpp/cache/DSV4CacheConfigHelper.h"

#include "rtp_llm/cpp/cache/DSV4KVCacheSpec.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <cstdlib>
#include <string>

namespace rtp_llm {

namespace {

// Kernel block size in tokens. DSV4 typed pools keep one logical
// seq_size_per_block for cache-key/block-table alignment, while FULL paged
// pools may use a larger internal physical entry count to satisfy HCA's
// 128-token compression unit.
constexpr uint32_t kDsv4KernelTokensPerBlock    = 256;
constexpr uint32_t kDsv4MinKernelTokensPerBlock = 128;
// SWA sliding-window length in tokens (SWA_KV ring base size; see swa_kv_eb).
constexpr uint32_t kDsv4SwaWindowEntries = 128;
// BF16 pool: head_dim=512 x 2B = 1024B per KV slot, 128 x 2B = 256B per
// indexer slot. FP8 pool packs the same logical KV into a smaller slot
// (canonical fp8_model1_mla layout: 448B fp8 NoPE + 64B bf16 RoPE + 8B
// UE8M0 scales = 584B; indexer is 128B fp8 + 4B fp32 scale = 132B).
// Selected at runtime from ``attn_config.kv_cache_dtype`` -- see
// ``buildDSV4PoolDescs``.
constexpr uint32_t kDsv4KvEntryBytesBf16         = 1024;
constexpr uint32_t kDsv4IndexerEntryBytesBf16    = 256;
constexpr uint32_t kDsv4KvEntryBytesFp8          = 584;
constexpr uint32_t kDsv4IndexerEntryBytesFp8     = 132;
constexpr uint32_t kDsv4AtomIndexerEntryBytesFp8 = 144;
constexpr size_t   kDsv4PoolNum                  = 7;

enum class DSV4IndexerKVLayout {
    BF16,
    RTP_FP8,
    ROCM_ATOM_FP8,
};

struct DSV4LayerSets {
    std::vector<int> csa_layers;
    std::vector<int> hca_layers;
    std::vector<int> swa_only_layers;
    std::vector<int> all_layers;
};

struct DSV4PoolDesc {
    KVCacheRegionName       region_name;
    const std::vector<int>* layer_ids;
    uint32_t                entry_elems;
    uint32_t                entries_per_block;
    uint32_t                tokens_per_block;
    DataType                store_dtype;
    bool                    is_paged;
    size_t                  block_size_bytes_override = 0;
};

constexpr int kCsaCompressRatio     = 4;
constexpr int kHcaCompressRatio     = 128;
constexpr int kIndexerCompressRatio = 4;
constexpr int kCsaOverlap           = 1;
constexpr int kHcaOverlap           = 0;
constexpr int kIndexerOverlap       = 1;

const char* dsv4RegionName(KVCacheRegionName region_name) {
    switch (region_name) {
        case KVCacheRegionName::DEFAULT:
            return "DEFAULT";
        case KVCacheRegionName::CSA_KV:
            return "CSA_KV";
        case KVCacheRegionName::HCA_KV:
            return "HCA_KV";
        case KVCacheRegionName::INDEXER_KV:
            return "INDEXER_KV";
        case KVCacheRegionName::INDEXER_STATE:
            return "INDEXER_STATE";
        case KVCacheRegionName::CSA_STATE:
            return "CSA_STATE";
        case KVCacheRegionName::HCA_STATE:
            return "HCA_STATE";
        case KVCacheRegionName::SWA_KV:
            return "SWA_KV";
        case KVCacheRegionName::REGION_COUNT:
            return "REGION_COUNT";
    }
    return "UNKNOWN";
}

const char* dsv4IndexerKVLayoutName(DSV4IndexerKVLayout layout) {
    switch (layout) {
        case DSV4IndexerKVLayout::BF16:
            return "BF16";
        case DSV4IndexerKVLayout::RTP_FP8:
            return "RTP_FP8_132B";
        case DSV4IndexerKVLayout::ROCM_ATOM_FP8:
            return "ROCM_ATOM_FP8_144B";
    }
    return "UNKNOWN";
}

bool externalModelPackagesContainAtomPlugin() {
    const char* packages = std::getenv("RTP_LLM_EXTERNAL_MODEL_PACKAGES");
    return packages != nullptr && std::string(packages).find("atom.plugin.rtpllm.models") != std::string::npos;
}

bool isDeepSeekV4Model(const ModelConfig& model_config) {
    return model_config.model_type == "deepseek_v4";
}

DSV4IndexerKVLayout resolveIndexerKVLayout(const ModelConfig&     model_config,
                                           const AttentionConfigs& attn,
                                           const KVCacheConfig&    kv_cache_config) {
    if (attn.kv_cache_dtype == KvCacheDataType::FP8) {
        if (kv_cache_config.rocm_atom_dsv4_indexer_fp8_kv_cache) {
            RTP_LLM_LOG_WARNING("ROCm ATOM DSV4 144B INDEXER_KV request is ignored because global fp8_kv_cache is "
                                "enabled; using RTP native 132B FP8 indexer layout.");
        }
        return DSV4IndexerKVLayout::RTP_FP8;
    }

    if (!kv_cache_config.rocm_atom_dsv4_indexer_fp8_kv_cache) {
        return DSV4IndexerKVLayout::BF16;
    }

#if defined(USING_ROCM)
    RTP_LLM_CHECK_WITH_INFO(isDeepSeekV4Model(model_config),
                            "ROCm ATOM DSV4 144B INDEXER_KV requires model_type=deepseek_v4, got %s",
                            model_config.model_type.c_str());
    RTP_LLM_CHECK_WITH_INFO(externalModelPackagesContainAtomPlugin(),
                            "ROCm ATOM DSV4 144B INDEXER_KV requires RTP_LLM_EXTERNAL_MODEL_PACKAGES to contain "
                            "atom.plugin.rtpllm.models");
    return DSV4IndexerKVLayout::ROCM_ATOM_FP8;
#else
    RTP_LLM_CHECK_WITH_INFO(false,
                            "rocm_atom_dsv4_indexer_fp8_kv_cache is ROCm ATOM plugin only and must not be enabled "
                            "on CUDA/NV native DSV4");
    return DSV4IndexerKVLayout::BF16;
#endif
}

uint32_t indexerKVEntryBytesForLayout(DSV4IndexerKVLayout layout) {
    switch (layout) {
        case DSV4IndexerKVLayout::BF16:
            return kDsv4IndexerEntryBytesBf16;
        case DSV4IndexerKVLayout::RTP_FP8:
            return kDsv4IndexerEntryBytesFp8;
        case DSV4IndexerKVLayout::ROCM_ATOM_FP8:
            return kDsv4AtomIndexerEntryBytesFp8;
    }
    return kDsv4IndexerEntryBytesBf16;
}

inline uint32_t alignUpToMultiple(uint32_t value, uint32_t multiple) {
    RTP_LLM_CHECK_WITH_INFO(multiple > 0, "DSV4 align multiple must be > 0");
    return ((value + multiple - 1) / multiple) * multiple;
}

inline uint32_t computeStateRing(int compress_ratio, int overlap, int gen_num_per_cycle) {
    RTP_LLM_CHECK_WITH_INFO(
        gen_num_per_cycle >= 0, "DSV4 state ring: gen_num_per_cycle must be >= 0, got %d", gen_num_per_cycle);
    const int window = (1 + overlap) * compress_ratio;
    const int raw    = window + gen_num_per_cycle;
    return static_cast<uint32_t>((raw + 1) & ~1);
}

uint32_t fixedRegionCpSize(const ParallelismConfig& parallelism_config) {
    if (!parallelism_config.prefill_cp_config.kv_cache_sharded) {
        return 1;
    }
    if (parallelism_config.role_type == RoleType::PREFILL && parallelism_config.tp_size > 1) {
        return static_cast<uint32_t>(parallelism_config.tp_size);
    }
    if (parallelism_config.role_type == RoleType::DECODE && parallelism_config.prefill_cp_config.is_prefill_enabled()) {
        RTP_LLM_CHECK_WITH_INFO(
            parallelism_config.prefill_cp_config.prefill_cp_size > 1,
            "DSV4 fixed/SWA CP sharding decode requires explicit prefill_cp_size when PREFILL_CP and kv_cache_sharded are enabled");
        return static_cast<uint32_t>(parallelism_config.prefill_cp_config.prefill_cp_size);
    }
    return 1;
}

bool isPrefillCpSliced(const ParallelismConfig& parallelism_config) {
    return parallelism_config.role_type == RoleType::PREFILL && fixedRegionCpSize(parallelism_config) > 1;
}

uint32_t maybeAdjustFixedEntriesForCpSharding(uint32_t                 entries,
                                              const ParallelismConfig& parallelism_config,
                                              KVCacheRegionName        region_name) {
    const auto cp_size = fixedRegionCpSize(parallelism_config);
    if (cp_size <= 1) {
        return entries;
    }

    // CP-aligned entries are the real ring capacity, not dead padding. Fixed
    // STATE rings stay entry-sliced under CP; SWA_KV uses separate physical-byte
    // slicing because its FP8 layout has a data/scale/padding block stride.
    const auto ring_capacity_entries = alignUpToMultiple(entries, cp_size);
    const bool prefill_sliced        = isPrefillCpSliced(parallelism_config);
    const auto entries_per_block     = prefill_sliced ? ring_capacity_entries / cp_size : ring_capacity_entries;
    RTP_LLM_LOG_INFO("DSV4 fixed STATE CP sharding region=%s(%d) min_entries=%u ring_capacity_entries=%u "
                     "entries_per_block=%u cp_size=%u prefill_sliced=%d expanded=%d role=%d",
                     dsv4RegionName(region_name),
                     static_cast<int>(region_name),
                     entries,
                     ring_capacity_entries,
                     entries_per_block,
                     cp_size,
                     prefill_sliced,
                     ring_capacity_entries != entries,
                     static_cast<int>(parallelism_config.role_type));
    return entries_per_block;
}

uint32_t maybeAdjustSwaEntriesForCpSharding(uint32_t entries, const ParallelismConfig& parallelism_config) {
    const auto cp_size = fixedRegionCpSize(parallelism_config);
    if (cp_size <= 1) {
        return entries;
    }
    return alignUpToMultiple(entries, cp_size);
}

size_t maybeSwaPrefillCpByteSliceBytes(uint32_t entries_per_block, const ParallelismConfig& parallelism_config) {
    const auto cp_size = fixedRegionCpSize(parallelism_config);
    if (cp_size <= 1 || !isPrefillCpSliced(parallelism_config)) {
        return 0;
    }
    const size_t full_natural_bytes = static_cast<size_t>(entries_per_block) * DSV4_FP8_KV_ENTRY_BYTES;
    const size_t full_stride_bytes  = alignDsv4Fp8KvBlockBytes(full_natural_bytes, cp_size);
    RTP_LLM_CHECK_WITH_INFO(full_stride_bytes % cp_size == 0,
                            "DSV4 SWA_KV full stride %zu must be divisible by cp_size %u",
                            full_stride_bytes,
                            cp_size);
    return full_stride_bytes / cp_size;
}

DSV4LayerSets classifyDSV4Layers(const std::vector<int>& compress_ratios) {
    DSV4LayerSets sets;
    const size_t  num_layers = compress_ratios.size();

    for (size_t i = 0; i < num_layers; ++i) {
        const int layer_id = static_cast<int>(i);
        const int ratio    = compress_ratios[i];
        sets.all_layers.push_back(layer_id);
        if (ratio == 4) {
            sets.csa_layers.push_back(layer_id);
        } else if (ratio == 128) {
            sets.hca_layers.push_back(layer_id);
        } else if (ratio == 0) {
            sets.swa_only_layers.push_back(layer_id);
        } else {
            RTP_LLM_LOG_WARNING("Unknown DSV4 compress_ratio %d at layer %zu, treating as HCA", ratio, i);
            sets.hca_layers.push_back(layer_id);
        }
    }

    RTP_LLM_LOG_INFO("DSV4 layer classification: %zu total, %zu CSA, %zu HCA, %zu SWA-only",
                     sets.all_layers.size(),
                     sets.csa_layers.size(),
                     sets.hca_layers.size(),
                     sets.swa_only_layers.size());
    return sets;
}

std::vector<DSV4PoolDesc> buildDSV4PoolDescs(const DSV4LayerSets&     sets,
                                             const ModelConfig&       model_config,
                                             const KVCacheConfig&     kv_cache_config,
                                             uint32_t                 kernel_tokens_per_block,
                                             uint32_t                 physical_tokens_per_block,
                                             const ParallelismConfig& parallelism_config,
                                             int                      gen_num_per_cycle) {
    const auto& attn         = model_config.attn_config;
    const auto  head_dim     = static_cast<uint32_t>(attn.size_per_head);
    const auto  idx_head_dim = static_cast<uint32_t>(attn.indexer_head_dim);

    const uint32_t idx_state_dim = 2 * idx_head_dim;
    const uint32_t csa_state_dim = 2 * head_dim;
    const uint32_t hca_state_dim = head_dim;

    const bool     fp8_kv         = (attn.kv_cache_dtype == KvCacheDataType::FP8);
    const uint32_t kv_entry_bytes = fp8_kv ? kDsv4KvEntryBytesFp8 : kDsv4KvEntryBytesBf16;
    // The V4 MTP draft is a single SWA-only layer and has no INDEXER_KV
    // consumers.  Do not validate or select the target-only ATOM 144B indexer
    // layout for an empty CSA pool: the global server option is shared by the
    // target and draft ModelConfigs, while the draft model_type is
    // deepseek_v4_mtp by design.
    const auto indexer_layout = sets.csa_layers.empty()
                                    ? DSV4IndexerKVLayout::BF16
                                    : resolveIndexerKVLayout(model_config, attn, kv_cache_config);
    const uint32_t indexer_entry_bytes = indexerKVEntryBytesForLayout(indexer_layout);
    RTP_LLM_LOG_INFO("DSV4 INDEXER_KV layout=%s entry_elems=%u global_fp8_kv=%d rocm_atom_indexer_fp8=%d",
                     dsv4IndexerKVLayoutName(indexer_layout),
                     indexer_entry_bytes,
                     fp8_kv,
                     kv_cache_config.rocm_atom_dsv4_indexer_fp8_kv_cache);

    const uint32_t csa_state_eb =
        maybeAdjustFixedEntriesForCpSharding(computeStateRing(kCsaCompressRatio, kCsaOverlap, gen_num_per_cycle),
                                             parallelism_config,
                                             KVCacheRegionName::CSA_STATE);
    const uint32_t hca_state_eb =
        maybeAdjustFixedEntriesForCpSharding(computeStateRing(kHcaCompressRatio, kHcaOverlap, gen_num_per_cycle),
                                             parallelism_config,
                                             KVCacheRegionName::HCA_STATE);
    const uint32_t indexer_state_eb = maybeAdjustFixedEntriesForCpSharding(
        computeStateRing(kIndexerCompressRatio, kIndexerOverlap, gen_num_per_cycle),
        parallelism_config,
        KVCacheRegionName::INDEXER_STATE);
    // SWA_KV ring = window + MTP draft slack, sized like the HCA state ring.
    // Without the +gen_num_per_cycle slack, a decode step's later draft writes
    // wrap onto ring slots still inside earlier tokens' SWA window -> MTP garble.
    const uint32_t swa_kv_eb = maybeAdjustSwaEntriesForCpSharding(
        computeStateRing(/*compress_ratio=*/static_cast<int>(kDsv4SwaWindowEntries),
                         /*overlap=*/0,
                         gen_num_per_cycle),
        parallelism_config);
    const size_t swa_kv_block_size_bytes_override =
        maybeSwaPrefillCpByteSliceBytes(swa_kv_eb, parallelism_config);
    const uint32_t fixed_cp_size = fixedRegionCpSize(parallelism_config);
    const uint32_t fixed_tokens_per_block =
        fixed_cp_size > 1 ? physical_tokens_per_block * fixed_cp_size : physical_tokens_per_block;
    return {
        {KVCacheRegionName::CSA_KV,
         &sets.csa_layers,
         kv_entry_bytes,
         kernel_tokens_per_block / 4,
         physical_tokens_per_block,
         DataType::TYPE_UINT8,
         true},
        {KVCacheRegionName::HCA_KV,
         &sets.hca_layers,
         kv_entry_bytes,
         kernel_tokens_per_block / 128,
         physical_tokens_per_block,
         DataType::TYPE_UINT8,
         true},
        {KVCacheRegionName::INDEXER_KV,
         &sets.csa_layers,
         indexer_entry_bytes,
         kernel_tokens_per_block / 4,
         physical_tokens_per_block,
         DataType::TYPE_UINT8,
         true},
        {KVCacheRegionName::INDEXER_STATE,
         &sets.csa_layers,
         idx_state_dim * 2,
         indexer_state_eb,
         fixed_tokens_per_block,
         DataType::TYPE_FP32,
         false},
        {KVCacheRegionName::CSA_STATE,
         &sets.csa_layers,
         csa_state_dim * 2,
         csa_state_eb,
         fixed_tokens_per_block,
         DataType::TYPE_FP32,
         false},
        {KVCacheRegionName::HCA_STATE,
         &sets.hca_layers,
         hca_state_dim * 2,
         hca_state_eb,
         fixed_tokens_per_block,
         DataType::TYPE_FP32,
         false},
        {KVCacheRegionName::SWA_KV,
         &sets.all_layers,
         kv_entry_bytes,
         swa_kv_eb,
         fixed_tokens_per_block,
         DataType::TYPE_UINT8,
         false,
         swa_kv_block_size_bytes_override},
    };
}

KVCacheSpecPtr makeDSV4Spec(const DSV4PoolDesc& pool) {
    const auto layer_count = static_cast<uint32_t>(pool.layer_ids->size());
    if (pool.is_paged) {
        return std::make_shared<DSV4KVSpec>(pool.region_name,
                                            layer_count,
                                            pool.entry_elems,
                                            pool.entries_per_block,
                                            pool.store_dtype,
                                            pool.tokens_per_block);
    }
    return std::make_shared<DSV4StateSpec>(pool.region_name,
                                           layer_count,
                                           pool.entry_elems,
                                           pool.entries_per_block,
                                           pool.store_dtype,
                                           pool.tokens_per_block,
                                           pool.block_size_bytes_override);
}

}  // namespace

void DSV4CacheConfigHelper::applyConfig(CacheConfig&             config,
                                        const ModelConfig&       model_config,
                                        const ParallelismConfig& parallelism_config,
                                        const KVCacheConfig&     kv_cache_config,
                                        int                      gen_num_per_cycle) {
    RTP_LLM_LOG_INFO("Creating DSV4 typed hybrid-pool cache config with %zu compress_ratios, "
                     "state ring slack=%d (gen_num_per_cycle)",
                     model_config.attn_config.layer_compress_ratios.size(),
                     gen_num_per_cycle);

    const auto user_seq_size        = kv_cache_config.seq_size_per_block;
    const auto user_kernel_seq_size = kv_cache_config.kernel_seq_size_per_block;
    const auto physical_tokens_per_block =
        user_seq_size > 0 ? static_cast<uint32_t>(user_seq_size) : kDsv4KernelTokensPerBlock;
    const auto kernel_tokens_per_block =
        user_kernel_seq_size > 0 ? static_cast<uint32_t>(user_kernel_seq_size) : physical_tokens_per_block;
    RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block >= kDsv4MinKernelTokensPerBlock
                                && kernel_tokens_per_block % kDsv4MinKernelTokensPerBlock == 0,
                            "DSV4 kernel_seq_size_per_block=%u must be >= %u and a multiple of %u",
                            kernel_tokens_per_block,
                            kDsv4MinKernelTokensPerBlock,
                            kDsv4MinKernelTokensPerBlock);
    RTP_LLM_CHECK_WITH_INFO(physical_tokens_per_block >= kernel_tokens_per_block
                                && physical_tokens_per_block % kernel_tokens_per_block == 0,
                            "DSV4 seq_size_per_block=%u must be >= kernel_seq_size_per_block=%u and divisible by it",
                            physical_tokens_per_block,
                            kernel_tokens_per_block);
    RTP_LLM_LOG_INFO("DSV4 physical block=%u, kernel block=%u (bpk=%u), "
                     "prefill_cp_fixed_sliced=%d (role=%d, cp_sharded=%d, tp_size=%ld)",
                     physical_tokens_per_block,
                     kernel_tokens_per_block,
                     physical_tokens_per_block / kernel_tokens_per_block,
                     isPrefillCpSliced(parallelism_config),
                     static_cast<int>(parallelism_config.role_type),
                     parallelism_config.prefill_cp_config.kv_cache_sharded,
                     parallelism_config.tp_size);

    const auto sets = classifyDSV4Layers(model_config.attn_config.layer_compress_ratios);
    const auto pools = buildDSV4PoolDescs(
        sets,
        model_config,
        kv_cache_config,
        kernel_tokens_per_block,
        physical_tokens_per_block,
        parallelism_config,
        gen_num_per_cycle);
    RTP_LLM_CHECK_WITH_INFO(pools.size() == kDsv4PoolNum, "DSV4 must produce %zu pools", kDsv4PoolNum);

    config.layer_num                                = static_cast<uint32_t>(sets.all_layers.size());
    config.layer_all_num                            = config.layer_num;
    config.use_mla                                  = false;
    config.is_sparse                                = true;
    config.seq_size_per_block                       = physical_tokens_per_block;
    config.kernel_seq_size_per_block                = kernel_tokens_per_block;
    config.use_typed_cache_regions                  = true;
    config.use_opaque_kv_cache_store                = true;
    config.disable_decode_first_malloc_device_reuse = true;

    config.cache_specs.clear();
    config.global_layer_ids.clear();
    config.layer_ids.clear();
    config.group_types.clear();
    config.group_region_names.clear();
    config.group_seq_size_per_block.clear();
    config.group_seq_size_per_block.reserve(pools.size());
    config.cache_specs.reserve(pools.size());
    config.global_layer_ids.reserve(pools.size());
    config.layer_ids.reserve(pools.size());
    config.group_types.reserve(pools.size());
    config.group_region_names.reserve(pools.size());
    for (size_t gid = 0; gid < pools.size(); ++gid) {
        const auto& pool = pools[gid];
        auto        spec = makeDSV4Spec(pool);

        RTP_LLM_LOG_INFO("DSV4 pool desc gid=%zu region=%s(%d) type=%s layer_count=%zu entry_elems=%u "
                         "entries_per_block=%u tokens_per_block=%u physical_tokens_per_block=%u "
                         "prefill_cp_fixed_sliced=%d",
                         gid,
                         dsv4RegionName(pool.region_name),
                         static_cast<int>(pool.region_name),
                         pool.is_paged ? "FULL" : "SWA/FIXED",
                         pool.layer_ids->size(),
                         pool.entry_elems,
                         pool.entries_per_block,
                         pool.tokens_per_block,
                         physical_tokens_per_block,
                         isPrefillCpSliced(parallelism_config));

        config.cache_specs.push_back(spec);
        config.group_seq_size_per_block.push_back(pool.tokens_per_block);
        config.global_layer_ids.push_back(*pool.layer_ids);
        config.layer_ids.push_back(*pool.layer_ids);
        config.group_types.push_back(pool.is_paged ? CacheGroupType::FULL : CacheGroupType::SWA);
        config.group_region_names.push_back(pool.region_name);
    }
}

}  // namespace rtp_llm
