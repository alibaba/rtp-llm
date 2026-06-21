#include "rtp_llm/cpp/models/dsv4/Dsv4CachePlanBuilder.h"

#include "rtp_llm/cpp/cache/OpaqueKVCacheSpec.h"
#include "rtp_llm/cpp/models/dsv4/Dsv4CacheLayout.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"

#include <array>
#include <map>
#include <set>
#include <utility>

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
constexpr size_t   kDsv4PoolNum               = 7;

struct DSV4LayerSets {
    std::vector<int> csa_layers;
    std::vector<int> hca_layers;
    std::vector<int> swa_only_layers;
    std::vector<int> all_layers;
};

struct DSV4PoolDesc {
    std::string             tag;
    const std::vector<int>* layer_ids;
    uint32_t                entry_elems;
    uint32_t                entries_per_block;
    uint32_t                tokens_per_block;
    uint32_t                compression_ratio = 1;
    DataType                store_dtype;
    bool                    is_paged;
    size_t                  block_size_bytes_override       = 0;
    size_t                  block_size_bytes_alignment      = 0;
    uint32_t                block_alignment_min_entries     = 0;
};

struct ExpectedDSV4Spec {
    const char* tag;
    bool        is_paged;
};

constexpr std::array<ExpectedDSV4Spec, kDsv4PoolNum> kExpectedDsv4Specs = {
    ExpectedDSV4Spec{"csa_kv", true},
    ExpectedDSV4Spec{"hca_kv", true},
    ExpectedDSV4Spec{"indexer_kv", true},
    ExpectedDSV4Spec{"indexer_state", false},
    ExpectedDSV4Spec{"csa_state", false},
    ExpectedDSV4Spec{"hca_state", false},
    ExpectedDSV4Spec{"swa_kv", false},
};

using DSV4SpecMap = std::map<std::string, KVCacheSpecPtr>;

constexpr int kCsaCompressRatio     = 4;
constexpr int kHcaCompressRatio     = 128;
constexpr int kIndexerCompressRatio = 4;
constexpr int kCsaOverlap           = 1;
constexpr int kHcaOverlap           = 0;
constexpr int kIndexerOverlap       = 1;

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
                                              const char*              tag) {
    const auto cp_size = fixedRegionCpSize(parallelism_config);
    if (cp_size <= 1) {
        return entries;
    }

    // CP-aligned entries are the real ring capacity, not dead padding. Prefill
    // stores one CP slice of that ring; decode stores the complete ring.
    const auto ring_capacity_entries = alignUpToMultiple(entries, cp_size);
    const bool prefill_sliced        = isPrefillCpSliced(parallelism_config);
    const auto entries_per_block     = prefill_sliced ? ring_capacity_entries / cp_size : ring_capacity_entries;
    RTP_LLM_LOG_INFO("DSV4 fixed/SWA CP sharding tag=%s min_entries=%u ring_capacity_entries=%u "
                     "entries_per_block=%u cp_size=%u prefill_sliced=%d expanded=%d role=%d",
                     tag,
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

bool sameLayers(const std::vector<int>& lhs, const std::vector<int>& rhs) {
    return lhs == rhs;
}

const KVCacheSpec& specForTag(const DSV4SpecMap& specs, const char* tag) {
    const auto it = specs.find(tag);
    RTP_LLM_CHECK_WITH_INFO(it != specs.end() && it->second != nullptr, "missing DSV4 kv_cache spec tag=%s", tag);
    return *it->second;
}

bool isDsv4PagedSpec(const KVCacheSpec& spec) {
    return spec.type == KVCacheSpecType::OpaqueKV;
}

const OpaqueKVCacheSpec& dsv4OpaqueSpec(const KVCacheSpec& spec) {
    const auto* opaque_spec = dynamic_cast<const OpaqueKVCacheSpec*>(&spec);
    RTP_LLM_CHECK_WITH_INFO(opaque_spec != nullptr,
                            "DSV4 kv_cache spec tag=%s must be a generic opaque spec",
                            spec.tag.c_str());
    return *opaque_spec;
}

uint32_t dsv4SpecEntryElems(const KVCacheSpec& spec) {
    return dsv4OpaqueSpec(spec).entry_elems;
}

uint32_t dsv4SpecCompressionRatio(const KVCacheSpec& spec) {
    const auto* kv_spec = dynamic_cast<const CompressedKVCacheSpec*>(&spec);
    RTP_LLM_CHECK_WITH_INFO(kv_spec != nullptr,
                            "DSV4 kv_cache spec tag=%s must be CompressedKVCacheSpec to have compression_ratio",
                            spec.tag.c_str());
    return kv_spec->compression_ratio;
}

DataType dsv4SpecStoreDtype(const KVCacheSpec& spec) {
    return dsv4OpaqueSpec(spec).store_dtype;
}

size_t dsv4SpecBlockSizeBytesAlignment(const KVCacheSpec& spec) {
    return dsv4OpaqueSpec(spec).block_size_bytes_alignment;
}

uint32_t dsv4SpecBlockAlignmentMinEntries(const KVCacheSpec& spec) {
    if (spec.type == KVCacheSpecType::OpaqueKV) {
        return 0;
    }
    return dsv4OpaqueSpec(spec).block_size_alignment_min_entries;
}

KVCacheSpecPtr makeFallbackDsv4Decl(const char* tag, const ExpectedDSV4Spec& expected, const ModelConfig& model_config) {
    const bool     fp8_kv              = model_config.attn_config.kv_cache_dtype == KvCacheDataType::FP8;
    const uint32_t head_dim            = static_cast<uint32_t>(model_config.attn_config.size_per_head);
    const uint32_t indexer_head_dim    = static_cast<uint32_t>(model_config.attn_config.indexer_head_dim);
    const uint32_t kv_entry_elems      = fp8_kv ? DSV4_FP8_KV_ENTRY_BYTES : head_dim * 2;
    const uint32_t indexer_entry_elems = fp8_kv ? DSV4_FP8_INDEXER_ENTRY_BYTES : indexer_head_dim * 2;

    KVCacheSpecPtr spec;
    if (expected.is_paged) {
        auto kv_spec               = std::make_shared<CompressedKVCacheSpec>();
        kv_spec->entry_elems       = std::string(tag) == "indexer_kv" ? indexer_entry_elems : kv_entry_elems;
        kv_spec->compression_ratio = std::string(tag) == "hca_kv" ? kHcaCompressRatio : kCsaCompressRatio;
        kv_spec->store_dtype       = DataType::TYPE_UINT8;
        spec                       = kv_spec;
    } else {
        auto state_spec        = std::make_shared<FixedStateCacheSpec>();
        state_spec->store_dtype = std::string(tag) == "swa_kv" ? DataType::TYPE_UINT8 : DataType::TYPE_FP32;
        if (std::string(tag) == "indexer_state") {
            state_spec->state_dim = 4 * indexer_head_dim;
        } else if (std::string(tag) == "csa_state") {
            state_spec->state_dim = 4 * head_dim;
        } else if (std::string(tag) == "hca_state") {
            state_spec->state_dim = 2 * head_dim;
        } else {
            state_spec->state_dim = kv_entry_elems;
        }
        spec = state_spec;
    }
    spec->tag   = tag;
    spec->dtype = expected.is_paged || std::string(tag) == "swa_kv" ? DataType::TYPE_UINT8 : DataType::TYPE_FP32;
    return spec;
}

std::pair<DSV4LayerSets, DSV4SpecMap> parseDSV4Specs(const ModelConfig& model_config) {
    RTP_LLM_CHECK_WITH_INFO(!model_config.kv_cache_specs.empty(),
                            "DSV4 cache config requires layer-wise model_config.kv_cache_specs; "
                            "layer_compress_ratios fallback is disabled");

    std::map<std::string, ExpectedDSV4Spec> expected_by_tag;
    for (const auto& expected : kExpectedDsv4Specs) {
        expected_by_tag.emplace(expected.tag, expected);
    }

    DSV4SpecMap                         specs;
    std::map<std::string, std::string>  fingerprints;
    std::map<std::string, std::vector<int>> layers_by_tag;
    for (int layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        const auto layer_it = model_config.kv_cache_specs.find(layer_id);
        RTP_LLM_CHECK_WITH_INFO(layer_it != model_config.kv_cache_specs.end(),
                                "DSV4 kv_cache_specs missing layer %d",
                                layer_id);
        RTP_LLM_CHECK_WITH_INFO(!layer_it->second.empty(),
                                "DSV4 kv_cache_specs layer %d has no specs",
                                layer_id);
        std::set<std::string> layer_tags;
        for (const auto& spec : layer_it->second) {
            RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "DSV4 kv_cache spec must not be null");
            const auto& decl = *spec;
            RTP_LLM_CHECK_WITH_INFO(!decl.tag.empty(), "DSV4 kv_cache spec has empty tag");
            RTP_LLM_CHECK_WITH_INFO(layer_tags.insert(decl.tag).second,
                                    "DSV4 layer %d has duplicate kv_cache spec tag=%s",
                                    layer_id,
                                    decl.tag.c_str());
            const auto expected_it = expected_by_tag.find(decl.tag);
            RTP_LLM_CHECK_WITH_INFO(expected_it != expected_by_tag.end(),
                                    "unknown DSV4 kv_cache spec tag=%s",
                                    decl.tag.c_str());
            const auto& expected = expected_it->second;
            RTP_LLM_CHECK_WITH_INFO(isDsv4PagedSpec(decl) == expected.is_paged,
                                    "DSV4 kv_cache spec tag=%s has mismatched concrete spec type",
                                    decl.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(dsv4SpecEntryElems(decl) > 0,
                                    "DSV4 kv_cache spec tag=%s must set entry_elems/state_dim",
                                    decl.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(dsv4SpecStoreDtype(decl) != DataType::TYPE_INVALID,
                                    "DSV4 kv_cache spec tag=%s must set store_dtype",
                                    decl.tag.c_str());
            if (expected.is_paged) {
                RTP_LLM_CHECK_WITH_INFO(dsv4SpecCompressionRatio(decl) > 0,
                                        "DSV4 kv_cache spec tag=%s must set compression_ratio",
                                        decl.tag.c_str());
            }
            const auto fingerprint = spec->fingerprint();
            const auto fp_it       = fingerprints.find(decl.tag);
            if (fp_it == fingerprints.end()) {
                fingerprints.emplace(decl.tag, fingerprint);
                specs.emplace(decl.tag, spec);
            } else {
                RTP_LLM_CHECK_WITH_INFO(fp_it->second == fingerprint,
                                        "DSV4 kv_cache spec tag=%s has multiple physical prototypes",
                                        decl.tag.c_str());
            }
            layers_by_tag[decl.tag].push_back(layer_id);
        }
    }
    for (const auto& expected : kExpectedDsv4Specs) {
        if (specs.count(expected.tag) == 0) {
            specs.emplace(expected.tag, makeFallbackDsv4Decl(expected.tag, expected, model_config));
        }
    }

    DSV4LayerSets sets;
    sets.csa_layers = layers_by_tag["csa_kv"];
    sets.hca_layers = layers_by_tag["hca_kv"];
    sets.all_layers = layers_by_tag["swa_kv"];
    const auto& csa = sets.csa_layers;
    const auto& hca = sets.hca_layers;
    const auto& all = sets.all_layers;

    RTP_LLM_CHECK_WITH_INFO(sameLayers(layers_by_tag["indexer_kv"], csa),
                            "DSV4 indexer_kv layers must match csa_kv layers");
    RTP_LLM_CHECK_WITH_INFO(sameLayers(layers_by_tag["indexer_state"], csa),
                            "DSV4 indexer_state layers must match csa_kv layers");
    RTP_LLM_CHECK_WITH_INFO(sameLayers(layers_by_tag["csa_state"], csa),
                            "DSV4 csa_state layers must match csa_kv layers");
    RTP_LLM_CHECK_WITH_INFO(sameLayers(layers_by_tag["hca_state"], hca),
                            "DSV4 hca_state layers must match hca_kv layers");

    std::set<int> compressed_layers;
    for (int layer_id : csa) {
        RTP_LLM_CHECK_WITH_INFO(compressed_layers.insert(layer_id).second,
                                "DSV4 layer %d appears in multiple compressed specs",
                                layer_id);
    }
    for (int layer_id : hca) {
        RTP_LLM_CHECK_WITH_INFO(compressed_layers.insert(layer_id).second,
                                "DSV4 layer %d appears in multiple compressed specs",
                                layer_id);
    }

    std::set<int> all_layer_set(all.begin(), all.end());
    RTP_LLM_CHECK_WITH_INFO(all_layer_set.size() == all.size(), "DSV4 swa_kv layers must be unique");
    RTP_LLM_CHECK_WITH_INFO(static_cast<int64_t>(all.size()) == model_config.num_layers,
                            "DSV4 swa_kv layer count %zu != num_layers %ld",
                            all.size(),
                            model_config.num_layers);
    for (int layer_id = 0; layer_id < model_config.num_layers; ++layer_id) {
        RTP_LLM_CHECK_WITH_INFO(all_layer_set.count(layer_id) == 1,
                                "DSV4 swa_kv layers must cover every model layer; missing %d",
                                layer_id);
    }
    for (int layer_id : compressed_layers) {
        RTP_LLM_CHECK_WITH_INFO(all_layer_set.count(layer_id) == 1,
                                "DSV4 compressed layer %d is absent from swa_kv",
                                layer_id);
    }
    for (int layer_id : all) {
        if (compressed_layers.count(layer_id) == 0) {
            sets.swa_only_layers.push_back(layer_id);
        }
    }

    RTP_LLM_LOG_INFO("DSV4 spec layer classification: %zu total, %zu CSA, %zu HCA, %zu SWA-only",
                     sets.all_layers.size(),
                     sets.csa_layers.size(),
                     sets.hca_layers.size(),
                     sets.swa_only_layers.size());
    return {sets, specs};
}

std::vector<DSV4PoolDesc> buildDSV4PoolDescs(const DSV4LayerSets&     sets,
                                             const DSV4SpecMap&       spec_decls,
                                             uint32_t                 kernel_tokens_per_block,
                                             uint32_t                 physical_tokens_per_block,
                                             const ParallelismConfig& parallelism_config,
                                             int                      gen_num_per_cycle) {
    const uint32_t csa_state_eb =
        maybeAdjustFixedEntriesForCpSharding(
            computeStateRing(kCsaCompressRatio, kCsaOverlap, gen_num_per_cycle), parallelism_config, "csa_state");
    const uint32_t hca_state_eb =
        maybeAdjustFixedEntriesForCpSharding(
            computeStateRing(kHcaCompressRatio, kHcaOverlap, gen_num_per_cycle), parallelism_config, "hca_state");
    const uint32_t indexer_state_eb = maybeAdjustFixedEntriesForCpSharding(
        computeStateRing(kIndexerCompressRatio, kIndexerOverlap, gen_num_per_cycle),
        parallelism_config,
        "indexer_state");
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
    auto paged_desc = [&](const char* tag, const std::vector<int>* layers) -> DSV4PoolDesc {
        const auto& decl             = specForTag(spec_decls, tag);
        const auto  compression_ratio = dsv4SpecCompressionRatio(decl);
        const auto  entry_elems       = dsv4SpecEntryElems(decl);
        const auto  alignment         = dsv4SpecBlockSizeBytesAlignment(decl);
        RTP_LLM_CHECK_WITH_INFO(kernel_tokens_per_block % compression_ratio == 0,
                                "DSV4 kv_cache spec tag=%s compression_ratio=%u must divide kernel block %u",
                                tag,
                                compression_ratio,
                                kernel_tokens_per_block);
        return {tag,
                layers,
                entry_elems,
                kernel_tokens_per_block / compression_ratio,
                physical_tokens_per_block,
                compression_ratio,
                dsv4SpecStoreDtype(decl),
                true,
                0,
                alignment > 0 ? alignment :
                                (entry_elems == DSV4_FP8_KV_ENTRY_BYTES ? DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES : 0),
                0};
    };
    auto fixed_desc = [&](const char* tag,
                          const std::vector<int>* layers,
                          uint32_t entries_per_block,
                          uint32_t tokens_per_block,
                          size_t block_size_bytes_override = 0) -> DSV4PoolDesc {
        const auto& decl        = specForTag(spec_decls, tag);
        const auto  entry_elems = dsv4SpecEntryElems(decl);
        const auto  alignment   = dsv4SpecBlockSizeBytesAlignment(decl);
        const auto  min_entries = dsv4SpecBlockAlignmentMinEntries(decl);
        return {tag,
                layers,
                entry_elems,
                entries_per_block,
                tokens_per_block,
                1,
                dsv4SpecStoreDtype(decl),
                false,
                block_size_bytes_override,
                alignment > 0 ? alignment :
                                (std::string(tag) == "swa_kv" && entry_elems == DSV4_FP8_KV_ENTRY_BYTES
                                     ? DSV4_FP8_MLA_BLOCK_ALIGNMENT_BYTES
                                     : 0),
                min_entries > 0 ? min_entries : DSV4_SWA_WINDOW_ENTRIES};
    };

    return {
        paged_desc("csa_kv", &sets.csa_layers),
        paged_desc("hca_kv", &sets.hca_layers),
        paged_desc("indexer_kv", &sets.csa_layers),
        fixed_desc("indexer_state", &sets.csa_layers, indexer_state_eb, fixed_tokens_per_block),
        fixed_desc("csa_state", &sets.csa_layers, csa_state_eb, fixed_tokens_per_block),
        fixed_desc("hca_state", &sets.hca_layers, hca_state_eb, fixed_tokens_per_block),
        fixed_desc("swa_kv", &sets.all_layers, swa_kv_eb, fixed_tokens_per_block, swa_kv_block_size_bytes_override),
    };
}

KVCacheSpecPtr makeDSV4Spec(const DSV4PoolDesc& pool) {
    KVCacheSpecPtr spec;
    if (pool.is_paged) {
        spec = std::make_shared<CompressedKVCacheSpec>(pool.tag,
                                                       pool.entry_elems,
                                                       pool.entries_per_block,
                                                       pool.store_dtype,
                                                       pool.tokens_per_block,
                                                       pool.compression_ratio,
                                                       pool.block_size_bytes_alignment);
    } else {
        const bool is_non_reusable_state = pool.tag == "hca_state";
        spec = std::make_shared<FixedStateCacheSpec>(pool.tag,
                                                     pool.entry_elems,
                                                     pool.entries_per_block,
                                                     pool.store_dtype,
                                                     pool.tokens_per_block,
                                                     pool.block_size_bytes_override,
                                                     pool.block_size_bytes_alignment,
                                                     pool.block_alignment_min_entries,
                                                     /*state_cache=*/true,
                                                     /*skip_reuse=*/is_non_reusable_state);
    }
    spec->tag    = pool.tag;
    spec->layers = *pool.layer_ids;
    return spec;
}

CacheGroupPolicy dsv4PolicyForTag(const KVCacheSpecPtr& spec, CacheGroupType group_type, uint32_t hca_state_blocks) {
    CacheGroupPolicy policy = CacheConfig::cacheGroupPolicyForSpec(spec, group_type);
    if (spec && spec->tag == "hca_state") {
        policy.explicit_block_num = hca_state_blocks;
    }
    return policy;
}

}  // namespace

void Dsv4CachePlanBuilder::applyConfig(CacheConfig&             config,
                                        const ModelConfig&       model_config,
                                        const ParallelismConfig& parallelism_config,
                                        const KVCacheConfig&     kv_cache_config,
                                        int                      gen_num_per_cycle) {
    RTP_LLM_LOG_INFO("Creating DSV4 typed hybrid-pool cache config with %zu declarative specs, "
                     "state ring slack=%d (gen_num_per_cycle)",
                     model_config.kv_cache_specs.size(),
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

    const auto parsed     = parseDSV4Specs(model_config);
    const auto& sets      = parsed.first;
    const auto& spec_decls = parsed.second;
    const auto pools = buildDSV4PoolDescs(sets,
                                          spec_decls,
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
    config.group_policies.clear();
    config.group_tags.clear();
    config.group_seq_size_per_block.clear();
    config.group_seq_size_per_block.reserve(pools.size());
    config.cache_specs.reserve(pools.size());
    config.global_layer_ids.reserve(pools.size());
    config.layer_ids.reserve(pools.size());
    config.group_types.reserve(pools.size());
    config.group_policies.reserve(pools.size());
    config.group_tags.reserve(pools.size());
    for (size_t gid = 0; gid < pools.size(); ++gid) {
        const auto& pool = pools[gid];
        auto        spec = makeDSV4Spec(pool);

        RTP_LLM_LOG_INFO("DSV4 pool desc gid=%zu tag=%s type=%s layer_count=%zu entry_elems=%u "
                         "entries_per_block=%u tokens_per_block=%u physical_tokens_per_block=%u "
                         "prefill_cp_fixed_sliced=%d",
                         gid,
                         pool.tag.c_str(),
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
        const auto group_type = pool.is_paged ? CacheGroupType::FULL : CacheGroupType::SWA;
        config.group_types.push_back(group_type);
        config.group_policies.push_back(
            dsv4PolicyForTag(spec, group_type, kv_cache_config.dsv4_hca_state_pool_blocks));
        config.group_tags.push_back(pool.tag);
    }
}

}  // namespace rtp_llm
