#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"
#include "rtp_llm/cpp/utils/StringUtil.h"

namespace rtp_llm {

// Residency (memory_placement) and budget (charge_to_paged_budget) are independent.
inline void checkGroupResidencyBudget(const CacheGroupPolicy& policy, const std::string& tag) {
    RTP_LLM_CHECK_WITH_INFO(
        !(policy.memory_placement != CacheMemoryPlacement::DEVICE && policy.charge_to_paged_budget),
        "cache group '%s' is host-resident (memory_placement=%d) but still charges the paged HBM budget; "
        "host-resident pools must set capacity.charge_to_paged_budget=false",
        tag.c_str(),
        static_cast<int>(policy.memory_placement));
}

struct CacheConfig {
private:
    std::shared_ptr<const CacheTopology> cache_topology;

public:
    int     cache_min_replay_tokens = 1;
    int64_t cache_key_hash_seed     = 0;

    int maxPrefixReuseTokens(int prompt_tokens) const {
        return std::max(0, prompt_tokens - std::max(1, cache_min_replay_tokens));
    }

    bool use_typed_cache_regions                  = false;
    bool use_opaque_kv_cache_store                = false;
    bool disable_decode_first_malloc_device_reuse = false;
    // Finalized by cache config creation. Physical block geometry remains in
    // each group's spec; warmup may intentionally use CP1 key topology.
    bool cache_key_layout_resolved = false;
    // Storage topology used for request key/slot mapping. This is distinct
    // from the producer topology used by the asynchronous cache writer.
    bool request_projection_required = false;
    int                             storage_cp_rank             = 0;
    int                             storage_cp_size             = 1;
    size_t                          reuse_unit_tokens           = 1;
    std::unordered_set<std::string> state_entry_tags;
    // Cache-owned layout facts for existing observability contracts. The
    // CacheInfo block size is the old virtual CP bundle span, while capacity
    // spans are per physical pool and may differ between FULL and STATE/SWA.
    std::optional<size_t> cache_info_block_tokens;
    std::unordered_map<std::string, size_t> capacity_tokens_per_block;

    rtp_llm::DataType dtype     = rtp_llm::DataType::TYPE_INVALID;
    uint32_t          layer_num = 0;  // the number of main model layers
    bool              use_mla   = false;
    bool              is_sparse = false;

    // Block configuration
    size_t seq_size_per_block = 1;  // tokens/base cache-key block; groups may cover multiple key blocks

    // Attention-specific configuration
    int linear_step = 1;  // For Linear attention: keep one cache block every `linear_step` blocks

    CacheConfig() {}

    uint32_t layer_all_num() const {
        if (cache_topology == nullptr) {
            return layer_num;
        }
        RTP_LLM_CHECK_WITH_INFO(cache_topology->layers().size() <= std::numeric_limits<uint32_t>::max(),
                                "CacheConfig layer count exceeds uint32_t range");
        return static_cast<uint32_t>(cache_topology->layers().size());
    }

    int groupNums() const {
        return cache_topology == nullptr ? 0 : static_cast<int>(cache_topology->groups().size());
    }

    const CacheTopology& topology() const {
        RTP_LLM_CHECK_WITH_INFO(cache_topology != nullptr, "CacheConfig topology is not initialized");
        return *cache_topology;
    }

    const std::shared_ptr<const CacheTopology>& topologyPtr() const {
        RTP_LLM_CHECK_WITH_INFO(cache_topology != nullptr, "CacheConfig topology is not initialized");
        return cache_topology;
    }

    const GroupBase& group(const std::string& tag) const {
        return topology().group(tag);
    }

    size_t groupKeyScale(const std::string& tag) const {
        const size_t group_tokens = group(tag).seqSizePerBlock();
        RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0 && group_tokens >= seq_size_per_block
                                    && group_tokens % seq_size_per_block == 0,
                                "cache group tag=%s has invalid key-token scale",
                                tag.c_str());
        return group_tokens / seq_size_per_block;
    }

    size_t logicalBlockCountForGroup(const std::string& tag, size_t key_count) const {
        const size_t scale = groupKeyScale(tag);
        return key_count / scale + (key_count % scale != 0);
    }

    size_t firstLogicalBlockAfterReuse(const std::string& tag, size_t reused_key_blocks) const {
        const size_t scale = groupKeyScale(tag);
        return group(tag).policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK && scale > 1 ?
                   reused_key_blocks :
                   reused_key_blocks / scale;
    }

    bool groupUsesCompactTransferTail(const std::string& tag) const {
        return group(tag).policy.cp_mapping == CpBlockMappingMode::COMPACT_LAST_RANK && groupKeyScale(tag) > 1;
    }

    std::vector<int64_t> canonicalCacheKeys(const std::vector<int64_t>& full_keys) const {
        if (storage_cp_size == 1) {
            return full_keys;
        }
        std::vector<int64_t> keys;
        for (size_t i = static_cast<size_t>(storage_cp_size - 1); i < full_keys.size();
             i += static_cast<size_t>(storage_cp_size)) {
            keys.push_back(full_keys[i]);
        }
        return keys;
    }

    size_t canonicalBlockPosition(const std::string& tag, size_t key_index) const {
        const auto& policy = group(tag).policy;
        const bool  sparse = storage_cp_size > 1
                            && (policy.group_type == CacheGroupType::LINEAR
                                || (policy.group_type == CacheGroupType::SWA
                                    && policy.cp_mapping != CpBlockMappingMode::COMPACT_LAST_RANK));
        if (!sparse) {
            return key_index;
        }
        const size_t cp_size = static_cast<size_t>(storage_cp_size);
        RTP_LLM_CHECK_WITH_INFO(key_index < std::numeric_limits<size_t>::max() / cp_size,
                                "cache canonical block position overflows size_t");
        return (key_index + 1) * cp_size - 1;
    }

    int localAllocTokens(const std::string& tag, int global_tokens) const {
        if (storage_cp_size == 1 || group(tag).policy.cp_mapping != CpBlockMappingMode::BLOCK_ROUND_ROBIN) {
            return global_tokens;
        }
        if (global_tokens <= 0) {
            return 0;
        }
        RTP_LLM_CHECK_WITH_INFO(seq_size_per_block > 0
                                    && seq_size_per_block <= static_cast<size_t>(std::numeric_limits<int>::max()),
                                "cache base token span must fit a positive int");
        const int64_t base  = static_cast<int64_t>(seq_size_per_block);
        const int64_t pages = (static_cast<int64_t>(global_tokens) + base - 1) / base;
        const int64_t slots = (pages + storage_cp_size - 1) / storage_cp_size;
        RTP_LLM_CHECK_WITH_INFO(slots <= std::numeric_limits<int>::max() / base,
                                "cache local allocation length exceeds int range");
        return static_cast<int>(slots * base);
    }

    const std::vector<GroupBase>& groups() const {
        return topology().groups();
    }

    CacheTopology::GroupRefs groupsForLayer(int layer_id) const {
        return topology().groupsForLayer(layer_id);
    }

    const GroupBase& groupForLayer(int layer_id, const std::string& tag) const {
        return topology().groupForLayer(layer_id, tag);
    }

    const GroupBase& soleGroupForLayer(int layer_id) const {
        return topology().soleGroupForLayer(layer_id);
    }

    std::vector<int> layerIdsForGroup(std::string_view group_tag) const {
        return topology().layerIdsForGroup(group_tag);
    }

    size_t blockSizeBytesForGroup(std::string_view group_tag) const {
        size_t      total = 0;
        const auto& tag   = topology().group(group_tag).tag;
        for (int layer_id : layerIdsForGroup(group_tag)) {
            const auto&  physical_group = physicalGroupForLayer(layer_id, tag);
            const size_t kv_bytes       = physical_group.kvBlockStrideBytes();
            const size_t scale_bytes    = physical_group.kvScaleStrideBytes();
            RTP_LLM_CHECK_WITH_INFO(scale_bytes <= std::numeric_limits<size_t>::max() - kv_bytes,
                                    "CacheConfig tag=%s layer=%d stride overflow",
                                    tag.c_str(),
                                    layer_id);
            const size_t layer_bytes = kv_bytes + scale_bytes;
            RTP_LLM_CHECK_WITH_INFO(layer_bytes <= std::numeric_limits<size_t>::max() - total,
                                    "CacheConfig tag=%s block size overflow",
                                    tag.c_str());
            total += layer_bytes;
        }
        return total;
    }

    size_t totalGroupBlockSizeBytes() const {
        size_t total = 0;
        for (const auto& group : topology().groups()) {
            const size_t group_bytes = blockSizeBytesForGroup(group.tag);
            RTP_LLM_CHECK_WITH_INFO(group_bytes <= std::numeric_limits<size_t>::max() - total,
                                    "CacheConfig total block size overflow");
            total += group_bytes;
        }
        return total;
    }

    size_t layerBlockStrideBytes(size_t layer_id) const {
        size_t total = 0;
        for (const auto& group : groupsForLayer(static_cast<int>(layer_id))) {
            const auto& physical_group = physicalGroupForLayer(static_cast<int>(layer_id), group.get().tag);
            total += physical_group.kvBlockStrideBytes() + physical_group.kvScaleStrideBytes();
        }
        return total;
    }

    // Physical descriptors are resolved within this module-local leaf.
    const GroupBase& physicalGroupForLayer(int layer_id, const std::string& tag) const;

    const std::vector<std::string>& groupTags() const {
        return topology().groupTags();
    }

    void setGroupBlockLayout(const std::vector<std::string>& group_tags,
                             const std::vector<uint32_t>&    block_nums,
                             const std::vector<size_t>&      kv_block_stride_bytes,
                             const std::vector<size_t>&      kv_scale_stride_bytes);

    static bool samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs);

    void        setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers);
    void        fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                                 const std::vector<std::vector<int>>& layers_by_group,
                                 const std::vector<CacheGroupType>&   types,
                                 const std::vector<std::string>&      tags     = {},
                                 const std::vector<CacheGroupPolicy>& policies = {});
    void        finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config);
    // Bind immutable cache-key and local allocation facts once after topology
    // construction. Warmup keeps CP1 keys while physical specs use the actual CP.
    void        finalizeStorageLayout(const ParallelismConfig& parallelism, bool warmup = false);
    std::string debugString(size_t indent = 0) const;
};

}  // namespace rtp_llm
