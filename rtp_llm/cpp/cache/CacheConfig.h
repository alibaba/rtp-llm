#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
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
    bool use_typed_cache_regions                  = false;
    bool use_opaque_kv_cache_store                = false;
    bool disable_decode_first_malloc_device_reuse = false;

    rtp_llm::DataType dtype     = rtp_llm::DataType::TYPE_INVALID;
    uint32_t          layer_num = 0;  // the number of main model layers
    bool              use_mla   = false;
    bool              is_sparse = false;

    // Block configuration
    uint32_t block_num          = 0;
    size_t   seq_size_per_block = 1;  // tokens/base cache-key block; groups may cover multiple key blocks

    size_t seqSizePerBlockForGroup(size_t gid) const {
        return topology().groupById(gid).seqSizePerBlock();
    }

    size_t kernelSeqSizePerBlockForGroup(size_t gid) const {
        return topology().groupById(gid).kernelSeqSizePerBlock();
    }

    size_t kernelBlocksPerKvBlockForGroup(size_t gid) const {
        return topology().groupById(gid).kernelBlocksPerKvBlock();
    }

    // Attention-specific configuration
    int linear_step = 1;  // For Linear attention: keep one cache block every `linear_step` blocks

    // mtp-model configurations
    std::vector<std::shared_ptr<CacheConfig>> mtp_sub_configs;

    CacheConfig() {}

    uint32_t layer_all_num() const {
        if (cache_topology == nullptr) {
            return layer_num;
        }
        RTP_LLM_CHECK_WITH_INFO(cache_topology->layers().size() <= std::numeric_limits<uint32_t>::max(),
                                "CacheConfig layer count exceeds uint32_t range");
        return static_cast<uint32_t>(cache_topology->layers().size());
    }

    static uint32_t
    mtpGlobalLayerId(uint32_t main_layer_num, int module_index, uint32_t module_layer_num, int local_layer_id) {
        constexpr uint32_t invalid = std::numeric_limits<uint32_t>::max();
        if (module_index < 0 || module_layer_num == 0 || local_layer_id < 0
            || static_cast<uint32_t>(local_layer_id) >= module_layer_num) {
            return invalid;
        }
        const uint64_t global_layer_id = static_cast<uint64_t>(main_layer_num)
                                         + static_cast<uint64_t>(module_index) * module_layer_num
                                         + static_cast<uint32_t>(local_layer_id);
        return global_layer_id < invalid ? static_cast<uint32_t>(global_layer_id) : invalid;
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

    CacheTopology::GroupRefs groupsForLayer(int layer_id) const {
        return topology().groupsForLayer(layer_id);
    }

    const GroupBase& groupForLayer(int layer_id, const std::string& tag) const {
        return topology().groupForLayer(layer_id, tag);
    }

    const GroupBase& soleGroupForLayer(int layer_id) const {
        return topology().soleGroupForLayer(layer_id);
    }

    const std::shared_ptr<const KVCacheSpec>& specForGroup(size_t gid) const {
        return topology().groupById(gid).spec;
    }

    CacheGroupType typeForGroup(size_t gid) const {
        return topology().groupById(gid).policy.group_type;
    }

    const std::string& tagForGroup(size_t gid) const {
        return topology().groupById(gid).tag;
    }

    int groupIdForTag(const std::string& tag) const {
        return static_cast<int>(topology().groupIdForTag(tag));
    }

    std::vector<int> layerIdsForGroup(size_t gid) const {
        return topology().layerIdsForGroup(gid);
    }

    std::vector<int> layerIdsForGroup(std::string_view group_tag) const {
        return topology().layerIdsForGroup(group_tag);
    }

    uint32_t blockNumForGroup(size_t gid) const {
        return topology().groupById(gid).block_num;
    }

    size_t kvBlockStrideBytesForGroup(size_t gid) const {
        return topology().groupById(gid).kvBlockStrideBytes();
    }

    size_t kvScaleStrideBytesForGroup(size_t gid) const {
        return topology().groupById(gid).kvScaleStrideBytes();
    }

    size_t blockSizeBytesForGroup(size_t gid) const {
        size_t      total = 0;
        const auto& tag   = tagForGroup(gid);
        for (int layer_id : layerIdsForGroup(gid)) {
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
        for (size_t gid = 0; gid < static_cast<size_t>(groupNums()); ++gid) {
            const size_t group_bytes = blockSizeBytesForGroup(gid);
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

    // The merged topology owns logical tag membership, while each MTP child
    // remains the authority for its physical byte layout. Resolve that owner
    // on demand instead of copying a second geometry table into the topology.
    const GroupBase& physicalGroupForLayer(int layer_id, const std::string& tag) const;

    uint32_t localKvHeadNumForGroup(size_t gid) const {
        return topology().groupById(gid).localKvHeadNum();
    }

    void setGroupBlockLayout(const std::vector<uint32_t>& block_nums,
                             const std::vector<size_t>&   kv_block_stride_bytes,
                             const std::vector<size_t>&   kv_scale_stride_bytes);

    std::shared_ptr<CacheConfig>
    mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num);

    uint32_t explicitIndependentBlocks(size_t gid) const {
        return policyForGroup(gid).explicit_block_num;
    }

    bool usesExplicitIndependentBlocks(size_t gid) const {
        return explicitIndependentBlocks(gid) > 0;
    }

    CacheGroupPolicy policyForGroup(size_t gid) const {
        return topology().groupById(gid).policy;
    }

    int groupIdFor(int layer_id) const {
        const auto gids = topology().groupIdsForLayer(layer_id);
        RTP_LLM_CHECK_WITH_INFO(gids.size() == 1,
                                "CacheConfig::groupIdFor requires exactly one cache tag for layer_id=%d, got %zu",
                                layer_id,
                                gids.size());
        return gids.front();
    }

    std::vector<int> groupIdsForLayer(int layer_id) const {
        return topology().groupIdsForLayer(layer_id);
    }

    static bool samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs);

    void        setTopology(std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers);
    void        fromGroupedSpecs(const std::vector<KVCacheSpecPtr>&   specs,
                                 const std::vector<std::vector<int>>& layers_by_group,
                                 const std::vector<CacheGroupType>&   types,
                                 const std::vector<std::string>&      tags     = {},
                                 const std::vector<CacheGroupPolicy>& policies = {});
    void        finalizeBlockNums(uint32_t global_block_num, const RuntimeConfig& runtime_config);
    std::string debugString(size_t indent = 0) const;
};

}  // namespace rtp_llm
