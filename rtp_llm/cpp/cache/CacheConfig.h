#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/CacheModuleConfig.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

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

// Module container: one main module plus N MTP modules, plus the cross-module
// shared runtime fields. `module_id == 0` is the main model; API callers only
// pass an explicit `module` argument when operating on MTP state.
struct CacheConfig {
    // === Module configs ===
    CacheModuleConfig              main_config;
    std::vector<CacheModuleConfig> mtp_sub_configs;

    // === Shared fields (cross-module invariants) ===
    int     cache_min_replay_tokens = 1;
    int64_t cache_key_hash_seed     = 0;
    // Global cache-key block granularity (tokens/base cache-key block).
    // Main and MTP share one token sequence, so the block granularity must be
    // identical across modules — kept here as a shared invariant, not per-module.
    size_t seq_size_per_block = 1;

    CacheConfig() {}

    // === Module access (0=main, 1..N=mtp) ===
    size_t moduleCount() const {
        return 1 + mtp_sub_configs.size();
    }

    const CacheModuleConfig& moduleConfig(size_t module_id) const {
        RTP_LLM_CHECK_WITH_INFO(
            module_id <= mtp_sub_configs.size(), "cache module id %zu out of range [0, %zu]", module_id, moduleCount());
        return module_id == 0 ? main_config : mtp_sub_configs[module_id - 1];
    }

    int maxPrefixReuseTokens(int prompt_tokens) const {
        return std::max(0, prompt_tokens - std::max(1, cache_min_replay_tokens));
    }

    uint32_t layer_all_num(size_t module = 0) const {
        return moduleConfig(module).layer_all_num();
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

    // === Topology accessors (module=0 default; old signatures unchanged) ===
    int groupNums(size_t module = 0) const {
        return moduleConfig(module).groupNums();
    }

    const CacheTopology& topology(size_t module = 0) const {
        return moduleConfig(module).topology();
    }

    const std::shared_ptr<const CacheTopology>& topologyPtr(size_t module = 0) const {
        return moduleConfig(module).topologyPtr();
    }

    const GroupBase& group(const std::string& tag, size_t module = 0) const {
        return moduleConfig(module).group(tag);
    }

    const std::vector<GroupBase>& groups(size_t module = 0) const {
        return moduleConfig(module).groups();
    }

    CacheTopology::GroupRefs groupsForLayer(int layer_id, size_t module = 0) const {
        return moduleConfig(module).groupsForLayer(layer_id);
    }

    const GroupBase& groupForLayer(int layer_id, const std::string& tag, size_t module = 0) const {
        return moduleConfig(module).groupForLayer(layer_id, tag);
    }

    const GroupBase& soleGroupForLayer(int layer_id, size_t module = 0) const {
        return moduleConfig(module).soleGroupForLayer(layer_id);
    }

    std::vector<int> layerIdsForGroup(std::string_view group_tag, size_t module = 0) const {
        return moduleConfig(module).layerIdsForGroup(group_tag);
    }

    const std::vector<std::string>& groupTags(size_t module = 0) const {
        return moduleConfig(module).groupTags();
    }

    // === Block sizing ===
    // module 0 keeps the merged main+MTP semantics (physical byte layout is
    // resolved through the MTP children); module > 0 is a single module.
    size_t blockSizeBytesForGroup(std::string_view group_tag, size_t module = 0) const {
        if (module != 0) {
            return moduleConfig(module).blockSizeBytesForGroup(group_tag);
        }
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

    size_t totalGroupBlockSizeBytes(size_t module = 0) const {
        if (module != 0) {
            return moduleConfig(module).totalGroupBlockSizeBytes();
        }
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

    // The merged topology owns logical tag membership, while each MTP child
    // remains the authority for its physical byte layout. Resolve that owner
    // on demand instead of copying a second geometry table into the topology.
    const GroupBase& physicalGroupForLayer(int layer_id, const std::string& tag) const;

    // === Field accessors (module=0 default; earlier public fields) ===
    uint32_t layer_num(size_t module = 0) const {
        return moduleConfig(module).layer_num;
    }
    rtp_llm::DataType dtype(size_t module = 0) const {
        return moduleConfig(module).dtype;
    }
    bool use_mla(size_t module = 0) const {
        return moduleConfig(module).use_mla;
    }
    bool is_sparse(size_t module = 0) const {
        return moduleConfig(module).is_sparse;
    }
    int reuse_cache_tail_blocks(size_t module = 0) const {
        return moduleConfig(module).reuse_cache_tail_blocks;
    }
    bool use_typed_cache_regions(size_t module = 0) const {
        return moduleConfig(module).use_typed_cache_regions;
    }
    bool use_opaque_kv_cache_store(size_t module = 0) const {
        return moduleConfig(module).use_opaque_kv_cache_store;
    }
    bool disable_decode_first_malloc_device_reuse(size_t module = 0) const {
        return moduleConfig(module).disable_decode_first_malloc_device_reuse;
    }

    void setGroupBlockLayout(const std::vector<std::string>& group_tags,
                             const std::vector<uint32_t>&    block_nums,
                             const std::vector<size_t>&      kv_block_stride_bytes,
                             const std::vector<size_t>&      kv_scale_stride_bytes);

    CacheModuleConfig mergeMTPModule(const CacheConfig& propose_config, int module_index, uint32_t main_layer_num);

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
