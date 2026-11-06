#pragma once

#include <algorithm>
#include <cstdint>
#include <iterator>
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

private:
    // Lazily-built union of every module's group tags (see allGroupTags()).
    mutable std::vector<std::string> all_group_tags_;
    // Lazily-built topology describing every module's groups (block-table rows).
    mutable std::shared_ptr<const CacheTopology> all_module_topology_;

public:
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

    // Union of every module's groups (as a single topology) — used as the block
    // table's tag-keyed row schema, not for addressing.
    const std::shared_ptr<const CacheTopology>& allModuleTopologyPtr() const;

    // === Block sizing (per-module; module=0 = main module) ===
    // Each module owns its layers exclusively now, so this is a pure per-module
    // sum over the module's own group geometry.
    size_t blockSizeBytesForGroup(std::string_view group_tag, size_t module = 0) const {
        return moduleConfig(module).blockSizeBytesForGroup(group_tag);
    }

    size_t totalGroupBlockSizeBytes(size_t module = 0) const {
        return moduleConfig(module).totalGroupBlockSizeBytes();
    }

    // Sum of every module's total block size; used for the paged-HBM budget.
    size_t totalAllModulesBlockSizeBytes() const {
        size_t total = 0;
        for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
            const size_t module_bytes = moduleConfig(module_id).totalGroupBlockSizeBytes();
            RTP_LLM_CHECK_WITH_INFO(module_bytes <= std::numeric_limits<size_t>::max() - total,
                                    "CacheConfig total block size overflow across modules");
            total += module_bytes;
        }
        return total;
    }

    // === Union accessors (all modules; block-table / snapshot / GroupSet) ===
    // Group tags are globally unique after the MTP rename, so the union is a
    // simple concatenation: main module groups first, then each MTP module's
    // groups in order.
    const std::vector<std::string>& allGroupTags() const {
        if (all_group_tags_.empty() && moduleCount() > 0) {
            for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
                const auto& tags = moduleConfig(module_id).groupTags();
                all_group_tags_.insert(all_group_tags_.end(), tags.begin(), tags.end());
            }
        }
        return all_group_tags_;
    }

    int allGroupNums() const {
        int total = 0;
        for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
            total += moduleConfig(module_id).groupNums();
        }
        return total;
    }

    // Flat index into allGroupTags() (main groups first, then MTP groups).
    int groupIdForTagGlobal(std::string_view tag) const {
        const auto& tags = allGroupTags();
        const auto  it   = std::find(tags.begin(), tags.end(), tag);
        RTP_LLM_CHECK_WITH_INFO(
            it != tags.end(), "unknown cache group tag %.*s", static_cast<int>(tag.size()), tag.data());
        return static_cast<int>(std::distance(tags.begin(), it));
    }

    // Module that owns a group tag (0 = main, 1..N = MTP).
    size_t moduleOfTag(std::string_view tag) const {
        for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
            const auto& tags = moduleConfig(module_id).groupTags();
            if (std::find(tags.begin(), tags.end(), tag) != tags.end()) {
                return module_id;
            }
        }
        RTP_LLM_FAIL("unknown cache group tag %.*s", static_cast<int>(tag.size()), tag.data());
    }

    // Cross-module group lookup: resolve a (globally unique) tag to its owning
    // module's GroupBase.
    const GroupBase& groupByTagGlobal(std::string_view tag) const {
        return moduleConfig(moduleOfTag(tag)).group(std::string(tag));
    }

    // === Cross-module invariants ===
    void validateCrossModuleInvariants() const;

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
