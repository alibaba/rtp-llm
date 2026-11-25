#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/cache/LinearReplayPlan.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

using PPBlockNumOverrides = std::unordered_map<std::string, uint32_t>;

// Residency (memory_placement) and budget (charge_to_paged_budget) are independent.
inline void checkGroupResidencyBudget(const CacheGroupPolicy& policy, const std::string& tag) {
    RTP_LLM_CHECK_WITH_INFO(
        !(policy.memory_placement != CacheMemoryPlacement::DEVICE && policy.charge_to_paged_budget),
        "cache group '%s' is host-resident (memory_placement=%d) but still charges the paged HBM budget; "
        "host-resident pools must set capacity.charge_to_paged_budget=false",
        tag.c_str(),
        static_cast<int>(policy.memory_placement));
}

// Narrow per-module runtime metadata that still has direct consumers. The
// module's logical topology itself lives in the immutable CacheTopology
// snapshot owned by CacheConfig; group geometry never appears here.
struct CacheModuleMetadata {
    rtp_llm::DataType dtype   = rtp_llm::DataType::TYPE_INVALID;
    bool              use_mla = false;
    bool              is_sparse = false;

    // Attention-specific configuration
    bool enable_linear_cache_reclaim = false;
    int  reuse_cache_tail_blocks     = std::numeric_limits<int>::max();

    bool use_opaque_kv_cache_store                = false;
    bool disable_decode_first_malloc_device_reuse = false;
};

// Module container: one main module plus N MTP modules, plus the cross-module
// shared runtime fields. Module 0 is the main model. Each module keeps an
// independently published, immutable topology snapshot with module-local layer
// IDs; module selection happens through the topology()/topology(module_id)
// queries and the tag-based cross-module accessors.
struct CacheConfig {
    // === Shared fields (cross-module invariants) ===
    int     cache_min_replay_tokens = 1;
    int64_t cache_key_hash_seed     = 0;
    // Global cache-key block granularity (tokens/base cache-key block).
    // Main and MTP share one token sequence, so the block granularity must be
    // identical across modules — kept here as a shared invariant, not per-module.
    size_t seq_size_per_block = 1;
    // FULL PageRR groups use compact local tables on this rank.
    bool local_kv_cache_sharded = false;
    // First main-model layer on this PP stage in the global layer space used by P2P and cache-store keys.
    uint32_t global_layer_begin = 0;
    // Deployment weights are keyed by physical pool tag.
    std::map<std::string, uint32_t> pool_block_weights;
    LinearReplayPlan linear_replay;

    CacheConfig() {}

public:
    // === Module slots (0=main, 1..N=mtp) ===
    size_t moduleCount() const {
        return modules_.size();
    }

    // === Module topology queries ===
    // The topology reference queries fail closed when the module slot was
    // never published; the owning-pointer queries expose the null state so
    // consumers can distinguish a default slot from a published snapshot.
    int groupNums() const {
        return groupNums(0);
    }

    int groupNums(size_t module_id) const {
        const auto& topology = topologyPtr(module_id);
        return topology == nullptr ? 0 : static_cast<int>(topology->groups().size());
    }

    const CacheTopology& topology(size_t module_id = 0) const {
        const auto& topology = topologyPtr(module_id);
        RTP_LLM_CHECK_WITH_INFO(topology != nullptr, "cache module %zu topology is not initialized", module_id);
        return *topology;
    }

    const std::shared_ptr<const CacheTopology>& topologyPtr(size_t module_id = 0) const {
        return moduleRecord(module_id).topology;
    }

    // === Module metadata (narrow runtime fields with direct consumers) ===
    CacheModuleMetadata& moduleMetadata(size_t module_id) {
        return moduleRecord(module_id).metadata;
    }

    const CacheModuleMetadata& moduleMetadata(size_t module_id) const {
        return moduleRecord(module_id).metadata;
    }

    bool anyLinearCacheReclaim() const {
        for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
            if (moduleMetadata(module_id).enable_linear_cache_reclaim) {
                return true;
            }
        }
        return false;
    }

    int maxPrefixReuseTokens(int prompt_tokens) const {
        return std::max(0, prompt_tokens - std::max(1, cache_min_replay_tokens));
    }

    // === Main-model topology accessors ===
    bool usesGroupCacheTransferPolicy() const {
        return usesGroupCacheTransferPolicy(0);
    }

    bool usesGroupCacheTransferPolicy(size_t module_id) const {
        return groupNums(module_id) > 1;
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

    bool isLinearReplayGroup(std::string_view tag) const {
        return linear_replay.containsGroup(tag);
    }

    bool isLegacyLinearGroup(std::string_view tag) const {
        return group(tag).policy.group_type == CacheGroupType::LINEAR && !isLinearReplayGroup(tag);
    }

    size_t layerBlockStrideBytes(size_t layer_id) const {
        size_t total = 0;
        for (const auto& group : groupsForLayer(static_cast<int>(layer_id))) {
            total += group.get().kvBlockStrideBytes() + group.get().kvScaleStrideBytes();
        }
        return total;
    }

    const GroupBase& soleGroupForLayer(int layer_id) const {
        return topology().soleGroupForLayer(layer_id);
    }

    std::vector<int> layerIdsForGroup(std::string_view group_tag) const {
        return layerIdsForGroup(0, group_tag);
    }

    std::vector<int> layerIdsForGroup(size_t module_id, std::string_view group_tag) const {
        const auto& topology = topologyPtr(module_id);
        if (topology == nullptr) {
            return {};
        }
        return topology->layerIdsForGroup(group_tag);
    }

    const std::vector<std::string>& groupTags() const {
        return topology().groupTags();
    }

    // === Main-model block sizing ===
    // Each module owns its layers exclusively now, so this is a pure per-module
    // sum over the module's own group geometry. Tags are globally unique, so
    // the query resolves the owning module first.
    size_t blockSizeBytesForGroup(std::string_view group_tag) const {
        return topology(moduleOfTag(group_tag)).blockSizeBytesForGroup(group_tag);
    }

    size_t totalGroupBlockSizeBytes(size_t module_id = 0) const {
        return topology(module_id).totalGroupBlockSizeBytes();
    }

    // Sum of every module's total block size; used for the paged-HBM budget.
    size_t totalAllModulesBlockSizeBytes() const {
        size_t total = 0;
        for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
            const auto& topology = topologyPtr(module_id);
            if (topology == nullptr) {
                continue;
            }
            const size_t module_bytes = topology->totalGroupBlockSizeBytes();
            RTP_LLM_CHECK_WITH_INFO(module_bytes <= std::numeric_limits<size_t>::max() - total,
                                    "CacheConfig total block size overflow across modules");
            total += module_bytes;
        }
        return total;
    }

    // === Cross-module accessors (all modules; block-table / snapshot / GroupSet) ===
    // Modules declare globally unique tags, so the union is a
    // simple concatenation: main module groups first, then each MTP module's
    // groups in order.

    // Union of every module's group tags, in main-then-sub traversal order.
    // Computed on demand from current module topologies — never cached.
    std::vector<std::string> allGroupTags() const {
        std::vector<std::string> tags;
        for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
            const auto& topology = topologyPtr(module_id);
            if (topology == nullptr || topology->groups().empty()) {
                continue;
            }
            const auto& module_tags = topology->groupTags();
            tags.insert(tags.end(), module_tags.begin(), module_tags.end());
        }
        return tags;
    }

    int allGroupNums() const {
        int total = 0;
        for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
            total += groupNums(module_id);
        }
        return total;
    }

    // Module that owns a group tag (0 = main, 1..N = MTP).
    size_t moduleOfTag(std::string_view tag) const {
        for (size_t module_id = 0; module_id < moduleCount(); ++module_id) {
            const auto& topology = topologyPtr(module_id);
            if (topology == nullptr) {
                continue;
            }
            const auto& tags = topology->groupTags();
            if (std::find(tags.begin(), tags.end(), tag) != tags.end()) {
                return module_id;
            }
        }
        RTP_LLM_FAIL("unknown cache group tag %.*s", static_cast<int>(tag.size()), tag.data());
    }

    const GroupBase& group(std::string_view tag) const {
        return topology(moduleOfTag(tag)).group(tag);
    }

    // === Cross-module invariants ===
    void validateCrossModuleInvariants() const;

    // === Main-model field accessors ===
    // Layer count derives from the published topology; a default-constructed or
    // layer-less slot answers zero.
    uint32_t layer_num() const {
        const auto& topology = topologyPtr(0);
        return topology == nullptr ? 0 : static_cast<uint32_t>(topology->layers().size());
    }
    rtp_llm::DataType dtype() const {
        return moduleMetadata(0).dtype;
    }
    bool use_mla() const {
        return moduleMetadata(0).use_mla;
    }
    bool is_sparse() const {
        return moduleMetadata(0).is_sparse;
    }
    bool enable_linear_cache_reclaim() const {
        return moduleMetadata(0).enable_linear_cache_reclaim;
    }
    int reuse_cache_tail_blocks() const {
        return moduleMetadata(0).reuse_cache_tail_blocks;
    }
    bool use_opaque_kv_cache_store() const {
        return moduleMetadata(0).use_opaque_kv_cache_store;
    }
    bool disable_decode_first_malloc_device_reuse() const {
        return moduleMetadata(0).disable_decode_first_malloc_device_reuse;
    }

    void setGroupBlockLayout(const std::vector<std::string>& group_tags,
                             const std::vector<uint32_t>&    block_nums,
                             const std::vector<size_t>&      kv_block_stride_bytes,
                             const std::vector<size_t>&      kv_scale_stride_bytes);

    static bool samePolicy(const CacheGroupPolicy& lhs, const CacheGroupPolicy& rhs);

    // Shared by budget estimation and final capacity publication. The
    // deployment weight scales a pool group's share of the baseline blocks.
    uint32_t blockWeight(const std::string& tag) const;
    uint32_t blockNumForGroup(const GroupBase& group, uint32_t baseline) const;
    uint32_t maxBaselineBlockNum() const;

    // Slot 0 is the main model and slots 1..N are drafts. Publication clones
    // the mutable Specs and validates the layout before swapping the snapshot.
    void setTopology(size_t module_id, std::vector<GroupBase> new_groups, std::vector<LayerBase> new_layers);

    void fromGroupedSpecs(size_t                               module_id,
                          const std::vector<KVCacheSpecPtr>&   specs,
                          uint32_t                             layer_num,
                          const std::vector<std::vector<int>>& layers_by_group,
                          const std::vector<CacheGroupType>&   types,
                          const std::vector<std::string>&      tags     = {},
                          const std::vector<CacheGroupPolicy>& policies = {});

    void        finalizeBlockNums(uint32_t                   global_block_num,
                                  const RuntimeConfig&       runtime_config,
                                  const PPBlockNumOverrides* pp_overrides = nullptr);
    std::string debugString(size_t indent = 0) const;

private:
    friend class CacheConfigCreator;

    void addMtpModule(std::shared_ptr<const CacheTopology> topology, CacheModuleMetadata metadata);

    // Per-module record: the immutable topology snapshot plus the narrow
    // runtime metadata. Slot 0 is the main model; the default-constructed
    // slot keeps a null topology and answers zero-group queries.
    struct ModuleRecord {
        std::shared_ptr<const CacheTopology> topology;
        CacheModuleMetadata                 metadata{};
    };

    ModuleRecord& moduleRecord(size_t module_id) {
        RTP_LLM_CHECK_WITH_INFO(module_id < modules_.size(),
                                "cache module id %zu out of range [0, %zu)",
                                module_id,
                                modules_.size());
        return modules_[module_id];
    }

    const ModuleRecord& moduleRecord(size_t module_id) const {
        RTP_LLM_CHECK_WITH_INFO(module_id < modules_.size(),
                                "cache module id %zu out of range [0, %zu)",
                                module_id,
                                modules_.size());
        return modules_[module_id];
    }

    std::vector<ModuleRecord> modules_{ModuleRecord{}};
};

}  // namespace rtp_llm
