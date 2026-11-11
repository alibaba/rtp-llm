#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"
#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/models_py/bindings/core/Types.h"

namespace rtp_llm {

// Per-module cache configuration: this module's own topology plus the runtime
// variables describing its K/V geometry and storage traits. CacheConfig owns
// one `main_config` plus a vector of MTP sub-module configs. Each module has
// its own independent topology with module-local layer IDs.
struct CacheModuleConfig {
private:
    std::shared_ptr<const CacheTopology> cache_topology;

public:
    rtp_llm::DataType dtype     = rtp_llm::DataType::TYPE_INVALID;
    uint32_t          layer_num = 0;  // the number of layers in this module
    bool              use_mla   = false;
    bool              is_sparse = false;

    // Cache-key block granularity (tokens/base cache-key block) for this module.
    // Production main+MTP share one token sequence so this is typically identical
    // across modules, but each module owns its own copy so tests that exercise
    // heterogeneous block sizes (e.g. draft tokens_per_block=2, main=4) work
    // correctly when runtimeWriteCacheStore compares group vs. base granularity.
    size_t seq_size_per_block = 1;

    // Attention-specific configuration
    int reuse_cache_tail_blocks = std::numeric_limits<int>::max();

    bool use_typed_cache_regions                  = false;
    bool use_opaque_kv_cache_store                = false;
    bool disable_decode_first_malloc_device_reuse = false;

    int groupNums() const {
        return cache_topology == nullptr ? 0 : static_cast<int>(cache_topology->groups().size());
    }

    const CacheTopology& topology() const {
        RTP_LLM_CHECK_WITH_INFO(cache_topology != nullptr, "CacheModuleConfig topology is not initialized");
        return *cache_topology;
    }

    const std::shared_ptr<const CacheTopology>& topologyPtr() const {
        RTP_LLM_CHECK_WITH_INFO(cache_topology != nullptr, "CacheModuleConfig topology is not initialized");
        return cache_topology;
    }

    const GroupBase& group(const std::string& tag) const {
        return topology().group(tag);
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

    const std::vector<std::string>& groupTags() const {
        return topology().groupTags();
    }

    // This module owns every layer in its topology, so the logical group is
    // authoritative for the physical byte layout (no merged MTP children).
    size_t blockSizeBytesForGroup(std::string_view group_tag) const;

    size_t totalGroupBlockSizeBytes() const;

    void setGroupBlockLayout(const std::vector<std::string>& group_tags,
                             const std::vector<uint32_t>&    block_nums,
                             const std::vector<size_t>&      kv_block_stride_bytes,
                             const std::vector<size_t>&      kv_scale_stride_bytes);

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
