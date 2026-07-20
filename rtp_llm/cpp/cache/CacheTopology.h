#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"

namespace rtp_llm {

// Immutable cache-group configuration published by CacheConfig. The tag is
// the semantic identity; numeric slots are private CacheTopology indices.
struct GroupBase {
    std::string                        tag;
    std::shared_ptr<const KVCacheSpec> spec;
    CacheGroupPolicy                   policy;
    std::vector<int>                   layer_ids;

    uint32_t block_num                 = 0;
    uint32_t local_kv_head_num         = 1;
    size_t   seq_size_per_block        = 0;
    size_t   kernel_seq_size_per_block = 0;
    size_t   kv_block_stride_bytes     = 0;
    size_t   kv_scale_stride_bytes     = 0;
};

// Order is deterministic but carries no business meaning.
struct LayerBase {
    int                      layer_id = -1;
    std::vector<std::string> group_tags;
};

struct CacheTopology {
public:
    using GroupRefs = std::vector<std::reference_wrapper<const GroupBase>>;

    static std::shared_ptr<const CacheTopology> create(std::vector<GroupBase> groups, std::vector<LayerBase> layers);

    const std::vector<GroupBase>& groups() const {
        return groups_;
    }

    const std::vector<LayerBase>& layers() const {
        return layers_;
    }

    const GroupBase& group(std::string_view tag) const;
    const GroupBase& groupBySlot(size_t slot) const;
    const LayerBase& layer(int layer_id) const;
    GroupRefs        groupsForLayer(int layer_id) const;
    const GroupBase& groupForLayer(int layer_id, std::string_view tag) const;
    const GroupBase& soleGroupForLayer(int layer_id) const;

    size_t slotForTag(std::string_view tag) const;
    bool   hasSingleGlobalGroup() const;
    bool   hasOneGroupPerLayer() const;

    // Lazily materialized compatibility projections. The same immutable
    // object is returned for the lifetime of this topology.
    const std::vector<std::string>&                groupTagsSnapshot() const;
    const std::vector<CacheGroupType>&             groupTypesSnapshot() const;
    const std::vector<KVCacheSpecType>&            groupSpecTypesSnapshot() const;
    const std::vector<std::vector<int>>&           layerGroupIdsSnapshot() const;
    const std::vector<std::map<std::string, int>>& layerTagToGroupIdSnapshot() const;

private:
    struct SnapshotCache {
        std::vector<std::string>                group_tags;
        std::vector<CacheGroupType>             group_types;
        std::vector<KVCacheSpecType>            group_spec_types;
        std::vector<std::vector<int>>           layer_group_ids;
        std::vector<std::map<std::string, int>> layer_tag_to_group_id;
    };

    CacheTopology(std::vector<GroupBase> groups, std::vector<LayerBase> layers);
    void validateAndBuildIndex();
    void buildSnapshots() const;

    std::vector<GroupBase>                  groups_;
    std::vector<LayerBase>                  layers_;
    std::unordered_map<std::string, size_t> tag_to_slot_;

    mutable std::once_flag                       snapshot_once_;
    mutable std::shared_ptr<const SnapshotCache> snapshots_;
};

}  // namespace rtp_llm
