#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheSpec.h"

namespace rtp_llm {

// Immutable cache-group configuration published by CacheConfig. The tag is
// the semantic identity.
struct GroupTopology {
    std::string                        tag;
    std::shared_ptr<const KVCacheSpec> spec;
    CacheGroupPolicy                   policy;
    std::vector<int>                   layer_ids;

    uint32_t local_kv_head_num     = 1;
    size_t   kv_block_stride_bytes = 0;
    size_t   kv_scale_stride_bytes = 0;
};

size_t storedKernelBlocksPerKvBlock(const GroupTopology& group);

// Order is deterministic but carries no business meaning.
struct LayerTopology {
    int                      layer_id = -1;
    std::vector<std::string> group_tags;
};

struct CacheTopology {
public:
    using GroupRefs = std::vector<std::reference_wrapper<const GroupTopology>>;

    static std::shared_ptr<const CacheTopology> create(std::vector<GroupTopology> groups,
                                                       std::vector<LayerTopology> layers);

    const std::vector<GroupTopology>& groups() const {
        return groups_;
    }

    const std::vector<LayerTopology>& layers() const {
        return layers_;
    }

    const GroupTopology& group(std::string_view tag) const;
    bool                 containsTag(std::string_view tag) const;
    const LayerTopology& layer(int layer_id) const;
    GroupRefs            groupsForLayer(int layer_id) const;
    const GroupTopology& groupForLayer(int layer_id, std::string_view tag) const;

    bool hasSingleGlobalGroup() const;

private:
    CacheTopology(std::vector<GroupTopology> groups, std::vector<LayerTopology> layers);
    void validateAndBuildIndex();

    std::vector<GroupTopology>              groups_;
    std::vector<LayerTopology>              layers_;
    std::unordered_map<std::string, size_t> tag_to_group_idx_;
};

}  // namespace rtp_llm
