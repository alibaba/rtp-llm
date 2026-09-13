#pragma once

#include <cstdint>
#include <algorithm>
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
// the semantic identity; numeric group ids are private CacheTopology indices.
struct GroupBase {
    std::string                        tag;
    std::shared_ptr<const KVCacheSpec> spec;
    CacheGroupPolicy                   policy;

    uint32_t block_num             = 0;

    size_t seqSizePerBlock() const;
    size_t kernelSeqSizePerBlock() const;
    size_t kernelBlocksPerKvBlock() const;
    size_t   kvBlockStrideBytes() const;
    size_t   kvScaleStrideBytes() const;
    uint32_t localKvHeadNum() const;
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
    const GroupBase& groupById(size_t group_id) const;
    const LayerBase& layer(int layer_id) const;
    GroupRefs        groupsForLayer(int layer_id) const;
    const GroupBase& groupForLayer(int layer_id, std::string_view tag) const;
    const GroupBase& soleGroupForLayer(int layer_id) const;

    size_t groupIdForTag(std::string_view tag) const;
    bool   hasSingleGlobalGroup() const;
    bool   hasOneGroupPerLayer() const;

    size_t totalGroupBlockSizeBytes() const;
    size_t           blockSizeBytesForGroup(size_t group_id) const;
    std::vector<int> layerIdsForGroup(size_t group_id) const;
    std::vector<int> groupIdsForLayer(int layer_id) const;

    size_t maxKernelBlocksPerKvBlock() const {
        size_t result = 1;
        for (const auto& group : groups_) {
            result = std::max(result, group.kernelBlocksPerKvBlock());
        }
        return result;
    }

    // Compatibility projections are values, never a second configuration source.
    std::vector<std::string>      groupTagsSnapshot() const;
    std::vector<CacheGroupType>   groupTypesSnapshot() const;
    std::vector<std::vector<int>> layerGroupIdsSnapshot() const;

private:
    CacheTopology(std::vector<GroupBase> groups, std::vector<LayerBase> layers);
    void validateAndBuildIndex();

    std::vector<GroupBase>                  groups_;
    std::vector<LayerBase>                  layers_;

    std::unordered_map<std::string, size_t> tag_to_group_id_;
};

}  // namespace rtp_llm
