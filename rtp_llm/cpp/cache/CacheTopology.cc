#include "rtp_llm/cpp/cache/CacheTopology.h"

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

size_t GroupBase::seqSizePerBlock() const {
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheTopology tag=%s has null spec", tag.c_str());
    return spec->seq_size_per_block;
}

size_t GroupBase::kernelSeqSizePerBlock() const {
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheTopology tag=%s has null spec", tag.c_str());
    return spec->kernel_seq_size_per_block;
}

size_t GroupBase::kernelBlocksPerKvBlock() const {
    const auto physical = seqSizePerBlock();
    const auto kernel   = kernelSeqSizePerBlock();
    RTP_LLM_CHECK_WITH_INFO(kernel > 0 && physical % kernel == 0,
                            "CacheTopology tag=%s seq_size_per_block=%zu is not divisible by kernel size=%zu",
                            tag.c_str(),
                            physical,
                            kernel);
    return std::max<size_t>(1, physical / kernel);
}

size_t GroupBase::kvBlockStrideBytes() const {
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheTopology tag=%s has null spec", tag.c_str());
    return spec->block_size_bytes();
}

size_t GroupBase::kvScaleStrideBytes() const {
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr, "CacheTopology tag=%s has null spec", tag.c_str());
    return spec->scale_block_size_bytes();
}

uint32_t GroupBase::localKvHeadNum() const {
    RTP_LLM_CHECK_WITH_INFO(spec != nullptr && spec->local_kv_head_num > 0,
                            "CacheTopology tag=%s has invalid local head count",
                            tag.c_str());
    return spec->local_kv_head_num;
}

size_t CacheTopology::blockSizeBytesForGroup(size_t group_id) const {
    const auto& group                 = groupById(group_id);
    const auto  kv_block_stride_bytes = group.kvBlockStrideBytes();
    const auto  kv_scale_stride_bytes = group.kvScaleStrideBytes();
    const auto  layer_count           = layerIdsForGroup(group_id).size();
    RTP_LLM_CHECK_WITH_INFO(kv_scale_stride_bytes <= std::numeric_limits<size_t>::max() - kv_block_stride_bytes,
                            "CacheTopology tag=%s stride overflow",
                            group.tag.c_str());
    const auto stride = kv_block_stride_bytes + kv_scale_stride_bytes;
    RTP_LLM_CHECK_WITH_INFO(layer_count == 0 || stride <= std::numeric_limits<size_t>::max() / layer_count,
                            "CacheTopology tag=%s block size overflow",
                            group.tag.c_str());
    return layer_count * stride;
}

std::shared_ptr<const CacheTopology> CacheTopology::create(std::vector<GroupBase> groups,
                                                           std::vector<LayerBase> layers) {
    return std::shared_ptr<const CacheTopology>(new CacheTopology(std::move(groups), std::move(layers)));
}

CacheTopology::CacheTopology(std::vector<GroupBase> groups, std::vector<LayerBase> layers):
    groups_(std::move(groups)), layers_(std::move(layers)) {
    validateAndBuildIndex();
}

void CacheTopology::validateAndBuildIndex() {
    RTP_LLM_CHECK_WITH_INFO(!groups_.empty(), "CacheTopology requires at least one cache group");
    RTP_LLM_CHECK_WITH_INFO(!layers_.empty(), "CacheTopology requires at least one cache layer");

    tag_to_group_id_.reserve(groups_.size());
    for (size_t group_id = 0; group_id < groups_.size(); ++group_id) {
        const auto& group = groups_[group_id];
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheTopology group_id=%zu has empty tag", group_id);
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheTopology tag=%s has null spec", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.spec->tag == group.tag,
                                "CacheTopology tag=%s does not match spec tag=%s",
                                group.tag.c_str(),
                                group.spec->tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(tag_to_group_id_.emplace(group.tag, group_id).second,
                                "CacheTopology has duplicate tag=%s",
                                group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            group.seqSizePerBlock() > 0, "CacheTopology tag=%s has zero seq_size_per_block", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.kernelSeqSizePerBlock() > 0,
                                "CacheTopology tag=%s has zero kernel_seq_size_per_block",
                                group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.seqSizePerBlock() % group.kernelSeqSizePerBlock() == 0,
                                "CacheTopology tag=%s seq_size_per_block=%zu is not divisible by kernel size=%zu",
                                group.tag.c_str(),
                                group.seqSizePerBlock(),
                                group.kernelSeqSizePerBlock());

    }

    for (size_t layer_index = 0; layer_index < layers_.size(); ++layer_index) {
        const auto& layer = layers_[layer_index];
        RTP_LLM_CHECK_WITH_INFO(layer.layer_id == static_cast<int>(layer_index),
                                "CacheTopology layer index=%zu has layer_id=%d",
                                layer_index,
                                layer.layer_id);
        RTP_LLM_CHECK_WITH_INFO(!layer.group_tags.empty(), "CacheTopology layer=%zu has no cache group", layer_index);
        std::unordered_set<std::string> seen_tags;
        for (const auto& tag : layer.group_tags) {
            RTP_LLM_CHECK_WITH_INFO(tag_to_group_id_.count(tag) != 0,
                                    "CacheTopology layer=%zu references unknown tag=%s",
                                    layer_index,
                                    tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(seen_tags.emplace(tag).second,
                                    "CacheTopology layer=%zu has duplicate tag=%s",
                                    layer_index,
                                    tag.c_str());
        }
    }

}

size_t CacheTopology::groupIdForTag(std::string_view tag) const {
    const std::string value(tag);
    const auto        it = tag_to_group_id_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != tag_to_group_id_.end(), "CacheTopology missing tag=%s", value.c_str());
    return it->second;
}

const GroupBase& CacheTopology::group(std::string_view tag) const {
    return groupById(groupIdForTag(tag));
}

const GroupBase& CacheTopology::groupById(size_t group_id) const {
    RTP_LLM_CHECK_WITH_INFO(
        group_id < groups_.size(), "CacheTopology invalid group_id=%zu size=%zu", group_id, groups_.size());
    return groups_[group_id];
}

const LayerBase& CacheTopology::layer(int layer_id) const {
    RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers_.size(),
                            "CacheTopology invalid layer_id=%d size=%zu",
                            layer_id,
                            layers_.size());
    return layers_[static_cast<size_t>(layer_id)];
}

CacheTopology::GroupRefs CacheTopology::groupsForLayer(int layer_id) const {
    const auto& layer_config = layer(layer_id);
    GroupRefs   result;
    result.reserve(layer_config.group_tags.size());
    for (const auto& tag : layer_config.group_tags) {
        result.emplace_back(group(tag));
    }
    return result;
}

const GroupBase& CacheTopology::groupForLayer(int layer_id, std::string_view tag) const {
    const auto&       layer_config = layer(layer_id);
    const std::string value(tag);
    const auto        it = std::find(layer_config.group_tags.begin(), layer_config.group_tags.end(), value);
    RTP_LLM_CHECK_WITH_INFO(
        it != layer_config.group_tags.end(), "CacheTopology layer=%d does not own tag=%s", layer_id, value.c_str());
    return group(tag);
}

const GroupBase& CacheTopology::soleGroupForLayer(int layer_id) const {
    const auto& layer_config = layer(layer_id);
    RTP_LLM_CHECK_WITH_INFO(layer_config.group_tags.size() == 1,
                            "CacheTopology layer=%d requires exactly one group, got %zu",
                            layer_id,
                            layer_config.group_tags.size());
    return group(layer_config.group_tags.front());
}

bool CacheTopology::hasSingleGlobalGroup() const {
    return groups_.size() == 1;
}

bool CacheTopology::hasOneGroupPerLayer() const {
    return std::all_of(
        layers_.begin(), layers_.end(), [](const LayerBase& layer) { return layer.group_tags.size() == 1; });
}

size_t CacheTopology::totalGroupBlockSizeBytes() const {
    size_t total = 0;
    for (size_t gid = 0; gid < groups_.size(); ++gid) {
        const auto bytes = blockSizeBytesForGroup(gid);
        RTP_LLM_CHECK_WITH_INFO(bytes <= std::numeric_limits<size_t>::max() - total,
                                "CacheTopology total block size overflow");
        total += bytes;
    }
    return total;
}

std::vector<std::string> CacheTopology::groupTagsSnapshot() const {
    std::vector<std::string> tags;
    tags.reserve(groups_.size());
    for (const auto& group : groups_) {
        tags.push_back(group.tag);
    }
    return tags;
}

std::vector<CacheGroupType> CacheTopology::groupTypesSnapshot() const {
    std::vector<CacheGroupType> types;
    types.reserve(groups_.size());
    for (const auto& group : groups_) {
        types.push_back(group.policy.group_type);
    }
    return types;
}

std::vector<int> CacheTopology::groupIdsForLayer(int layer_id) const {
    std::vector<int> group_ids;
    for (const auto& tag : layer(layer_id).group_tags) {
        group_ids.push_back(static_cast<int>(groupIdForTag(tag)));
    }
    return group_ids;
}

std::vector<int> CacheTopology::layerIdsForGroup(size_t group_id) const {
    const auto&      tag = groupById(group_id).tag;
    std::vector<int> ids;
    for (const auto& layer : layers_) {
        if (std::find(layer.group_tags.begin(), layer.group_tags.end(), tag) != layer.group_tags.end()) {
            ids.push_back(layer.layer_id);
        }
    }
    return ids;
}

std::vector<std::vector<int>> CacheTopology::layerGroupIdsSnapshot() const {
    std::vector<std::vector<int>> ids;
    ids.reserve(layers_.size());
    for (const auto& layer : layers_) {
        ids.push_back(groupIdsForLayer(layer.layer_id));
    }
    return ids;
}

}  // namespace rtp_llm
