#include "rtp_llm/cpp/cache/CacheTopology.h"

#include <algorithm>
#include <unordered_set>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

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

    tag_to_group_index_.reserve(groups_.size());
    std::vector<std::unordered_set<int>> group_layers(groups_.size());
    for (size_t group_index = 0; group_index < groups_.size(); ++group_index) {
        const auto& group = groups_[group_index];
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheTopology group_index=%zu has empty tag", group_index);
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheTopology tag=%s has null spec", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.spec->tag == group.tag,
                                "CacheTopology tag=%s does not match spec tag=%s",
                                group.tag.c_str(),
                                group.spec->tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(tag_to_group_index_.emplace(group.tag, group_index).second,
                                "CacheTopology has duplicate tag=%s",
                                group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            group.seq_size_per_block > 0, "CacheTopology tag=%s has zero seq_size_per_block", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.kernel_seq_size_per_block > 0,
                                "CacheTopology tag=%s has zero kernel_seq_size_per_block",
                                group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.seq_size_per_block % group.kernel_seq_size_per_block == 0,
                                "CacheTopology tag=%s seq_size_per_block=%zu is not divisible by kernel size=%zu",
                                group.tag.c_str(),
                                group.seq_size_per_block,
                                group.kernel_seq_size_per_block);

        for (int layer_id : group.layer_ids) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers_.size(),
                                    "CacheTopology tag=%s has invalid layer_id=%d",
                                    group.tag.c_str(),
                                    layer_id);
            RTP_LLM_CHECK_WITH_INFO(group_layers[group_index].emplace(layer_id).second,
                                    "CacheTopology tag=%s has duplicate layer_id=%d",
                                    group.tag.c_str(),
                                    layer_id);
        }
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
            const auto group_index_it = tag_to_group_index_.find(tag);
            RTP_LLM_CHECK_WITH_INFO(group_index_it != tag_to_group_index_.end(),
                                    "CacheTopology layer=%zu references unknown tag=%s",
                                    layer_index,
                                    tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(seen_tags.emplace(tag).second,
                                    "CacheTopology layer=%zu has duplicate tag=%s",
                                    layer_index,
                                    tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(group_layers[group_index_it->second].count(static_cast<int>(layer_index)) != 0,
                                    "CacheTopology layer=%zu tag=%s is missing reverse group membership",
                                    layer_index,
                                    tag.c_str());
        }
    }

    for (const auto& group : groups_) {
        for (int layer_id : group.layer_ids) {
            const auto& tags = layers_[static_cast<size_t>(layer_id)].group_tags;
            RTP_LLM_CHECK_WITH_INFO(std::find(tags.begin(), tags.end(), group.tag) != tags.end(),
                                    "CacheTopology tag=%s layer=%d is missing reverse layer membership",
                                    group.tag.c_str(),
                                    layer_id);
        }
    }
}

size_t CacheTopology::groupIndex(std::string_view tag) const {
    const std::string value(tag);
    const auto        it = tag_to_group_index_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != tag_to_group_index_.end(), "CacheTopology missing tag=%s", value.c_str());
    return it->second;
}

const GroupBase& CacheTopology::group(std::string_view tag) const {
    return groups_[groupIndex(tag)];
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

void CacheTopology::buildSnapshots() const {
    auto snapshots = std::make_shared<SnapshotCache>();
    snapshots->group_tags.reserve(groups_.size());
    snapshots->group_types.reserve(groups_.size());
    snapshots->group_spec_types.reserve(groups_.size());
    for (const auto& group : groups_) {
        snapshots->group_tags.push_back(group.tag);
        snapshots->group_types.push_back(group.policy.group_type);
        snapshots->group_spec_types.push_back(group.spec->type);
    }

    snapshots_ = std::move(snapshots);
}

const std::vector<std::string>& CacheTopology::groupTagsSnapshot() const {
    std::call_once(snapshot_once_, [this]() { buildSnapshots(); });
    return snapshots_->group_tags;
}

const std::vector<CacheGroupType>& CacheTopology::groupTypesSnapshot() const {
    std::call_once(snapshot_once_, [this]() { buildSnapshots(); });
    return snapshots_->group_types;
}

const std::vector<KVCacheSpecType>& CacheTopology::groupSpecTypesSnapshot() const {
    std::call_once(snapshot_once_, [this]() { buildSnapshots(); });
    return snapshots_->group_spec_types;
}

}  // namespace rtp_llm
