#include "rtp_llm/cpp/cache/CacheTopology.h"

#include <algorithm>
#include <unordered_set>

#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

size_t storedKernelBlocksPerKvBlock(const GroupBase& group) {
    RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "cache group tag=%s has null spec", group.tag.c_str());
    if (group.policy.group_type != CacheGroupType::FULL) {
        return 1;
    }

    const size_t physical_seq_size = group.spec->seq_size_per_block;
    const size_t kernel_seq_size   = group.spec->kernel_seq_size_per_block;
    RTP_LLM_CHECK_WITH_INFO(kernel_seq_size > 0 && physical_seq_size >= kernel_seq_size
                                && physical_seq_size % kernel_seq_size == 0,
                            "invalid block subdivision for tag=%s: physical=%zu kernel=%zu",
                            group.tag.c_str(),
                            physical_seq_size,
                            kernel_seq_size);
    return physical_seq_size / kernel_seq_size;
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

    tag_to_group_idx_.clear();
    tag_to_group_idx_.reserve(groups_.size());
    std::unordered_map<std::string, std::unordered_set<int>> group_layers;
    for (size_t offset = 0; offset < groups_.size(); ++offset) {
        const auto& group = groups_[offset];
        RTP_LLM_CHECK_WITH_INFO(!group.tag.empty(), "CacheTopology group has empty tag");
        RTP_LLM_CHECK_WITH_INFO(tag_to_group_idx_.emplace(group.tag, offset).second,
                                "CacheTopology has duplicate tag=%s",
                                group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.spec != nullptr, "CacheTopology tag=%s has null spec", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.spec->tag == group.tag,
                                "CacheTopology tag=%s does not match spec tag=%s",
                                group.tag.c_str(),
                                group.spec->tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(
            group.spec->seq_size_per_block > 0, "CacheTopology tag=%s has zero seq_size_per_block", group.tag.c_str());
        RTP_LLM_CHECK_WITH_INFO(group.spec->kernel_seq_size_per_block > 0,
                                "CacheTopology tag=%s has zero kernel_seq_size_per_block",
                                group.tag.c_str());

        for (int layer_id : group.layer_ids) {
            RTP_LLM_CHECK_WITH_INFO(layer_id >= 0 && static_cast<size_t>(layer_id) < layers_.size(),
                                    "CacheTopology tag=%s has invalid layer_id=%d",
                                    group.tag.c_str(),
                                    layer_id);
            RTP_LLM_CHECK_WITH_INFO(group_layers[group.tag].emplace(layer_id).second,
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
        RTP_LLM_CHECK_WITH_INFO(
            !layer.group_tags.empty(), "CacheTopology layer_id=%d requires at least one cache group", layer.layer_id);
        std::unordered_set<std::string> seen_tags;
        for (const auto& tag : layer.group_tags) {
            RTP_LLM_CHECK_WITH_INFO(tag_to_group_idx_.find(tag) != tag_to_group_idx_.end(),
                                    "CacheTopology layer=%zu references unknown tag=%s",
                                    layer_index,
                                    tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(seen_tags.emplace(tag).second,
                                    "CacheTopology layer=%zu has duplicate tag=%s",
                                    layer_index,
                                    tag.c_str());
            const auto group_layers_it = group_layers.find(tag);
            RTP_LLM_CHECK_WITH_INFO(group_layers_it != group_layers.end()
                                        && group_layers_it->second.count(static_cast<int>(layer_index)) != 0,
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

const GroupBase& CacheTopology::group(std::string_view tag) const {
    const std::string value(tag);
    const auto        it = tag_to_group_idx_.find(value);
    RTP_LLM_CHECK_WITH_INFO(it != tag_to_group_idx_.end(), "CacheTopology missing tag=%s", value.c_str());
    return groups_[it->second];
}

bool CacheTopology::containsTag(std::string_view tag) const {
    return tag_to_group_idx_.find(std::string(tag)) != tag_to_group_idx_.end();
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

bool CacheTopology::hasSingleGlobalGroup() const {
    return groups_.size() == 1;
}

}  // namespace rtp_llm
