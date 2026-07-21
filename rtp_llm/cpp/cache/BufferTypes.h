#pragma once

#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <torch/extension.h>

#include "rtp_llm/cpp/cache/CacheTopology.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

struct BlockBufferPtrInfo {
    torch::Tensor kv_addr;
    torch::Tensor kv_scale_addr;
};

// Dense, immutable all-layer view for one cache group. A group that does not
// own a layer stores an undefined kv_addr at that layer. Scale storage is
// optional even for active layers.
class CacheLayerLayout {
public:
    CacheLayerLayout() = default;

    explicit CacheLayerLayout(std::vector<BlockBufferPtrInfo> layers): layers_(std::move(layers)) {
        for (const auto& layer : layers_) {
            active_layer_count_ += layer.kv_addr.defined() ? 1 : 0;
        }
    }

    bool empty() const noexcept {
        return active_layer_count_ == 0;
    }

    size_t activeLayerCount() const noexcept {
        return active_layer_count_;
    }

    size_t size() const noexcept {
        return layers_.size();
    }

    bool hasLayer(size_t layer_id) const {
        RTP_LLM_CHECK_WITH_INFO(
            layer_id < layers_.size(), "CacheLayerLayout invalid layer_id=%zu size=%zu", layer_id, layers_.size());
        return layers_[layer_id].kv_addr.defined();
    }

    const BlockBufferPtrInfo& at(size_t layer_id) const {
        RTP_LLM_CHECK_WITH_INFO(
            layer_id < layers_.size(), "CacheLayerLayout invalid layer_id=%zu size=%zu", layer_id, layers_.size());
        return layers_[layer_id];
    }

    const std::vector<BlockBufferPtrInfo>& layers() const noexcept {
        return layers_;
    }

private:
    std::vector<BlockBufferPtrInfo> layers_;
    size_t                          active_layer_count_ = 0;
};

// Canonical KV-cache buffer layout: semantic group tag -> dense all-layer
// layout. CacheTopology is the sole owner of group metadata and numeric group ids.
class GroupedCacheLayerLayout {
public:
    using GroupLayouts = std::map<std::string, CacheLayerLayout>;

    GroupedCacheLayerLayout() = default;

    GroupedCacheLayerLayout(std::shared_ptr<const CacheTopology> topology, GroupLayouts groups):
        topology_(std::move(topology)), groups_(std::move(groups)) {
        RTP_LLM_CHECK_WITH_INFO(topology_ != nullptr, "GroupedCacheLayerLayout requires a topology");
        RTP_LLM_CHECK_WITH_INFO(groups_.size() == topology_->groups().size(),
                                "GroupedCacheLayerLayout group count=%zu topology count=%zu",
                                groups_.size(),
                                topology_->groups().size());
        for (const auto& group_config : topology_->groups()) {
            const auto it = groups_.find(group_config.tag);
            RTP_LLM_CHECK_WITH_INFO(
                it != groups_.end(), "GroupedCacheLayerLayout missing topology tag=%s", group_config.tag.c_str());
            RTP_LLM_CHECK_WITH_INFO(it->second.size() == topology_->layers().size(),
                                    "GroupedCacheLayerLayout tag=%s layer count=%zu topology count=%zu",
                                    group_config.tag.c_str(),
                                    it->second.size(),
                                    topology_->layers().size());
        }
    }

    const CacheLayerLayout& group(std::string_view tag) const {
        const std::string value(tag);
        const auto        it = groups_.find(value);
        RTP_LLM_CHECK_WITH_INFO(it != groups_.end(), "GroupedCacheLayerLayout missing tag=%s", value.c_str());
        return it->second;
    }

    const CacheLayerLayout& group(size_t group_id) const {
        return group(topology().groupById(group_id).tag);
    }

    const BlockBufferPtrInfo& at(std::string_view tag, size_t layer_id) const {
        return group(tag).at(layer_id);
    }

    const BlockBufferPtrInfo& at(size_t group_id, size_t layer_id) const {
        return group(group_id).at(layer_id);
    }

    // Layer-only access is valid only when exactly one group has data for the
    // requested layer.
    const BlockBufferPtrInfo& at(size_t layer_id) const {
        const BlockBufferPtrInfo* result = nullptr;
        size_t                    count  = 0;
        for (const auto& [tag, layout] : groups_) {
            (void)tag;
            if (layout.hasLayer(layer_id)) {
                result = &layout.at(layer_id);
                ++count;
            }
        }
        RTP_LLM_CHECK_WITH_INFO(count == 1,
                                "GroupedCacheLayerLayout layer=%zu requires exactly one active group, got %zu",
                                layer_id,
                                count);
        return *result;
    }

    const GroupLayouts& groups() const noexcept {
        return groups_;
    }

    bool hasGroupData(std::string_view tag) const {
        return !group(tag).empty();
    }

    size_t groupId(std::string_view tag) const {
        return topology().groupIdForTag(tag);
    }

    const CacheTopology& topology() const {
        RTP_LLM_CHECK_WITH_INFO(topology_ != nullptr, "GroupedCacheLayerLayout has no topology");
        return *topology_;
    }

    const std::shared_ptr<const CacheTopology>& topologyPtr() const {
        RTP_LLM_CHECK_WITH_INFO(topology_ != nullptr, "GroupedCacheLayerLayout has no topology");
        return topology_;
    }

private:
    std::shared_ptr<const CacheTopology> topology_;
    GroupLayouts                         groups_;
};

struct KVCacheBuffer {
    torch::Tensor kv_blocks;
    torch::Tensor kv_scale_blocks;
};

}  // namespace rtp_llm
