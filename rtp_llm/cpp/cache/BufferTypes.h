#pragma once

#include <algorithm>
#include <functional>
#include <iterator>
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

class GroupedCacheLayerLayout;

// Non-owning view of every tagged cache buffer active at one layer.
class LayerCacheGroupView {
public:
    struct GroupRef {
        std::string_view                                 tag;
        std::reference_wrapper<const BlockBufferPtrInfo> value;
    };

    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type        = GroupRef;
        using difference_type   = std::ptrdiff_t;
        using pointer           = void;
        using reference         = value_type;

        Iterator() = default;
        value_type operator*() const;

        Iterator& operator++() {
            ++tag_it_;
            return *this;
        }

        Iterator operator++(int) {
            auto previous = *this;
            ++(*this);
            return previous;
        }

        bool operator==(const Iterator& other) const {
            return owner_ == other.owner_ && layer_id_ == other.layer_id_ && tag_it_ == other.tag_it_;
        }

        bool operator!=(const Iterator& other) const {
            return !(*this == other);
        }

    private:
        friend class LayerCacheGroupView;

        Iterator(const GroupedCacheLayerLayout*           owner,
                 size_t                                   layer_id,
                 std::vector<std::string>::const_iterator tag_it):
            owner_(owner), layer_id_(layer_id), tag_it_(tag_it) {}

        const GroupedCacheLayerLayout*           owner_    = nullptr;
        size_t                                   layer_id_ = 0;
        std::vector<std::string>::const_iterator tag_it_;
    };

    const BlockBufferPtrInfo& at(std::string_view tag) const;
    bool                      contains(std::string_view tag) const;
    size_t                    size() const;
    Iterator                  begin() const;
    Iterator                  end() const;

private:
    friend class GroupedCacheLayerLayout;
    LayerCacheGroupView(const GroupedCacheLayerLayout* owner, size_t layer_id): owner_(owner), layer_id_(layer_id) {}

    const GroupedCacheLayerLayout* owner_;
    size_t                         layer_id_;
};

// Canonical KV-cache buffer layout: semantic group tag -> dense all-layer
// layout. CacheTopology is the sole owner of group metadata.
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
            for (size_t layer_id = 0; layer_id < topology_->layers().size(); ++layer_id) {
                const auto& layer_tags = topology_->layer(static_cast<int>(layer_id)).group_tags;
                const bool  owns_layer =
                    std::find(layer_tags.begin(), layer_tags.end(), group_config.tag) != layer_tags.end();
                RTP_LLM_CHECK_WITH_INFO(it->second.hasLayer(layer_id) == owns_layer,
                                        "GroupedCacheLayerLayout tag=%s layer=%zu active=%d topology membership=%d",
                                        group_config.tag.c_str(),
                                        layer_id,
                                        it->second.hasLayer(layer_id),
                                        owns_layer);
            }
        }
    }

    const CacheLayerLayout& group(std::string_view tag) const {
        const std::string value(tag);
        const auto        it = groups_.find(value);
        RTP_LLM_CHECK_WITH_INFO(it != groups_.end(), "GroupedCacheLayerLayout missing tag=%s", value.c_str());
        return it->second;
    }

    const BlockBufferPtrInfo& at(std::string_view tag, size_t layer_id) const {
        topology().groupForLayer(static_cast<int>(layer_id), tag);
        return group(tag).at(layer_id);
    }

    LayerCacheGroupView at(size_t layer_id) const {
        topology().layer(static_cast<int>(layer_id));
        return LayerCacheGroupView(this, layer_id);
    }

    const GroupLayouts& groups() const noexcept {
        return groups_;
    }

    bool hasGroupData(std::string_view tag) const {
        return !group(tag).empty();
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

inline LayerCacheGroupView::GroupRef LayerCacheGroupView::Iterator::operator*() const {
    return {*tag_it_, std::cref(owner_->group(*tag_it_).at(layer_id_))};
}

inline LayerCacheGroupView::Iterator LayerCacheGroupView::begin() const {
    const auto& tags = owner_->topology().layer(static_cast<int>(layer_id_)).group_tags;
    return Iterator(owner_, layer_id_, tags.begin());
}

inline LayerCacheGroupView::Iterator LayerCacheGroupView::end() const {
    const auto& tags = owner_->topology().layer(static_cast<int>(layer_id_)).group_tags;
    return Iterator(owner_, layer_id_, tags.end());
}

inline const BlockBufferPtrInfo& LayerCacheGroupView::at(std::string_view tag) const {
    return owner_->at(tag, layer_id_);
}

inline bool LayerCacheGroupView::contains(std::string_view tag) const {
    const auto& tags = owner_->topology().layer(static_cast<int>(layer_id_)).group_tags;
    return std::any_of(
        tags.begin(), tags.end(), [tag](const std::string& candidate) { return std::string_view(candidate) == tag; });
}

inline size_t LayerCacheGroupView::size() const {
    return owner_->topology().layer(static_cast<int>(layer_id_)).group_tags.size();
}

struct KVCacheBuffer {
    torch::Tensor kv_blocks;
    torch::Tensor kv_scale_blocks;
};

}  // namespace rtp_llm
