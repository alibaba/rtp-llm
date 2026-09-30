#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

#include "rtp_llm/cpp/cache/block_tree_cache/group_set/GroupSet.h"

namespace rtp_llm {

// Enumerate one backing before copy plans are merged by device. Production
// GroupSets use the pool's per-layer physical geometry, including MTP layers
// whose strides differ from the logical tag's spec. GroupSets without an explicit
// physical payload keep their logical strides and exclude extra pool padding.
template<typename Visitor>
size_t visitDeviceHostCopyTiles(const GroupSet& group, const std::vector<BlockIdxType>& blocks, Visitor&& visitor) {
    const auto& tags  = group.groupTags();
    const auto& pools = group.devicePools();
    if (tags.empty() || pools.size() != tags.size() || blocks.size() != pools.size()) {
        throw std::invalid_argument("device-host backing tag/pool/block count mismatch");
    }
    size_t offset = 0;
    size_t tiles  = 0;
    for (size_t member = 0; member < tags.size(); ++member) {
        if (!pools[member]) {
            throw std::invalid_argument("device-host backing has a null pool");
        }
        const auto& base      = group.group(tags[member]);
        const auto  layer_ids = group.topologyPtr()->layerIdsForGroup(tags[member]);
        if (layer_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("device-host backing layer count overflow");
        }
        for (size_t layer = 0; layer < layer_ids.size(); ++layer) {
            // Device pools use group-local layer ordinals, not model-global IDs.
            const auto buffers = pools[member]->convertIndexToBuffer(static_cast<int>(layer), blocks[member]);
            const auto append  = [&](size_t index, size_t bytes) {
                if (!bytes) {
                    return;
                }
                if (index >= buffers.size() || !buffers[index].addr || buffers[index].size_bytes < bytes
                    || offset > group.payloadBytes() || bytes > group.payloadBytes() - offset) {
                    throw std::invalid_argument("device-host backing tile exceeds its buffer or payload");
                }
                auto buffer       = buffers[index];
                buffer.size_bytes = bytes;
                visitor(buffer, offset, member, layer);
                offset += bytes;
                ++tiles;
            };
            if (group.usesPhysicalPayloadGeometry()) {
                for (size_t index = 0; index < buffers.size(); ++index) {
                    append(index, buffers[index].size_bytes);
                }
            } else {
                append(0, base.kvBlockStrideBytes());
                append(1, base.kvScaleStrideBytes());
            }
        }
    }
    if (offset != group.payloadBytes()) {
        throw std::invalid_argument("device-host backing payload geometry mismatch");
    }
    return tiles;
}

}  // namespace rtp_llm
