#include "rtp_llm/cpp/cache/block_tree_cache/transfer/DeviceHostCopyCoalescer.h"

#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace rtp_llm {
namespace {

void validateAddressRange(void* address, size_t bytes) {
    if (address == nullptr) {
        throw std::invalid_argument("nonzero copy tile has a null address");
    }

    const auto value = reinterpret_cast<uintptr_t>(address);
    if (bytes > std::numeric_limits<uintptr_t>::max() - value) {
        throw std::overflow_error("copy tile address range overflows");
    }
}

bool hasCompleteIdentity(const CopyCoalescingTile& tile) {
    return tile.descriptor_index != SIZE_MAX && tile.layout_index != SIZE_MAX && tile.member_group_id != SIZE_MAX
           && tile.component_index != SIZE_MAX && tile.local_layer_index != SIZE_MAX && tile.device_index >= 0;
}

using IdentityKey = std::tuple<size_t, size_t, size_t, size_t, int>;

IdentityKey identityKey(const CopyCoalescingTile& tile) {
    return {tile.descriptor_index,
            tile.layout_index,
            tile.member_group_id,
            tile.component_index,
            tile.device_index};
}

bool checkedPitch(void* previous, void* next, size_t width, size_t& pitch) {
    const auto previous_value = reinterpret_cast<uintptr_t>(previous);
    const auto next_value     = reinterpret_cast<uintptr_t>(next);
    if (next_value <= previous_value) {
        return false;
    }

    const auto difference = next_value - previous_value;
    if (difference < width || difference > std::numeric_limits<size_t>::max()) {
        return false;
    }
    pitch = static_cast<size_t>(difference);
    return true;
}

struct TileGroup {
    std::vector<const CopyCoalescingTile*> tiles;
    bool                                   mergeable{false};
};

void appendRegion(std::vector<CopyCoalescingRegion>& regions,
                  const CopyCoalescingTile&          first,
                  size_t                             height,
                  size_t                             src_pitch,
                  size_t                             dst_pitch) {
    regions.push_back({first.src,
                       first.dst,
                       first.bytes,
                       height,
                       height == 1 ? first.bytes : src_pitch,
                       height == 1 ? first.bytes : dst_pitch});
}

void appendCoalescedGroup(const TileGroup& group, std::vector<CopyCoalescingRegion>& regions) {
    if (!group.mergeable) {
        const auto& tile = *group.tiles.front();
        appendRegion(regions, tile, 1, tile.bytes, tile.bytes);
        return;
    }

    size_t run_begin = 0;
    size_t run_height = 1;
    size_t src_pitch = 0;
    size_t dst_pitch = 0;
    for (size_t index = 1; index < group.tiles.size(); ++index) {
        const auto& previous = *group.tiles[index - 1];
        const auto& current  = *group.tiles[index];
        size_t     next_src_pitch = 0;
        size_t     next_dst_pitch = 0;
        const bool consecutive = current.local_layer_index == previous.local_layer_index + 1;
        const bool compatible = consecutive && current.bytes == previous.bytes
                                && checkedPitch(previous.src, current.src, current.bytes, next_src_pitch)
                                && checkedPitch(previous.dst, current.dst, current.bytes, next_dst_pitch)
                                && (run_height == 1
                                    || (next_src_pitch == src_pitch && next_dst_pitch == dst_pitch));
        if (compatible) {
            if (run_height == 1) {
                src_pitch = next_src_pitch;
                dst_pitch = next_dst_pitch;
            }
            ++run_height;
            continue;
        }

        appendRegion(regions, *group.tiles[run_begin], run_height, src_pitch, dst_pitch);
        run_begin  = index;
        run_height = 1;
        src_pitch  = 0;
        dst_pitch  = 0;
    }
    appendRegion(regions, *group.tiles[run_begin], run_height, src_pitch, dst_pitch);
}

}  // namespace

std::vector<CopyCoalescingRegion> coalesceDeviceHostTiles(const std::vector<CopyCoalescingTile>& tiles) {
    for (const auto& tile : tiles) {
        if (tile.bytes == 0) {
            continue;
        }
        validateAddressRange(tile.src, tile.bytes);
        validateAddressRange(tile.dst, tile.bytes);
    }

    std::vector<TileGroup> groups;
    groups.reserve(tiles.size());
    std::map<IdentityKey, size_t> group_indices;
    for (const auto& tile : tiles) {
        if (tile.bytes == 0) {
            continue;
        }
        if (!hasCompleteIdentity(tile)) {
            groups.push_back({{&tile}, false});
            continue;
        }

        const auto [matching, inserted] = group_indices.emplace(identityKey(tile), groups.size());
        if (inserted) {
            groups.push_back({{&tile}, true});
        } else {
            groups[matching->second].tiles.push_back(&tile);
        }
    }

    std::vector<CopyCoalescingRegion> regions;
    regions.reserve(tiles.size());
    for (const auto& group : groups) {
        appendCoalescedGroup(group, regions);
    }
    return regions;
}

}  // namespace rtp_llm
