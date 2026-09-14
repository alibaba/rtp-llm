#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rtp_llm {

struct CopyCoalescingTile {
    void*  src{nullptr};
    void*  dst{nullptr};
    size_t bytes{0};
    size_t descriptor_index{SIZE_MAX};
    size_t layout_index{SIZE_MAX};
    size_t member_group_id{SIZE_MAX};
    size_t component_index{SIZE_MAX};
    size_t local_layer_index{SIZE_MAX};
    int    device_index{-1};
};

struct CopyCoalescingRegion {
    void*  src{nullptr};
    void*  dst{nullptr};
    size_t width{0};
    size_t height{0};
    size_t src_pitch{0};
    size_t dst_pitch{0};
};

std::vector<CopyCoalescingRegion> coalesceDeviceHostTiles(const std::vector<CopyCoalescingTile>& tiles);

}  // namespace rtp_llm
