#pragma once

#include <cstddef>
#include <vector>

namespace rtp_llm {

// Pointer-only rows; this low-level binding has no cache/layout dependency.
// Each region copies width bytes per row, without copying inter-row gaps.
struct MemoryCopyRegion {
    const void* src{nullptr};
    void*       dst{nullptr};
    size_t      width{0};
    size_t      height{0};
    size_t      src_pitch{0};
    size_t      dst_pitch{0};
};

struct Batched3DMemoryCopyParams {
    std::vector<MemoryCopyRegion> regions;
    int                           device_index{-1};
};

}  // namespace rtp_llm
