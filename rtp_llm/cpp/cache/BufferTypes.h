#pragma once

#include <vector>

#include "rtp_llm/cpp/core/Buffer.h"
#include "rtp_llm/cpp/cache/Types.h"

namespace rtp_llm {

struct BlockBufferPtrInfo {
    BufferPtr kv_addr       = nullptr;
    BufferPtr kv_scale_addr = nullptr;
};

struct CacheLayerLayout {
    std::vector<int>       layer_to_groups;
    std::vector<BufferPtr> layers_to_buffer_ptrs;
    std::vector<BufferPtr> layers_to_scale_buffer_ptrs;
};

struct KVCacheBuffer {
    rtp_llm::BufferPtr kv_blocks       = nullptr;
    rtp_llm::BufferPtr kv_scale_blocks = nullptr;
};

}  // namespace rtp_llm
