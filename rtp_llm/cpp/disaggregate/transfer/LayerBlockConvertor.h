#pragma once

#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

/// @brief 封装 KVCacheAllocator 的 convertIndexToBuffer 接口，转换 blockid 到 bufferptr
class LayerBlockConvertor {
public:
    virtual std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count = 1, int partition_id = 0) const = 0;

    virtual std::vector<std::pair<BufferPtr, size_t>> getAllBuffers() const = 0;
};

}  // namespace rtp_llm
