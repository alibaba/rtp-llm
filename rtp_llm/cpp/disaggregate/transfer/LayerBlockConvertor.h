#pragma once

#include "rtp_llm/cpp/core/Buffer.h"

namespace rtp_llm {

/// @brief 封装 KVCacheAllocator 的 convertIndexToBuffer 接口，转换 blockid 到 bufferptr
class LayerBlockConvertor {
public:
    /// @brief 将 layer_id 和 block_id 转换为 BufferPtr
    /// @param layer_id 层ID
    /// @param block_id 块ID
    /// @param partition_count 分区数量
    /// @param partition_id 分区ID
    /// @return BlockBufferInfo 包含 k_addr 和 v_addr
    virtual std::vector<BufferPtr>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count = 1, int partition_id = 0) const = 0;
};

}  // namespace rtp_llm
