#pragma once

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace rtp_llm {

/// @brief 封装 KVCacheAllocator 的 convertIndexToBuffer 接口，将 block_id 转换为 BlockInfo
class LayerBlockConverter {
public:
    virtual ~LayerBlockConverter() = default;

    /// @return each element describes one KV or scale sub-buffer.
    virtual std::vector<BlockInfo> convertIndexToBufferByTag(
        int layer_id, const std::string& tag, int block_id, int partition_count, int partition_id) const = 0;

    /// @brief 返回所有需要注册 RDMA MR 的 buffer 及其对齐大小
    virtual std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const = 0;
};

}  // namespace rtp_llm
