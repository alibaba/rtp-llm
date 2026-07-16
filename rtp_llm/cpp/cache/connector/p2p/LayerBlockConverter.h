#pragma once

#include "rtp_llm/cpp/cache/BlockInfo.h"
#include <cstddef>
#include <utility>
#include <vector>

namespace rtp_llm {

/// @brief 封装 KVCacheAllocator 的 convertIndexToBuffer 接口，将 block_id 转换为 BlockInfo
class LayerBlockConverter {
public:
    virtual ~LayerBlockConverter() = default;

    /// @brief 将 (layer_id, block_id, partition) 转换为实际内存地址描述符列表
    /// @return 每个元素对应一个 sub-buffer（kv 或 kv_scale），addr==nullptr 表示无效
    virtual std::vector<BlockInfo>
    convertIndexToBuffer(int layer_id, int block_id, int partition_count = 1, int partition_id = 0) const = 0;

    /// @brief 返回所有需要注册 RDMA MR 的 buffer 及其对齐大小
    virtual std::vector<std::pair<BlockInfo, size_t>> getAllBuffers() const = 0;
};

}  // namespace rtp_llm
