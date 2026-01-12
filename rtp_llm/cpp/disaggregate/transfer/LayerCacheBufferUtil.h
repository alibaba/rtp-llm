#pragma once

#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include <vector>
#include <memory>

namespace rtp_llm {

/// @brief LayerCacheBuffer 转换工具类
/// 提供 KVCacheResource 到 LayerCacheBuffer 的转换功能
class LayerCacheBufferUtil {
public:
    /// @brief 将 KVCacheResource 转换为所有层的 LayerCacheBuffer 列表
    /// @param resource KVCacheResource 引用
    /// @param batch_id Batch ID（用于索引或计算偏移，当前实现中主要用于兼容性）
    /// @return LayerCacheBuffer 列表，按层 ID 顺序
    static std::vector<std::shared_ptr<LayerCacheBuffer>>
    convert(KVCacheResource& resource, int batch_id, int start_block_idx = 0, int block_count = -1);

    /// @brief 将 KVCacheResource 的指定层转换为单个 LayerCacheBuffer
    /// @param resource KVCacheResource 引用
    /// @param batch_id Batch ID（用于索引或计算偏移，当前实现中主要用于兼容性）
    /// @param layer_id 层 ID
    /// @return LayerCacheBuffer，如果层 ID 无效则返回 nullptr
    static std::shared_ptr<LayerCacheBuffer>
    convertLayer(KVCacheResource& resource, int batch_id, int layer_id, int start_block_idx, int block_count);
};

}  // namespace rtp_llm
