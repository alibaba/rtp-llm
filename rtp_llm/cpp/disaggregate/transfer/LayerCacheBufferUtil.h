#pragma once

#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include <vector>
#include <memory>

namespace rtp_llm {

/// @brief LayerCacheBuffer 转换工具类
/// 提供 KVCacheResourceV1 到 LayerCacheBuffer 的转换功能
class LayerCacheBufferUtil {
public:
    /// @brief 将 KVCacheResourceV1 转换为所有层的 LayerCacheBuffer 列表
    /// @param resource KVCacheResourceV1 引用
    /// @param batch_id Batch ID（用于索引或计算偏移，当前实现中主要用于兼容性）
    /// @return LayerCacheBuffer 列表，按层 ID 顺序
    static std::vector<std::shared_ptr<LayerCacheBuffer>> convert(KVCacheResourceV1& resource, int batch_id);

    /// @brief 将 KVCacheResourceV1 的指定层转换为单个 LayerCacheBuffer
    /// @param resource KVCacheResourceV1 引用
    /// @param batch_id Batch ID（用于索引或计算偏移，当前实现中主要用于兼容性）
    /// @param layer_id 层 ID
    /// @return LayerCacheBuffer，如果层 ID 无效则返回 nullptr
    static std::shared_ptr<LayerCacheBuffer> convert(KVCacheResourceV1& resource, int batch_id, int layer_id);
};

}  // namespace rtp_llm
