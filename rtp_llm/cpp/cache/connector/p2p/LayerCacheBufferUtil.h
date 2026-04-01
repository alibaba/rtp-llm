#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include <vector>
#include <memory>

namespace rtp_llm {

/// @brief LayerCacheBuffer 转换工具类
/// 提供 KVCacheResource 到 LayerCacheBuffer 的转换功能
class LayerCacheBufferUtil {
public:
    /// @brief 将 KVCacheResource 转换为所有层的 LayerCacheBuffer 列表
    static std::vector<std::shared_ptr<LayerCacheBuffer>>
    convert(KVCacheResource& resource, int batch_id, int start_block_idx = 0, int block_count = -1);

    /// @brief 将 KVCacheResource 的指定层转换为单个 LayerCacheBuffer
    static std::shared_ptr<LayerCacheBuffer>
    convertLayer(KVCacheResource& resource, int batch_id, int layer_id, int start_block_idx, int block_count);

    /// @brief 将 LayerCacheBuffer 转换为 transfer 层需要的 KeyBlockInfoMap
    static transfer::KeyBlockInfoMap buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                                        const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                                        int                                         partition_count = 1,
                                                        int                                         partition_id = 0);
};

}  // namespace rtp_llm
