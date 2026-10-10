#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include <cstddef>
#include <vector>
#include <memory>

namespace rtp_llm {

/// @brief LayerCacheBuffer 转换工具类
/// 提供 KVCacheResource 到 LayerCacheBuffer 的转换功能
class LayerCacheBufferUtil {
public:
    /// @brief 将 KVCacheResource 转换为所有层的 LayerCacheBuffer 列表
    /// @param key_first/key_stride  externally resolved indexes in the global
    ///                              B-key array. The local block at index i
    ///                              is paired with key_first + i*key_stride.
    ///                              The ordinary P1 path uses {0,1}.
    static std::vector<std::shared_ptr<LayerCacheBuffer>> convert(KVCacheResource& resource,
                                                                  int              batch_id,
                                                                  int              start_block_idx = 0,
                                                                  int              block_count     = -1,
                                                                  size_t           key_first       = 0,
                                                                  size_t           key_stride      = 1);

    /// @brief 将 KVCacheResource 的指定层转换为单个 LayerCacheBuffer
    static std::shared_ptr<LayerCacheBuffer> convertLayer(KVCacheResource& resource,
                                                          int              batch_id,
                                                          int              layer_id,
                                                          int              start_block_idx,
                                                          int              block_count,
                                                          size_t           key_first = 0,
                                                          size_t           key_stride = 1);
    static std::shared_ptr<LayerCacheBuffer> convertLayer(KVCacheResource&   resource,
                                                          int                batch_id,
                                                          int                layer_id,
                                                          const std::string& cache_tag,
                                                          int                start_block_idx,
                                                          int                block_count,
                                                          size_t             key_first,
                                                          size_t             key_stride);

    /// @brief Return whether the selected layer/tag window contains a transferable block.
    /// Uses the same argument validation, CP key bounds, and start/count semantics as convertLayer().
    static bool hasTransferableBlocks(const KVCacheResource& resource,
                                      int                    layer_id,
                                      const std::string&     cache_tag,
                                      int                    start_block_idx,
                                      int                    block_count,
                                      size_t                 key_first,
                                      size_t                 key_stride);

    /// @brief 将 LayerCacheBuffer 转换为 transfer 层需要的 KeyBlockInfoMap
    static transfer::KeyBlockInfoMap buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                                        const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                                        int                                         partition_count = 1,
                                                        int                                         partition_id = 0);
};

}  // namespace rtp_llm
