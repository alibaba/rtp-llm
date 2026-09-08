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
    /// @param cp_rank/cp_size  prefill CP page-RR shard. cp_size==1 (default) =
    ///                         no sharding; cp_size>1 maps the i-th LOCAL owned
    ///                         block on this rank to logical position
    ///                         (cp_rank + (start_block_idx + i) * cp_size) so
    ///                         the registered cache_key matches the decode
    ///                         side's per-peer block_pos lookup.
    static std::vector<std::shared_ptr<LayerCacheBuffer>> convert(KVCacheResource& resource,
                                                                  int              batch_id,
                                                                  int              start_block_idx = 0,
                                                                  int              block_count     = -1,
                                                                  int              cp_rank         = 0,
                                                                  int              cp_size         = 1);

    /// @brief 将 KVCacheResource 的指定层转换为单个 LayerCacheBuffer
    static std::shared_ptr<LayerCacheBuffer> convertLayer(KVCacheResource& resource,
                                                          int              batch_id,
                                                          int              layer_id,
                                                          int              start_block_idx,
                                                          int              block_count,
                                                          int              cp_rank = 0,
                                                          int              cp_size = 1);
    static std::shared_ptr<LayerCacheBuffer> convertLayer(KVCacheResource&   resource,
                                                          int                batch_id,
                                                          int                layer_id,
                                                          const std::string& cache_tag,
                                                          int                start_block_idx,
                                                          int                block_count,
                                                          int                cp_rank,
                                                          int                cp_size);

    /// @brief Return whether the selected layer/tag window contains a transferable block.
    /// Uses the same argument validation, CP key bounds, and start/count semantics as convertLayer().
    static bool hasTransferableBlocks(const KVCacheResource& resource,
                                      int                    layer_id,
                                      const std::string&     cache_tag,
                                      int                    start_block_idx,
                                      int                    block_count,
                                      int                    cp_rank,
                                      int                    cp_size);

    /// @brief 将 LayerCacheBuffer 转换为 transfer 层需要的 KeyBlockInfoMap
    static transfer::KeyBlockInfoMap buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                                        const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                                        int                                         partition_count = 1,
                                                        int                                         partition_id = 0);
};

}  // namespace rtp_llm
