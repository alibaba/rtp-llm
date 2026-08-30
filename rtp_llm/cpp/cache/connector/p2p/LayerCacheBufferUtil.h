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
    static std::vector<std::shared_ptr<LayerCacheBuffer>> convert(const CacheConfig& config,
                                                                  KVCacheResource&   resource,
                                                                  int                batch_id,
                                                                  int                start_key_ordinal = 0,
                                                                  int                key_count         = -1,
                                                                  int                cp_rank           = 0,
                                                                  int                cp_size           = 1);

    /// @brief 将 KVCacheResource 的指定层转换为单个 LayerCacheBuffer
    static std::shared_ptr<LayerCacheBuffer> convertLayer(const CacheConfig& config,
                                                          KVCacheResource&   resource,
                                                          int                batch_id,
                                                          int                layer_id,
                                                          std::string_view   tag,
                                                          int                start_key_ordinal,
                                                          int                key_count,
                                                          int                cp_rank,
                                                          int                cp_size);
    /// @brief Return whether the selected layer/tag window contains a transferable block.
    /// Uses the same argument validation, CP key bounds, and start/count semantics as convertLayer().
    static bool hasTransferableBlocks(const CacheConfig&     config,
                                      const KVCacheResource& resource,
                                      int                    layer_id,
                                      std::string_view       tag,
                                      int                    start_key_ordinal,
                                      int                    key_count,
                                      int                    cp_rank,
                                      int                    cp_size);

    /// @brief 将 LayerCacheBuffer 转换为 transfer 层需要的 KeyBlockInfoMap
    static transfer::KeyBlockInfoMap buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                                        const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                                        int                                         partition_count = 1,
                                                        int                                         partition_id = 0);

protected:
    // Test-only observer for proving invalid selection sets are rejected before
    // a LayerCacheBuffer is constructed or published. Normal callers use the
    // public overloads above and always pass no observer.
    class ConversionObserver {
    public:
        virtual ~ConversionObserver()                = default;
        virtual void onLayerCacheBufferConstructed() = 0;
        virtual void onLayerCacheBufferPublished()   = 0;
    };

    static std::vector<std::shared_ptr<LayerCacheBuffer>> convert(const CacheConfig&  config,
                                                                  KVCacheResource&    resource,
                                                                  int                 batch_id,
                                                                  int                 start_key_ordinal,
                                                                  int                 key_count,
                                                                  int                 cp_rank,
                                                                  int                 cp_size,
                                                                  ConversionObserver* observer);
    static std::shared_ptr<LayerCacheBuffer>              convertLayer(const CacheConfig&  config,
                                                                       KVCacheResource&    resource,
                                                                       int                 batch_id,
                                                                       int                 layer_id,
                                                                       std::string_view    tag,
                                                                       int                 start_key_ordinal,
                                                                       int                 key_count,
                                                                       int                 cp_rank,
                                                                       int                 cp_size,
                                                                       ConversionObserver* observer);
};

}  // namespace rtp_llm
