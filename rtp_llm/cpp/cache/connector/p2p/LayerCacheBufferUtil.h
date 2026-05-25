#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/Types.h"
#include <vector>
#include <memory>

namespace rtp_llm {

class LayerCacheBufferUtil {
public:
    static std::vector<std::shared_ptr<LayerCacheBuffer>> convert(KVCacheResource& resource,
                                                                  int              batch_id,
                                                                  int              start_block_idx = 0,
                                                                  int              block_count     = -1,
                                                                  int              cp_rank         = 0,
                                                                  int              cp_size         = 1);

    static std::vector<std::shared_ptr<LayerCacheBuffer>>
    convertWithRegions(KVCacheResource&                     resource,
                       int                                  batch_id,
                       const std::vector<std::vector<int>>& layer_region_to_group_id,
                       const std::vector<CacheGroupType>&   group_types,
                       int                                  start_block_idx = 0,
                       int                                  block_count     = -1);

    static std::shared_ptr<LayerCacheBuffer> convertLayer(KVCacheResource& resource,
                                                          int              batch_id,
                                                          int              layer_id,
                                                          int              start_block_idx,
                                                          int              block_count,
                                                          int              cp_rank = 0,
                                                          int              cp_size = 1);

    static std::shared_ptr<LayerCacheBuffer> convertLayerRegion(KVCacheResource&  resource,
                                                                int               batch_id,
                                                                int               layer_id,
                                                                KVCacheRegionName region_name,
                                                                int               start_block_idx,
                                                                int               block_count);

    static transfer::KeyBlockInfoMap buildKeyBlockInfos(const std::shared_ptr<LayerBlockConverter>& converter,
                                                        const std::shared_ptr<LayerCacheBuffer>&    layer_cache_buffer,
                                                        int                                         partition_count = 1,
                                                        int                                         partition_id = 0);
};

}  // namespace rtp_llm
