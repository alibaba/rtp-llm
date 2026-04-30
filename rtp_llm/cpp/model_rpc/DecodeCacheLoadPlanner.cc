#include "rtp_llm/cpp/model_rpc/DecodeCacheLoadPlanner.h"

#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

namespace rtp_llm {

bool useFullBlockRemoteLoad(const CacheConfig& cache_config) {
    return cache_config.use_mla || cache_config.is_sparse || cache_config.groupNums() > 1;
}

std::vector<size_t> blockPositionsForRpc(size_t         block_num,
                                         size_t         reuse_block_size,
                                         bool           use_hybrid,
                                         CacheGroupType group_type,
                                         bool           hybrid_full_from_begin) {
    return blockPositionsForCacheTransfer(block_num, reuse_block_size, use_hybrid, group_type, hybrid_full_from_begin);
}

std::string layerRegionRequestKey(size_t request_id, size_t layer_id, KVCacheRegionName region_name) {
    return layerRegionCacheTransferKey(request_id, layer_id, region_name);
}

}  // namespace rtp_llm
