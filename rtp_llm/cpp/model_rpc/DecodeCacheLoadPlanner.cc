#include "rtp_llm/cpp/model_rpc/DecodeCacheLoadPlanner.h"

namespace rtp_llm {

bool useFullBlockRemoteLoad(const CacheConfig& cache_config) {
    return cache_config.use_mla || cache_config.is_sparse || cache_config.groupNums() > 1;
}

std::vector<size_t> blockPositionsForRpc(size_t         block_num,
                                         size_t         reuse_block_size,
                                         bool           use_hybrid,
                                         CacheGroupType group_type,
                                         bool           hybrid_full_from_begin) {
    std::vector<size_t> block_pos_list;
    block_pos_list.reserve(block_num);
    if (use_hybrid && block_num > 0 && group_type == CacheGroupType::LINEAR) {
        const size_t start = block_num > 2 ? block_num - 2 : 0;
        for (size_t block_pos = start; block_pos < block_num; ++block_pos) {
            block_pos_list.push_back(block_pos);
        }
        return block_pos_list;
    }
    const size_t start = use_hybrid && hybrid_full_from_begin ? 0 : reuse_block_size;
    for (size_t block_pos = start; block_pos < block_num; ++block_pos) {
        block_pos_list.push_back(block_pos);
    }
    return block_pos_list;
}

std::string layerRegionRequestKey(size_t request_id, size_t layer_id, KVCacheRegionName region_name) {
    auto key = std::to_string(request_id) + "-" + std::to_string(layer_id);
    if (region_name != KVCacheRegionName::DEFAULT) {
        key += "-" + std::to_string(static_cast<int>(region_name));
    }
    return key;
}

}  // namespace rtp_llm
