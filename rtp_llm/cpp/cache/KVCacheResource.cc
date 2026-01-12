#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

void KVCacheResource::initGroups(int group_nums, int layer_num) {
    group_block_ids.reserve(group_block_ids.size() + static_cast<size_t>(group_nums));
    for (int i = 0; i < group_nums; i++) {
        group_block_ids.push_back(std::make_shared<BlockIds>());
    }

    if (layer_num > 0) {
        layer_block_ids.resize(layer_num);
        for (int i = 0; i < layer_num; ++i) {
            layer_block_ids[i] = group_block_ids.front();
        }
    }
}

void KVCacheResource::resizeBlocks(int reserver_blocks, int value) {
    for (auto& group : group_block_ids) {
        group->resize(reserver_blocks, value);
    }
}

int KVCacheResource::blocksNum(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return static_cast<int>(group_block_ids[group_id]->blocksNum());
}

BlockIndicesType& KVCacheResource::blocks(int group_id) const {
    RTP_LLM_CHECK(group_block_ids.size() > static_cast<size_t>(group_id));
    return group_block_ids[group_id]->blocks();
}

int KVCacheResource::groupNums() const {
    return static_cast<int>(group_block_ids.size());
}

GroupBlockIds& KVCacheResource::groupBlocks() {
    return group_block_ids;
}

const GroupBlockIds& KVCacheResource::groupBlocks() const {
    return group_block_ids;
}

LayerBlockIds& KVCacheResource::layerBlockIds() {
    return layer_block_ids;
}

const LayerBlockIds& KVCacheResource::layerBlockIds() const {
    return layer_block_ids;
}

CacheKeysType& KVCacheResource::cacheKeys() {
    return cache_keys;
}

const CacheKeysType& KVCacheResource::cacheKeys() const {
    return cache_keys;
}

std::string KVCacheResource::debugString() const {
    std::stringstream debug_string;
    const int         group_nums = static_cast<int>(group_block_ids.size());
    for (int group_id = 0; group_id < group_nums; group_id++) {
        debug_string << "group:[" << group_id << "], block:[";
        auto& block_indices = blocks(group_id);
        for (auto& block : block_indices) {
            debug_string << block << ", ";
        }
        debug_string << "], ";
    }

    return debug_string.str();
}

}  // namespace rtp_llm
