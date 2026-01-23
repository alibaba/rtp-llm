#include "rtp_llm/cpp/cache/KVCacheResource.h"

namespace rtp_llm {

void KVCacheResource::initGroups(int group_num, int layer_num) {
    group_block_ids.reserve(group_block_ids.size() + static_cast<size_t>(group_num));
    for (int i = 0; i < group_num; i++) {
        group_block_ids.push_back(std::make_shared<BlockIds>());
    }
    if (!group_block_ids.empty()) {
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
    if (group_block_ids.empty()) {
        layer_block_ids.clear();
    } else {
        for (auto& layer : layer_block_ids) {
            layer = group_block_ids.front();
        }
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

const LayerBlockIds& KVCacheResource::layerBlocks() const {
    return layer_block_ids;
}

CacheKeysType& KVCacheResource::cacheKeys() {
    return cache_keys;
}

const CacheKeysType& KVCacheResource::cacheKeys() const {
    return cache_keys;
}

size_t KVCacheResource::reuseBlocksNum() const {
    return reuse_blocks_num;
}

void KVCacheResource::setReuseBlocksNum(size_t reuse_blocks_num) {
    this->reuse_blocks_num = reuse_blocks_num;
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
        debug_string << "], cache_keys:[";
        for (auto& key : cache_keys) {
            debug_string << key << ", ";
        }
        debug_string << "], ";
    }
    debug_string << "layer_num:[" << layer_block_ids.size() << "] ";
    debug_string << "reuse_blocks_num:[" << reuse_blocks_num << "] ";
    debug_string << "last_block_aligned:[" << last_block_aligned_ << "]";

    return debug_string.str();
}

bool KVCacheResource::lastBlockAligned() const {
    return last_block_aligned_;
}

void KVCacheResource::setLastBlockAligned(bool aligned) {
    last_block_aligned_ = aligned;
}

}  // namespace rtp_llm
