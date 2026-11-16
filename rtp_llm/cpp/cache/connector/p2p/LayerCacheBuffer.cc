#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"

#include <utility>

namespace rtp_llm {

LayerCacheBuffer::LayerCacheBuffer(int layer_id, std::string cache_tag, int chunk_id):
    layer_id_(layer_id), cache_tag_(std::move(cache_tag)), chunk_id_(chunk_id) {}

void LayerCacheBuffer::addBlockId(int64_t key, int block_id) {
    block_id_map_[key] = block_id;
}

int LayerCacheBuffer::getBlockId(int64_t cache_key) const {
    auto it = block_id_map_.find(cache_key);
    if (it == block_id_map_.end()) {
        return -1;
    }
    return it->second;
}

}  // namespace rtp_llm
