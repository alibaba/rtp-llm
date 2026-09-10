#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"

#include <utility>

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

// ==================== LayerCacheBuffer ====================

LayerCacheBuffer::LayerCacheBuffer(int layer_id, std::string cache_tag):
    layer_id_(layer_id), cache_tag_(std::move(cache_tag)) {
    RTP_LLM_CHECK_WITH_INFO(!cache_tag_.empty(), "LayerCacheBuffer requires a non-empty cache tag");
}

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

// ==================== LayerCacheBufferStore ====================

LayerCacheBufferStore::LayerCacheBufferStore(uint64_t timeout_ms): timeout_ms_(timeout_ms) {}

void LayerCacheBufferStore::addLayerCacheBuffer(const std::string&                       unique_key,
                                                const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    layer_cache_buffer_map_[unique_key][{layer_cache_buffer->getLayerId(), layer_cache_buffer->cacheTag()}] =
        layer_cache_buffer;
    expired_time_map_[unique_key] = currentTimeMs() + timeout_ms_;
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferStore::getLayerCacheBuffer(const std::string& unique_key,
                                                                             int                layer_id,
                                                                             const std::string& cache_tag) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = layer_cache_buffer_map_.find(unique_key);
    if (it == layer_cache_buffer_map_.end()) {
        return nullptr;
    }
    auto it2 = it->second.find({layer_id, cache_tag});
    if (it2 == it->second.end()) {
        return nullptr;
    }
    return it2->second;
}

void LayerCacheBufferStore::checkTimeout() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        now = currentTimeMs();
    for (auto iter = expired_time_map_.begin(); iter != expired_time_map_.end();) {
        if (iter->second < now) {
            layer_cache_buffer_map_.erase(iter->first);
            iter = expired_time_map_.erase(iter);
        } else {
            ++iter;
        }
    }
}

}  // namespace rtp_llm
