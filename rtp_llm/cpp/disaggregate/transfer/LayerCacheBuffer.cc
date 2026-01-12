#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"

#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <chrono>

namespace rtp_llm {

// ==================== LayerCacheBuffer ====================

LayerCacheBuffer::LayerCacheBuffer(int layer_id): layer_id_(layer_id) {}

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
    layer_cache_buffer_map_[unique_key][layer_cache_buffer->getLayerId()] = layer_cache_buffer;
    expired_time_map_[unique_key]                                         = currentTimeMs() + timeout_ms_;
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferStore::getLayerCacheBuffer(const std::string& unique_key,
                                                                             int                layer_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = layer_cache_buffer_map_.find(unique_key);
    if (it == layer_cache_buffer_map_.end()) {
        return nullptr;
    }
    auto layer_cache_buffer_map = it->second;
    auto layer_cache_buffer_it  = layer_cache_buffer_map.find(layer_id);
    if (layer_cache_buffer_it == layer_cache_buffer_map.end()) {
        return nullptr;
    }
    return layer_cache_buffer_it->second;
}

void LayerCacheBufferStore::checkTimeout() {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        now = currentTimeMs();
    for (auto it = expired_time_map_.begin(); it != expired_time_map_.end();) {
        if (now >= it->second) {
            layer_cache_buffer_map_.erase(it->first);
            it = expired_time_map_.erase(it);
        } else {
            ++it;
        }
    }
}
}  // namespace rtp_llm
