#include "rtp_llm/cpp/disaggregate/cache_store_new/LayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

LayerCacheBuffer::LayerCacheBuffer(int layer_id): layer_id_(layer_id) {}

void LayerCacheBuffer::addBlockCacheBuffer(int64_t key, BufferPtr k_buffer, BufferPtr v_buffer) {
    block_cache_buffers_[key] = std::make_shared<BlockCacheBuffer>(key, k_buffer, v_buffer);
}

std::shared_ptr<BlockCacheBuffer> LayerCacheBuffer::getBlockCacheBuffer(int64_t key) {
    auto iter = block_cache_buffers_.find(key);
    if (iter != block_cache_buffers_.end()) {
        return iter->second;
    }
    return nullptr;
}

SingleLayerCacheBufferStore::SingleLayerCacheBufferStore(int layer_id): layer_id_(layer_id) {}

bool SingleLayerCacheBufferStore::setLayerCacheBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                                      int64_t                                  deadline_ms) {
    {
        std::unique_lock<std::mutex> lock(layer_cache_buffers_mutex_);
        layer_cache_buffer_map_[layer_cache_buffer] = deadline_ms;
    }
    notifyAllWatchers(layer_cache_buffer);
    return true;
}

void SingleLayerCacheBufferStore::notifyAllWatchers(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) {
    std::vector<std::shared_ptr<Watcher>> watchers;
    {
        std::unique_lock<std::mutex> lock(layer_cache_buffer_watchers_mutex_);
        watchers.reserve(layer_cache_buffer_watcher_map_.size());
        for (auto& watcher : layer_cache_buffer_watcher_map_) {
            watchers.push_back(watcher.first);
        }
    }
    for (auto& watcher : watchers) {
        if (watcher->notify(layer_cache_buffer)) {
            delLayerCacheBufferWatchFunc(watcher);
            delLayerCacheBuffer(layer_cache_buffer);
            break;
        }
    }
}

void SingleLayerCacheBufferStore::delLayerCacheBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer) {
    std::unique_lock<std::mutex> lock(layer_cache_buffers_mutex_);
    layer_cache_buffer_map_.erase(layer_cache_buffer);
}

void SingleLayerCacheBufferStore::setLayerCacheBufferWatchFunc(std::shared_ptr<Watcher> watcher, int64_t deadline_ms) {
    {
        std::unique_lock<std::mutex> lock(layer_cache_buffer_watchers_mutex_);
        layer_cache_buffer_watcher_map_[watcher] = deadline_ms;
    }
    watchAllCacheBuffers(watcher);
}

void SingleLayerCacheBufferStore::watchAllCacheBuffers(const std::shared_ptr<Watcher>& watcher) {
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    {
        std::unique_lock<std::mutex> lock(layer_cache_buffers_mutex_);
        layer_cache_buffers.reserve(layer_cache_buffer_map_.size());
        for (auto& layer_cache_buffer : layer_cache_buffer_map_) {
            layer_cache_buffers.push_back(layer_cache_buffer.first);
        }
    }
    for (auto& layer_cache_buffer : layer_cache_buffers) {
        if (watcher->notify(layer_cache_buffer)) {
            delLayerCacheBufferWatchFunc(watcher);
            delLayerCacheBuffer(layer_cache_buffer);
            break;
        }
    }
}

void SingleLayerCacheBufferStore::delLayerCacheBufferWatchFunc(std::shared_ptr<Watcher> watcher) {
    std::unique_lock<std::mutex> lock(layer_cache_buffer_watchers_mutex_);
    layer_cache_buffer_watcher_map_.erase(watcher);
}

void SingleLayerCacheBufferStore::checkTimeout() {
    auto current_time_us = currentTimeUs();
    {
        std::unique_lock<std::mutex> lock(layer_cache_buffers_mutex_);
        auto                         it = layer_cache_buffer_map_.begin();
        while (it != layer_cache_buffer_map_.end()) {
            if (current_time_us >= it->second) {
                it = layer_cache_buffer_map_.erase(it);
            } else {
                ++it;
            }
        }
    }
    {
        std::unique_lock<std::mutex> lock(layer_cache_buffer_watchers_mutex_);
        auto                         it = layer_cache_buffer_watcher_map_.begin();
        while (it != layer_cache_buffer_watcher_map_.end()) {
            if (current_time_us >= it->second) {
                it = layer_cache_buffer_watcher_map_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

LayerCacheBufferStore::LayerCacheBufferStore(int layer_num): layer_num_(layer_num) {
    for (int i = 0; i < layer_num; ++i) {
        single_layer_cache_buffer_stores_.push_back(std::make_shared<SingleLayerCacheBufferStore>(i));
    }
}

std::shared_ptr<SingleLayerCacheBufferStore> LayerCacheBufferStore::getSingleLayerCacheBufferStore(int layer_id) const {
    RTP_LLM_CHECK(layer_id >= 0 && layer_id < layer_num_);
    return single_layer_cache_buffer_stores_[layer_id];
}

void LayerCacheBufferStore::checkTimeoutThread() {
    while (!check_timeout_thread_stop_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        for (int i = 0; i < layer_num_; ++i) {
            single_layer_cache_buffer_stores_[i]->checkTimeout();
        }
    }
}

}  // namespace rtp_llm