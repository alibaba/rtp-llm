#include "rtp_llm/cpp/disaggregate/p2p_connector/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

ComputedLayerCacheBufferStore::ComputedLayerCacheBufferStore() {}

ComputedLayerCacheBufferStore::~ComputedLayerCacheBufferStore() {}

void ComputedLayerCacheBufferStore::addBuffer(int64_t                                  request_id,
                                              const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                              int64_t                                  deadline_ms) {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);

    auto iter = computed_buffers_.find(request_id);
    if (iter != computed_buffers_.end()) {
        iter->second->layer_cache_buffers[layer_cache_buffer->getLayerId()] = layer_cache_buffer;
        return;
    }

    auto new_computed_layer_cache_buffer =
        std::make_shared<ComputedLayerCacheBuffer>(request_id, layer_cache_buffer, deadline_ms);
    computed_buffers_[request_id] = new_computed_layer_cache_buffer;
}

std::shared_ptr<ComputedLayerCacheBuffer> ComputedLayerCacheBufferStore::getBuffer(int64_t request_id) const {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);
    auto                        iter = computed_buffers_.find(request_id);
    if (iter == computed_buffers_.end()) {
        return nullptr;
    }
    return iter->second;
}

void ComputedLayerCacheBufferStore::removeBuffer(int64_t request_id) {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);
    computed_buffers_.erase(request_id);
}

void ComputedLayerCacheBufferStore::checkTimeout() {
    std::unique_lock<std::mutex> lock(computed_buffers_mutex_);
    int64_t                      current_time_ms = currentTimeMs();
    for (auto iter = computed_buffers_.begin(); iter != computed_buffers_.end();) {
        if (current_time_ms >= iter->second->deadline_ms) {
            RTP_LLM_LOG_INFO("P2PConnectorPrefillWorker storeWaitThread erase computed_buffers_, request_id: %ld",
                             iter->first);
            iter = computed_buffers_.erase(iter);
        } else {
            ++iter;
        }
    }
}

}  // namespace rtp_llm