#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

namespace {

constexpr int64_t kRemovedIdRetentionMs = 3600000;

}  // namespace

ComputedLayerCacheBuffer::ComputedLayerCacheBuffer(int64_t                                  request_id,
                                                   const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                                   int64_t                                  deadline_ms):
    request_id_(request_id), deadline_ms_(deadline_ms) {
    if (layer_cache_buffer) {
        layer_cache_buffers_[layer_cache_buffer->getLayerId()] = layer_cache_buffer;
    }
}

void ComputedLayerCacheBuffer::addBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                         int64_t                                  deadline_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_cache_buffer) {
        layer_cache_buffers_[layer_cache_buffer->getLayerId()] = layer_cache_buffer;
    }
    int64_t cur = deadline_ms_.load(std::memory_order_relaxed);
    if (deadline_ms > cur) {
        deadline_ms_.store(deadline_ms, std::memory_order_relaxed);
    }
    condition_variable_.notify_all();
}

std::pair<int, std::vector<std::shared_ptr<LayerCacheBuffer>>>
ComputedLayerCacheBuffer::getBuffers(const std::set<int>& layer_ids) {
    std::lock_guard<std::mutex>                    lock(mutex_);
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    for (auto layer_id : layer_ids) {
        auto iter = layer_cache_buffers_.find(layer_id);
        if (iter != layer_cache_buffers_.end()) {
            layer_cache_buffers.push_back(iter->second);
        }
    }
    return {static_cast<int>(layer_cache_buffers_.size()), layer_cache_buffers};
}

void ComputedLayerCacheBuffer::waitChange(int last_layer_num, int timeout_ms) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (static_cast<int>(layer_cache_buffers_.size()) != last_layer_num) {
        return;
    }
    condition_variable_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, last_layer_num] {
        return static_cast<int>(layer_cache_buffers_.size()) > last_layer_num;
    });
}

ComputedLayerCacheBufferStore::ComputedLayerCacheBufferStore() {}

ComputedLayerCacheBufferStore::~ComputedLayerCacheBufferStore() {}

std::shared_ptr<ComputedLayerCacheBuffer> ComputedLayerCacheBufferStore::addBuffer(
    int64_t request_id, const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms) {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);

    if (removed_request_ids_.count(request_id)) {
        return nullptr;
    }

    auto iter = computed_buffers_.find(request_id);
    if (iter != computed_buffers_.end()) {
        // 使用现有的 ComputedLayerCacheBuffer 的 addBuffer 方法
        iter->second->addBuffer(layer_cache_buffer, deadline_ms);
        return iter->second;
    }

    auto new_computed_layer_cache_buffer =
        std::make_shared<ComputedLayerCacheBuffer>(request_id, layer_cache_buffer, deadline_ms);
    computed_buffers_[request_id] = new_computed_layer_cache_buffer;
    return new_computed_layer_cache_buffer;
}

std::shared_ptr<ComputedLayerCacheBuffer> ComputedLayerCacheBufferStore::getBuffer(int64_t request_id) const {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);
    auto                        iter = computed_buffers_.find(request_id);
    if (iter != computed_buffers_.end()) {
        return iter->second;
    }
    return nullptr;
}

void ComputedLayerCacheBufferStore::removeBuffer(int64_t request_id) {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);
    computed_buffers_.erase(request_id);
    markRemovedLocked(request_id, currentTimeMs());
}

int64_t ComputedLayerCacheBufferStore::getBuffersCount() const {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);
    return static_cast<int64_t>(computed_buffers_.size());
}

void ComputedLayerCacheBufferStore::checkTimeout() {
    std::unique_lock<std::mutex> lock(computed_buffers_mutex_);
    int64_t                      current_time_ms = currentTimeMs();
    for (auto iter = computed_buffers_.begin(); iter != computed_buffers_.end();) {
        if (current_time_ms >= iter->second->deadlineMs()) {
            markRemovedLocked(iter->first, current_time_ms);
            iter                              = computed_buffers_.erase(iter);
        } else {
            ++iter;
        }
    }
    while (!removed_request_expiry_queue_.empty()) {
        const auto& expiry = removed_request_expiry_queue_.top();
        if (expiry.expire_at_ms > current_time_ms) {
            break;
        }
        auto it = removed_request_ids_.find(expiry.request_id);
        if (it != removed_request_ids_.end() && it->second == expiry.removed_at_ms) {
            removed_request_ids_.erase(it);
        }
        removed_request_expiry_queue_.pop();
    }
}

void ComputedLayerCacheBufferStore::markRemovedLocked(int64_t request_id, int64_t removed_at_ms) {
    removed_request_ids_[request_id] = removed_at_ms;
    removed_request_expiry_queue_.push(
        RemovedRequestExpiry{removed_at_ms + kRemovedIdRetentionMs, request_id, removed_at_ms});
}

}  // namespace rtp_llm
