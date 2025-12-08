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

// ==================== LayerCacheBufferTask ====================

void LayerCacheBufferTask::waitDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    while (true) {
        if (cancelled_) {
            RTP_LLM_LOG_WARNING("LayerCacheBufferTask waitDone cancelled");
            all_success_ = false;
            return;
        }
        if (done_layer_ids_.size() == layer_cache_buffers_.size()) {
            return;
        }
        // 计算剩余时间
        int64_t now_ms = currentTimeMs();
        if (now_ms >= deadline_ms_) {
            all_success_ = false;
            RTP_LLM_LOG_WARNING("LayerCacheBufferTask waitDone timeout, deadline_ms: %lld", deadline_ms_);
            return;
        }
        int64_t remaining_ms = deadline_ms_ - now_ms;
        // 使用 wait_until 等待到截止时间
        auto deadline_time = std::chrono::steady_clock::now() + std::chrono::milliseconds(remaining_ms);
        bool finished      = cond_.wait_until(lock, deadline_time, [this] {
            return done_layer_ids_.size() == layer_cache_buffers_.size() || cancelled_;
        });
    }
}

void LayerCacheBufferTask::waitLoadingDone() {
    std::unique_lock<std::mutex> lock(mutex_);
    cond_.wait(lock, [this] { return loading_layer_ids_.empty(); });
}

bool LayerCacheBufferTask::success() const {
    return all_success_;
}

bool LayerCacheBufferTask::cancelled() const {
    return cancelled_;
}

void LayerCacheBufferTask::setCancelled() {
    std::lock_guard<std::mutex> lock(mutex_);
    cancelled_ = true;
    cond_.notify_all();
}

void LayerCacheBufferTask::setLoading(int layer_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (auto iter = layer_cache_buffers_.find(layer_id); iter == layer_cache_buffers_.end()) {
        RTP_LLM_LOG_WARNING("LayerCacheBufferTask setLoading failed: layer_id not found, layer_id: %d", layer_id);
        return;
    }
    loading_layer_ids_.insert(layer_id);
}

void LayerCacheBufferTask::notifyDone(int layer_id, bool success) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (auto iter = layer_cache_buffers_.find(layer_id); iter == layer_cache_buffers_.end()) {
        RTP_LLM_LOG_WARNING("LayerCacheBufferTask notifyDone failed: layer_id not found, layer_id: %d", layer_id);
        return;
    }
    if (!success) {
        all_success_ = false;
    }
    done_layer_ids_.insert(layer_id);
    loading_layer_ids_.erase(layer_id);
    cond_.notify_all();
}

std::shared_ptr<LayerCacheBuffer> LayerCacheBufferTask::getLayerCacheBuffer(int layer_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        iter = layer_cache_buffers_.find(layer_id);
    if (iter == layer_cache_buffers_.end()) {
        return nullptr;
    }
    return iter->second;
}

// ==================== LayerCacheBufferTaskStore ====================

LayerCacheBufferTaskStore::LayerCacheBufferTaskStore() {}

std::shared_ptr<LayerCacheBufferTask>
LayerCacheBufferTaskStore::addTask(const std::string&                                      unique_key,
                                   const std::map<int, std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                                   int64_t                                                 deadline_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    task_map_[unique_key] = std::make_shared<LayerCacheBufferTask>(layer_cache_buffers, deadline_ms);
    return task_map_[unique_key];
}

std::shared_ptr<LayerCacheBufferTask> LayerCacheBufferTaskStore::getTask(const std::string& unique_key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = task_map_.find(unique_key);
    if (it == task_map_.end()) {
        return nullptr;
    }
    return it->second;
}

std::shared_ptr<LayerCacheBufferTask> LayerCacheBufferTaskStore::stealTask(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto                        it = task_map_.find(unique_key);
    if (it == task_map_.end()) {
        return nullptr;
    }
    auto task = it->second;
    task_map_.erase(it->first);
    return task;
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
