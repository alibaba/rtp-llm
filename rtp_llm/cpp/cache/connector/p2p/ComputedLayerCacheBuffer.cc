#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <limits>

namespace rtp_llm {

namespace {

bool hasFiniteDeadline(int64_t deadline_ms) {
    return deadline_ms > 0 && deadline_ms != std::numeric_limits<int64_t>::max();
}

}  // namespace

ComputedLayerCacheBuffer::ComputedLayerCacheBuffer(int64_t                                  request_id,
                                                   const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                                   int64_t                                  deadline_ms):
    request_id_(request_id), deadline_ms_(deadline_ms) {
    if (layer_cache_buffer) {
        layer_cache_buffers_[layer_cache_buffer->bufferKey()] = layer_cache_buffer;
    }
}

void ComputedLayerCacheBuffer::addBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                         int64_t                                  deadline_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_cache_buffer) {
        layer_cache_buffers_[layer_cache_buffer->bufferKey()] = layer_cache_buffer;
    }
    int64_t cur = deadline_ms_.load(std::memory_order_relaxed);
    if (deadline_ms < cur) {
        deadline_ms_.store(deadline_ms, std::memory_order_relaxed);
    }
    condition_variable_.notify_all();
}

std::pair<int, std::vector<std::shared_ptr<LayerCacheBuffer>>>
ComputedLayerCacheBuffer::getBuffers(const std::set<std::string>& buffer_keys) {
    std::lock_guard<std::mutex>                    lock(mutex_);
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    for (const auto& buffer_key : buffer_keys) {
        auto iter = layer_cache_buffers_.find(buffer_key);
        if (iter != layer_cache_buffers_.end()) {
            layer_cache_buffers.push_back(iter->second);
        }
    }
    // The count is the total number already stored, not only the number that
    // matched this lookup. dispatchPendingLayerTransfers uses it as the
    // waitChange baseline; returning only matches would make a request spin
    // when an unrelated layer/tag is already present.
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

ComputedLayerCacheBufferStore::ComputedLayerCacheBufferStore() = default;

ComputedLayerCacheBufferStore::~ComputedLayerCacheBufferStore() {}

std::shared_ptr<ComputedLayerCacheBuffer> ComputedLayerCacheBufferStore::addBuffer(
    int64_t request_id, const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms) {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);

    if (removed_request_ids_.count(request_id) || !hasFiniteDeadline(deadline_ms)) {
        return nullptr;
    }
    // Callbacks may carry the request deadline from before StartLoad.
    // The store's phase deadline is authoritative while holding this lock.
    auto horizon = request_horizons_.find(request_id);
    if (horizon == request_horizons_.end()) {
        return nullptr;
    }
    deadline_ms = horizon->second.horizon_ms;
    if (currentTimeMs() >= deadline_ms) {
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

std::optional<int64_t> ComputedLayerCacheBufferStore::registerRequestHorizon(int64_t request_id,
                                                                             int64_t horizon_ms,
                                                                             int64_t request_deadline_ms) {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);
    if (removed_request_ids_.count(request_id) || !hasFiniteDeadline(request_deadline_ms)
        || !hasFiniteDeadline(horizon_ms) || horizon_ms > request_deadline_ms
        || currentTimeMs() >= horizon_ms) {
        return std::nullopt;
    }
    auto result = request_horizons_.emplace(request_id, RequestHorizon{horizon_ms, request_deadline_ms});
    if (result.first->second.request_deadline_ms != request_deadline_ms
        || currentTimeMs() >= result.first->second.horizon_ms) {
        return std::nullopt;
    }
    return result.first->second.horizon_ms;
}

std::optional<int64_t> ComputedLayerCacheBufferStore::activateRequestHorizon(int64_t request_id,
                                                                             int64_t horizon_ms,
                                                                             int64_t request_deadline_ms) {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);
    if (removed_request_ids_.count(request_id) || !hasFiniteDeadline(request_deadline_ms)
        || !hasFiniteDeadline(horizon_ms) || horizon_ms > request_deadline_ms
        || currentTimeMs() >= horizon_ms) {
        return std::nullopt;
    }
    auto [it, inserted] = request_horizons_.emplace(request_id, RequestHorizon{horizon_ms, request_deadline_ms});
    if (!inserted) {
        if (it->second.request_deadline_ms != request_deadline_ms || currentTimeMs() >= it->second.horizon_ms) {
            return std::nullopt;
        }
        it->second.horizon_ms = std::min(it->second.horizon_ms, horizon_ms);
    }
    auto buffer_it = computed_buffers_.find(request_id);
    if (buffer_it != computed_buffers_.end()) {
        // Update the stored buffer under the same store lock so checkTimeout
        // cannot remove it between StartLoad activation and sendKVCache's
        // subsequent lookup.
        buffer_it->second->addBuffer(nullptr, it->second.horizon_ms);
    }
    return it->second.horizon_ms;
}

std::optional<int64_t> ComputedLayerCacheBufferStore::requestHorizon(int64_t request_id) const {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);
    auto                        it = request_horizons_.find(request_id);
    return it == request_horizons_.end() || removed_request_ids_.count(request_id) ?
               std::nullopt : std::make_optional(it->second.horizon_ms);
}

void ComputedLayerCacheBufferStore::removeBuffer(int64_t request_id, int64_t request_deadline_ms) {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);
    std::optional<int64_t>      finite_request_deadline;
    if (hasFiniteDeadline(request_deadline_ms)) {
        finite_request_deadline = request_deadline_ms;
    }
    auto horizon_it = request_horizons_.find(request_id);
    if (horizon_it != request_horizons_.end()) {
        if (hasFiniteDeadline(horizon_it->second.request_deadline_ms)) {
            finite_request_deadline = finite_request_deadline.has_value() ?
                                          std::max(*finite_request_deadline,
                                                   horizon_it->second.request_deadline_ms) :
                                          horizon_it->second.request_deadline_ms;
        }
        request_horizons_.erase(horizon_it);
    }
    auto buffer_it = computed_buffers_.find(request_id);
    if (buffer_it != computed_buffers_.end()) {
        computed_buffers_.erase(buffer_it);
    }
    if (finite_request_deadline) {
        markRemovedLocked(request_id, *finite_request_deadline);
    }
}

int64_t ComputedLayerCacheBufferStore::getBuffersCount() const {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);
    return static_cast<int64_t>(computed_buffers_.size());
}

void ComputedLayerCacheBufferStore::checkTimeout() {
    std::unique_lock<std::mutex> lock(computed_buffers_mutex_);
    int64_t                      current_time_ms = currentTimeMs();
    // Clean tombstones created by earlier checks first. Entries created below
    // remain visible for at least one checker cycle even when their horizon is
    // exactly the current time.
    while (!removed_request_expiry_queue_.empty()) {
        const auto& expiry = removed_request_expiry_queue_.top();
        if (expiry.expire_at_ms > current_time_ms) {
            break;
        }
        auto it = removed_request_ids_.find(expiry.request_id);
        if (it != removed_request_ids_.end() && it->second == expiry.expire_at_ms) {
            removed_request_ids_.erase(it);
        }
        removed_request_expiry_queue_.pop();
    }
    for (auto it = request_horizons_.begin(); it != request_horizons_.end();) {
        if (current_time_ms >= it->second.horizon_ms) {
            markRemovedLocked(it->first, it->second.request_deadline_ms);
            computed_buffers_.erase(it->first);
            it = request_horizons_.erase(it);
        } else {
            ++it;
        }
    }
}

void ComputedLayerCacheBufferStore::markRemovedLocked(int64_t request_id, int64_t expire_at_ms) {
    auto [it, inserted] = removed_request_ids_.emplace(request_id, expire_at_ms);
    if (!inserted && expire_at_ms > it->second) {
        it->second = expire_at_ms;
    }
    removed_request_expiry_queue_.push(RemovedRequestExpiry{it->second, request_id});
}

}  // namespace rtp_llm
