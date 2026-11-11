#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

namespace rtp_llm {

ComputedLayerCacheBuffer::ComputedLayerCacheBuffer(int64_t                                  request_id,
                                                   const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                                   int64_t                                  deadline_ms):
    request_id_(request_id), deadline_ms_(deadline_ms) {
    if (layer_cache_buffer) {
        layer_cache_buffers_[layer_cache_buffer->descriptor()] = layer_cache_buffer;
    }
}

void ComputedLayerCacheBuffer::addBuffer(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                         int64_t                                  deadline_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (layer_cache_buffer) {
        const auto descriptor = layer_cache_buffer->descriptor();
        if (expected_descriptors_.has_value() && expected_descriptors_->count(descriptor) == 0) {
            RTP_LLM_LOG_WARNING(
                "P2P publication rejected: descriptor module=%zu layer=%d tag=%s is not in the sealed set",
                descriptor.module_id,
                descriptor.layer_id,
                descriptor.cache_tag.c_str());
        } else {
            if (layer_cache_buffers_.count(descriptor) != 0) {
                RTP_LLM_LOG_DEBUG("P2P duplicate publication for module=%zu layer=%d tag=%s ignored",
                                  descriptor.module_id,
                                  descriptor.layer_id,
                                  descriptor.cache_tag.c_str());
            }
            layer_cache_buffers_[descriptor] = layer_cache_buffer;
        }
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
    for (const auto& [descriptor, buffer] : layer_cache_buffers_) {
        if (layer_ids.count(descriptor.layer_id)) {
            layer_cache_buffers.push_back(buffer);
        }
    }
    return {static_cast<int>(layer_cache_buffers_.size()), layer_cache_buffers};
}

std::vector<std::shared_ptr<LayerCacheBuffer>> ComputedLayerCacheBuffer::getAllBuffers() {
    std::lock_guard<std::mutex>                    lock(mutex_);
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    layer_cache_buffers.reserve(layer_cache_buffers_.size());
    for (const auto& [descriptor, buffer] : layer_cache_buffers_) {
        layer_cache_buffers.push_back(buffer);
    }
    return layer_cache_buffers;
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

void ComputedLayerCacheBuffer::sealExpectedDescriptors(std::set<LayerCacheDescriptor> expected_descriptors) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        RTP_LLM_CHECK_WITH_INFO(!expected_descriptors_.has_value() || *expected_descriptors_ == expected_descriptors,
                                "inconsistent sealed P2P descriptor set: old=%zu new=%zu",
                                expected_descriptors_.has_value() ? expected_descriptors_->size() : 0,
                                expected_descriptors.size());
        expected_descriptors_ = std::move(expected_descriptors);
        if (!expected_buffer_count_.has_value()) {
            expected_buffer_count_ = expected_descriptors_->size();
        } else {
            RTP_LLM_CHECK_WITH_INFO(*expected_buffer_count_ == expected_descriptors_->size(),
                                    "sealed P2P descriptor count=%zu disagrees with expected buffer count=%zu",
                                    expected_descriptors_->size(),
                                    *expected_buffer_count_);
        }
    }
    condition_variable_.notify_all();
}

std::optional<std::set<LayerCacheDescriptor>> ComputedLayerCacheBuffer::expectedDescriptors() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return expected_descriptors_;
}

bool ComputedLayerCacheBuffer::hasSealedDescriptors() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return expected_descriptors_.has_value();
}

bool ComputedLayerCacheBuffer::descriptorsComplete() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!expected_descriptors_.has_value() || layer_cache_buffers_.size() != expected_descriptors_->size()) {
        return false;
    }
    for (const auto& descriptor : *expected_descriptors_) {
        if (layer_cache_buffers_.count(descriptor) == 0) {
            return false;
        }
    }
    return true;
}

void ComputedLayerCacheBuffer::setExpectedBufferCount(size_t expected_buffer_count) {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        RTP_LLM_CHECK_WITH_INFO(!expected_buffer_count_.has_value() || *expected_buffer_count_ == expected_buffer_count,
                                "inconsistent expected P2P layer cache buffer count: old=%zu new=%zu",
                                expected_buffer_count_.value_or(0),
                                expected_buffer_count);
        RTP_LLM_CHECK_WITH_INFO(!expected_descriptors_.has_value()
                                    || expected_descriptors_->size() == expected_buffer_count,
                                "expected P2P layer cache buffer count=%zu disagrees with sealed descriptors=%zu",
                                expected_buffer_count,
                                expected_descriptors_->size());
        expected_buffer_count_ = expected_buffer_count;
    }
    condition_variable_.notify_all();
}

std::optional<size_t> ComputedLayerCacheBuffer::expectedBufferCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return expected_buffer_count_;
}

ComputedLayerCacheBufferStore::ComputedLayerCacheBufferStore() {}

ComputedLayerCacheBufferStore::~ComputedLayerCacheBufferStore() {}

std::shared_ptr<ComputedLayerCacheBuffer> ComputedLayerCacheBufferStore::addBuffer(
    int64_t request_id, const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer, int64_t deadline_ms) {
    std::lock_guard<std::mutex> lock(computed_buffers_mutex_);

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
            iter = computed_buffers_.erase(iter);
        } else {
            ++iter;
        }
    }
}

}  // namespace rtp_llm
