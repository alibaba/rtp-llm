#include "rtp_llm/cpp/cache/events/LogPublisher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <sstream>
#include <thread>
#include <utility>

#include "rtp_llm/cpp/cache/events/KVCacheEventQueue.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace {

const char* eventTypeName(KVCacheEventType type) {
    switch (type) {
        case KVCacheEventType::BLOCK_ADD:
            return "BLOCK_ADD";
        case KVCacheEventType::BLOCK_DELETE:
            return "BLOCK_DELETE";
    }
    return "UNKNOWN";
}

}  // namespace

class LogPublisher::Impl {
public:
    Impl(KVCacheEventPublisherConfig config, KVCacheEventPublisherContext context):
        config_(std::move(config)), context_(std::move(context)), queue_(config_.queue_capacity) {}

    ~Impl() {
        stop();
    }

    bool start() noexcept {
        bool expected = false;
        if (!started_.compare_exchange_strong(expected, true)) {
            return true;
        }
        state_.store(PublisherState::STARTING, std::memory_order_relaxed);
        try {
            worker_ = std::thread(&Impl::workerLoop, this);
        } catch (const std::exception& e) {
            started_.store(false, std::memory_order_relaxed);
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("start LogPublisher failed: %s", e.what());
            return false;
        }
        return true;
    }

    PublishResult tryPublish(KVCacheEvent event) noexcept {
        if (!started_.load(std::memory_order_relaxed) || stopping_.load(std::memory_order_relaxed)) {
            return PublishResult::NOT_RUNNING;
        }
        const auto result = queue_.tryPush(std::move(event));
        if (result == detail::QueuePushResult::ACCEPTED) {
            accepted_count_.fetch_add(1, std::memory_order_relaxed);
        } else {
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
        }
        return detail::toPublishResult(result);
    }

    void stop() noexcept {
        if (!started_.load(std::memory_order_relaxed)) {
            return;
        }
        stopping_.store(true, std::memory_order_relaxed);
        queue_.stop();
        if (worker_.joinable()) {
            worker_.join();
        }
        started_.store(false, std::memory_order_relaxed);
        state_.store(PublisherState::STOPPED, std::memory_order_relaxed);
    }

    PublisherStatus status() const noexcept {
        return {state_.load(std::memory_order_relaxed),
                queue_.size(),
                accepted_count_.load(std::memory_order_relaxed),
                dropped_count_.load(std::memory_order_relaxed)};
    }

private:
    void workerLoop() noexcept {
        state_.store(PublisherState::LOGGING, std::memory_order_relaxed);
        try {
            while (!stopping_.load(std::memory_order_relaxed)) {
                const auto batch = queue_.waitPop(config_.report_batch_size,
                                                  std::chrono::milliseconds(std::max(config_.flush_interval_ms, 1)));
                if (batch.empty()) {
                    continue;
                }

                size_t             add_count    = 0;
                size_t             delete_count = 0;
                std::ostringstream samples;
                const size_t       sample_count = std::min(batch.size(), config_.log_max_keys_per_batch);
                for (size_t i = 0; i < batch.size(); ++i) {
                    if (batch[i].type == KVCacheEventType::BLOCK_ADD) {
                        ++add_count;
                    } else {
                        ++delete_count;
                    }
                    if (i < sample_count) {
                        if (i > 0) {
                            samples << ',';
                        }
                        samples << eventTypeName(batch[i].type) << ':' << batch[i].block_key;
                    }
                }
                RTP_LLM_LOG_INFO("kv_cache_event publisher=log instance_id=%s host=%s dp_rank=%d batch_size=%zu "
                                 "add=%zu delete=%zu sequence_begin=%llu sequence_end=%llu samples=%s",
                                 context_.instance_id.c_str(),
                                 context_.host_ip_port.c_str(),
                                 context_.dp_rank,
                                 batch.size(),
                                 add_count,
                                 delete_count,
                                 static_cast<unsigned long long>(batch.front().sequence),
                                 static_cast<unsigned long long>(batch.back().sequence),
                                 samples.str().c_str());
            }
        } catch (const std::exception& e) {
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("LogPublisher worker stopped after exception: %s", e.what());
        } catch (...) {
            state_.store(PublisherState::DEGRADED, std::memory_order_relaxed);
            RTP_LLM_LOG_WARNING("LogPublisher worker stopped after unknown exception");
        }
    }

private:
    KVCacheEventPublisherConfig  config_;
    KVCacheEventPublisherContext context_;
    detail::KVCacheEventQueue    queue_;
    std::thread                  worker_;
    std::atomic<bool>            started_{false};
    std::atomic<bool>            stopping_{false};
    std::atomic<PublisherState>  state_{PublisherState::DISABLED};
    std::atomic<uint64_t>        accepted_count_{0};
    std::atomic<uint64_t>        dropped_count_{0};
};

LogPublisher::LogPublisher(KVCacheEventPublisherConfig config, KVCacheEventPublisherContext context):
    impl_(std::make_unique<Impl>(std::move(config), std::move(context))) {}

LogPublisher::~LogPublisher() = default;

bool LogPublisher::start() noexcept {
    return impl_->start();
}

PublishResult LogPublisher::tryPublish(KVCacheEvent event) noexcept {
    return impl_->tryPublish(std::move(event));
}

void LogPublisher::stop() noexcept {
    impl_->stop();
}

PublisherStatus LogPublisher::status() const noexcept {
    return impl_->status();
}

bool LogPublisher::enabled() const noexcept {
    return true;
}

}  // namespace rtp_llm
