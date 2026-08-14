#include "rtp_llm/cpp/cache/events/LogPublisher.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <mutex>
#include <sstream>
#include <thread>
#include <utility>

#include "rtp_llm/cpp/cache/events/KVCacheEventAdmissionGate.h"
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
        std::lock_guard<std::mutex> lock(lifecycle_mu_);
        if (circuit_open_.load(std::memory_order_acquire)) {
            RTP_LLM_LOG_WARNING("LogPublisher cannot restart after its circuit opened");
            return false;
        }
        if (started_.load(std::memory_order_relaxed)) {
            return true;
        }
        if (stopped_permanently_) {
            RTP_LLM_LOG_WARNING("LogPublisher cannot restart after stop");
            return false;
        }
        if (config_.queue_capacity == 0 || config_.queue_capacity > kKVCacheEventMaxQueueCapacity
            || config_.report_batch_size == 0 || config_.report_batch_size > kKVCacheEventMaxReportBatchSize
            || config_.flush_interval_ms <= 0) {
            tripCircuit();
            stopped_permanently_ = true;
            RTP_LLM_LOG_WARNING("LogPublisher is disabled by invalid resource configuration");
            return false;
        }
        stopping_.store(false, std::memory_order_release);
        available_.store(true, std::memory_order_release);
        // Publish the accepting state before launching the worker. Once
        // start() succeeds, the manager may install this publisher and
        // concurrent producers can safely queue before the worker is
        // scheduled. Publishing availability after thread
        // creation would be unsafe: an immediately failing worker could mark
        // itself unavailable, only for start() to overwrite that terminal
        // state and leave a dead exporter advertised as healthy.
        accepting_.store(true, std::memory_order_release);
        state_.store(PublisherState::STARTING, std::memory_order_relaxed);
        try {
            worker_ = std::thread(&Impl::workerLoop, this);
        } catch (const std::exception& e) {
            tripCircuit();
            publication_gate_.quiesce();
            queue_.discardPending();
            started_.store(false, std::memory_order_release);
            stopped_permanently_ = true;
            RTP_LLM_LOG_WARNING("start LogPublisher failed: %s", e.what());
            return false;
        }
        started_.store(true, std::memory_order_release);
        return true;
    }

    PublishResult tryPublish(KVCacheEvent event) noexcept {
        auto publication = publication_gate_.tryEnter();
        if (!publication || !accepting_.load(std::memory_order_acquire)) {
            return PublishResult::NOT_RUNNING;
        }
        const auto result = queue_.tryPush(std::move(event));
        if (result == detail::QueuePushResult::ACCEPTED) {
            accepted_count_.fetch_add(1, std::memory_order_relaxed);
        } else if (result == detail::QueuePushResult::FULL) {
            // A producer can enter immediately before stop() closes the
            // admission gate. Do not turn that shutdown race into a spurious
            // overflow circuit or dropped-event metric.
            if (!accepting_.load(std::memory_order_acquire)) {
                return PublishResult::NOT_RUNNING;
            }
            dropped_count_.fetch_add(1, std::memory_order_relaxed);
            tripCircuit();
        }
        return detail::toPublishResult(result);
    }

    void stop() noexcept {
        std::lock_guard<std::mutex> lock(lifecycle_mu_);
        if (!started_.load(std::memory_order_acquire) && !worker_.joinable()) {
            accepting_.store(false, std::memory_order_release);
            available_.store(false, std::memory_order_release);
            stopping_.store(true, std::memory_order_release);
            stopped_permanently_ = true;
            publication_gate_.close();
            queue_.stop();
            publication_gate_.quiesce();
            queue_.quiescePushes();
            // start() can fail after opening admission but before installing
            // the worker. The owning factory does not publish the instance in
            // that state, but draining here also makes direct concurrent use
            // lossless and keeps both stop paths equivalent.
            drainQueueAfterStop();
            state_.store(PublisherState::STOPPED, std::memory_order_relaxed);
            return;
        }
        accepting_.store(false, std::memory_order_release);
        available_.store(false, std::memory_order_release);
        stopping_.store(true, std::memory_order_release);
        publication_gate_.close();
        queue_.stop();
        publication_gate_.quiesce();
        queue_.quiescePushes();
        if (worker_.joinable()) {
            worker_.join();
        }
        // stop() closes and quiesces both publication gates before joining,
        // so the queue is stable here and this thread is its sole consumer.
        // Drain the bounded remainder instead of silently abandoning events
        // that were accepted immediately before shutdown.
        drainQueueAfterStop();
        started_.store(false, std::memory_order_release);
        stopped_permanently_ = true;
        state_.store(PublisherState::STOPPED, std::memory_order_relaxed);
    }

    PublisherStatus status() const noexcept {
        auto state = state_.load(std::memory_order_relaxed);
        // A producer can open the circuit while the worker is leaving its
        // startup transition. Keep the terminal flag authoritative even if a
        // concurrent diagnostic state store wins the last-write race.
        const bool circuit_open = circuit_open_.load(std::memory_order_acquire);
        if (circuit_open && state != PublisherState::STOPPED) {
            state = PublisherState::CIRCUIT_OPEN;
        }
        PublisherStatus status;
        status.state                = state;
        status.queue_size           = circuit_open ? 0 : queue_.size();
        status.accepted_count       = accepted_count_.load(std::memory_order_relaxed);
        status.dropped_count        = dropped_count_.load(std::memory_order_relaxed);
        status.queue_high_watermark = queue_.highWatermark();
        return status;
    }

    bool enabled() const noexcept {
        return available_.load(std::memory_order_acquire);
    }

private:
    void logBatch(const std::vector<KVCacheEvent>& batch) {
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

    void drainQueueAfterStop() noexcept {
        try {
            for (;;) {
                const auto batch = queue_.waitPop(config_.report_batch_size, std::chrono::milliseconds::zero());
                if (batch.empty()) {
                    return;
                }
                logBatch(batch);
            }
        } catch (const std::exception& e) {
            RTP_LLM_LOG_WARNING("LogPublisher could not drain its shutdown queue: %s", e.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("LogPublisher could not drain its shutdown queue after an unknown exception");
        }
    }

    void workerLoop() noexcept {
        auto expected_state = PublisherState::STARTING;
        // Do not overwrite CIRCUIT_OPEN if the startup queue filled before this
        // thread was scheduled.
        state_.compare_exchange_strong(
            expected_state, PublisherState::LOGGING, std::memory_order_relaxed, std::memory_order_relaxed);
        try {
            while (!stopping_.load(std::memory_order_acquire)) {
                const auto batch = queue_.waitPop(config_.report_batch_size,
                                                  std::chrono::milliseconds(std::max(config_.flush_interval_ms, 1)));
                if (batch.empty()) {
                    continue;
                }
                logBatch(batch);
            }
        } catch (const std::exception& e) {
            tripCircuit();
            RTP_LLM_LOG_WARNING("LogPublisher worker stopped after exception: %s", e.what());
        } catch (...) {
            tripCircuit();
            RTP_LLM_LOG_WARNING("LogPublisher worker stopped after unknown exception");
        }
        if (circuit_open_.load(std::memory_order_acquire)) {
            // Accepted entries are no longer actionable after a terminal
            // exporter failure. Release them on the sole consumer before
            // reporting the failure; an explicit healthy stop still drains
            // its queue through stop() instead.
            queue_.discardPending();
            // Logging stays on the worker: queue overflow is detected from the
            // cache mutation path, which must remain bounded and free of I/O.
            RTP_LLM_LOG_WARNING("LogPublisher circuit opened; cache-event logging is disabled for this process, "
                                "accepted=%llu dropped=%llu queue_capacity=%zu",
                                static_cast<unsigned long long>(accepted_count_.load(std::memory_order_relaxed)),
                                static_cast<unsigned long long>(dropped_count_.load(std::memory_order_relaxed)),
                                config_.queue_capacity);
        }
    }

    void tripCircuit() noexcept {
        // Keep every terminal failure path on the same transition. The flag
        // is published first so a concurrent start() can never advertise an
        // already-failed one-shot publisher as healthy again.
        circuit_open_.store(true, std::memory_order_release);
        accepting_.store(false, std::memory_order_release);
        available_.store(false, std::memory_order_release);
        stopping_.store(true, std::memory_order_release);
        publication_gate_.close();
        state_.store(PublisherState::CIRCUIT_OPEN, std::memory_order_relaxed);
        queue_.stop();
    }

private:
    KVCacheEventPublisherConfig       config_;
    KVCacheEventPublisherContext      context_;
    detail::KVCacheEventQueue         queue_;
    std::thread                       worker_;
    std::mutex                        lifecycle_mu_;
    std::atomic<bool>                 started_{false};
    std::atomic<bool>                 accepting_{false};
    std::atomic<bool>                 stopping_{false};
    std::atomic<bool>                 available_{true};
    std::atomic<bool>                 circuit_open_{false};
    detail::KVCacheEventAdmissionGate publication_gate_;
    bool                              stopped_permanently_{false};
    std::atomic<PublisherState>       state_{PublisherState::DISABLED};
    std::atomic<uint64_t>             accepted_count_{0};
    std::atomic<uint64_t>             dropped_count_{0};
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
    return impl_->enabled();
}

}  // namespace rtp_llm
