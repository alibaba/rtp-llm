#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <thread>

namespace {

constexpr int64_t kSideChannelMaxTtlMs = 3600000;  // 1 hour

std::chrono::system_clock::time_point deadlineToTimeoutPoint(int64_t deadline_ms, int64_t start_time_us) {
    const int64_t remaining_us = deadline_ms * 1000 - start_time_us;
    return std::chrono::system_clock::now() + std::chrono::microseconds(remaining_us);
}

// Generic backoff wait: polls `predicate` under `lock`, using `cv` with exponential backoff (capped at 8ms).
//
// Returns true if either `predicate()` or `is_cancelled()` became true before timeout.
// IMPORTANT: Caller must re-check `is_cancelled()` when this function returns true to distinguish
// between "predicate satisfied" vs "operation cancelled". Return value `false` always means timeout.
template<typename Lock>
bool waitWithBackoff(Lock&                                 lock,
                     std::condition_variable&              cv,
                     std::chrono::system_clock::time_point timeout_tp,
                     const std::function<bool()>&          predicate,
                     const std::function<bool()>&          is_cancelled) {
    int           sleep_ms    = 1;
    constexpr int kBackoffCap = 8;
    while (true) {
        if (is_cancelled && is_cancelled()) {
            return true;
        }
        if (predicate()) {
            return true;
        }
        const auto now = std::chrono::system_clock::now();
        if (now >= timeout_tp) {
            return false;
        }
        auto next_wake = now + std::chrono::milliseconds(std::min(sleep_ms, kBackoffCap));
        if (next_wake > timeout_tp) {
            next_wake = timeout_tp;
        }
        cv.wait_until(lock, next_wake);
        sleep_ms = std::min(sleep_ms * 2, kBackoffCap);
    }
}

}  // namespace

namespace rtp_llm {

P2PConnectorResourceStore::P2PConnectorResourceStore(const kmonitor::MetricsReporterPtr& metrics_reporter,
                                                     int                                 timeout_check_interval_ms,
                                                     int64_t                             prefill_resource_hold_ms,
                                                     int64_t                             cancelled_keys_ttl_ms):
    metrics_reporter_(metrics_reporter),
    timeout_check_interval_ms_(timeout_check_interval_ms),
    prefill_resource_hold_ms_(prefill_resource_hold_ms),
    cancelled_keys_ttl_ms_(cancelled_keys_ttl_ms) {}

P2PConnectorResourceStore::~P2PConnectorResourceStore() {
    if (check_timeout_thread_) {
        check_timeout_thread_->stop();
    }
}

bool P2PConnectorResourceStore::isMarkedCancelled(const std::string& unique_key) const {
    std::lock_guard<std::mutex> lock(resource_map_mutex_);
    return cancelled_keys_.find(unique_key) != cancelled_keys_.end();
}

bool P2PConnectorResourceStore::init() {
    check_timeout_thread_ =
        autil::LoopThread::createLoopThread(std::bind(&P2PConnectorResourceStore::checkTimeout, this),
                                            timeout_check_interval_ms_,
                                            "P2PConnectorResourceStoreCheckTimeoutThread");
    if (!check_timeout_thread_) {
        RTP_LLM_LOG_ERROR("P2PConnectorResourceStore init failed: check_timeout_thread is null");
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorResourceStore init success");
    return true;
}

bool P2PConnectorResourceStore::addResource(const std::shared_ptr<Meta>& meta,
                                            const KVCacheResourcePtr&    kv_cache_resource) {
    // Extract routing from Meta::p2pRouting()
    auto routing = meta->p2pRouting();
    if (!routing.has_value()) {
        RTP_LLM_LOG_WARNING("P2PConnectorResourceStore::addResource failed: meta->p2pRouting() returned nullopt");
        return false;
    }

    const std::string& unique_key = routing->unique_key;
    if (unique_key.empty()) {
        RTP_LLM_LOG_WARNING("P2PConnectorResourceStore::addResource failed: unique_key is empty");
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto                        cancelled_it = cancelled_keys_.find(unique_key);
        if (cancelled_it != cancelled_keys_.end()) {
            // Decode already cancelled this request. Drop the resource immediately instead of
            // letting it sit until checkTimeout(), so blocks are freed without delay.
            cancelled_keys_.erase(cancelled_it);
            RTP_LLM_LOG_INFO("P2PConnectorResourceStore::addResource: rejected cancelled key, unique_key: %s",
                             unique_key.c_str());
            return false;
        }
        auto entry               = std::make_shared<P2PConnectorResourceEntry>();
        entry->request_id        = routing->request_id;
        entry->unique_key        = unique_key;
        entry->kv_cache_resource = kv_cache_resource;
        // Use the request's business deadline directly. The previous 60s cap
        // (prefill_resource_hold_ms_) caused premature resource expiration when
        // decode was slow to schedule — requests stuck in WAITING never got a
        // chance to call asyncRead before the resource was reclaimed, producing
        // cascading error_code=8312 failures.
        entry->deadline_ms        = routing->deadline_ms;
        entry->add_time_us        = currentTimeUs();
        resource_map_[unique_key] = entry;
    }
    // 通知所有等待的线程
    resource_cv_.notify_all();
    return true;
}

bool P2PConnectorResourceStore::waitForResourceOrCancellation(std::unique_lock<std::mutex>&         lock,
                                                              const std::string&                    unique_key,
                                                              std::chrono::system_clock::time_point timeout_tp,
                                                              const std::function<bool()>&          is_cancelled) {
    return waitWithBackoff(
        lock,
        resource_cv_,
        timeout_tp,
        [&]() {
            // Wake on either:
            //   (a) resource arrived → handleRead steals it
            //   (b) entry expired and was tombstoned by checkTimeout →
            //       handleRead surfaces this as GENERATE_TIMEOUT instead of
            //       waiting out the full business deadline (~1h).
            return resource_map_.find(unique_key) != resource_map_.end()
                   || cancelled_keys_.find(unique_key) != cancelled_keys_.end();
        },
        is_cancelled);
}

std::shared_ptr<P2PConnectorResourceEntry>
P2PConnectorResourceStore::stealResourceEntryLocked(const std::string& unique_key) {
    auto it = resource_map_.find(unique_key);
    if (it == resource_map_.end()) {
        RTP_LLM_LOG_WARNING(
            "P2PConnectorResourceStore::waitAndStealResource failed: resource not found, unique_key: %s",
            unique_key.c_str());
        return nullptr;
    }
    auto entry = it->second;
    resource_map_.erase(it);
    reportMetrics(false, false, entry->add_time_us);
    return entry;
}

void P2PConnectorResourceStore::markCancelled(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(resource_map_mutex_);
    auto                        it = resource_map_.find(unique_key);
    if (it != resource_map_.end()) {
        // Resource is already in the store — remove it now rather than waiting for checkTimeout().
        auto wait_start_time_us = it->second->add_time_us;
        resource_map_.erase(it);
        reportMetrics(false, true, wait_start_time_us);
        RTP_LLM_LOG_INFO("P2PConnectorResourceStore::markCancelled: removed existing resource, unique_key: %s",
                         unique_key.c_str());
    } else {
        // Resource not yet in store. Record cancellation so addResource() rejects it on arrival.
        cancelled_keys_[unique_key] = currentTimeMs();
        RTP_LLM_LOG_DEBUG("P2PConnectorResourceStore::markCancelled: recorded pending cancel, unique_key: %s",
                          unique_key.c_str());
    }
}

std::shared_ptr<P2PConnectorResourceEntry> P2PConnectorResourceStore::waitAndStealResource(
    const std::string& unique_key, int64_t deadline_ms, std::function<bool()> is_cancelled) {
    std::unique_lock<std::mutex> lock(resource_map_mutex_);

    const int64_t start_time_us = currentTimeUs();
    const int64_t remaining_us  = deadline_ms * 1000 - start_time_us;
    if (remaining_us <= 0) {
        RTP_LLM_LOG_WARNING("P2PConnectorResourceStore::waitAndStealResource already past deadline, unique_key: %s",
                            unique_key.c_str());
        reportMetrics(true, false, start_time_us);
        return nullptr;
    }
    const auto timeout_tp = deadlineToTimeoutPoint(deadline_ms, start_time_us);

    if (!waitForResourceOrCancellation(lock, unique_key, timeout_tp, is_cancelled)) {
        reportMetrics(true, false, start_time_us);
        RTP_LLM_LOG_WARNING("P2PConnectorResourceStore::waitAndStealResource timeout, unique_key: %s, deadline_ms: %ld",
                            unique_key.c_str(),
                            deadline_ms);
        return nullptr;
    }

    if (is_cancelled && is_cancelled()) {
        reportMetrics(false, true, start_time_us);
        return nullptr;  // 因取消退出，不取资源
    }

    // Check if the resource was expired by checkTimeout() (hold_ms exceeded).
    // waitForResourceOrCancellation wakes on cancelled_keys_ too, so we must
    // distinguish "resource arrived" from "resource expired".
    if (cancelled_keys_.find(unique_key) != cancelled_keys_.end()) {
        reportMetrics(true, false, start_time_us);
        RTP_LLM_LOG_WARNING(
            "P2PConnectorResourceStore::waitAndStealResource: resource expired (hold_ms), unique_key: %s",
            unique_key.c_str());
        return nullptr;
    }

    return stealResourceEntryLocked(unique_key);
}

void P2PConnectorResourceStore::checkTimeout() {
    int64_t                  current_time_ms = currentTimeMs();
    bool                     any_expired     = false;
    std::vector<std::string> expired_keys;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        for (auto it = resource_map_.begin(); it != resource_map_.end();) {
            auto& [unique_key, entry] = *it;
            if (entry && current_time_ms >= entry->deadline_ms) {
                RTP_LLM_LOG_WARNING(
                    "P2PConnectorResourceStore: resource timeout, unique_key: %s, deadline_ms: %ld, current_time_ms: %ld",
                    unique_key.c_str(),
                    entry->deadline_ms,
                    current_time_ms);
                auto wait_start_time_us = entry->add_time_us;
                // Mark this key cancelled so a late-arriving handleRead's
                // waitForResourceOrCancellation() returns immediately instead
                // of waiting until the business deadline (~1h). See predicate
                // in waitForResourceOrCancellation.
                cancelled_keys_[unique_key] = current_time_ms;
                expired_keys.push_back(unique_key);
                it          = resource_map_.erase(it);
                any_expired = true;
                reportMetrics(true, false, wait_start_time_us);
            } else {
                ++it;
            }
        }
        // Clean up cancelled_keys_ entries older than the configured TTL
        // (default 1h). The TTL must comfortably exceed the longest expected
        // skew between addResource and a late StartLoad arrival, otherwise
        // handleRead won't be able to tell "expired here" from "never seen".
        for (auto it = cancelled_keys_.begin(); it != cancelled_keys_.end();) {
            if (current_time_ms >= it->second + cancelled_keys_ttl_ms_) {
                it = cancelled_keys_.erase(it);
            } else {
                ++it;
            }
        }
        if (metrics_reporter_) {
            auto collector          = std::make_shared<StreamStoreCountMetricsCollector>();
            collector->stream_count = resource_map_.size();
            metrics_reporter_->report<P2PConnectorMetrics, StreamStoreCountMetricsCollector>(nullptr, collector.get());
        }
    }
    if (any_expired) {
        // Wake up any handleRead currently sitting in waitForResourceOrCancellation
        // for one of the keys we just marked cancelled.
        resource_cv_.notify_all();
    }

    {
        std::lock_guard<std::mutex> lock(side_channel_map_mutex_);
        // Clean up side-channel data for resources that just expired, so it
        // doesn't linger with the original business deadline (~1h).
        for (const auto& key : expired_keys) {
            clearSideChannelDataLocked(key);
        }
        for (auto it = side_channel_data_map_.begin(); it != side_channel_data_map_.end();) {
            const auto& [unique_key, entry] = *it;
            if (entry.deadline_ms > 0 && current_time_ms >= entry.deadline_ms) {
                RTP_LLM_LOG_WARNING(
                    "P2PConnectorResourceStore: side-channel timeout, unique_key: %s, deadline_ms: %ld, current_time_ms: %ld",
                    unique_key.c_str(),
                    entry.deadline_ms,
                    current_time_ms);
                it = side_channel_data_map_.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void P2PConnectorResourceStore::reportMetrics(bool timeout, bool cancelled, int64_t wait_start_time_us) {
    if (metrics_reporter_) {
        auto collector                 = std::make_shared<StreamStoreWaitMetricsCollector>();
        collector->timeout             = timeout;
        collector->cancelled           = cancelled;
        collector->stream_wait_time_us = currentTimeUs() - wait_start_time_us;
        metrics_reporter_->report<P2PConnectorMetrics, StreamStoreWaitMetricsCollector>(nullptr, collector.get());
    }
}

void P2PConnectorResourceStore::notifySideChannelReady(const std::string&                                unique_key,
                                                       int64_t                                           deadline_ms,
                                                       const P2PConnectorResourceEntry::SideChannelData& data) {
    int64_t add_time_us = currentTimeUs();
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        if (cancelled_keys_.find(unique_key) != cancelled_keys_.end()) {
            RTP_LLM_LOG_DEBUG("notifySideChannelReady: skipped cancelled key, unique_key: %s", unique_key.c_str());
            return;
        }
        auto it = resource_map_.find(unique_key);
        if (it != resource_map_.end() && it->second) {
            // Always use the resource entry's capped deadline so side-channel
            // data expires together with the resource (~hold_ms), not the
            // caller-provided business deadline (~1h).
            deadline_ms = it->second->deadline_ms;
            add_time_us = it->second->add_time_us;
        } else if (deadline_ms <= 0 || deadline_ms == INT64_MAX) {
            deadline_ms = currentTimeMs() + kSideChannelMaxTtlMs;
        }
    }
    {
        std::lock_guard<std::mutex> lock(side_channel_map_mutex_);
        side_channel_data_map_[unique_key] = P2PSideChannelStoreEntry{data, deadline_ms, add_time_us};
    }
    side_channel_cv_.notify_all();

    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto                        it = resource_map_.find(unique_key);
        if (it != resource_map_.end() && it->second) {
            std::lock_guard<std::mutex> sc_lock(it->second->side_channel_mutex);
            it->second->side_channel_data  = data;
            it->second->side_channel_ready = true;
            it->second->side_channel_cv.notify_all();
        }
    }
    RTP_LLM_LOG_DEBUG(
        "notifySideChannelReady: unique_key: %s, first_token: %ld", unique_key.c_str(), data.first_token_id);
}

bool P2PConnectorResourceStore::consumeSideChannelData(const std::string&                          unique_key,
                                                       P2PConnectorResourceEntry::SideChannelData& out_data) {
    std::lock_guard<std::mutex> lock(side_channel_map_mutex_);
    auto                        it = side_channel_data_map_.find(unique_key);
    if (it != side_channel_data_map_.end()) {
        out_data = it->second.data;
        side_channel_data_map_.erase(it);
        return true;
    }
    return false;
}

void P2PConnectorResourceStore::clearSideChannelDataLocked(const std::string& unique_key) {
    auto it = side_channel_data_map_.find(unique_key);
    if (it != side_channel_data_map_.end()) {
        side_channel_data_map_.erase(it);
        RTP_LLM_LOG_DEBUG("P2PConnectorResourceStore: cleared side-channel, unique_key=%s", unique_key.c_str());
    }
}

void P2PConnectorResourceStore::clearSideChannelData(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(side_channel_map_mutex_);
    clearSideChannelDataLocked(unique_key);
}

bool P2PConnectorResourceStore::waitSideChannelReady(const std::string&    unique_key,
                                                     int64_t               deadline_ms,
                                                     std::function<bool()> is_cancelled) {
    std::shared_ptr<P2PConnectorResourceEntry> entry;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto                        it = resource_map_.find(unique_key);
        if (it != resource_map_.end()) {
            entry = it->second;
        }
    }
    if (!entry) {
        RTP_LLM_LOG_WARNING("waitSideChannelReady: entry not found, unique_key: %s", unique_key.c_str());
        return false;
    }

    std::unique_lock<std::mutex> lock(entry->side_channel_mutex);
    const int64_t                start_time_us = currentTimeUs();
    const int64_t                remaining_us  = deadline_ms * 1000 - start_time_us;
    if (remaining_us <= 0) {
        RTP_LLM_LOG_WARNING("waitSideChannelReady: past deadline, unique_key: %s", unique_key.c_str());
        return false;
    }

    const auto timeout_tp = deadlineToTimeoutPoint(deadline_ms, start_time_us);
    bool       ready      = waitWithBackoff(
        lock, entry->side_channel_cv, timeout_tp, [&]() { return entry->side_channel_ready; }, is_cancelled);

    if (!ready) {
        RTP_LLM_LOG_WARNING("waitSideChannelReady: timeout, unique_key: %s", unique_key.c_str());
        clearSideChannelData(unique_key);
    } else if (is_cancelled && is_cancelled()) {
        RTP_LLM_LOG_DEBUG("waitSideChannelReady: cancelled, unique_key: %s", unique_key.c_str());
        clearSideChannelData(unique_key);
        return false;
    }
    return ready;
}

}  // namespace rtp_llm
