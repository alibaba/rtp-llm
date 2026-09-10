#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PRequestDeadline.h"

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>
#include <tuple>

namespace {

std::chrono::system_clock::time_point deadlineToTimeoutPoint(int64_t deadline_ms, int64_t start_time_us) {
    if (deadline_ms > INT64_MAX / 1000) {
        return std::chrono::system_clock::time_point::max();
    }
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
                                            int64_t(timeout_check_interval_ms_) * 1000,
                                            "P2PConnectorResourceStoreCheckTimeoutThread");
    if (!check_timeout_thread_) {
        RTP_LLM_LOG_ERROR("P2PConnectorResourceStore init failed: check_timeout_thread is null");
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorResourceStore init success");
    return true;
}

void P2PConnectorResourceStore::setOnRequestReleased(
    std::function<void(const std::string&, int64_t, int64_t)> on_request_released) {
    on_request_released_ = std::move(on_request_released);
}

void P2PConnectorResourceStore::setOnRequestRegistered(
    std::function<bool(const std::string&, int64_t, int64_t)> on_request_registered) {
    on_request_registered_ = std::move(on_request_registered);
}

void P2PConnectorResourceStore::setOnRequestAcquired(
    std::function<bool(const std::string&, int64_t)> on_request_acquired) {
    on_request_acquired_ = std::move(on_request_acquired);
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
    const int64_t      request_id = routing->request_id;
    if (unique_key.empty()) {
        RTP_LLM_LOG_WARNING("P2PConnectorResourceStore::addResource failed: unique_key is empty");
        return false;
    }
    const int64_t now_ms = currentTimeMs();
    const int64_t request_deadline_ms =
        normalizeP2PRequestDeadline(routing->deadline_ms, now_ms, cancelled_keys_ttl_ms_);
    if (routing->deadline_ms > 0 && routing->deadline_ms != std::numeric_limits<int64_t>::max()
        && routing->deadline_ms <= now_ms) {
        RTP_LLM_LOG_WARNING("P2PConnectorResourceStore::addResource rejected expired request, unique_key: %s",
                            unique_key.c_str());
        if (on_request_released_) {
            on_request_released_(unique_key, request_id, request_deadline_ms);
        }
        return false;
    }

    bool rejected = false;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto                        cancelled_it = cancelled_keys_.find(unique_key);
        if (cancelled_it != cancelled_keys_.end()) {
            // Decode already cancelled this request. Drop the resource immediately instead of
            // letting it sit until checkTimeout(), so blocks are freed without delay.
            rejected = true;
            RTP_LLM_LOG_INFO("P2PConnectorResourceStore::addResource: rejected cancelled key, unique_key: %s",
                             unique_key.c_str());
        } else {
            auto entry               = std::make_shared<P2PConnectorResourceEntry>();
            entry->request_id        = request_id;
            entry->unique_key        = unique_key;
            entry->kv_cache_resource = kv_cache_resource;
            entry->request_deadline_ms = request_deadline_ms;
            entry->deadline_ms = p2pResourceHoldDeadline(request_deadline_ms, now_ms, prefill_resource_hold_ms_);
            entry->add_time_us        = currentTimeUs();
            if (on_request_registered_
                && !on_request_registered_(unique_key, entry->deadline_ms, entry->request_deadline_ms)) {
                cancelled_keys_[unique_key] = request_deadline_ms;
                rejected = true;
            } else {
                resource_map_[unique_key] = entry;
            }
        }
    }
    if (rejected) {
        if (on_request_released_) {
            on_request_released_(unique_key, request_id, request_deadline_ms);
        }
        resource_cv_.notify_all();
        return false;
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

void P2PConnectorResourceStore::markCancelled(const std::string& unique_key, int64_t request_deadline_ms) {
    markTerminal(unique_key, request_deadline_ms);
}

void P2PConnectorResourceStore::markTerminal(const std::string& unique_key, int64_t request_deadline_ms) {
    int64_t released_request_id          = -1;
    int64_t released_request_deadline_ms = request_deadline_ms;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        const int64_t               now_ms = currentTimeMs();
        int64_t tombstone_expire_at = normalizeP2PRequestDeadline(request_deadline_ms, now_ms, cancelled_keys_ttl_ms_);
        auto                        it = resource_map_.find(unique_key);
        if (it != resource_map_.end()) {
            // Resource is already in the store — remove it now rather than waiting for checkTimeout().
            released_request_id          = it->second->request_id;
            tombstone_expire_at          = it->second->request_deadline_ms;
            released_request_deadline_ms = it->second->request_deadline_ms;
            auto wait_start_time_us = it->second->add_time_us;
            resource_map_.erase(it);
            reportMetrics(false, true, wait_start_time_us);
            RTP_LLM_LOG_INFO("P2PConnectorResourceStore::markTerminal: removed existing resource, unique_key: %s",
                             unique_key.c_str());
        } else {
            RTP_LLM_LOG_DEBUG("P2PConnectorResourceStore::markTerminal: recorded terminal key, unique_key: %s",
                              unique_key.c_str());
        }
        // Record terminal state even when the resource was already present. A
        // duplicate or late StartLoad for this request must not wait again.
        cancelled_keys_[unique_key] = tombstone_expire_at;
        released_request_deadline_ms = tombstone_expire_at;
    }
    resource_cv_.notify_all();
    if (on_request_released_) {
        on_request_released_(unique_key, released_request_id, released_request_deadline_ms);
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

    const auto it = resource_map_.find(unique_key);
    if (it == resource_map_.end()) {
        return nullptr;
    }
    auto entry = it->second;
    const int64_t transfer_deadline_ms = std::min(deadline_ms, entry->request_deadline_ms);
    const int64_t now_ms = currentTimeMs();
    // Hold expiry and transfer activation have one decision point. ResultStore
    // does not expire resource-owned entries independently while this lock is held.
    if (now_ms >= entry->deadline_ms || now_ms >= transfer_deadline_ms
        || (on_request_acquired_ && !on_request_acquired_(unique_key, transfer_deadline_ms))) {
        resource_map_.erase(it);
        cancelled_keys_[unique_key] = entry->request_deadline_ms;
        reportMetrics(true, false, entry->add_time_us);
        lock.unlock();
        if (on_request_released_) {
            on_request_released_(unique_key, entry->request_id, entry->request_deadline_ms);
        }
        resource_cv_.notify_all();
        return nullptr;
    }
    entry->deadline_ms = transfer_deadline_ms;
    return stealResourceEntryLocked(unique_key);
}

void P2PConnectorResourceStore::checkTimeout() {
    int64_t                  current_time_ms = currentTimeMs();
    bool                     any_expired     = false;
    std::vector<std::tuple<std::string, int64_t, int64_t>> released_requests;
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
                cancelled_keys_[unique_key] = entry->request_deadline_ms;
                released_requests.emplace_back(unique_key, entry->request_id, entry->request_deadline_ms);
                it          = resource_map_.erase(it);
                any_expired = true;
                reportMetrics(true, false, wait_start_time_us);
            } else {
                ++it;
            }
        }
        // Tombstones expire at the original request deadline. Invalid legacy
        // deadlines are normalized to cancelled_keys_ttl_ms_ when inserted.
        for (auto it = cancelled_keys_.begin(); it != cancelled_keys_.end();) {
            if (current_time_ms >= it->second) {
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
    if (on_request_released_) {
        for (const auto& [unique_key, request_id, request_deadline_ms] : released_requests) {
            on_request_released_(unique_key, request_id, request_deadline_ms);
        }
    }
    if (any_expired) {
        // Wake up any handleRead currently sitting in waitForResourceOrCancellation
        // for one of the keys we just marked cancelled.
        resource_cv_.notify_all();
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

}  // namespace rtp_llm
