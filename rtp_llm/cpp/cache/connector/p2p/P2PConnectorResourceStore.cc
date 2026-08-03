#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>

namespace {

constexpr int64_t kSideChannelMaxTtlMs     = 3600000;  // 1 hour
constexpr int64_t kMaxResourceLifetimeMs   = 3600000;  // 1 hour cap to prevent permanently pinned blocks

int64_t normalizeRequestDeadline(int64_t request_deadline_ms, int64_t now_ms, int64_t fallback_ttl_ms) {
    if (request_deadline_ms <= 0 || request_deadline_ms == std::numeric_limits<int64_t>::max()) {
        return now_ms + fallback_ttl_ms;
    }
    return std::max(request_deadline_ms, now_ms);
}

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
                                            timeout_check_interval_ms_,
                                            "P2PConnectorResourceStoreCheckTimeoutThread");
    if (!check_timeout_thread_) {
        RTP_LLM_LOG_ERROR("P2PConnectorResourceStore init failed: check_timeout_thread is null");
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorResourceStore init success");
    return true;
}

void P2PConnectorResourceStore::setOnRequestReleased(std::function<void(int64_t, int64_t)> on_request_released) {
    on_request_released_ = std::move(on_request_released);
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
    if (routing->deadline_ms > 0 && routing->deadline_ms != std::numeric_limits<int64_t>::max()
        && routing->deadline_ms <= now_ms) {
        RTP_LLM_LOG_WARNING("P2PConnectorResourceStore::addResource rejected expired request, unique_key: %s",
                            unique_key.c_str());
        if (on_request_released_) {
            on_request_released_(request_id, routing->deadline_ms);
        }
        return false;
    }

    bool rejected_cancelled = false;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto                        cancelled_it = cancelled_keys_.find(unique_key);
        if (cancelled_it != cancelled_keys_.end()) {
            // Decode already cancelled this request. Drop the resource immediately instead of
            // letting it sit until checkTimeout(), so blocks are freed without delay.
            rejected_cancelled = true;
            RTP_LLM_LOG_INFO("P2PConnectorResourceStore::addResource: rejected cancelled key, unique_key: %s",
                             unique_key.c_str());
        } else {
            auto entry               = std::make_shared<P2PConnectorResourceEntry>();
            entry->request_id        = request_id;
            entry->unique_key        = unique_key;
            entry->kv_cache_resource = kv_cache_resource;
            entry->request_deadline_ms =
                normalizeRequestDeadline(routing->deadline_ms, now_ms, cancelled_keys_ttl_ms_);
            entry->deadline_ms = std::min(
                {entry->request_deadline_ms, now_ms + prefill_resource_hold_ms_, now_ms + kMaxResourceLifetimeMs});
            entry->add_time_us        = currentTimeUs();
            resource_map_[unique_key] = entry;
        }
    }
    if (rejected_cancelled) {
        if (on_request_released_) {
            on_request_released_(
                request_id, normalizeRequestDeadline(routing->deadline_ms, currentTimeMs(), cancelled_keys_ttl_ms_));
        }
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
        int64_t tombstone_expire_at = normalizeRequestDeadline(request_deadline_ms, now_ms, cancelled_keys_ttl_ms_);
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
    }
    if (released_request_id >= 0 && on_request_released_) {
        on_request_released_(released_request_id, released_request_deadline_ms);
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

    auto entry = stealResourceEntryLocked(unique_key);
    if (entry) {
        std::lock_guard<std::mutex> side_channel_lock(side_channel_map_mutex_);
        active_side_channel_deadlines_[unique_key] = deadline_ms;
        auto side_channel_it = side_channel_data_map_.find(unique_key);
        if (side_channel_it != side_channel_data_map_.end()) {
            side_channel_it->second.deadline_ms = deadline_ms;
        }
    }
    return entry;
}

void P2PConnectorResourceStore::checkTimeout() {
    int64_t                  current_time_ms = currentTimeMs();
    bool                     any_expired     = false;
    std::vector<std::string> expired_keys;
    std::vector<std::pair<int64_t, int64_t>> released_requests;
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
                released_requests.emplace_back(entry->request_id, entry->request_deadline_ms);
                expired_keys.push_back(unique_key);
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
        for (const auto& [request_id, request_deadline_ms] : released_requests) {
            on_request_released_(request_id, request_deadline_ms);
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
        for (auto it = active_side_channel_deadlines_.begin(); it != active_side_channel_deadlines_.end();) {
            if (current_time_ms >= it->second) {
                it = active_side_channel_deadlines_.erase(it);
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
    std::shared_ptr<P2PConnectorResourceEntry> entry;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        if (cancelled_keys_.find(unique_key) != cancelled_keys_.end()) {
            RTP_LLM_LOG_DEBUG("notifySideChannelReady: skipped cancelled key, unique_key: %s", unique_key.c_str());
            return;
        }
        auto it = resource_map_.find(unique_key);
        if (it != resource_map_.end() && it->second) {
            entry = it->second;
            // Always use the resource entry's capped deadline so side-channel
            // data expires together with the resource (~hold_ms), not the
            // caller-provided business deadline (~1h).
            deadline_ms = it->second->deadline_ms;
            add_time_us = it->second->add_time_us;
        }

        {
            std::lock_guard<std::mutex> sc_map_lock(side_channel_map_mutex_);
            if (!entry) {
                if (auto active_it = active_side_channel_deadlines_.find(unique_key);
                    active_it != active_side_channel_deadlines_.end()) {
                    deadline_ms = active_it->second;
                } else {
                    const int64_t now_ms = currentTimeMs();
                    const int64_t hold_ms =
                        std::max<int64_t>(0, std::min(prefill_resource_hold_ms_, kSideChannelMaxTtlMs));
                    const int64_t hold_deadline_ms =
                        now_ms > std::numeric_limits<int64_t>::max() - hold_ms ?
                            std::numeric_limits<int64_t>::max() :
                            now_ms + hold_ms;
                    if (deadline_ms <= 0 || deadline_ms == INT64_MAX) {
                        deadline_ms = hold_deadline_ms;
                    } else {
                        deadline_ms = std::min(deadline_ms, hold_deadline_ms);
                    }
                }
            }
            side_channel_data_map_[unique_key] = P2PSideChannelStoreEntry{data, deadline_ms, add_time_us};
        }
        if (entry) {
            std::lock_guard<std::mutex> sc_lock(entry->side_channel_mutex);
            entry->side_channel_data  = data;
            entry->side_channel_ready = true;
            entry->side_channel_cv.notify_all();
        }
    }
    side_channel_cv_.notify_all();
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
    active_side_channel_deadlines_.erase(unique_key);
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
