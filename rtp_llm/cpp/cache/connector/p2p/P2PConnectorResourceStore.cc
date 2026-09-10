#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <limits>

namespace {

bool validDeadline(int64_t deadline_ms) {
    return deadline_ms > 0 && deadline_ms != std::numeric_limits<int64_t>::max();
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
                                                   int timeout_check_interval_ms):
    metrics_reporter_(metrics_reporter), timeout_check_interval_ms_(timeout_check_interval_ms) {}

P2PConnectorResourceStore::~P2PConnectorResourceStore() {
    if (check_timeout_thread_) {
        check_timeout_thread_->stop();
    }
}

bool P2PConnectorResourceStore::init() {
    check_timeout_thread_ = autil::LoopThread::createLoopThread(
        [this]() { checkTimeout(); }, timeout_check_interval_ms_, "P2PConnectorResourceStoreCheckTimeoutThread");
    return check_timeout_thread_ != nullptr;
}

void P2PConnectorResourceStore::setOnRequestReleased(std::function<void(int64_t, int64_t)> callback) {
    on_request_released_ = std::move(callback);
}

bool P2PConnectorResourceStore::isMarkedCancelled(const std::string& unique_key) const {
    std::lock_guard<std::mutex> lock(resource_map_mutex_);
    auto it = request_states_.find(unique_key);
    return it != request_states_.end() && (it->second.terminal || currentTimeMs() >= it->second.deadlineMs());
}

bool P2PConnectorResourceStore::addResource(const std::shared_ptr<Meta>& meta,
                                          const KVCacheResourcePtr& resource) {
    const auto routing = meta ? meta->p2pRouting() : std::nullopt;
    if (!routing || routing->unique_key.empty() || !validDeadline(routing->deadline_ms) || !resource) {
        return false;
    }
    bool accepted = false;
    bool release_layers = false;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto [it, inserted] = request_states_.try_emplace(routing->unique_key, RequestState{routing->deadline_ms});
        auto& state = it->second;
        release_layers = state.terminal || currentTimeMs() >= state.deadlineMs();
        if (!state.terminal && !state.consumed && currentTimeMs() < state.deadlineMs()
            && state.request_deadline_ms == routing->deadline_ms && !resource_map_.count(routing->unique_key)) {
            auto entry = std::make_shared<P2PConnectorResourceEntry>();
            entry->request_id = routing->request_id;
            entry->unique_key = routing->unique_key;
            entry->kv_cache_resource = resource;
            entry->request_deadline_ms = state.request_deadline_ms;
            entry->deadline_ms = state.deadlineMs();
            entry->add_time_us = currentTimeUs();
            resource_map_.emplace(routing->unique_key, std::move(entry));
            accepted = true;
        }
    }
    if (release_layers && on_request_released_) {
        on_request_released_(routing->request_id, routing->deadline_ms);
    }
    resource_cv_.notify_all();
    return accepted;
}

void P2PConnectorResourceStore::markCancelled(const std::string& unique_key, int64_t request_deadline_ms) {
    markTerminal(unique_key, request_deadline_ms);
}

void P2PConnectorResourceStore::markTerminal(const std::string& unique_key, int64_t request_deadline_ms) {
    int64_t request_id = -1;
    std::optional<P2PConnectorResourceEntry::SideChannelData> retired;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto it = request_states_.find(unique_key);
        if (it == request_states_.end()) {
            if (!validDeadline(request_deadline_ms)) {
                return;
            }
            it = request_states_.emplace(unique_key, RequestState{request_deadline_ms}).first;
        }
        it->second.terminal = true;
        retired.swap(it->second.side_channel_data);
        request_deadline_ms = it->second.request_deadline_ms;
        auto resource = resource_map_.find(unique_key);
        if (resource != resource_map_.end()) {
            request_id = resource->second->request_id;
            reportMetrics(false, true, resource->second->add_time_us);
            resource_map_.erase(resource);
        }
    }
    retired.reset();
    resource_cv_.notify_all();
    if (request_id >= 0 && on_request_released_) {
        on_request_released_(request_id, request_deadline_ms);
    }
}

std::shared_ptr<P2PConnectorResourceEntry> P2PConnectorResourceStore::waitAndStealResource(
    const std::string& unique_key, int64_t deadline_ms, int64_t request_deadline_ms,
    std::function<bool()> is_cancelled) {
    if (!validDeadline(request_deadline_ms) || !validDeadline(deadline_ms)
        || deadline_ms > request_deadline_ms || currentTimeMs() >= deadline_ms) {
        return nullptr;
    }
    const auto start_time_us = currentTimeUs();
    std::unique_lock<std::mutex> lock(resource_map_mutex_);
    auto [it, inserted] = request_states_.try_emplace(unique_key, RequestState{request_deadline_ms});
    auto& state = it->second;
    if (state.terminal || state.consumed || state.request_deadline_ms != request_deadline_ms) {
        return nullptr;
    }
    state.load_deadline_ms = state.load_deadline_ms > 0 ? std::min(state.load_deadline_ms, deadline_ms) : deadline_ms;
    auto resource = resource_map_.find(unique_key);
    if (resource != resource_map_.end()) {
        resource->second->deadline_ms = state.deadlineMs();
    }
    resource_cv_.notify_all();
    // Re-find state on each wake: the scanner may remove an expired request.
    const auto stopped = [&]() {
        auto current = request_states_.find(unique_key);
        return current == request_states_.end() || current->second.terminal || current->second.consumed
               || currentTimeMs() >= current->second.deadlineMs();
    };
    const auto ready = waitWithBackoff(lock, resource_cv_,
        std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms)),
        [&]() { return stopped() || resource_map_.count(unique_key); }, is_cancelled);
    if (!ready || stopped() || (is_cancelled && is_cancelled())) {
        reportMetrics(!ready, is_cancelled && is_cancelled(), start_time_us);
        return nullptr;
    }
    auto entry = resource_map_.at(unique_key);
    auto& current = request_states_.at(unique_key);
    current.consumed = true;
    entry->deadline_ms = current.deadlineMs();
    resource_map_.erase(unique_key);
    reportMetrics(false, false, entry->add_time_us);
    resource_cv_.notify_all();
    return entry;
}

void P2PConnectorResourceStore::checkTimeout() {
    std::vector<std::pair<int64_t, int64_t>> released;
    std::vector<P2PConnectorResourceEntry::SideChannelData> retired;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        const auto now = currentTimeMs();
        for (auto it = request_states_.begin(); it != request_states_.end();) {
            auto& state = it->second;
            if (now >= state.deadlineMs()) {
                state.terminal = true;
                if (state.side_channel_data) {
                    retired.push_back(std::move(*state.side_channel_data));
                    state.side_channel_data.reset();
                }
                auto resource = resource_map_.find(it->first);
                if (resource != resource_map_.end()) {
                    released.emplace_back(resource->second->request_id, state.request_deadline_ms);
                    reportMetrics(true, false, resource->second->add_time_us);
                    resource_map_.erase(resource);
                }
            }
            if (now >= state.request_deadline_ms) {
                it = request_states_.erase(it);
            } else {
                ++it;
            }
        }
        if (metrics_reporter_) {
            auto collector = std::make_shared<StreamStoreCountMetricsCollector>();
            collector->stream_count = resource_map_.size();
            metrics_reporter_->report<P2PConnectorMetrics, StreamStoreCountMetricsCollector>(nullptr, collector.get());
        }
    }
    retired.clear();
    resource_cv_.notify_all();
    if (on_request_released_) {
        for (const auto& [request_id, request_deadline_ms] : released) {
            on_request_released_(request_id, request_deadline_ms);
        }
    }
}

void P2PConnectorResourceStore::notifySideChannelReady(const std::string&                           unique_key,
                                                       int64_t                                      request_deadline_ms,
                                                       P2PConnectorResourceEntry::SideChannelData&& data) {
    if (!validDeadline(request_deadline_ms) || currentTimeMs() >= request_deadline_ms) {
        return;
    }
    std::optional<P2PConnectorResourceEntry::SideChannelData> retired;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto [it, inserted] = request_states_.try_emplace(unique_key, RequestState{request_deadline_ms});
        auto& state = it->second;
        if (state.terminal || currentTimeMs() >= state.deadlineMs()
            || state.request_deadline_ms != request_deadline_ms) {
            return;
        }
        retired.swap(state.side_channel_data);
        state.side_channel_data.emplace(std::move(data));
    }
    resource_cv_.notify_all();
}

bool P2PConnectorResourceStore::consumeSideChannelData(
    const std::string& unique_key, P2PConnectorResourceEntry::SideChannelData& out_data) {
    std::optional<P2PConnectorResourceEntry::SideChannelData> consumed;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto                        it = request_states_.find(unique_key);
        if (it == request_states_.end() || it->second.terminal || currentTimeMs() >= it->second.deadlineMs()
            || !it->second.side_channel_data) {
            return false;
        }
        consumed.swap(it->second.side_channel_data);
    }
    // Replacing an existing output may free its tensors; do that outside the store lock.
    out_data = std::move(*consumed);
    return true;
}

void P2PConnectorResourceStore::clearSideChannelData(const std::string& unique_key) {
    std::optional<P2PConnectorResourceEntry::SideChannelData> retired;
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto                        it = request_states_.find(unique_key);
        if (it != request_states_.end()) {
            retired.swap(it->second.side_channel_data);
        }
    }
}

bool P2PConnectorResourceStore::waitSideChannelReady(const std::string& unique_key, int64_t deadline_ms,
                                                   std::function<bool()> is_cancelled) {
    std::unique_lock<std::mutex> lock(resource_map_mutex_);
    const auto ready = [&]() {
        auto it = request_states_.find(unique_key);
        return it == request_states_.end() || it->second.terminal || currentTimeMs() >= it->second.deadlineMs()
               || it->second.side_channel_data.has_value();
    };
    if (!validDeadline(deadline_ms) || currentTimeMs() >= deadline_ms) {
        return false;
    }
    waitWithBackoff(lock, resource_cv_,
        std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms)), ready, is_cancelled);
    auto it = request_states_.find(unique_key);
    return !(is_cancelled && is_cancelled()) && it != request_states_.end() && !it->second.terminal
           && currentTimeMs() < it->second.deadlineMs() && it->second.side_channel_data.has_value();
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
