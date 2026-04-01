#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorResourceStore.h"

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <thread>

namespace {

std::chrono::system_clock::time_point deadlineToTimeoutPoint(int64_t deadline_ms, int64_t start_time_us) {
    const int64_t remaining_us = deadline_ms * 1000 - start_time_us;
    return std::chrono::system_clock::now() + std::chrono::microseconds(remaining_us);
}

}  // namespace

namespace rtp_llm {

P2PConnectorResourceStore::P2PConnectorResourceStore(const kmonitor::MetricsReporterPtr& metrics_reporter,
                                                     int                                 timeout_check_interval_ms):
    metrics_reporter_(metrics_reporter), timeout_check_interval_ms_(timeout_check_interval_ms) {}

P2PConnectorResourceStore::~P2PConnectorResourceStore() {
    if (check_timeout_thread_) {
        check_timeout_thread_->stop();
    }
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

bool P2PConnectorResourceStore::addResource(const IGenerateStreamPtr& generate_stream,
                                            const KVCacheResourcePtr& kv_cache_resource) {
    const std::string unique_key = generate_stream->uniqueKey();
    if (unique_key.empty()) {
        RTP_LLM_LOG_WARNING("P2PConnectorResourceStore::addResource failed: unique_key is empty");
        return false;
    }
    int64_t deadline_ms = generate_stream->deadlineMs();
    {
        std::lock_guard<std::mutex> lock(resource_map_mutex_);
        auto                        entry = std::make_shared<P2PConnectorResourceEntry>();
        entry->request_id                 = generate_stream->requestId();
        entry->generate_stream            = generate_stream;
        entry->kv_cache_resource          = kv_cache_resource;
        entry->deadline_ms                = deadline_ms;
        entry->add_time_us                = currentTimeUs();
        resource_map_[unique_key]         = entry;
    }
    // 通知所有等待的线程
    resource_cv_.notify_all();
    return true;
}

bool P2PConnectorResourceStore::waitForResourceOrCancellation(std::unique_lock<std::mutex>&         lock,
                                                              const std::string&                    unique_key,
                                                              std::chrono::system_clock::time_point timeout_tp,
                                                              const std::function<bool()>&          is_cancelled) {
    // is_cancelled 由外部线程设置时不会 notify resource_cv_；用带退避、有上界的 wait_until 轮询
    // （与 prefill waitForBroadcastCompletion 类似），addResource 仍会 notify_all。
    bool          satisfied   = false;
    int           sleep_ms    = 1;
    constexpr int kBackoffCap = 8;
    while (true) {
        if (is_cancelled && is_cancelled()) {
            satisfied = true;
            break;
        }
        if (resource_map_.find(unique_key) != resource_map_.end()) {
            satisfied = true;
            break;
        }
        const auto now = std::chrono::system_clock::now();
        if (now >= timeout_tp) {
            satisfied = false;
            break;
        }
        auto next_wake = now + std::chrono::milliseconds(std::min(sleep_ms, kBackoffCap));
        if (next_wake > timeout_tp) {
            next_wake = timeout_tp;
        }
        resource_cv_.wait_until(lock, next_wake);
        sleep_ms = std::min(sleep_ms * 2, kBackoffCap);
    }
    return satisfied;
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

    return stealResourceEntryLocked(unique_key);
}

void P2PConnectorResourceStore::checkTimeout() {
    std::lock_guard<std::mutex> lock(resource_map_mutex_);
    int64_t                     current_time_ms = currentTimeMs();
    for (auto it = resource_map_.begin(); it != resource_map_.end();) {
        auto& [unique_key, entry] = *it;
        if (entry && current_time_ms >= entry->deadline_ms) {
            RTP_LLM_LOG_WARNING(
                "P2PConnectorResourceStore: resource timeout, unique_key: %s, deadline_ms: %ld, current_time_ms: %ld",
                unique_key.c_str(),
                entry->deadline_ms,
                current_time_ms);
            auto wait_start_time_us = entry->add_time_us;
            it                      = resource_map_.erase(it);
            reportMetrics(true, false, wait_start_time_us);
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
