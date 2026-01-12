#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorStreamStore.h"

#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorMetrics.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <thread>
#include <chrono>

namespace rtp_llm {

P2PConnectorStreamStore::P2PConnectorStreamStore(const kmonitor::MetricsReporterPtr& metrics_reporter):
    metrics_reporter_(metrics_reporter) {}

P2PConnectorStreamStore::~P2PConnectorStreamStore() {
    if (check_timeout_thread_) {
        check_timeout_thread_->stop();
    }
}

bool P2PConnectorStreamStore::init() {
    check_timeout_thread_ = autil::LoopThread::createLoopThread(std::bind(&P2PConnectorStreamStore::checkTimeout, this),
                                                                100,  // 100ms
                                                                "P2PConnectorStreamStoreCheckTimeoutThread");
    if (!check_timeout_thread_) {
        RTP_LLM_LOG_ERROR("P2PConnectorStreamStore init failed: check_timeout_thread is null");
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorStreamStore init success");
    return true;
}

void P2PConnectorStreamStore::addResource(const std::string&        unique_key,
                                          int64_t                   request_id,
                                          const IGenerateStreamPtr& generate_stream,
                                          const KVCacheResourcePtr& kv_cache_resource,
                                          int64_t                   deadline_ms) {
    std::lock_guard<std::mutex> lock(resource_map_mutex_);
    auto                        entry = std::make_shared<P2PConnectorResourceEntry>();
    entry->request_id                 = request_id;
    entry->generate_stream            = generate_stream;
    entry->kv_cache_resource          = kv_cache_resource;
    entry->deadline_ms                = deadline_ms;
    entry->add_time_us                = currentTimeUs();
    resource_map_[unique_key]         = entry;
    RTP_LLM_LOG_INFO("P2PConnectorStreamStore::addResource: unique_key: %s, request_id: %ld, deadline_ms: %ld",
                     unique_key.c_str(),
                     request_id,
                     deadline_ms);
}

std::shared_ptr<P2PConnectorResourceEntry> P2PConnectorStreamStore::stealResource(const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(resource_map_mutex_);
    auto                        it = resource_map_.find(unique_key);
    if (it == resource_map_.end()) {
        return nullptr;
    }
    auto entry              = it->second;
    auto wait_start_time_us = entry->add_time_us;
    resource_map_.erase(it);
    reportMetrics(false, wait_start_time_us);
    RTP_LLM_LOG_INFO(
        "P2PConnectorStreamStore::stealResource success, unique_key: %s, request_id: %ld, deadline_ms: %ld, add_time_us: %ld, size: %zu",
        unique_key.c_str(),
        entry->request_id,
        entry->deadline_ms,
        entry->add_time_us,
        resource_map_.size());
    return entry;
}

void P2PConnectorStreamStore::checkTimeout() {
    std::lock_guard<std::mutex> lock(resource_map_mutex_);
    int64_t                     current_time_ms = currentTimeMs();
    for (auto it = resource_map_.begin(); it != resource_map_.end();) {
        auto& [unique_key, entry] = *it;
        if (entry && current_time_ms >= entry->deadline_ms) {
            RTP_LLM_LOG_WARNING(
                "P2PConnectorStreamStore: resource timeout, unique_key: %s, deadline_ms: %ld, current_time_ms: %ld",
                unique_key.c_str(),
                entry->deadline_ms,
                current_time_ms);
            auto wait_start_time_us = entry->add_time_us;
            it                      = resource_map_.erase(it);
            reportMetrics(true, wait_start_time_us);
        } else {
            ++it;
        }
    }
    if (metrics_reporter_) {
        auto collector          = std::make_shared<P2PConnectorStreamStoreMetricsCollector1>();
        collector->stream_count = resource_map_.size();
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorStreamStoreMetricsCollector1>(nullptr,
                                                                                                 collector.get());
    }
}

void P2PConnectorStreamStore::reportMetrics(bool timeout, int64_t wait_start_time_us) {
    if (metrics_reporter_) {
        auto collector                 = std::make_shared<P2PConnectorStreamStoreMetricsCollector2>();
        collector->timeout             = timeout;
        collector->stream_wait_time_us = currentTimeUs() - wait_start_time_us;
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorStreamStoreMetricsCollector2>(nullptr,
                                                                                                 collector.get());
    }
}

}  // namespace rtp_llm
