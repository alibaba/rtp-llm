#include "rtp_llm/cpp/disaggregate/p2p_connector/StoreWaitContext.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <functional>

namespace rtp_llm {

StoreWaitContextChecker::StoreWaitContextChecker(
    const kmonitor::MetricsReporterPtr&                   metrics_reporter,
    const std::shared_ptr<ComputedLayerCacheBufferStore>& computed_buffers):
    metrics_reporter_(metrics_reporter), computed_buffers_(computed_buffers) {}

StoreWaitContextChecker::~StoreWaitContextChecker() {}

void StoreWaitContextChecker::addContext(const StoreWaitContext& context) {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    contexts_.push_back(context);
}

size_t StoreWaitContextChecker::getContextCount() const {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    return contexts_.size();
}

void StoreWaitContextChecker::checkOnce() {
    std::lock_guard<std::mutex> lock(contexts_mutex_);
    auto                        iter = contexts_.begin();
    while (iter != contexts_.end()) {
        auto& context = *iter;

        // check timeout
        if (currentTimeMs() >= context.deadline_ms) {
            RTP_LLM_LOG_WARNING("StoreWaitContextChecker: wait timeout, request_id: %ld, deadline_ms: %ld",
                                context.request_id,
                                context.deadline_ms);
            if (context.collector) {
                context.collector->success                 = false;
                context.collector->store_wait_done_time_us = currentTimeUs() - context.collector->start_time_us;
            }
            if (metrics_reporter_ && context.collector) {
                metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorServerWorkerStoreMetricsCollector>(
                    nullptr, context.collector.get());
            }
            iter = contexts_.erase(iter);
            continue;
        }

        // check event readiness
        if (context.event == nullptr || context.event->checkReadiness()) {
            if (computed_buffers_) {
                computed_buffers_->addBuffer(context.request_id, context.layer_cache_buffer, context.deadline_ms);
            }
            if (context.collector) {
                context.collector->store_wait_done_time_us = currentTimeUs() - context.collector->start_time_us;
            }
            if (metrics_reporter_ && context.collector) {
                metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorServerWorkerStoreMetricsCollector>(
                    nullptr, context.collector.get());
            }
            iter = contexts_.erase(iter);
            continue;
        }
        ++iter;
    }
}

}  // namespace rtp_llm
