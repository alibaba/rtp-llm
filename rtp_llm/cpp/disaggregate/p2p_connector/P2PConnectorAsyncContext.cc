#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorAsyncContext.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <functional>

namespace rtp_llm {

/*----------------------------------------------- P2PConnectorAsyncMatchContext
 * -------------------------------------------------*/
size_t P2PConnectorAsyncMatchContext::matchedBlockCount() const {
    auto& layer_block_ids = resource_->layerBlockIds();
    if (!layer_block_ids.empty() && layer_block_ids.at(0)) {
        return layer_block_ids.at(0)->blocksNum();
    }
    return 0;
}

ConnectorType P2PConnectorAsyncMatchContext::connectorType() const {
    return ConnectorType::P2P;
}

bool P2PConnectorAsyncMatchContext::done() const {
    return true;
}

bool P2PConnectorAsyncMatchContext::success() const {
    return true;
}

/*----------------------------------------------- P2PConnectorAsyncReadContext
 * -------------------------------------------------*/
bool P2PConnectorAsyncReadContext::done() const {
    return tp_sync_result_->done() && server_call_result_->done();
}

bool P2PConnectorAsyncReadContext::success() const {
    return tp_sync_result_->success() && server_call_result_->success();
}

void P2PConnectorAsyncReadContext::checkDone() {
    if (!tp_sync_result_->done()) {
        tp_sync_result_->checkDone();
    }
    if (!server_call_result_->done()) {
        server_call_result_->checkDone();
    }
    if (done()) {
        collector_->success                  = success();
        collector_->total_cost_time_us       = currentTimeUs() - collector_->start_time_us;
        collector_->tp_sync_cost_time_us     = tp_sync_result_->totalCostTimeUs();
        collector_->server_call_cost_time_us = server_call_result_->totalCostTimeUs();
    }
}

/*----------------------------------------------- P2PConnectorAsyncWriteByLayerContext
 * -------------------------------------------------*/
bool P2PConnectorAsyncWriteByLayerContext::done() const {
    return true;
}

bool P2PConnectorAsyncWriteByLayerContext::success() const {
    return true;
}

/*----------------------------------------------- P2PConnectorAsyncReadContextChecker
 * -------------------------------------------------*/
P2PConnectorAsyncReadContextChecker::~P2PConnectorAsyncReadContextChecker() {
    stop();
}

bool P2PConnectorAsyncReadContextChecker::init(const kmonitor::MetricsReporterPtr& metrics_reporter) {
    metrics_reporter_ = metrics_reporter;
    check_done_thread_ =
        autil::LoopThread::createLoopThread(std::bind(&P2PConnectorAsyncReadContextChecker::checkOnce, this),
                                            5 * 1000,  // 5ms
                                            "P2PConnectorAsyncReadContextCheckerThread");
    if (!check_done_thread_) {
        RTP_LLM_LOG_ERROR("P2PConnectorAsyncReadContextChecker init failed: check_done_thread is null");
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorAsyncReadContextChecker init success");
    return true;
}

void P2PConnectorAsyncReadContextChecker::stop() {
    if (check_done_thread_) {
        check_done_thread_->stop();
        check_done_thread_.reset();
    }
}

void P2PConnectorAsyncReadContextChecker::addContext(const std::shared_ptr<P2PConnectorAsyncReadContext>& context) {
    if (!context) {
        return;
    }
    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    async_contexts_.push_back(context);
}

size_t P2PConnectorAsyncReadContextChecker::inflightContextCount() const {
    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    return async_contexts_.size();
}

void P2PConnectorAsyncReadContextChecker::checkOnce() {
    int64_t start_time_us = currentTimeUs();

    std::lock_guard<std::mutex> lock(async_contexts_mutex_);
    for (auto& async_context : async_contexts_) {
        async_context->checkDone();
        // RTP_LLM_LOG_INFO("P2PConnectorAsyncReadContextChecker::checkOnce: async_context: %p, done_: %d, success_:
        // %d", async_context.get(), async_context->done(), async_context->success());
    }
    async_contexts_.erase(
        std::remove_if(async_contexts_.begin(),
                       async_contexts_.end(),
                       [](const std::shared_ptr<P2PConnectorAsyncReadContext>& async_context) -> bool {
                           // RTP_LLM_LOG_INFO("P2PConnectorAsyncReadContextChecker::checkOnce: async_context: %p,
                           // done_: %d, success_: %d", async_context.get(), async_context->done(),
                           // async_context->success());
                           return async_context->done();
                       }),
        async_contexts_.end());

    if (metrics_reporter_) {
        auto collector                     = std::make_shared<P2PConnectorClientSchedulerStatusMetricsCollector>();
        collector->check_once_cost_time_us = currentTimeUs() - start_time_us;
        collector->inflight_context_count  = async_contexts_.size();
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorClientSchedulerStatusMetricsCollector>(
            nullptr, collector.get());
    }
}

}  // namespace rtp_llm