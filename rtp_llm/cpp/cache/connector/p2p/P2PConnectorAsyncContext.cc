#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <algorithm>
#include <functional>

namespace rtp_llm {

/*----------------------------------------------- P2PConnectorAsyncMatchContext
 * -------------------------------------------------*/
size_t P2PConnectorAsyncMatchContext::matchedBlockCount() const {
    auto& layer_block_ids = resource_->layerBlocks();
    if (!layer_block_ids.empty() && layer_block_ids.at(0)) {
        return layer_block_ids.at(0)->blocksNum();
    }
    return 0;
}

KVCacheConnector::ConnectorType P2PConnectorAsyncMatchContext::connectorType() const {
    return KVCacheConnector::ConnectorType::P2P;
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
    bool prefill_done = true;
    if (prefill_context_) {
        prefill_done = prefill_context_->done();
    }
    return prefill_done && tp_sync_result_->done() && server_call_result_->done();
}

bool P2PConnectorAsyncReadContext::success() const {
    bool prefill_success = true;
    if (prefill_context_) {
        prefill_success = prefill_context_->success();
    }
    return prefill_success && tp_sync_result_->success() && server_call_result_->success();
}

void P2PConnectorAsyncReadContext::checkDone() {
    // 检查 prefill_context 状态（如果存在）
    if (prefill_context_) {
        if (!prefill_context_->done()) {
            // 非阻塞检查 prefill 是否完成
            prefill_context_->checkDone();
        }
    }
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
        RTP_LLM_LOG_DEBUG("P2PConnectorAsyncReadContext checkDone: async_context: [%p], success: %d", this, success());

        // 根据请求结果设置 error_code
        if (!success()) {
            // 检查 prefill_context 是否失败
            if (prefill_context_ && prefill_context_->done() && !prefill_context_->success()) {
                error_code_ = ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED;
            }
            // 检查 server_call_result 是否失败
            else if (server_call_result_->done() && !server_call_result_->success()) {
                error_code_ = ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED;
            }
            // 检查 tp_sync_result 是否失败，从 result 获取 error_code 和 error_msg
            else if (tp_sync_result_->done() && !tp_sync_result_->success()) {
                error_code_ = tp_sync_result_->errorCode();
                // error_msg 可以从 tp_sync_result_->errorMessage() 获取，但这里只设置 error_code_
            }
        }
    }
}

bool P2PConnectorAsyncReadContext::needCancel() const {
    if (done()) {
        return false;
    }
    // 检查 prefill_context 是否失败
    if (prefill_context_ && prefill_context_->done() && !prefill_context_->success()) {
        return true;
    }
    // 当其中一个完成并且失败时，需要取消另一个
    if (tp_sync_result_->done() && !tp_sync_result_->success()) {
        return true;
    }
    if (server_call_result_->done() && !server_call_result_->success()) {
        return true;
    }
    return false;
}

void P2PConnectorAsyncReadContext::cancel(const std::shared_ptr<TPBroadcastClient>& tp_broadcast_client) {
    std::string unique_key = uniqueKey();
    RTP_LLM_LOG_DEBUG("P2PConnectorAsyncReadContext cancel: unique_key: %s", unique_key.c_str());

    // 如果 prefill_context_ 存在且未完成，取消 prefill 请求
    if (prefill_context_ && !prefill_context_->done() && prefill_context_->client_context) {
        RTP_LLM_LOG_DEBUG("P2PConnectorAsyncReadContext cancel: cancelling prefill_context, unique_key: %s",
                          unique_key.c_str());
        prefill_context_->client_context->TryCancel();
    }

    // 如果 server_call_result_ 未完成，取消 grpc 请求
    if (!server_call_result_->done()) {
        RTP_LLM_LOG_DEBUG("P2PConnectorAsyncReadContext cancel: cancelling server_call_result, unique_key: %s",
                          unique_key.c_str());
        server_call_result_->cancel();
    }

    // 如果 tp_sync_result_ 未完成，通过 TPBroadcastClient 发送 CANCEL 请求
    if (!tp_sync_result_->done() && tp_broadcast_client && !cancel_result_) {
        RTP_LLM_LOG_DEBUG(
            "P2PConnectorAsyncReadContext cancel: cancelling tp_sync_result via broadcast, unique_key: %s",
            unique_key.c_str());
        cancel_result_ = tp_broadcast_client->cancel(unique_key, P2PConnectorBroadcastType::CANCEL_READ);
    }
    if (cancel_result_ && !cancel_result_->done()) {
        cancel_result_->checkDone();
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

bool P2PConnectorAsyncReadContextChecker::init(const kmonitor::MetricsReporterPtr&       metrics_reporter,
                                               const std::shared_ptr<TPBroadcastClient>& tp_broadcast_client) {
    metrics_reporter_    = metrics_reporter;
    tp_broadcast_client_ = tp_broadcast_client;
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
        // 检查是否需要取消另一个未完成的请求
        if (async_context->needCancel()) {
            RTP_LLM_LOG_DEBUG("P2PConnectorAsyncReadContextChecker checkOnce: needCancel, unique_key: %s",
                              async_context->uniqueKey().c_str());
            async_context->cancel(tp_broadcast_client_);
        }
    }
    // 当 async_context done && !success 时，调用 generate_stream->setStop 设置停止并设置 errorcode
    for (auto& async_context : async_contexts_) {
        if (async_context->done() && !async_context->success()) {
            auto generate_stream = async_context->generateStream();
            if (generate_stream) {
                ErrorCode   error_code = async_context->errorCode();
                std::string error_msg  = "P2P connector async read failed: " + ErrorCodeToString(error_code);
                generate_stream->setStop(error_code, error_msg);
                RTP_LLM_LOG_WARNING(
                    "P2PConnectorAsyncReadContextChecker checkOnce: setStop called, unique_key: %s, error_code: %s",
                    async_context->uniqueKey().c_str(),
                    ErrorCodeToString(error_code).c_str());
            }
        }
    }

    async_contexts_.erase(
        std::remove_if(async_contexts_.begin(),
                       async_contexts_.end(),
                       [](const std::shared_ptr<P2PConnectorAsyncReadContext>& async_context) -> bool {
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