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

bool P2PConnectorAsyncMatchContext::done() const {
    return true;
}

bool P2PConnectorAsyncMatchContext::success() const {
    return true;
}

/*----------------------------------------------- P2PConnectorAsyncReadContext
 * -------------------------------------------------*/
bool P2PConnectorAsyncReadContext::done() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return done_;
}

bool P2PConnectorAsyncReadContext::success() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return success_;
}

void P2PConnectorAsyncReadContext::waitDone() {
    std::unique_lock<std::mutex> lock(state_mutex_);
    done_cv_.wait(lock, [this]() { return done_; });
}

// 生产路径由 P2PConnectorAsyncReadContextChecker 单线程按间隔调用
// `checkDone()`，不存在与其它调用方并发重入，故无实际竞态。UT 为同线程同步调用。若未来多线程驱动
// checkDone，需整体重审。
void P2PConnectorAsyncReadContext::checkDone() {
    if (done()) {
        return;
    }
    if (tryFinishExpiredTransferNotDoneHold()) {
        return;
    }
    if (!tp_sync_result_->done()) {
        tp_sync_result_->checkDone();
    }
    if (!server_call_result_->done()) {
        server_call_result_->checkDone();
    }
    const bool both_done = tp_sync_result_->done() && server_call_result_->done();
    if (!both_done) {
        return;
    }

    applyMergedReadOutcome(mergeReadResultsWhenBothDone());
}

// `TRANSFER_NOT_DONE` 保留窗口在 `checkDone()` 里的闸门（方法名中 tryFinish 仅对应**到期**分支）。
//
// 当 `applyMergedReadOutcome` 因 `P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE` 进入 hold 时，`done_` 仍为 false，
// 在窗口结束前不走常规 merge（避免反复失败/取消）。本函数仅在 `checkDone()` 开头调用，语义：
// - 未 hold → 返回 false，调用方继续 `tp_sync` / `server_call` 的 checkDone 与 merge。
// - hold 且当前时间仍早于 until_ms → 返回 true，调用方必须直接 return（短路），不推进子 result、不 merge。
// - hold 且已到期 → 清 hold、刷新两侧 result；若都已 done 则 applyMergedReadOutcome(..., false)
// 终态合并（含成功补救），
//   否则仅 done_=true；返回 true，调用方 return。
bool P2PConnectorAsyncReadContext::tryFinishExpiredTransferNotDoneHold() {
    if (!transfer_not_done_hold_pending_.load(std::memory_order_acquire)) {
        return false;
    }
    const int64_t until_ms = transfer_not_done_hold_until_ms_.load(std::memory_order_relaxed);
    if (currentTimeMs() < until_ms) {
        return true;
    }

    transfer_not_done_hold_pending_.store(false, std::memory_order_release);
    transfer_not_done_hold_until_ms_.store(0, std::memory_order_relaxed);

    if (!tp_sync_result_->done()) {
        tp_sync_result_->checkDone();
    }
    if (!server_call_result_->done()) {
        server_call_result_->checkDone();
    }
    const bool both_done = tp_sync_result_->done() && server_call_result_->done();
    if (both_done) {
        applyMergedReadOutcome(mergeReadResultsWhenBothDone(), false);
    } else {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!done_) {
            done_ = true;
        }
        done_cv_.notify_all();
    }
    return true;
}

P2PConnectorAsyncReadContext::MergedReadOutcome P2PConnectorAsyncReadContext::mergeReadResultsWhenBothDone() const {
    MergedReadOutcome outcome;
    outcome.success = tp_sync_result_->success() && server_call_result_->success();
    if (!outcome.success) {
        if (tp_sync_result_->done() && !tp_sync_result_->success()) {
            outcome.error_code    = tp_sync_result_->errorCode();
            outcome.error_message = tp_sync_result_->errorMessage();
        } else if (server_call_result_->done() && !server_call_result_->success()) {
            outcome.error_code    = server_call_result_->error_code;
            outcome.error_message = server_call_result_->error_message;
        }
    }
    return outcome;
}

void P2PConnectorAsyncReadContext::applyMergedReadOutcome(const MergedReadOutcome& outcome,
                                                          bool                     allow_transfer_not_done_hold) {
    const bool  success    = outcome.success;
    ErrorCode   error_code = outcome.error_code;
    std::string error_message{outcome.error_message};

    if (allow_transfer_not_done_hold && !success && error_code == ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE
        && transfer_not_done_hold_ms_ > 0) {
        transfer_not_done_hold_until_ms_.store(currentTimeMs() + transfer_not_done_hold_ms_, std::memory_order_relaxed);
        transfer_not_done_hold_pending_.store(true, std::memory_order_release);
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            success_       = false;
            error_code_    = error_code;
            error_message_ = std::move(error_message);
        }
        collector_->success                  = false;
        collector_->total_cost_time_us       = currentTimeUs() - collector_->start_time_us;
        collector_->tp_sync_cost_time_us     = tp_sync_result_->totalCostTimeUs();
        collector_->server_call_cost_time_us = server_call_result_->totalCostTimeUs();
        return;
    }

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        done_       = true;
        success_    = success;
        error_code_ = error_code;
        error_message_.assign(error_message);
        done_cv_.notify_all();
    }
    collector_->success                  = success_;
    collector_->total_cost_time_us       = currentTimeUs() - collector_->start_time_us;
    collector_->tp_sync_cost_time_us     = tp_sync_result_->totalCostTimeUs();
    collector_->server_call_cost_time_us = server_call_result_->totalCostTimeUs();
}

ErrorInfo P2PConnectorAsyncReadContext::errorInfo() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return ErrorInfo(error_code_, error_message_);
}

bool P2PConnectorAsyncReadContext::needCancel() const {
    if (transfer_not_done_hold_pending_.load()) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (done_) {
            return false;
        }
    }
    if (tp_sync_result_->done() && !tp_sync_result_->success()) {
        return true;
    }
    if (server_call_result_->done() && !server_call_result_->success()) {
        return true;
    }
    return false;
}

void P2PConnectorAsyncReadContext::cancel(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
    std::string unique_key = uniqueKey();

    if (!server_call_result_->done()) {
        server_call_result_->cancel();
    }

    // 如果 tp_sync_result_ 未完成，通过 P2PBroadcastClient 发送 CANCEL 请求（至多成功发起一次）
    if (!tp_sync_result_->done() && tp_broadcast_client) {
        bool expected = false;
        if (tp_cancel_broadcast_triggered_.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
            auto cancel_result = tp_broadcast_client->cancel(unique_key, P2PConnectorBroadcastType::CANCEL_READ);
            if (!cancel_result) {
                tp_cancel_broadcast_triggered_.store(false, std::memory_order_release);
            } else if (!cancel_result->done()) {
                cancel_result->checkDone();
            }
        }
    }
}

/*----------------------------------------------- P2PConnectorAsyncWriteByLayerContext
 * -------------------------------------------------*/
void P2PConnectorAsyncWriteByLayerContext::waitDone() {
    // done() is always true, no blocking
}

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

bool P2PConnectorAsyncReadContextChecker::init(const kmonitor::MetricsReporterPtr&        metrics_reporter,
                                               const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
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
    for (auto& async_context : async_contexts_) {
        if (async_context->done() && !async_context->success()) {
            auto error = async_context->errorInfo();
            RTP_LLM_LOG_WARNING(
                "P2PConnectorAsyncReadContextChecker checkOnce: async read failed, unique_key: %s, error: %s",
                async_context->uniqueKey().c_str(),
                error.ToString().c_str());
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
        auto collector                     = std::make_shared<DecodeSchedulerStatusMetricsCollector>();
        collector->check_once_cost_time_us = currentTimeUs() - start_time_us;
        collector->inflight_context_count  = async_contexts_.size();
        metrics_reporter_->report<P2PConnectorMetrics, DecodeSchedulerStatusMetricsCollector>(nullptr, collector.get());
    }
}

}  // namespace rtp_llm
