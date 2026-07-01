#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorAsyncContext.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <algorithm>
#include <functional>

namespace rtp_llm {

namespace {
constexpr int64_t kLeasePollInitialIntervalMs = 10;
constexpr int64_t kLeasePollMaxIntervalMs     = 100;
constexpr int64_t kLeasePollRpcTimeoutMs      = 500;

bool shouldHoldReadOutcomeUntilLeaseStops(ErrorCode error_code) {
    return error_code == ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE
        || error_code == ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED;
}

const char* readOutcomeHoldReason(ErrorCode error_code) {
    switch (error_code) {
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE:
            return "TRANSFER_NOT_DONE";
        case ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED:
            return "READ_CANCELLED";
        default:
            return "UNKNOWN";
    }
}
}  // namespace

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
    if (tryFinishExpiredLeaseHold()) {
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

// lease 保留窗口在 `checkDone()` 里的闸门（方法名中 tryFinish 仅对应**到期**分支）。
//
// 当 `applyMergedReadOutcome` 因 `P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE` /
// `P2P_CONNECTOR_WORKER_READ_CANCELLED` 进入 hold 时，`done_` 仍为 false，
// 在窗口结束前不走常规 merge（避免反复失败/取消）。本函数仅在 `checkDone()` 开头调用，语义：
// - 未 hold → 返回 false，调用方继续 `tp_sync` / `server_call` 的 checkDone 与 merge。
// - hold 且当前时间仍早于 until_ms → 返回 true，调用方必须直接 return（短路），不推进子 result、不 merge。
// - hold 且已到期 → 清 hold、刷新两侧 result；若都已 done 则 applyMergedReadOutcome(..., false)
// 终态合并（含成功补救），
//   否则仅 done_=true；返回 true，调用方 return。
bool P2PConnectorAsyncReadContext::tryFinishExpiredLeaseHold() {
    if (!lease_hold_pending_.load(std::memory_order_acquire)) {
        return false;
    }
    const int64_t until_ms    = lease_hold_until_ms_.load(std::memory_order_relaxed);
    const bool    timed_out   = currentTimeMs() >= until_ms;
    const bool    all_stopped = lease_all_ranks_stopped_.load(std::memory_order_acquire);

    if (!timed_out && !all_stopped) {
        return true;  // still in polling window, short-circuit
    }

    if (all_stopped) {
        RTP_LLM_LOG_INFO(
            "tryFinishExpiredLeaseHold: all ranks stopped via lease poll, unique_key=%s retries=%d",
            uniqueKey().c_str(),
            lease_poll_retry_count_.load());
    } else {
        RTP_LLM_LOG_WARNING(
            "tryFinishExpiredLeaseHold: final_timeout reached without all ranks stopped, unique_key=%s retries=%d",
            uniqueKey().c_str(),
            lease_poll_retry_count_.load());
    }

    lease_hold_pending_.store(false, std::memory_order_release);
    lease_hold_until_ms_.store(0, std::memory_order_relaxed);

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

    if (allow_transfer_not_done_hold && !success && transfer_not_done_hold_ms_ > 0
        && shouldHoldReadOutcomeUntilLeaseStops(error_code)) {
        RTP_LLM_LOG_WARNING("[PD-DIAG] %s, entering %ldms lease hold, unique_key=%s, "
                            "tp_sync_cost_us=%ld, server_call_cost_us=%ld",
                            readOutcomeHoldReason(error_code),
                            transfer_not_done_hold_ms_,
                            uniqueKey().c_str(),
                            tp_sync_result_->totalCostTimeUs(),
                            server_call_result_->totalCostTimeUs());
        lease_hold_until_ms_.store(currentTimeMs() + transfer_not_done_hold_ms_, std::memory_order_relaxed);
        // Initialise lease polling state for this hold window.
        lease_all_ranks_stopped_.store(false, std::memory_order_relaxed);
        lease_poll_interval_ms_.store(kLeasePollInitialIntervalMs, std::memory_order_relaxed);
        lease_poll_next_ms_.store(currentTimeMs() + kLeasePollInitialIntervalMs, std::memory_order_relaxed);
        lease_poll_retry_count_.store(0, std::memory_order_relaxed);
        lease_hold_pending_.store(true, std::memory_order_release);
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
    RTP_LLM_LOG_DEBUG("[PD-DIAG] P2PAsyncRead done, unique_key=%s, success=%d, error_code=%d, "
                      "total_cost_us=%ld, tp_sync_cost_us=%ld, server_call_cost_us=%ld",
                     uniqueKey().c_str(),
                     success_,
                     static_cast<int>(error_code_),
                     currentTimeUs() - collector_->start_time_us,
                     tp_sync_result_->totalCostTimeUs(),
                     server_call_result_->totalCostTimeUs());
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
    if (lease_hold_pending_.load()) {
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

void P2PConnectorAsyncReadContext::pollLeaseIfNeeded(const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client) {
    if (!lease_hold_pending_.load(std::memory_order_acquire)) {
        return;
    }
    if (done()) {
        return;
    }
    if (lease_all_ranks_stopped_.load(std::memory_order_acquire)) {
        return;
    }
    if (!tp_broadcast_client) {
        return;
    }

    const int64_t now = currentTimeMs();
    if (now < lease_poll_next_ms_.load(std::memory_order_relaxed)) {
        return;
    }

    const std::string unique_key = uniqueKey();
    const int         retry      = lease_poll_retry_count_.fetch_add(1, std::memory_order_relaxed);

    auto result = tp_broadcast_client->queryLeaseStatus(unique_key, kLeasePollRpcTimeoutMs);
    if (!result.success) {
        RTP_LLM_LOG_WARNING("pollLeaseIfNeeded: QUERY_LEASE_STATUS broadcast failed, unique_key=%s retry=%d",
                            unique_key.c_str(),
                            retry);
        // Backoff: double interval up to max.
        const int64_t interval =
            std::min(lease_poll_interval_ms_.load(std::memory_order_relaxed) * 2, kLeasePollMaxIntervalMs);
        lease_poll_interval_ms_.store(interval, std::memory_order_relaxed);
        lease_poll_next_ms_.store(now + interval, std::memory_order_relaxed);
        return;
    }

    if (result.allStopped()) {
        RTP_LLM_LOG_DEBUG("pollLeaseIfNeeded: all ranks stopped, unique_key=%s retry=%d", unique_key.c_str(), retry);
        lease_all_ranks_stopped_.store(true, std::memory_order_release);
        return;
    }

    // Not yet stopped — continue polling with backoff.
    const int64_t interval =
        std::min(lease_poll_interval_ms_.load(std::memory_order_relaxed) * 2, kLeasePollMaxIntervalMs);
    lease_poll_interval_ms_.store(interval, std::memory_order_relaxed);
    lease_poll_next_ms_.store(now + interval, std::memory_order_relaxed);
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

    // Three-phase structure to keep async_contexts_mutex_ off the slow check/cancel path —
    // see DingTalk doc §7 for the 8-min production stall this fixes:
    //   Phase 1 (under lock): snapshot the shared_ptr list only.
    //   Phase 2 (no lock):    run checkDone / lease poll / cancel decisions on the snapshot.
    //   Phase 3 (under lock): reclaim done contexts from the live vector.
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> to_poll;
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> to_cancel;
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> snapshot;
    {
        std::lock_guard<std::mutex> lock(async_contexts_mutex_);
        snapshot = async_contexts_;
    }

    for (const auto& async_context : snapshot) {
        async_context->checkDone();
        if (async_context->needLeasePoll()) {
            to_poll.push_back(async_context);
        }
        if (async_context->needCancel()) {
            RTP_LLM_LOG_DEBUG("P2PConnectorAsyncReadContextChecker checkOnce: needCancel, unique_key: %s",
                              async_context->uniqueKey().c_str());
            to_cancel.push_back(async_context);
        }
    }

    // Drive lease polling outside the lock to avoid blocking addContext on synchronous RPC.
    for (auto& ctx : to_poll) {
        ctx->pollLeaseIfNeeded(tp_broadcast_client_);
    }

    // cancel() is idempotent (server_call_result_->done() / tp_sync_result_->done() guards inside).
    // shared_ptr held in to_cancel keeps each context alive even if Phase 3's erase removes it.
    for (auto& async_context : to_cancel) {
        async_context->cancel(tp_broadcast_client_);
    }

    size_t inflight_after = 0;
    std::vector<std::shared_ptr<P2PConnectorAsyncReadContext>> failed_contexts;
    {
        std::lock_guard<std::mutex> lock(async_contexts_mutex_);
        auto it = async_contexts_.begin();
        while (it != async_contexts_.end()) {
            if ((*it)->done()) {
                if (!(*it)->success()) {
                    failed_contexts.push_back(*it);
                }
                it = async_contexts_.erase(it);
                continue;
            }
            ++it;
        }
        inflight_after = async_contexts_.size();
    }

    for (const auto& async_context : failed_contexts) {
        auto error = async_context->errorInfo();
        RTP_LLM_LOG_WARNING("P2PConnectorAsyncReadContextChecker checkOnce: async read failed, unique_key: %s, error: %s",
                            async_context->uniqueKey().c_str(),
                            error.ToString().c_str());
    }

    if (metrics_reporter_) {
        auto collector                     = std::make_shared<DecodeSchedulerStatusMetricsCollector>();
        collector->check_once_cost_time_us = currentTimeUs() - start_time_us;
        collector->inflight_context_count  = inflight_after;
        metrics_reporter_->report<P2PConnectorMetrics, DecodeSchedulerStatusMetricsCollector>(nullptr, collector.get());
    }
}

}  // namespace rtp_llm
