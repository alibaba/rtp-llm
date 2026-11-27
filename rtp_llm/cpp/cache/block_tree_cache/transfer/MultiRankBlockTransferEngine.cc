#include "rtp_llm/cpp/cache/block_tree_cache/transfer/MultiRankBlockTransferEngine.h"
#include "rtp_llm/cpp/cache/block_tree_cache/metrics/CacheObserver.h"

#include <algorithm>
#include <exception>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include <type_traits>

#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferBatchAsyncContext.h"
#include "rtp_llm/cpp/cache/block_tree_cache/transfer/TransferStageTimer.h"
#include "rtp_llm/cpp/cache/block_tree_cache/diagnostic/CacheDiagnostics.h"

namespace rtp_llm {

namespace {

class MultiRankTransferAsyncContext final:
    public TransferAsyncContext,
    public std::enable_shared_from_this<MultiRankTransferAsyncContext> {
public:
    static std::shared_ptr<MultiRankTransferAsyncContext> create(const std::vector<TransferCompletionPeer>& peers,
                                                                 size_t                                     block_count,
                                                                 SharedTransferStageTimer                   wait_timer,
                                                                 std::shared_ptr<CacheObserver>             reporter,
                                                                 int64_t                                    request_id,
                                                                 uint64_t                                   transfer_id,
                                                                 CacheStageContext                          identity,
                                                                 const char* completion_operation) {
        auto context = std::shared_ptr<MultiRankTransferAsyncContext>(new MultiRankTransferAsyncContext(block_count));
        context->wait_timer_          = std::move(wait_timer);
        context->reporter_            = std::move(reporter);
        context->request_id_          = request_id;
        context->transfer_id_         = transfer_id;
        context->identity_            = identity;
        context->completion_identity_ = identity;
        if (completion_operation)
            context->completion_identity_.operation = completion_operation;
        auto diagnostics      = std::make_shared<TransferExecutionDiagnostics>();
        diagnostics->trace_id = transfer_id;
        diagnostics->ranks.reserve(peers.size());
        for (const auto& peer : peers) {
            context->expected_ranks_.push_back(peer.rank);
            TransferRankSnapshot snapshot;
            snapshot.rank           = peer.rank;
            snapshot.server_address = peer.address;
            snapshot.dispatch_state = TransferRpcDispatchState::POSSIBLY_SUBMITTED;
            snapshot.dispatch_error = "broadcast dispatch, subscription, or result publication failed";
            diagnostics->ranks.push_back(std::move(snapshot));
        }
        std::sort(context->expected_ranks_.begin(), context->expected_ranks_.end());
        if (std::adjacent_find(context->expected_ranks_.begin(), context->expected_ranks_.end())
            != context->expected_ranks_.end()) {
            throw std::invalid_argument("duplicate multi-rank transfer participant");
        }
        context->unknown_result_ = TransferResult(
            ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "transfer broadcast completion could not be established"),
            CacheCopyError::RPC_FAILED);
        context->unknown_result_.expandBlockErrors(block_count);
        context->unknown_result_.execution_diagnostics = std::move(diagnostics);
        context->unknown_publication_                  = context->unknown_result_;
        context->prepareCallback();
        return context;
    }

    void attach(std::shared_ptr<RankTransferBroadcast> result) {
        broadcast_result_ = std::move(result);
        try {
            if (auto probe = block_tree_cache_detail::before_transfer_broadcast_subscription_for_test.load(
                    std::memory_order_acquire)) {
                probe();
            }
            broadcast_result_->onDone(std::move(completion_callback_));
        } catch (...) {
            completeUnknown();
        }
    }

    void completeUnknown() {
        // Both results and the callback were prepared before submitting RPCs.
        // Registration failure must not escape as an ordinary execution error.
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (completion_.done() || finishing_ || fallback_started_) {
                return;
            }
            fallback_started_ = true;
        }
        completion_callback_ = {};
        if (broadcast_result_) {
            try {
                auto diagnostics                           = collectDiagnostics();
                unknown_result_.execution_diagnostics      = diagnostics;
                unknown_publication_.execution_diagnostics = std::move(diagnostics);
            } catch (...) {
                // The preallocated snapshot conservatively marks every rank
                // possibly submitted if collecting diagnostics also fails.
            }
        }
        publish(std::move(unknown_result_), std::move(unknown_publication_));
    }

    void discardUnsubmittedCallback() {
        completion_callback_ = {};
    }

    void waitDone() override {
        if (done()) {
            return;
        }
        try {
            broadcast_result_->waitDone();
        } catch (...) {
            completeUnknown();
        }
        if (!done()) {
            evaluateWithFallback();
        }
        completion_.waitDone();
    }

    void onDone(DoneCallback callback) override {
        completion_.onDone(std::move(callback));
    }

    bool done() const override {
        return completion_.done();
    }

    bool success() const override {
        return completion_.success();
    }

    TransferResult result() const override {
        return completion_.result();
    }

    std::shared_ptr<const TransferExecutionDiagnostics> executionDiagnostics() const override {
        if (completion_.done())
            return completion_.result().execution_diagnostics;
        return broadcast_result_ ? collectDiagnostics() : unknown_result_.execution_diagnostics;
    }

private:
    explicit MultiRankTransferAsyncContext(size_t block_count): block_count_(block_count), completion_(block_count) {}

    void prepareCallback() {
        std::shared_ptr<MultiRankTransferAsyncContext> self = shared_from_this();
        completion_callback_                                = [self = std::move(self)] {
            TransferStageTimer total(self->reporter_,
                                     "completion_total",
                                     self->request_id_,
                                     0,
                                     self->block_count_,
                                     -1,
                                     "coordinator_completion",
                                     self->transfer_id_,
                                     self->completion_identity_);
            total.finish(self->evaluateWithFallback());
        };
    }

    bool evaluateWithFallback() {
        try {
            return evaluate();
        } catch (...) {
            // A transient allocation failure before the completion claim can
            // be retried immediately. Async consumers must not depend on a
            // future waitDone() call to retry the only CQ notification.
            try {
                return evaluate();
            } catch (...) {
                completeUnknown();
                return false;
            }
        }
    }

    bool evaluate() {
        if (!broadcast_result_->done()) {
            return false;
        }
        auto result = TransferResult::emptyAggregate();
        if (auto reason = broadcast_result_->failureSummary(); !reason.empty()) {
            result = {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, std::move(reason)), CacheCopyError::RPC_FAILED};
        }

        TransferStageTimer merge(reporter_,
                                 "rank_result_merge",
                                 request_id_,
                                 0,
                                 block_count_,
                                 -1,
                                 "coordinator_completion",
                                 transfer_id_,
                                 completion_identity_);
        result.execution_diagnostics = collectDiagnostics(&result);
        // Copy before claiming completion: allocation failure must leave the
        // context retryable. Once claimed, publication cannot allocate.
        if (auto probe =
                block_tree_cache_detail::before_transfer_publication_copy_for_test.load(std::memory_order_acquire))
            probe();
        auto published_result = result;
        merge.finish(result.ok());
        return publish(std::move(result), std::move(published_result));
    }

    std::shared_ptr<const TransferExecutionDiagnostics> collectDiagnostics(TransferResult* aggregate = nullptr) const {
        auto  ignored         = TransferResult::emptyAggregate();
        auto& result          = aggregate ? *aggregate : ignored;
        auto  replies         = broadcast_result_->replies();
        auto  diagnostics     = std::make_shared<TransferExecutionDiagnostics>();
        diagnostics->trace_id = transfer_id_;
        diagnostics->ranks.reserve(replies.size());
        std::vector<size_t> observed;
        observed.reserve(replies.size());
        for (const auto& reply : replies)
            observed.push_back(reply.snapshot.rank);
        std::sort(observed.begin(), observed.end());
        if (observed != expected_ranks_) {
            result.merge(
                {ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "multi-rank transfer participants do not match"),
                 CacheCopyError::RPC_FAILED});
        }
        result.expandBlockErrors(block_count_);
        for (auto& reply : replies) {
            const auto rank = reply.snapshot.rank;
            if (!reply.result
                || (!reply.result->block_errors.empty() && reply.result->block_errors.size() != block_count_)) {
                reply.snapshot.response_valid = false;
                reply.result.emplace(
                    ErrorInfo(ErrorCode::TRANSFER_COMPLETION_UNKNOWN, "missing or malformed rank completion"),
                    CacheCopyError::RPC_FAILED);
            }
            diagnostics->ranks.push_back(std::move(reply.snapshot));
            auto& rank_result = *reply.result;
            for (size_t index = 0; index < block_count_; ++index) {
                result.block_errors[index] = std::max(result.block_errors[index], rank_result.blockError(index));
            }
            if (!rank_result.ok()) {
                rank_result.status =
                    ErrorInfo(rank_result.status.code(),
                              "rank=" + std::to_string(rank) + " {" + rank_result.status.ToString() + "}");
            }
            result.merge(std::move(rank_result));
        }
        return diagnostics;
    }

    bool publish(TransferResult result, TransferResult published_result) {
        static_assert(std::is_nothrow_move_assignable_v<TransferResult>);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (completion_.done() || finishing_) {
                // An unpublished result owned by another evaluator is not a
                // failure of this completed response evaluation.
                return result.ok();
            }
            finishing_ = true;
        }
        if (auto probe =
                block_tree_cache_detail::after_transfer_completion_claim_for_test.load(std::memory_order_acquire))
            probe();
        if (wait_timer_) {
            wait_timer_.finish(result.ok());
            try {
                const auto times = broadcast_result_ ? broadcast_result_->completionTimesUs() : std::vector<int64_t>{};
                if (times.size() > 1 && std::all_of(times.begin(), times.end(), [](int64_t t) { return t > 0; })) {
                    const auto    bounds  = std::minmax_element(times.begin(), times.end());
                    const int64_t skew_us = *bounds.second - *bounds.first;
                    if (reporter_) {
                        reporter_->reportTransferStage("rank_completion_skew",
                                                       skew_us,
                                                       result.ok(),
                                                       -1,
                                                       "coordinator",
                                                       identity_.operation,
                                                       identity_.source,
                                                       identity_.target);
                    }
                    const auto       wall_times = broadcast_result_->completionWallTimesUs();
                    const auto       end_us     = wall_times[std::distance(times.begin(), bounds.second)];
                    AsyncProfileSpan span("rank_completion_skew",
                                          request_id_,
                                          0,
                                          block_count_,
                                          end_us - skew_us,
                                          -1,
                                          transfer_id_,
                                          identity_.path,
                                          AsyncProfileKind::Derived);
                    span.finish(end_us);
                }
            } catch (...) {
                // Diagnostics must not prevent completion notification.
            }
        }
        TransferStageTimer dispatch(reporter_,
                                    "callback_dispatch",
                                    request_id_,
                                    0,
                                    block_count_,
                                    -1,
                                    "coordinator_completion",
                                    transfer_id_,
                                    completion_identity_);
        completion_.complete(std::move(published_result));
        dispatch.finish(result.ok());
        return result.ok();
    }

    std::shared_ptr<RankTransferBroadcast> broadcast_result_;
    std::vector<size_t>                    expected_ranks_;
    size_t                                 block_count_{0};
    mutable std::mutex                     mutex_;
    bool                                   finishing_{false};
    TransferBatchAsyncContext              completion_;
    TransferResult                         unknown_result_;
    TransferResult                         unknown_publication_;
    RankTransferBroadcast::DoneCallback    completion_callback_;
    bool                                   fallback_started_{false};
    SharedTransferStageTimer               wait_timer_;
    std::shared_ptr<CacheObserver>         reporter_;
    int64_t                                request_id_{-1};
    uint64_t                               transfer_id_{0};
    CacheStageContext                      identity_;
    CacheStageContext                      completion_identity_;
};

}  // namespace

MultiRankBlockTransferEngine::MultiRankBlockTransferEngine(std::vector<GroupSetPtr>       groups,
                                                           RankTransferTransport          transport,
                                                           std::shared_ptr<CacheObserver> reporter):
    group_sets_(std::move(groups)), transport_(std::move(transport)), metrics_reporter_(std::move(reporter)) {}

const std::vector<std::string>& MultiRankBlockTransferEngine::participantAddresses() const {
    return transport_.addresses;
}

std::shared_ptr<TransferAsyncContext> MultiRankBlockTransferEngine::execute(TransferTask task,
                                                                            const char*  completion_operation) const {
    const auto descriptors = task.descriptors();
    const size_t block_count = descriptors.size();
    const auto reject = [&](TransferResult result) {
        result.buffer_safe = true;
        return std::make_shared<CompletedTransferAsyncContext>(std::move(result), block_count, task.completionProof());
    };
    const auto expired = [&] {
        return reject({ErrorInfo(ErrorCode::DEADLINE_EXCEEDED, "transfer deadline exceeded before multi-rank submission"),
                       CacheCopyError::NOT_EXECUTED});
    };
    if (task.expired()) return expired();
    if (!transport_.prepare || descriptors.empty()) {
        return reject(ErrorInfo(ErrorCode::INVALID_PARAMS, "multi-rank transfer requires a transport and descriptors"));
    }
    const auto& first = descriptors.front();
    const auto identity = CacheStageContext::local(first.source_tier, first.target_tier);
    auto wait_timer = TransferStageTimer::create(metrics_reporter_, "multi_rank_wait", task.requestId().value_or(-1),
                                                0, block_count, -1, "coordinator", task.traceId(), identity, AsyncProfileKind::Wait);
    auto request = transport_.prepare(task);
    const auto remaining = task.remainingTimeout();
    if (!remaining) return expired();
    if (!request || request->peers.empty() || !request->send) {
        return reject(ErrorInfo(ErrorCode::INVALID_PARAMS, "failed to encode transfer batch"));
    }
    auto context = MultiRankTransferAsyncContext::create(request->peers, block_count, std::move(wait_timer),
                                                         metrics_reporter_, task.requestId().value_or(-1),
                                                         task.traceId(), identity, completion_operation);
    std::shared_ptr<RankTransferBroadcast> broadcast_result;
    try {
        broadcast_result = request->send(*remaining);
    } catch (...) {
        context->completeUnknown();
        return context;
    }
    if (broadcast_result == nullptr) {
        context->discardUnsubmittedCallback();
        BTC_LOG_WARNING("failed to start broadcast");
        return reject({ErrorInfo(ErrorCode::EXECUTION_EXCEPTION, "failed to start transfer broadcast"), CacheCopyError::RPC_FAILED});
    }
    context->attach(std::move(broadcast_result));
    return context;
}

}  // namespace rtp_llm
