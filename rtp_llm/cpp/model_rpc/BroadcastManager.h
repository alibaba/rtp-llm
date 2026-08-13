#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

template<typename RequestPB, typename ResponsePB>
class BroadcastResult {
    friend class BroadcastManager;

public:
    struct WorkerRpcContext {
        std::shared_ptr<RpcService::Stub>    stub;
        std::shared_ptr<grpc::ClientContext> client_context;
        RequestPB                            request;
        ResponsePB                           response;
        grpc::CompletionQueue                completion_queue;
        grpc::Status                         status;
        std::string                          server_addr;
        int                                  timeout_ms;
        bool                                 finish_posted{false};
        void*                                expected_finish_tag{nullptr};
        bool                                 finish_observed{false};
        bool                                 completion_queue_drained{false};
    };

public:
    explicit BroadcastResult(const std::vector<std::shared_ptr<WorkerRpcContext>>& worker_rpc_contexts):
        worker_contexts_(worker_rpc_contexts), finished_(worker_rpc_contexts.size(), false) {}
    ~BroadcastResult() {
        drainNoexcept();
    }

public:
    /// Snapshot of internal completion counters, not a live probe of RPC completion.
    ///
    /// Progress (polling gRPC completion queues and updating `finished_*`) happens only inside
    /// `waitDone()` / `waitDone(int)`. This method does not poll or advance completion; it only
    /// reflects state already updated by prior `waitDone()` work. Callers must not infer that
    /// work is still in flight from `done() == false` without also driving `waitDone()`, nor
    /// assume `done() == true` reflects anything that `waitDone()` has not yet observed.
    bool done() const {
        return already_done_.load(std::memory_order_acquire);
    }

    /// Polls completion queues until all workers finish or `timeout_ms` elapses (0 = no limit).
    /// This is what advances completion state observed by `done()`.
    bool waitDone(int timeout_ms) {
        if (already_done_.load(std::memory_order_acquire)) {
            throwTerminalExceptionIfNeeded();
            return true;
        }

        std::unique_lock<std::mutex> lock(wait_done_mutex_);
        if (already_done_.load(std::memory_order_relaxed)) {
            throwTerminalExceptionIfNeeded();
            return true;
        }

        const int  worker_size = worker_contexts_.size();
        const auto deadline    = (timeout_ms > 0) ?
                                     std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms) :
                                     std::chrono::steady_clock::time_point::max();

        while (true) {
            if (finished_count_.load() == worker_size) {
                break;
            }
            if (!terminal_exception_seen_ && timeout_ms > 0 && std::chrono::steady_clock::now() >= deadline) {
                return false;
            }
            const int  once_timeout_ms = 1;
            const auto once_deadline   = std::chrono::system_clock::now() + std::chrono::milliseconds(once_timeout_ms);
            for (int rank = 0; rank < worker_size; ++rank) {
                if (finished_[rank]) {
                    continue;
                }

                auto& ctx         = worker_contexts_.at(rank);
                if (!ctx->finish_posted) {
                    shutdownAndDrainCompletionQueue(*ctx);
                    markRankFinished(rank);
                    grpc_status_failure_seen_ = true;
                    recordTerminalException("Finish was not posted");
                    continue;
                }
                void* got_tag     = nullptr;
                bool  ok          = false;
                auto  next_status = ctx->completion_queue.AsyncNext(&got_tag, &ok, once_deadline);
                if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                    continue;
                }
                if (next_status == grpc::CompletionQueue::NextStatus::SHUTDOWN) {
                    ctx->completion_queue_drained = true;
                    markRankFinished(rank);
                    grpc_status_failure_seen_ = true;
                    recordTerminalException("completion queue shutdown before Finish");
                    continue;
                }
                if (got_tag != ctx->expected_finish_tag) {
                    continue;
                }

                ctx->finish_observed = true;
                markRankFinished(rank);
                if (!ok) {
                    RTP_LLM_LOG_WARNING(
                        "broadcast rpc Finish failed, rank=%d status=%d", rank, static_cast<int>(next_status));
                    grpc_status_failure_seen_ = true;
                    recordTerminalException("Finish completion failure");
                    shutdownAndDrainCompletionQueue(*ctx);
                    continue;
                }

                const auto& status = ctx->status;
                if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                    RTP_LLM_LOG_WARNING("broadcast rpc timeout, timeout_ms=%d rank=%d err=%d(%s) addr=%s",
                                        ctx->timeout_ms,
                                        rank,
                                        status.error_code(),
                                        status.error_message().c_str(),
                                        ctx->server_addr.c_str());
                    recordTerminalException("deadline exceeded");
                }
                if (!status.ok()) {
                    RTP_LLM_LOG_WARNING("broadcast rpc failed, rank=%d err=%d(%s) addr=%s",
                                        rank,
                                        status.error_code(),
                                        status.error_message().c_str(),
                                        ctx->server_addr.c_str());
                    grpc_status_failure_seen_ = true;
                }
                shutdownAndDrainCompletionQueue(*ctx);
            }
        }

        // Match pre-refactor semantics: finalize once all ranks are finished (same as local bool + single store).
        all_request_success_.store(!grpc_status_failure_seen_);
        already_done_.store(true, std::memory_order_release);
        throwTerminalExceptionIfNeeded();
        return true;
    }

    /// Same as `waitDone(0)`; drives completion state for `done()` / `success()`.
    void waitDone() {
        (void)waitDone(/*timeout_ms=*/0);
    }

    void cancelAndDrain() noexcept {
        try {
            std::unique_lock<std::mutex> lock(wait_done_mutex_);
            if (already_done_.load(std::memory_order_relaxed)) {
                return;
            }
            for (size_t rank = 0; rank < worker_contexts_.size(); ++rank) {
                if (!finished_[rank] && worker_contexts_[rank]->finish_posted
                    && worker_contexts_[rank]->client_context) {
                    worker_contexts_[rank]->client_context->TryCancel();
                }
            }
            drainToTerminalLocked();
            all_request_success_.store(false);
            already_done_.store(true, std::memory_order_release);
        } catch (...) {
            std::abort();
        }
    }

    void drainNoexcept() noexcept {
        try {
            waitDone();
        } catch (...) {
            cancelAndDrain();
        }
    }

    bool success() const {
        return all_request_success_.load();
    }

    std::vector<ResponsePB> responses() const {
        std::unique_lock<std::mutex> lock(wait_done_mutex_);
        std::vector<ResponsePB>      responses;
        responses.reserve(worker_contexts_.size());
        for (const auto& worker_rpc_context : worker_contexts_) {
            responses.push_back(worker_rpc_context->response);
        }
        return responses;
    }

private:
    void drainToTerminalLocked() {
        for (size_t rank = 0; rank < worker_contexts_.size(); ++rank) {
            if (finished_[rank]) {
                continue;
            }
            auto& context = worker_contexts_[rank];
            if (!context->finish_posted) {
                shutdownAndDrainCompletionQueue(*context);
                markRankFinished(rank);
                continue;
            }
            void* tag     = nullptr;
            bool  ok      = false;
            while (context->completion_queue.Next(&tag, &ok)) {
                if (tag != context->expected_finish_tag) {
                    continue;
                }
                context->finish_observed = true;
                markRankFinished(rank);
                if (!ok) {
                    grpc_status_failure_seen_ = true;
                    recordTerminalException("Finish completion failure while draining");
                }
                shutdownAndDrainCompletionQueue(*context);
                break;
            }
            if (!finished_[rank]) {
                markRankFinished(rank);
                grpc_status_failure_seen_ = true;
                recordTerminalException("completion queue shutdown before Finish while draining");
                context->completion_queue_drained = true;
            }
        }
    }

    void markRankFinished(size_t rank) {
        if (finished_[rank]) {
            return;
        }
        finished_[rank] = true;
        ++finished_count_;
    }

    void shutdownAndDrainCompletionQueue(WorkerRpcContext& context) {
        if (context.completion_queue_drained) {
            return;
        }
        context.completion_queue.Shutdown();
        void* tag = nullptr;
        bool  ok  = false;
        while (context.completion_queue.Next(&tag, &ok)) {
        }
        context.completion_queue_drained = true;
    }

    void throwTerminalExceptionIfNeeded() const {
        if (terminal_exception_seen_) {
            RTP_LLM_FAIL("broadcast rpc failed after all ranks reached a terminal state: %s",
                         terminal_exception_message_.c_str());
        }
    }

    void recordTerminalException(const char* message) {
        if (terminal_exception_seen_) {
            return;
        }
        terminal_exception_seen_    = true;
        terminal_exception_message_ = message;
        for (size_t rank = 0; rank < worker_contexts_.size(); ++rank) {
            if (!finished_[rank] && worker_contexts_[rank]->finish_posted
                && worker_contexts_[rank]->client_context) {
                worker_contexts_[rank]->client_context->TryCancel();
            }
        }
    }

    std::vector<std::shared_ptr<WorkerRpcContext>> worker_contexts_;
    std::vector<bool>                              finished_;
    std::atomic<int>                               finished_count_{0};
    std::atomic<bool>                              already_done_{false};
    std::atomic<bool>                              all_request_success_{false};
    bool                                           grpc_status_failure_seen_{false};
    bool                                           terminal_exception_seen_{false};
    std::string                                    terminal_exception_message_;
    mutable std::mutex                             wait_done_mutex_;
};

class BroadcastManager {
public:
    explicit BroadcastManager(const std::vector<std::string>& worker_addrs): worker_addrs_(worker_addrs) {}
    ~BroadcastManager() {
        rpc_pool_.reset();
    }

public:
    bool init() {
        if (worker_addrs_.empty()) {
            RTP_LLM_LOG_WARNING("init failed, worker_addrs is empty");
            return false;
        }

        rpc_pool_ = std::make_shared<RPCPool>();
        return true;
    }

    template<typename RequestPB, typename ResponsePB, typename RpcCall>
    std::shared_ptr<BroadcastResult<RequestPB, ResponsePB>>
    broadcast(const std::vector<RequestPB>& requests, int timeout_ms, const RpcCall& rpc_call) const {
        // Compatibility contract: rpc_call returns a reader created by Async*, which has already started the RPC.
        return broadcastImpl<RequestPB, ResponsePB>(requests, timeout_ms, rpc_call, /*reader_prepared=*/false);
    }

    template<typename RequestPB, typename ResponsePB, typename RpcCall>
    std::shared_ptr<BroadcastResult<RequestPB, ResponsePB>>
    broadcastPrepared(const std::vector<RequestPB>& requests, int timeout_ms, const RpcCall& rpc_call) const {
        // Quiesce-safe contract: rpc_call returns a reader created by PrepareAsync*. Dispatch starts only after the
        // callback successfully returns a non-null reader.
        return broadcastImpl<RequestPB, ResponsePB>(requests, timeout_ms, rpc_call, /*reader_prepared=*/true);
    }

    size_t workerNum() const {
        return worker_addrs_.size();
    }

private:
    template<typename RequestPB, typename ResponsePB, typename RpcCall>
    std::shared_ptr<BroadcastResult<RequestPB, ResponsePB>> broadcastImpl(const std::vector<RequestPB>& requests,
                                                                          int timeout_ms,
                                                                          const RpcCall& rpc_call,
                                                                          bool reader_prepared) const {
        const auto worker_size = worker_addrs_.size();
        if (worker_size == 0 || requests.size() != worker_size) {
            RTP_LLM_LOG_WARNING(
                "broadcast failed, requests size mismatch, req: %zu, worker size: %zu", requests.size(), worker_size);
            return nullptr;
        }

        using CtxT = typename BroadcastResult<RequestPB, ResponsePB>::WorkerRpcContext;
        std::vector<std::shared_ptr<CtxT>> contexts(worker_size);
        const auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);

        for (int rank = 0; rank < worker_size; ++rank) {
            const auto& addr        = worker_addrs_[rank];
            auto        conn_status = rpc_pool_->getConnection(addr);
            if (!conn_status.ok()) {
                RTP_LLM_LOG_WARNING("broadcast: getConnection failed rank=%d addr=%s", rank, addr.c_str());
                return nullptr;
            }

            contexts[rank]      = std::make_shared<CtxT>();
            auto& ctx           = contexts.at(rank);
            ctx->stub           = conn_status.value().stub;
            ctx->request        = requests.at(rank);
            ctx->server_addr    = addr;
            ctx->timeout_ms     = timeout_ms;
            ctx->client_context = std::make_shared<grpc::ClientContext>();
            ctx->client_context->set_deadline(deadline);
        }

        auto result = std::make_shared<BroadcastResult<RequestPB, ResponsePB>>(std::move(contexts));
        try {
            for (int rank = 0; rank < worker_size; ++rank) {
                auto& ctx    = result->worker_contexts_.at(rank);
                auto  reader = rpc_call(ctx->stub, ctx->client_context, ctx->request, &ctx->completion_queue);
                if (!reader) {
                    RTP_LLM_LOG_WARNING("broadcast: rpc call returned no reader, rank=%d addr=%s",
                                        rank,
                                        ctx->server_addr.c_str());
                    result->cancelAndDrain();
                    return nullptr;
                }
                try {
                    if (reader_prepared) {
                        reader->StartCall();
                    }
                    ctx->expected_finish_tag = static_cast<void*>(ctx.get());
                    reader->Finish(&ctx->response,
                                   &ctx->status,
                                   ctx->expected_finish_tag);
                } catch (...) {
                    for (int started_rank = 0; started_rank < rank; ++started_rank) {
                        auto& started_ctx = result->worker_contexts_.at(started_rank);
                        if (started_ctx->finish_posted && started_ctx->client_context) {
                            started_ctx->client_context->TryCancel();
                        }
                    }
                    std::abort();
                }
                ctx->finish_posted = true;
            }
        } catch (...) {
            result->cancelAndDrain();
            throw;
        }

        return result;
    }
    std::vector<std::string> worker_addrs_;
    std::shared_ptr<RPCPool> rpc_pool_;
};

}  // namespace rtp_llm
