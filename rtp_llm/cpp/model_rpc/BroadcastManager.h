#pragma once

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

enum class BroadcastDeadlinePolicy {
    FATAL_ABORT,
    EXIT_WITHOUT_CORE,
};

template<typename RequestPB, typename ResponsePB>
class BroadcastResult {
public:
    struct WorkerRpcContext {
        std::shared_ptr<RpcService::Stub>     stub;
        std::shared_ptr<grpc::ClientContext>  client_context;
        RequestPB                             request;
        ResponsePB                            response;
        grpc::CompletionQueue                 completion_queue;
        grpc::Status                          status;
        std::string                           server_addr;
        int                                   timeout_ms;
        std::chrono::steady_clock::time_point start_time{std::chrono::steady_clock::now()};
    };

public:
    explicit BroadcastResult(const std::vector<std::shared_ptr<WorkerRpcContext>>& worker_rpc_contexts,
                             BroadcastDeadlinePolicy deadline_policy = BroadcastDeadlinePolicy::FATAL_ABORT,
                             uint64_t                operation_id    = 0):
        worker_contexts_(worker_rpc_contexts),
        finished_(worker_rpc_contexts.size(), false),
        deadline_policy_(deadline_policy),
        operation_id_(operation_id) {}
    ~BroadcastResult() = default;

public:
    /// Snapshot of internal completion counters, not a live probe of RPC completion.
    ///
    /// Progress (polling gRPC completion queues and updating `finished_*`) happens only inside
    /// `waitDone()` / `waitDone(int)`. This method does not poll or advance completion; it only
    /// reflects state already updated by prior `waitDone()` work. Callers must not infer that
    /// work is still in flight from `done() == false` without also driving `waitDone()`, nor
    /// assume `done() == true` reflects anything that `waitDone()` has not yet observed.
    bool done() const {
        return finished_count_.load() == static_cast<int>(worker_contexts_.size());
    }

    /// Polls completion queues until all workers finish or `timeout_ms` elapses (0 = no limit).
    /// This is what advances completion state observed by `done()`.
    bool waitDone(int timeout_ms) {
        if (already_done_.load()) {
            return true;
        }

        std::unique_lock<std::mutex> lock(wait_done_mutex_);
        if (already_done_.load()) {
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
            if (timeout_ms > 0 && std::chrono::steady_clock::now() >= deadline) {
                RTP_LLM_LOG_ERROR("broadcast wait timeout, timeout_ms=%d state=%s",
                                  timeout_ms,
                                  completionStateStringLocked().c_str());
                return false;
            }
            const int  once_timeout_ms = 1;
            const auto once_deadline   = std::chrono::system_clock::now() + std::chrono::milliseconds(once_timeout_ms);
            for (int rank = 0; rank < worker_size; ++rank) {
                if (finished_[rank]) {
                    continue;
                }

                auto& ctx         = worker_contexts_.at(rank);
                void* got_tag     = nullptr;
                bool  ok          = false;
                auto  next_status = ctx->completion_queue.AsyncNext(&got_tag, &ok, once_deadline);
                if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
                    continue;
                }
                if (!ok) {
                    // Preserve the original CQ-failure behavior. At this point gRPC did not
                    // deliver a trustworthy completion event, so remote execution state is
                    // unknown and must not be reclassified as an ordinary cache miss.
                    RTP_LLM_FAIL("broadcast rpc cq failed, operation_id=%lu rank=%d status=%d state=%s",
                                 operation_id_,
                                 rank,
                                 static_cast<int>(next_status),
                                 completionStateStringLocked().c_str());
                }
                ++finished_count_;
                finished_[rank] = true;

                const auto& status     = ctx->status;
                const auto  elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                            std::chrono::steady_clock::now() - ctx->start_time)
                                            .count();
                if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                    if (deadline_policy_ == BroadcastDeadlinePolicy::FATAL_ABORT) {
                        RTP_LLM_FATAL_ABORT("broadcast rpc timeout, timeout_ms=%d rank=%d err=%d(%s) addr=%s",
                                            ctx->timeout_ms,
                                            rank,
                                            status.error_code(),
                                            status.error_message().c_str(),
                                            ctx->server_addr.c_str());
                    }
                    RTP_LLM_LOG_ERROR("broadcast rpc timeout, exit_without_core=1 operation_id=%lu timeout_ms=%d "
                                      "rank=%d elapsed_ms=%ld err=%d(%s) addr=%s state=%s request=[%s]",
                                      operation_id_,
                                      ctx->timeout_ms,
                                      rank,
                                      elapsed_ms,
                                      status.error_code(),
                                      status.error_message().c_str(),
                                      ctx->server_addr.c_str(),
                                      completionStateStringLocked().c_str(),
                                      ctx->request.DebugString().c_str());
                    Logger::getEngineLogger().flush();
                    Logger::getStackTraceLogger().flush();
                    Logger::getAccessLogger().flush();
                    std::fflush(stdout);
                    std::fflush(stderr);
                    std::_Exit(EXIT_FAILURE);
                }
                if (!status.ok()) {
                    RTP_LLM_LOG_WARNING("broadcast rpc failed, rank=%d elapsed_ms=%ld err=%d(%s) addr=%s state=%s",
                                        rank,
                                        elapsed_ms,
                                        status.error_code(),
                                        status.error_message().c_str(),
                                        ctx->server_addr.c_str(),
                                        completionStateStringLocked().c_str());
                    grpc_status_failure_seen_ = true;
                }
            }
        }

        // Match pre-refactor semantics: finalize once all ranks are finished (same as local bool + single store).
        all_request_success_.store(!grpc_status_failure_seen_);
        already_done_.store(true);
        return true;
    }

    /// Same as `waitDone(0)`; drives completion state for `done()` / `success()`.
    void waitDone() {
        (void)waitDone(/*timeout_ms=*/0);
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
    std::string completionStateStringLocked() const {
        std::ostringstream done;
        std::ostringstream pending;
        done << "[";
        pending << "[";
        bool first_done    = true;
        bool first_pending = true;
        for (size_t rank = 0; rank < finished_.size(); ++rank) {
            auto& out   = finished_[rank] ? done : pending;
            auto& first = finished_[rank] ? first_done : first_pending;
            if (!first) {
                out << ",";
            }
            out << rank;
            if (finished_[rank]) {
                const auto& status = worker_contexts_[rank]->status;
                out << ":" << static_cast<int>(status.error_code());
                if (!status.error_message().empty()) {
                    out << "(" << status.error_message() << ")";
                }
            }
            first = false;
        }
        done << "]";
        pending << "]";
        return "done=" + done.str() + " pending=" + pending.str();
    }

private:
    std::vector<std::shared_ptr<WorkerRpcContext>> worker_contexts_;
    std::vector<bool>                              finished_;
    std::atomic<int>                               finished_count_{0};
    std::atomic<bool>                              already_done_{false};
    std::atomic<bool>                              all_request_success_{false};
    bool                                           grpc_status_failure_seen_{false};
    BroadcastDeadlinePolicy                        deadline_policy_{BroadcastDeadlinePolicy::FATAL_ABORT};
    uint64_t                                       operation_id_{0};
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
    broadcast(const std::vector<RequestPB>& requests,
              int                           timeout_ms,
              const RpcCall&                rpc_call,
              BroadcastDeadlinePolicy       deadline_policy = BroadcastDeadlinePolicy::FATAL_ABORT,
              uint64_t                      operation_id    = 0) const {
        const auto worker_size = worker_addrs_.size();
        if (requests.size() != worker_size) {
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
            ctx->start_time     = std::chrono::steady_clock::now();
            ctx->client_context = std::make_shared<grpc::ClientContext>();
            ctx->client_context->set_deadline(deadline);
        }

        for (int rank = 0; rank < worker_size; ++rank) {
            auto& ctx    = contexts.at(rank);
            auto  reader = rpc_call(ctx->stub, ctx->client_context, ctx->request, &ctx->completion_queue);
            reader->Finish(&ctx->response, &ctx->status, reinterpret_cast<void*>(static_cast<intptr_t>(rank)));
        }

        return std::make_shared<BroadcastResult<RequestPB, ResponsePB>>(
            std::move(contexts), deadline_policy, operation_id);
    }

    size_t workerNum() const {
        return worker_addrs_.size();
    }

private:
    std::vector<std::string> worker_addrs_;
    std::shared_ptr<RPCPool> rpc_pool_;
};

}  // namespace rtp_llm
