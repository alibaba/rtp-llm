#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>
#include <type_traits>
#include <utility>
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"

#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"
#include "rtp_llm/cpp/utils/RpcCompletionQueue.h"

namespace rtp_llm {

template<typename RequestPB, typename ResponsePB>
class BroadcastResult {
public:
    struct WorkerRpcContext {
        std::shared_ptr<RpcService::Stub>                            stub;
        std::shared_ptr<grpc::ClientContext>                         client_context;
        RequestPB                                                    request;
        ResponsePB                                                   response;
        std::unique_ptr<grpc::ClientAsyncResponseReader<ResponsePB>> reader;
        grpc::Status                                                 status;
        std::string                                                  server_addr;
        int                                                          timeout_ms{0};
    };

    explicit BroadcastResult(const std::vector<std::shared_ptr<WorkerRpcContext>>& contexts):
        worker_contexts_(contexts), finished_(contexts.size(), false), remaining_(contexts.size()) {}
    ~BroadcastResult() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t rank = 0; rank < worker_contexts_.size(); ++rank) {
            if (!finished_[rank] && worker_contexts_[rank] && worker_contexts_[rank]->client_context) {
                worker_contexts_[rank]->client_context->TryCancel();
            }
        }
        // CQ tags retain individual contexts through physical Finish, not this result.
    }

    bool done() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return remaining_ == 0;
    }
    bool success() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return remaining_ == 0 && all_success_;
    }
    bool waitDone(int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (timeout_ms <= 0) {
            done_cv_.wait(lock, [this] { return remaining_ == 0; });
            return true;
        }
        return done_cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return remaining_ == 0; });
    }
    void waitDone() {
        (void)waitDone(0);
    }

    void setDoneCallback(std::function<void()> callback) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (remaining_ != 0) {
                done_callback_ = std::move(callback);
                return;
            }
        }
        if (callback) {
            callback();
        }
    }

    FirstError::Snapshot firstError() const {
        return first_error_.snapshot();
    }
    void setProgressCallback(std::function<void()> callback) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            progress_callback_ = callback;
        }
        // Covers completion before registration without reading in-flight PBs.
        if (callback)
            callback();
    }

    // Only the registered Finish callback (or a rank that was never dispatched)
    // may report completion. Non-OK client status does not prove server quiescence.
    void complete(size_t rank, bool ok) {
        std::function<void()> callback;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (finished_.at(rank)) {
                return;
            }
            const auto&       ctx = worker_contexts_[rank];
            ErrorInfo         error;
            const std::string location =
                "ExecuteFunction rank=" + std::to_string(rank) + " peer=" + (ctx ? ctx->server_addr : "<null>");
            if (ctx && !ctx->status.ok()) {
                error = errorInfoFromGrpcStatus(ctx->status, location);
            } else if (!ok || !ctx) {
                error = ErrorInfo(ErrorCode::RPC_FINISH_FAILED, location + ": Finish event failed");
            } else if constexpr (std::is_same_v<ResponsePB, FunctionResponsePB>) {
                if (ctx->request.has_p2p_request() || ctx->response.has_p2p_response()) {
                    if (!ctx->response.has_p2p_response()) {
                        error = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                                          location + ": missing p2p_response");
                    } else if (ctx->response.p2p_response().error_code() != ErrorCodePB::NONE_ERROR) {
                        const auto& response = ctx->response.p2p_response();
                        error                = ErrorInfo(transRPCErrorCode(response.error_code()),
                                          location + " key=" + ctx->request.p2p_request().unique_key() + ": "
                                              + response.error_message());
                    }
                }
            }
            first_error_.record(error);
            all_success_    = all_success_ && error.ok();
            finished_[rank] = true;
            if (--remaining_ == 0) {
                callback = std::move(done_callback_);
            }
        }
        std::function<void()> progress;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            progress = progress_callback_;
        }
        done_cv_.notify_all();
        if (progress)
            progress();
        if (callback) {
            callback();
        }
    }

    std::vector<ResponsePB> responses() const {
        std::lock_guard<std::mutex> lock(mutex_);
        // gRPC may still be writing response buffers before Finish.
        if (remaining_ != 0) {
            return {};
        }
        std::vector<ResponsePB> responses;
        responses.reserve(worker_contexts_.size());
        for (const auto& ctx : worker_contexts_) {
            responses.push_back(ctx ? ctx->response : ResponsePB{});
        }
        return responses;
    }

private:
    std::vector<std::shared_ptr<WorkerRpcContext>> worker_contexts_;
    std::vector<bool>                              finished_;
    size_t                                         remaining_;
    bool                                           all_success_{true};
    mutable std::mutex                             mutex_;
    std::condition_variable                        done_cv_;
    std::function<void()>                          done_callback_;
    std::function<void()>                          progress_callback_;
    FirstError                                     first_error_;
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
    broadcast(std::vector<RequestPB> requests, int timeout_ms, const RpcCall& rpc_call) const {
        const auto worker_size = worker_addrs_.size();
        if (requests.size() != worker_size) {
            RTP_LLM_LOG_WARNING(
                "broadcast failed, requests size mismatch, req: %zu, worker size: %zu", requests.size(), worker_size);
            return nullptr;
        }

        using CtxT = typename BroadcastResult<RequestPB, ResponsePB>::WorkerRpcContext;
        std::vector<std::shared_ptr<CtxT>> contexts(worker_size);
        const auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);

        // [PD-DIAG] track total getConnection time across all workers — when one
        // peer's channel is in TRANSIENT_FAILURE this loop can serialize behind
        // RpcPool's mutex and the entire broadcast appears slow.
        const auto t_get_conn_start = std::chrono::steady_clock::now();

        for (int rank = 0; rank < worker_size; ++rank) {
            const auto& addr        = worker_addrs_[rank];
            auto        conn_status = rpc_pool_->getConnection(addr);
            contexts[rank]      = std::make_shared<CtxT>();
            auto& ctx           = contexts.at(rank);
            if (conn_status.ok()) {
                ctx->stub = conn_status.value().stub;
            } else {
                ctx->status =
                    grpcStatusFromErrorInfo(ErrorInfo(ErrorCode::GET_CONNECTION_FAILED,
                                                      "ExecuteFunction getConnection rank=" + std::to_string(rank)
                                                          + " peer=" + addr + ": " + conn_status.status().ToString()));
            }
            ctx->request        = std::move(requests.at(rank));
            ctx->server_addr    = addr;
            ctx->timeout_ms     = timeout_ms;
            ctx->client_context = std::make_shared<grpc::ClientContext>();
            ctx->client_context->set_deadline(deadline);
        }

        const auto get_conn_loop_us = std::chrono::duration_cast<std::chrono::microseconds>(
                                          std::chrono::steady_clock::now() - t_get_conn_start)
                                          .count();
        if (get_conn_loop_us >= 100 * 1000) {
            RTP_LLM_LOG_WARNING(
                "[PD-DIAG] BroadcastManager::broadcast slow getConnection loop, worker_count=%zu, total_us=%lld",
                worker_size,
                (long long)get_conn_loop_us);
        }

        using Result                      = BroadcastResult<RequestPB, ResponsePB>;
        auto                  result      = std::make_shared<Result>(contexts);
        std::weak_ptr<Result> weak_result = result;
        for (size_t rank = 0; rank < worker_size; ++rank) {
            const auto ctx = contexts[rank];
            if (!ctx->status.ok()) {
                result->complete(rank, false);
                continue;
            }
            const bool started = RpcCompletionQueue::instance().submit(
                ctx->client_context,
                [&](grpc::CompletionQueue* cq, void* tag) {
                    ctx->reader = rpc_call(ctx->stub, ctx->client_context, ctx->request, cq);
                    if (!ctx->reader) {
                        return false;
                    }
                    ctx->reader->Finish(&ctx->response, &ctx->status, tag);
                    return true;
                },
                [ctx, weak_result, rank](bool ok) {
                    (void)ctx;  // Keep RPC buffers alive even if the result was abandoned.
                    if (auto result = weak_result.lock()) {
                        result->complete(rank, ok);
                    }
                });
            if (!started) {
                RTP_LLM_LOG_WARNING(
                    "broadcast: create async reader failed rank=%zu addr=%s", rank, ctx->server_addr.c_str());
                ctx->status = grpcStatusFromErrorInfo(ErrorInfo(
                    ErrorCode::RPC_FINISH_FAILED,
                    "ExecuteFunction async start failed rank=" + std::to_string(rank) + " peer=" + ctx->server_addr));
                for (size_t pending = rank; pending < worker_size; ++pending) {
                    result->complete(pending, false);
                }
                break;
            }
        }
        return result;
    }

    size_t workerNum() const {
        return worker_addrs_.size();
    }

private:
    std::vector<std::string> worker_addrs_;
    std::shared_ptr<RPCPool> rpc_pool_;
};

}  // namespace rtp_llm
