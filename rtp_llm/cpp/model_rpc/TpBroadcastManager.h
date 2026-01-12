#pragma once

#include <chrono>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

class TPBroadcastResult {
public:
    struct WorkerRpcContext {
        std::shared_ptr<RpcService::Stub>      stub;
        std::shared_ptr<grpc::ClientContext>   client_context;
        std::shared_ptr<BroadcastTpRequestPB>  request;
        std::shared_ptr<BroadcastTpResponsePB> response;
        grpc::CompletionQueue                  completion_queue;
        grpc::Status                           status;
        std::string                            server_addr;
        int                                    timeout_ms;

        WorkerRpcContext(): response(new BroadcastTpResponsePB()) {}
    };

public:
    explicit TPBroadcastResult(const std::vector<std::shared_ptr<WorkerRpcContext>>& worker_rpc_contexts):
        worker_contexts_(worker_rpc_contexts), finished_(worker_rpc_contexts.size(), false) {}
    ~TPBroadcastResult() = default;

public:
    bool done() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return finished_count_ == worker_contexts_.size();
    }

    bool success() const {
        if (!done()) {
            return false;
        }
        std::unique_lock<std::mutex> lock(mutex_);
        return all_request_success_;
    }

    void waitDone(int timeout_ms) {
        auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);
        std::unique_lock<std::mutex> lock(mutex_);
        int                          once_timeout_ms = 5;
        while (true) {
            if (finished_count_ == worker_contexts_.size()) {
                break;
            }

            if (std::chrono::system_clock::now() >= deadline) {
                break;
            }

            auto once_deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(once_timeout_ms);
            for (size_t rank = 0; rank < worker_contexts_.size(); ++rank) {
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
                    // tp sync should not fail
                    RTP_LLM_FAIL("tp broadcast rpc cq failed, rank=%zu status=%d", rank, static_cast<int>(next_status));
                }
                ++finished_count_;
                finished_[rank] = true;

                // tp sync should not timeout, or may cause kvcache value error
                const auto& status = ctx->status;
                if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                    RTP_LLM_FAIL("tp broadcast rpc timeout, timeout_ms=%d rank=%zu err=%d(%s) addr=%s",
                                 ctx->timeout_ms,
                                 rank,
                                 status.error_code(),
                                 status.error_message().c_str(),
                                 ctx->server_addr.c_str());
                }

                if (!status.ok()) {
                    RTP_LLM_LOG_WARNING("tp broadcast rpc failed, rank=%zu err=%d(%s) addr=%s",
                                        rank,
                                        status.error_code(),
                                        status.error_message().c_str(),
                                        ctx->server_addr.c_str());
                    all_request_success_ = false;
                    cancelAllRequests();
                }
            }
        }
    }

    std::vector<std::shared_ptr<BroadcastTpResponsePB>> responses() const {
        std::vector<std::shared_ptr<BroadcastTpResponsePB>> responses;
        responses.reserve(worker_contexts_.size());
        for (const auto& worker_rpc_context : worker_contexts_) {
            responses.push_back(worker_rpc_context->response);
        }
        return responses;
    }

private:
    void cancelAllRequests() const {
        for (const auto& context : worker_contexts_) {
            context->client_context->TryCancel();
        }
    }

private:
    std::vector<std::shared_ptr<WorkerRpcContext>> worker_contexts_;
    mutable std::mutex                             mutex_;
    mutable std::vector<bool>                      finished_;
    mutable bool                                   all_request_success_{true};
    mutable size_t                                 finished_count_{0};
};

class TpBroadcastManager {
public:
    explicit TpBroadcastManager(const std::vector<std::string>& worker_addrs): worker_addrs_(worker_addrs) {}
    ~TpBroadcastManager() {
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

    std::shared_ptr<TPBroadcastResult> broadcast(const std::vector<std::shared_ptr<BroadcastTpRequestPB>>& requests,
                                                 int64_t timeout_ms) const {
        RTP_LLM_LOG_INFO("timeout is %ld", timeout_ms);
        const auto worker_size = worker_addrs_.size();
        if (requests.size() != worker_size) {
            RTP_LLM_LOG_WARNING(
                "broadcast failed, requests size mismatch, req: %zu, worker size: %zu", requests.size(), worker_size);
            return nullptr;
        }

        using CtxT = TPBroadcastResult::WorkerRpcContext;
        std::vector<std::shared_ptr<CtxT>> contexts(worker_size);
        const auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);

        for (size_t rank = 0; rank < worker_size; ++rank) {
            const auto& addr        = worker_addrs_[rank];
            auto        conn_status = rpc_pool_->getConnection(addr);
            if (!conn_status.ok()) {
                RTP_LLM_LOG_WARNING("broadcast: getConnection failed rank=%zu addr=%s", rank, addr.c_str());
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

        for (size_t rank = 0; rank < worker_size; ++rank) {
            auto& ctx   = contexts.at(rank);
            auto reader = ctx->stub->AsyncBroadcastTp(ctx->client_context.get(), *ctx->request, &ctx->completion_queue);
            reader->Finish(ctx->response.get(), &ctx->status, reinterpret_cast<void*>(static_cast<intptr_t>(rank)));
        }

        return std::make_shared<TPBroadcastResult>(std::move(contexts));
    }

    size_t workerNum() const {
        return worker_addrs_.size();
    }

private:
    std::vector<std::string> worker_addrs_;
    std::shared_ptr<RPCPool> rpc_pool_;
};

}  // namespace rtp_llm
