#pragma once

#include <chrono>
#include <mutex>
#include <vector>

#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/utils/AssertUtils.h"

namespace rtp_llm {

template<typename RequestPB, typename ResponsePB>
class BroadcastResult {
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
    };

public:
    explicit BroadcastResult(const std::vector<std::shared_ptr<WorkerRpcContext>>& worker_rpc_contexts):
        worker_contexts_(worker_rpc_contexts) {}
    ~BroadcastResult() = default;

public:
    void waitDone() {
        if (already_done_.load()) {
            return;
        }

        std::unique_lock<std::mutex> lock(wait_done_mutex_);
        if (already_done_.load()) {
            return;
        }

        bool              all_request_success = true;
        const int         worker_size         = worker_contexts_.size();
        std::vector<bool> finished(worker_size, false);
        int               finished_count = 0;

        while (true) {
            if (finished_count == worker_size) {
                break;
            }
            const int  once_timeout_ms = 1;
            const auto once_deadline   = std::chrono::system_clock::now() + std::chrono::milliseconds(once_timeout_ms);
            for (int rank = 0; rank < worker_size; ++rank) {
                if (finished[rank]) {
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
                    RTP_LLM_FAIL("broadcast rpc cq failed, rank=%d status=%d", rank, static_cast<int>(next_status));
                }
                ++finished_count;
                finished[rank] = true;

                const auto& status = ctx->status;
                if (status.error_code() == grpc::StatusCode::DEADLINE_EXCEEDED) {
                    RTP_LLM_FAIL("broadcast rpc timeout, timeout_ms=%d rank=%d err=%d(%s) addr=%s",
                                 ctx->timeout_ms,
                                 rank,
                                 status.error_code(),
                                 status.error_message().c_str(),
                                 ctx->server_addr.c_str());
                }
                if (!status.ok()) {
                    RTP_LLM_LOG_WARNING("broadcast rpc failed, rank=%d err=%d(%s) addr=%s",
                                        rank,
                                        status.error_code(),
                                        status.error_message().c_str(),
                                        ctx->server_addr.c_str());
                    all_request_success = false;
                }
            }
        }

        all_request_success_.store(all_request_success);
        already_done_.store(true);
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
    std::vector<std::shared_ptr<WorkerRpcContext>> worker_contexts_;
    std::atomic<bool>                              already_done_{false};
    std::atomic<bool>                              all_request_success_{false};
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
            ctx->client_context = std::make_shared<grpc::ClientContext>();
            ctx->client_context->set_deadline(deadline);
        }

        for (int rank = 0; rank < worker_size; ++rank) {
            auto& ctx    = contexts.at(rank);
            auto  reader = rpc_call(ctx->stub, ctx->client_context, ctx->request, &ctx->completion_queue);
            reader->Finish(&ctx->response, &ctx->status, reinterpret_cast<void*>(static_cast<intptr_t>(rank)));
        }

        return std::make_shared<BroadcastResult<RequestPB, ResponsePB>>(std::move(contexts));
    }

    size_t workerNum() const {
        return worker_addrs_.size();
    }

private:
    std::vector<std::string> worker_addrs_;
    std::shared_ptr<RPCPool> rpc_pool_;
};

}  // namespace rtp_llm
