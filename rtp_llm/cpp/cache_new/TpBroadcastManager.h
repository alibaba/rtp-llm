#pragma once

#include "rtp_llm/cpp/model_rpc/RPCPool.h"

namespace rtp_llm {

class TPBroadcastResult {
public:
    struct WorkerRpcContext {
        std::shared_ptr<RpcService::Stub>    stub;
        std::shared_ptr<grpc::ClientContext> client_context;
        BroadcastTpRequestPB                 request;
        BroadcastTpResponsePB                response;
        grpc::CompletionQueue                completion_queue;
        grpc::Status                         status;
        std::string                          server_addr;
        int                                  timeout_ms;
    };

public:
    TPBroadcastResult(const std::vector<std::shared_ptr<WorkerRpcContext>>& worker_rpc_contexts):
        worker_rpc_contexts_(worker_rpc_contexts) {}
    ~TPBroadcastResult() = default;

public:
    void                               waitDone();
    bool                               success() const;
    std::vector<BroadcastTpResponsePB> responses() const;

private:
    void cancelAllRequests() const;

private:
    std::vector<std::shared_ptr<WorkerRpcContext>> worker_rpc_contexts_;
    bool                                           all_request_success_{false};
    int                                            finished_count_{0};
};

class TpBroadcastManager {
public:
    explicit TpBroadcastManager(const std::vector<std::string>& worker_addrs): worker_addrs_(worker_addrs) {}
    ~TpBroadcastManager() {
        rpc_pool_.reset();
    }

public:
    bool                               init();
    std::shared_ptr<TPBroadcastResult> broadcast(const std::vector<BroadcastTpRequestPB>& requests,
                                                 int                                      timeout_ms) const;
    size_t                             workerNum() const {
        return worker_addrs_.size();
    }

private:
    std::vector<std::string> worker_addrs_;
    std::shared_ptr<RPCPool> rpc_pool_;
};

}  // namespace rtp_llm
