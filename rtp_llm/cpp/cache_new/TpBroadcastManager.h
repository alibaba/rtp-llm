#pragma once

#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"

namespace rtp_llm {

class TPBroadcastResult {
public:
    struct WorkerRpcContext {
        std::shared_ptr<RpcService::Stub>      stub;
        std::shared_ptr<grpc::ClientContext>   client_context;
        BroadcastTpRequestPB                   request;
        std::shared_ptr<BroadcastTpResponsePB> response;
        grpc::CompletionQueue                  completion_queue;
        grpc::Status                           status;
        std::string                            server_addr;
        int                                    timeout_ms;
    };

public:
    TPBroadcastResult(const std::vector<std::shared_ptr<WorkerRpcContext>>& worker_rpc_contexts):
        worker_rpc_contexts_(worker_rpc_contexts) {}
    ~TPBroadcastResult() = default;

public:
    virtual void                                        waitDone();
    virtual bool                                        success() const;
    std::vector<std::shared_ptr<BroadcastTpResponsePB>> responses() const;

private:
    void cancelAllRequests() const;

private:
    std::vector<std::shared_ptr<WorkerRpcContext>> worker_rpc_contexts_;
    bool                                           all_request_success_{false};
    int                                            finished_count_{0};
};

class TpBroadcastManager {
public:
    explicit TpBroadcastManager(std::shared_ptr<RPCPool> rpc_pool, const std::vector<std::string>& worker_addrs):
        rpc_pool_(rpc_pool), worker_addrs_(worker_addrs) {}
    ~TpBroadcastManager() {
        rpc_pool_.reset();
    }

public:
    bool                                       init();
    virtual std::shared_ptr<TPBroadcastResult> broadcast(const std::vector<BroadcastTpRequestPB>& requests,
                                                         int                                      timeout_ms) const;
    size_t                                     workerNum() const {
        return worker_addrs_.size();
    }

private:
    std::vector<std::string> worker_addrs_;
    std::shared_ptr<RPCPool> rpc_pool_;
};

class TPBroadcastService {
public:
    class Callback {
    public:
        virtual bool         shouldProcess(const BroadcastTpRequestPB& request)                                  = 0;
        virtual grpc::Status onBroadcastTp(const BroadcastTpRequestPB& request, BroadcastTpResponsePB& response) = 0;
    };
    void registerCallback(std::shared_ptr<Callback> callback);

    grpc::Status
    broadcast(::grpc::ServerContext* context, const ::BroadcastTpRequestPB* request, ::BroadcastTpResponsePB* response);

private:
    std::vector<std::shared_ptr<Callback>> callbacks_;
};

}  // namespace rtp_llm
