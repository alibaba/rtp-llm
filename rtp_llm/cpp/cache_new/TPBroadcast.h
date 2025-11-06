#pragma once

#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"

namespace rtp_llm {

struct TPBroadcastWorkerRpcContext {
    std::shared_ptr<RpcService::Stub>    stub;
    std::shared_ptr<grpc::ClientContext> client_context;
    std::string                          server_addr;
    BroadcastAllTpRequestPB              request_pb;
    BroadcastAllTpResponsePB             response_pb;
    grpc::CompletionQueue                completion_queue;
    grpc::Status                         status;
};

class TPBroadcastResult {
public:
    TPBroadcastResult(uint32_t                                                         once_timeout_ms,
                      const std::vector<std::shared_ptr<TPBroadcastWorkerRpcContext>>& worker_rpc_contexts);
    ~TPBroadcastResult() = default;

public:
    bool               success() const;
    const std::string& error_info() const;
    void               waitDone();

private:
    uint32_t                                                  once_timeout_ms_;
    std::vector<std::shared_ptr<TPBroadcastWorkerRpcContext>> worker_rpc_contexts_;

    int         finished_count_      = 0;
    bool        all_request_success_ = true;
    std::string error_info_;
};

class TPBroadcast {
public:
    TPBroadcast(const std::vector<std::string>& grpc_workers,
                const std::shared_ptr<RPCPool>& rpc_pool,
                uint32_t                        once_timeout_ms = 1);
    ~TPBroadcast() = default;

public:
    template<typename BroadcastInfoPB>
    std::shared_ptr<TPBroadcastResult> broadcast(const BroadcastInfoPB& broadcast_info_pb, int timeout_ms = 10000);

private:
    std::vector<std::string> grpc_workers_;
    std::shared_ptr<RPCPool> rpc_pool_;
    uint32_t                 once_timeout_ms_ = 1;
};

}  // namespace rtp_llm