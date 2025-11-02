#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "grpc++/grpc++.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {

// TP broadcaster: provides a unified API to register named handlers and to broadcast
// requests that indicate which handler to execute on peers.
class TpBroadcaster {
public:
    TpBroadcaster(std::shared_ptr<RPCPool> rpc_pool, const std::vector<std::string>& peers):
        rpc_pool_(std::move(rpc_pool)), peers_(peers) {}

public:
    // Handler signature: execute action on this rank and fill response
    using HandlerFn = std::function<bool(const BroadcastAllTpRequest&, BroadcastAllTpResponse&)>;
    void registerHandler(const std::string& handler_name, HandlerFn handler) {
        handlers_[handler_name] = std::move(handler);
    }

    // Execute a registered handler on this rank.
    bool executeHandler(const std::string&           handler_name,
                        const BroadcastAllTpRequest& request,
                        BroadcastAllTpResponse&      response);

    // Broadcast (calls RpcService::AsyncBroadcastAllTp)
    // - handler_name: indicates which handler to execute on peers (stored in request.method)
    // - requests: size==1 → same request to all peers; size==peers.size() → per-rank request
    // - responses: ordered by peers
    bool broadcast(const std::vector<BroadcastAllTpRequest>& requests,
                   std::vector<BroadcastAllTpResponse>&      responses,
                   const std::string&                        action,
                   int                                       timeout_ms);

private:
    std::shared_ptr<RPCPool>                   rpc_pool_;
    std::vector<std::string>                   peers_;
    std::unordered_map<std::string, HandlerFn> handlers_;
};

}  // namespace rtp_llm
