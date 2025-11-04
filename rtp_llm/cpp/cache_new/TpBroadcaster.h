#pragma once

#include "rtp_llm/cpp/model_rpc/RPCPool.h"

namespace rtp_llm {

class TpBroadcastHandler {
public:
    TpBroadcastHandler()          = default;
    virtual ~TpBroadcastHandler() = default;

public:
    virtual bool run(const BroadcastAllTpRequestPB& request, BroadcastAllTpResponse& response) = 0;
};
using TpBroadcastHandlerPtr = std::shared_ptr<TpBroadcastHandler>;

class TpBroadcaster {
public:
    TpBroadcaster(const std::vector<std::string>& peers): peers_(peers) {}

public:
    void registerHandler(const std::string& handler_name, TpBroadcastHandlerPtr handler);
    bool executeHandler(const std::string&           handler_name,
                        const BroadcastAllTpRequestPB& request,
                        BroadcastAllTpResponse&      response);
    bool broadcast(const std::vector<BroadcastAllTpRequestPB>& requests,
                   std::vector<BroadcastAllTpResponse>&      responses,
                   const std::string&                        action,
                   int                                       timeout_ms);

private:
    std::vector<std::string>                   peers_;
    std::shared_ptr<RPCPool>                   rpc_pool_;
    std::unordered_map<std::string, TpBroadcastHandlerPtr> handlers_;
};

}  // namespace rtp_llm
