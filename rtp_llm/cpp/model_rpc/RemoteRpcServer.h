#pragma once
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RemoteServerResource.h"

namespace rtp_llm {

class RemoteRpcServer: public LocalRpcServer {
public:
    RemoteRpcServer() {}
    virtual ~RemoteRpcServer() {}
    void startServices() override;

    auto& resource() {
        return resource_;
    }

private:
    void initLocalHostInfo();
    void initLocalPeerInfo();
    void initCacheStore(const EngineInitParams& params);

protected:
    std::string                 process_id_;
    RemoteServerResource        resource_;
    std::atomic<size_t>         loading_cache_requests_{0};
};

}  // namespace rtp_llm
