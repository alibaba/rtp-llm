#pragma once
#include "maga_transformer/cpp/model_rpc/RPCPool.h"
#include "maga_transformer/cpp/utils/RpcErrorCode.h"
#include "maga_transformer/cpp/model_rpc/LocalRpcServer.h"
#include "maga_transformer/cpp/model_rpc/RemoteServerResource.h"

namespace rtp_llm {

class RemoteRpcServer : public LocalRpcServer {
public:
    RemoteRpcServer() {}
    virtual ~RemoteRpcServer() {}
    grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);

    auto& resource() {
        return resource_;
    }

private:
    void initLocalHostInfo();
    void initLocalPeerInfo();
    void initCacheStore();

protected:
    std::string process_id_;
    RemoteServerResource resource_;
    std::atomic<size_t> loading_cache_requests_{0};
};

}