#pragma once
#include "maga_transformer/cpp/model_rpc/RPCPool.h"
#include "maga_transformer/cpp/model_rpc/LocalRpcServer.h"

namespace rtp_llm {

class RemoteRpcServer : public LocalRpcServer {
public:
    RemoteRpcServer() {}
    virtual ~RemoteRpcServer() {}
    grpc::Status init(const EngineInitParams& maga_init_params, py::object mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);

    const auto& workers() const {
        return workers_;
    }
    auto& cacheStore() {
        return cache_store_;
    }
    auto& rpcPool() {
        return rpc_pool_;
    }

private:
    void initLocalHostInfo();
    void initLocalPeerInfo();
    void initCacheStore();

protected:
    std::string process_id_;
    std::vector<std::string> workers_;
    std::shared_ptr<NormalCacheStore> cache_store_;
    RPCPool rpc_pool_;
};

}