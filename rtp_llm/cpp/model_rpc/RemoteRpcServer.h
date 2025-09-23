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
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      py::object                                             mm_process_engine,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params);

    auto& resource() {
        return resource_;
    }

private:
    void initLocalHostInfo();
    void initLocalPeerInfo();
    void initCacheStore(const rtp_llm::GptInitParameter& params, rtp_llm::ProposeModelEngineInitParams* propose_params);

protected:
    std::string                 process_id_;
    RemoteServerResource        resource_;
    std::atomic<size_t>         loading_cache_requests_{0};
    std::shared_ptr<CacheStore> cache_store_;
};

}  // namespace rtp_llm