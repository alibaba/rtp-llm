#pragma once
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "rtp_llm/cpp/model_rpc/LocalRpcServer.h"
#include "rtp_llm/cpp/model_rpc/RemoteServerResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"

namespace rtp_llm {

class RemoteRpcServer: public LocalRpcServer {
public:
    RemoteRpcServer() {}
    virtual ~RemoteRpcServer() {}
    grpc::Status init(const EngineInitParams&                                maga_init_params,
                      std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params,
                      py::object                                             mm_process_engine);

    auto& resource() {
        return resource_;
    }

private:
    void initLocalHostInfo();
    void initLocalPeerInfo();
    void initCacheStore(const EngineInitParams& params, rtp_llm::ProposeModelEngineInitParams* propose_params);

protected:
    virtual grpc::Status validateBeforeCacheStoreInit() const;
    static grpc::Status  validateNormalCacheStoreWireSpan(const CacheConfig& config,
                                                          size_t             expected_span,
                                                          size_t             cp_size,
                                                          const std::string& topology_name);
    static grpc::Status  validateNormalCacheStoreTopologies(const CacheConfig& config, size_t cp_size = 1);

    std::string                 process_id_;
    RemoteServerResource        resource_;
    std::atomic<size_t>         loading_cache_requests_{0};
    std::shared_ptr<CacheStore> cache_store_;
};

}  // namespace rtp_llm
