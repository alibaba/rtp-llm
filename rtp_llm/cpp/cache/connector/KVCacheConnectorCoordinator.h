#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "autil/LoopThread.h"
#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/connector/AsyncContext.h"
#include "rtp_llm/cpp/cache/connector/KVCacheConnector.h"
#include "rtp_llm/cpp/cache/KVCacheAllocator.h"
#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnector.h"
#include "rtp_llm/cpp/cache/connector/IKVCacheConnectorCoordinator.h"
#include <functional>

namespace rtp_llm {

class DeviceBase;
class KVCacheAllocator;
class KVCacheMemoryConnector;
class KVCacheConnectorReadWriteContext;
class P2PConnector;

class KVCacheConnectorCoordinator: public IKVCacheConnectorCoordinator {
public:
    KVCacheConnectorCoordinator(const CacheConfig&                       cache_config,
                                const KVCacheConfig&                     kv_cache_config,
                                const RuntimeConfig&                     runtime_config,
                                const CacheStoreConfig&                  cache_store_config,
                                const ParallelismConfig&                 parallelism_config,
                                const PDSepConfig&                       pd_sep_config,
                                const ModelConfig&                       model_config,
                                const std::shared_ptr<KVCacheAllocator>& allocator,
                                rtp_llm::DeviceBase*                     device,
                                const kmonitor::MetricsReporterPtr&      metrics_reporter = nullptr);
    virtual ~KVCacheConnectorCoordinator();

public:
    bool init();

    // virtual for test
    using Meta = KVCacheConnector::Meta;
    virtual std::shared_ptr<AsyncContext>
    asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
              const std::shared_ptr<KVCacheConnector::Meta>&           meta) override;
    virtual std::shared_ptr<AsyncContext>
    asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
               const std::shared_ptr<KVCacheConnector::Meta>&           meta) override;
    virtual std::shared_ptr<AsyncContext>
    asyncWriteByLayer(int                                                      layer_id,
                      const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
                      const std::shared_ptr<KVCacheConnector::Meta>&           meta) override;

    virtual bool executeFunction(const FunctionRequestPB& request, FunctionResponsePB& response);
    bool         handleRead(const P2PConnectorStartLoadRequestPB& request,
                            P2PConnectorStartLoadResponsePB&      response,
                            std::function<bool()>                 is_cancelled = nullptr);
    uint32_t     convertToGlobalLayerId(size_t model_id, int local_layer_id) const override;

private:
    bool initMemoryConnector();
    bool initP2PConnector();
    bool initUpdateThread();
    void updateOnce();
    void processReadContexts();
    void processWriteContexts();
    void asyncReadAfterMatch(std::shared_ptr<FusedAsyncReadContext>  fused_read_context,
                             std::shared_ptr<KVCacheConnector::Meta> meta);

private:
    const CacheConfig                 cache_config_;
    const KVCacheConfig               kv_cache_config_;
    const RuntimeConfig               runtime_config_;
    const CacheStoreConfig            cache_store_config_;
    const ParallelismConfig           parallelism_config_;
    const PDSepConfig                 pd_sep_config_;
    const ModelConfig                 model_config_;
    std::shared_ptr<KVCacheAllocator> allocator_;
    rtp_llm::DeviceBase*              device_{nullptr};
    kmonitor::MetricsReporterPtr      metrics_reporter_;

    std::shared_ptr<KVCacheMemoryConnector>                                      memory_connector_;
    std::shared_ptr<P2PConnector>                                                p2p_connector_;
    std::map<KVCacheConnector::ConnectorType, std::shared_ptr<KVCacheConnector>> connectors_;

    mutable std::mutex update_mutex_;
    std::list<std::pair<std::shared_ptr<FusedAsyncReadContext>, std::shared_ptr<KVCacheConnector::Meta>>>
                                                  fused_async_read_context_list_;
    std::list<std::shared_ptr<FusedAsyncContext>> fused_async_write_context_list_;
    autil::LoopThreadPtr                          update_thread_;
    const int                                     update_interval_ms_{1};
    std::atomic<bool>                             stop_{false};
};

}  // namespace rtp_llm
