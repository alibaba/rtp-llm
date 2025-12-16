#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "autil/LoopThread.h"
#include "rtp_llm/cpp/cache_new/AsyncContext.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache_new/CacheConfig.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.grpc.pb.h"

namespace rtp_llm {

class DeviceBase;
class KVCacheMemoryConnector;
class KVCacheConnectorReadWriteContext;
class StreamCacheResource;

class FusedAsyncContext: public AsyncContext {
public:
    FusedAsyncContext(const std::vector<std::shared_ptr<AsyncContext>>& contexts);
    ~FusedAsyncContext() override = default;

public:
    bool done() const override;
    bool success() const override;

    const std::vector<std::shared_ptr<AsyncContext>>& contexts() const {
        return contexts_;
    }

private:
    std::vector<std::shared_ptr<AsyncContext>> contexts_;
};

class FusedAsyncReadContext: public AsyncContext {
public:
    FusedAsyncReadContext(const std::shared_ptr<FusedAsyncContext>& fused_match_context,
                          const std::shared_ptr<KVCacheResourceV1>& resource);
    ~FusedAsyncReadContext() override;

public:
    bool done() const override;
    bool success() const override;
    void setFusedReadContext(const std::shared_ptr<FusedAsyncContext>& fused_read_context);
    const std::shared_ptr<FusedAsyncContext>& fusedMatchContext() const {
        return fused_match_context_;
    }
    const std::shared_ptr<FusedAsyncContext>& fusedReadContext() const {
        return fused_read_context_;
    }
    const std::shared_ptr<KVCacheResourceV1>& resource() const {
        return resource_;
    }

private:
    std::shared_ptr<FusedAsyncContext> fused_match_context_;
    std::shared_ptr<FusedAsyncContext> fused_read_context_;
    std::shared_ptr<KVCacheResourceV1> resource_;
};

class KVCacheConnectorCoordinator: public std::enable_shared_from_this<KVCacheConnectorCoordinator> {
public:
    KVCacheConnectorCoordinator(const CacheConfig&                  config,
                                const KVCacheAllocatorPtr&          allocator,
                                rtp_llm::DeviceBase*                device,
                                const GptInitParameter&             params,
                                const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);
    ~KVCacheConnectorCoordinator();

public:
    bool init();

    using Meta = KVCacheConnector::Meta;
    std::shared_ptr<AsyncContext> asyncRead(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
                                            const std::shared_ptr<Meta>&                             meta);
    std::shared_ptr<AsyncContext> asyncWrite(const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
                                             const std::shared_ptr<Meta>&                             meta);
    std::shared_ptr<AsyncContext>
                 asyncWriteByLayer(int                                                      layer_id,
                                   const std::shared_ptr<KVCacheConnectorReadWriteContext>& connector_context,
                                   const std::shared_ptr<Meta>&                             meta);
    virtual bool copyCache(const CopyCacheRequestPB& request, CopyCacheResponsePB& response);
    virtual void clearMemoryCache();

private:
    bool initMemoryConnector();
    bool initUpdateThread();
    void updateOnce();

private:
    CacheConfig                  config_;
    KVCacheAllocatorPtr          allocator_;
    rtp_llm::DeviceBase*         device_{nullptr};
    const GptInitParameter       params_;
    kmonitor::MetricsReporterPtr metrics_reporter_;

    std::shared_ptr<KVCacheMemoryConnector> memory_connector_;
    std::shared_ptr<KVCacheConnector>       remote_connector_;
    std::shared_ptr<KVCacheConnector>       p2p_connector_;

    std::map<KVCacheConnector::ConnectorType, std::shared_ptr<KVCacheConnector>> connectors_;

    mutable std::mutex                                update_mutex_;
    std::list<std::shared_ptr<FusedAsyncReadContext>> fused_async_read_context_list_;
    std::list<std::shared_ptr<FusedAsyncContext>>     fused_async_write_context_list_;
    autil::LoopThreadPtr                              update_thread_;
    const int                                         update_interval_ms_{1};
    std::atomic<bool>                                 stop_{false};
};

}  // namespace rtp_llm
