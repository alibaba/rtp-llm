#pragma once

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpClient.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpServer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreClient.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreServer.h"
#include "autil/ThreadPoolBase.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class KVCacheP2PConnector: public KVCacheConnector {
public:
    KVCacheP2PConnector(const GptInitParameter&             gpt_init_params,
                        CacheStoreConfig&                   cache_store_config,
                        DeviceBase*                         device,
                        const kmonitor::MetricsReporterPtr& metrics_reporter,
                        KVCacheAllocatorPtr                 kv_cache_allocator);
    ~KVCacheP2PConnector();

public:
    bool init() override;
    std::shared_ptr<AsyncContext>
    asyncRead(const BatchKVCacheResourcePtr& resource, const std::string& ip, uint32_t port) override;
    std::shared_ptr<AsyncContext> asyncWrite(const BatchKVCacheResourcePtr& resource, DeviceEventPtr event) override;
    std::shared_ptr<AsyncContext>
    asyncWriteByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, DeviceEventPtr event) override;

    void regUserMr(size_t model_id);
    void deregUserMr();

private:
    const GptInitParameter&      gpt_init_params_;
    DeviceBase*                  device_;
    kmonitor::MetricsReporterPtr metrics_reporter_;
    KVCacheAllocatorPtr          kv_cache_allocator_;

    std::shared_ptr<MemoryUtil> memory_util_;
    std::shared_ptr<TcpClient>  tcp_client_;
    std::shared_ptr<TcpServer>  tcp_server_;
    autil::ThreadPoolBasePtr    thread_pool_;  // task executor

    std::shared_ptr<CacheStoreClient> cache_store_client_;
    std::shared_ptr<CacheStoreServer> cache_store_server_;
    bool                              kvcache_reg_mr_ = false;
};

}  // namespace rtp_llm