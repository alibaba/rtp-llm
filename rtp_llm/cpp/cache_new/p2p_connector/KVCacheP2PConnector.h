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
#include "rtp_llm/cpp/cache_new/TPBroadcast.h"
#include "kmonitor/client/MetricsReporter.h"

namespace rtp_llm {

class KVCacheP2PLoadAsyncContext: public KVCacheConnector::AsyncContext {
public:
    KVCacheP2PLoadAsyncContext();
    ~KVCacheP2PLoadAsyncContext();

public:
    void addTPBroadcastResult(const std::shared_ptr<TPBroadcastResult>& tp_broadcast_result);
    bool success() const override;
    void cancel() override;
    void waitDone() override;

    void setAllSuccess(bool all_success);

private:
    bool                                            all_success_ = true;
    std::vector<std::shared_ptr<TPBroadcastResult>> tp_broadcast_results_;
};

class KVCacheP2PStoreAsyncContext: public KVCacheConnector::AsyncContext {
public:
    KVCacheP2PStoreAsyncContext();
    ~KVCacheP2PStoreAsyncContext();

public:
    bool success() const override {
        return true;
    }
    void cancel() override {}
    void waitDone() override {}
};

class KVCacheP2PConnector: public KVCacheConnector {
public:
    KVCacheP2PConnector(const GptInitParameter&             gpt_init_params,
                        CacheStoreConfig&                   cache_store_config,
                        DeviceBase*                         device,
                        const kmonitor::MetricsReporterPtr& metrics_reporter,
                        KVCacheAllocatorPtr                 kv_cache_allocator,
                        const std::shared_ptr<TPBroadcast>& tp_broadcast);
    ~KVCacheP2PConnector();

public:
    bool init() override;
    std::shared_ptr<KVCacheConnector::AsyncContext>
    asyncRead(const BatchKVCacheResourcePtr& resource, const std::string& ip, uint32_t port) override;

    std::shared_ptr<KVCacheConnector::AsyncContext> asyncWrite(const BatchKVCacheResourcePtr& resource,
                                                               DeviceEventPtr                 event) override;

    std::shared_ptr<KVCacheConnector::AsyncContext>
    asyncWriteByLayer(const BatchKVCacheResourcePtr& resource, int layer_id, DeviceEventPtr event) override;

    void regUserMr();
    void deregUserMr();

private:
    std::shared_ptr<TPBroadcastResult> asyncReadOneBatch(const std::vector<int64_t>&                   cache_keys,
                                                         const std::vector<std::shared_ptr<BlockIds>>& layer_block_ids,
                                                         const std::vector<CacheStoreServerWorker>& peer_worker_infos);
    bool                               read(const std::vector<int64_t>&                cache_keys,
                                            const std::map<int, std::vector<int>>&     layer_block_ids,
                                            const std::vector<CacheStoreServerWorker>& peer_worker_infos);
    std::shared_ptr<CacheStoreClientLoadContext>
    loadFromPeerWorker(const std::vector<int64_t>&            cache_keys,
                       const std::map<int, std::vector<int>>& layer_block_ids,
                       const CacheStoreServerWorker&          peer_worker_addr,
                       int                                    local_partition_count,
                       int                                    local_partition_id,
                       int                                    peer_partition_count,
                       int                                    peer_partition_id,
                       std::vector<std::set<void*>>&          layer_loading_buffer_set);

private:
    const GptInitParameter&             gpt_init_params_;
    const CacheStoreConfig&             cache_store_config_;
    DeviceBase*                         device_;
    kmonitor::MetricsReporterPtr        metrics_reporter_;
    KVCacheAllocatorPtr                 kv_cache_allocator_;
    std::vector<CacheStoreServerWorker> local_worker_infos_;
    int                                 tp_rank_;

    std::shared_ptr<MemoryUtil>             memory_util_;
    std::shared_ptr<cache_store::TcpClient> tcp_client_;
    std::shared_ptr<cache_store::TcpServer> tcp_server_;
    autil::ThreadPoolBasePtr                thread_pool_;  // task executor
    std::shared_ptr<TPBroadcast>            tp_broadcast_;

    std::shared_ptr<cache_store::CacheStoreClient> cache_store_client_;
    std::shared_ptr<cache_store::CacheStoreServer> cache_store_server_;
    bool                                           kvcache_reg_mr_ = false;
};

}  // namespace rtp_llm