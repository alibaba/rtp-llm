#pragma once

#include "rtp_llm/cpp/cache_new/p2p_connector/LayerCacheBuffer.h"
#include "autil/ThreadPoolBase.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpClient.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/TcpServer.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/LayerCacheBufferStore.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreServerService.h"
#include "rtp_llm/cpp/cache_new/p2p_connector/DeviceEvent.h"

namespace rtp_llm {

class CacheStoreServer {
public:
    CacheStoreServer(const std::shared_ptr<TcpClient>&          tcp_client,
                     const std::shared_ptr<TcpServer>&          tcp_server,
                     int                                        layer_num,
                     const std::vector<CacheStoreServerWorker>& worker_addrs);
    ~CacheStoreServer();

public:
    bool init();
    void asyncStore(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                    DeviceEventPtr                           device_event,
                    int64_t                                  timeout_ms);

private:
    void storeWaitThread();

private:
    std::shared_ptr<TcpClient>          tcp_client_;
    std::shared_ptr<TcpServer>          tcp_server_;
    int                                 layer_num_;
    std::vector<CacheStoreServerWorker> worker_addrs_;

    autil::ThreadPoolBasePtr               thread_pool_;  // task executor
    std::shared_ptr<LayerCacheBufferStore> layer_cache_buffer_store_;

    std::unique_ptr<CacheStoreServerService> cache_store_server_service_;

    // store wait
    std::atomic<bool>                                                                   store_wait_thread_stop_{false};
    std::mutex                                                                          store_wait_mutex_;
    std::vector<std::tuple<DeviceEventPtr, std::shared_ptr<LayerCacheBuffer>, int64_t>> store_wait_device_events_;
    std::thread                                                                         store_wait_thread_;
};

}  // namespace rtp_llm