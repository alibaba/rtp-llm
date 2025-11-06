#include "rtp_llm/cpp/cache_new/p2p_connector/CacheStoreServer.h"

namespace rtp_llm {

CacheStoreServer::CacheStoreServer(const std::shared_ptr<TcpClient>&          tcp_client,
                                   const std::shared_ptr<TcpServer>&          tcp_server,
                                   int                                        layer_num,
                                   const std::vector<CacheStoreServerWorker>& worker_addrs):
    tcp_client_(tcp_client), tcp_server_(tcp_server), layer_num_(layer_num), worker_addrs_(worker_addrs) {}

CacheStoreServer::~CacheStoreServer() {
    store_wait_thread_stop_ = true;
    store_wait_thread_.join();
}

bool CacheStoreServer::init() {
    layer_cache_buffer_store_   = std::make_shared<LayerCacheBufferStore>(layer_num_);
    cache_store_server_service_ = std::make_unique<CacheStoreServerService>(tcp_client_, layer_cache_buffer_store_);

    if (!tcp_server_->registerService(cache_store_server_service_.get())) {
        RTP_LLM_LOG_ERROR("cache store server init failed : register service failed");
        return false;
    }

    store_wait_thread_ = std::thread([this]() { storeWaitThread(); });
    return true;
}

void CacheStoreServer::asyncStore(const std::shared_ptr<LayerCacheBuffer>& layer_cache_buffer,
                                  DeviceEventPtr                           device_event,
                                  int64_t                                  timeout_ms) {
    layer_cache_buffer_store_->store(layer_cache_buffer, device_event, timeout_ms);
}

void CacheStoreServer::storeWaitThread() {
    while (!store_wait_thread_stop_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        auto iter = store_wait_device_events_.begin();
        while (iter != store_wait_device_events_.end()) {
            auto& [device_event, layer_cache_buffer, timeout_ms] = *iter;
            if (device_event == nullptr || device_event->checkReadiness()) {
                layer_cache_buffer_store_->store(layer_cache_buffer, device_event, timeout_ms);
                iter = store_wait_device_events_.erase(iter);
            } else {
                ++iter;
            }
        }
    }
}

}  // namespace rtp_llm