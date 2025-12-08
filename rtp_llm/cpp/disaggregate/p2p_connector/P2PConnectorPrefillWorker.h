#pragma once

#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/cache_new/BatchKVCacheResource.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/TransferClient.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/AsymmetricTpUtil.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/PrefillWorkerLoadContext.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/cache_new/TpBroadcastManager.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/core/Event.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <atomic>

namespace rtp_llm {

class P2PConnectorPrefillWorker {
public:
    P2PConnectorPrefillWorker(const GptInitParameter&                  gpt_init_parameter,
                              DeviceBase*                              device_base,
                              const std::shared_ptr<KVCacheAllocator>& kv_cache_allocator);
    ~P2PConnectorPrefillWorker();

public:
    bool init();

public:
    bool writeByLayer(int                                       layer_id,
                      const std::shared_ptr<KVCacheResourceV1>& resource,
                      int64_t                                   request_id,
                      DeviceEventPtr                            event);
    bool write(int64_t                                              request_id,
               const std::string&                                   unique_key,
               int64_t                                              deadline_ms,
               const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers);
    void cancelWrite(int64_t request_id, const std::string& unique_key);

private:
    std::shared_ptr<ComputedLayerCacheBufferStore> getComputedBuffersStore() const {
        return computed_buffers_;
    }

    std::shared_ptr<PrefillWorkerLoadContextStore> getLoadContexts() const {
        return load_contexts_;
    }

    void setStoreWaitTimeoutMs(int64_t store_wait_timeout_ms) {
        store_wait_timeout_ms_ = store_wait_timeout_ms;
    }

    void setTransferClient(const std::shared_ptr<TransferClient>& transfer_client) {
        transfer_client_ = transfer_client;
    }

private:
    void storeWaitThread();
    void storeWaitThreadProcess();

private:
    const GptInitParameter&           gpt_init_parameter_;
    DeviceBase*                       device_base_;
    std::shared_ptr<KVCacheAllocator> kv_cache_allocator_;
    int64_t                           store_wait_timeout_ms_ = 10 * 1000;

    std::shared_ptr<TransferClient>   transfer_client_;
    std::shared_ptr<AsymmetricTpUtil> asymmetric_tp_util_;

    std::shared_ptr<ComputedLayerCacheBufferStore> computed_buffers_;
    std::shared_ptr<PrefillWorkerLoadContextStore> load_contexts_;

    std::atomic<bool>  store_wait_thread_stop_{false};
    mutable std::mutex store_wait_mutex_;
    std::vector<std::tuple<int64_t, DeviceEventPtr, std::shared_ptr<LayerCacheBuffer>, int64_t>>
                store_wait_contexts_;  // [request_id, event, layer_cache_buffer, deadline_ms]
    std::thread store_wait_thread_;
};

class P2PConnectorPrefillWorkerTPCallback: public TPBroadcastService::Callback {
public:
    P2PConnectorPrefillWorkerTPCallback(const std::shared_ptr<P2PConnectorPrefillWorker>& p2p_connector_prefill_worker);
    ~P2PConnectorPrefillWorkerTPCallback() = default;

public:
    bool         shouldProcess(const BroadcastTpRequestPB& request) override;
    grpc::Status onBroadcastTp(const BroadcastTpRequestPB& request, BroadcastTpResponsePB& response) override;

private:
    std::shared_ptr<P2PConnectorPrefillWorker> p2p_connector_prefill_worker_;
};

}  // namespace rtp_llm
