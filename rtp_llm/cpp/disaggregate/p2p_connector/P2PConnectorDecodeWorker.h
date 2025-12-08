#pragma once

#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/disaggregate/transfer/TransferServer.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/cache_new/TpBroadcastManager.h"
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnectorDecodeWorker {
public:
    P2PConnectorDecodeWorker(const GptInitParameter&                  gpt_init_parameter,
                             DeviceBase*                              device_base,
                             const std::shared_ptr<KVCacheAllocator>& kv_cache_allocator);
    ~P2PConnectorDecodeWorker();

public:
    bool init();

public:
    bool read(int64_t                                               request_id,
              const std::string&                                    unique_key,
              int64_t                                               deadline_ms,
              const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers);
    void cancelRead(int64_t request_id, const std::string& unique_key);

    std::shared_ptr<TPBroadcastService::Callback> makeCallback();

private:
    void setLayerCacheBufferTaskStore(const std::shared_ptr<LayerCacheBufferTaskStore>& layer_cache_buffer_task_store);

private:
    const GptInitParameter&                    gpt_init_parameter_;
    DeviceBase*                                device_base_;
    std::shared_ptr<KVCacheAllocator>          kv_cache_allocator_;
    std::shared_ptr<TransferServer>            transfer_server_;
    std::shared_ptr<LayerCacheBufferTaskStore> layer_cache_buffer_task_store_;
};

class P2PConnectorDecodeWorkerTPCallback: public TPBroadcastService::Callback {
public:
    P2PConnectorDecodeWorkerTPCallback(const std::shared_ptr<P2PConnectorDecodeWorker>& p2p_connector_decode_worker);
    ~P2PConnectorDecodeWorkerTPCallback() = default;

public:
    bool         shouldProcess(const BroadcastTpRequestPB& request) override;
    grpc::Status onBroadcastTp(const BroadcastTpRequestPB& request, BroadcastTpResponsePB& response) override;

private:
    std::shared_ptr<P2PConnectorDecodeWorker> p2p_connector_decode_worker_;
};

}  // namespace rtp_llm
