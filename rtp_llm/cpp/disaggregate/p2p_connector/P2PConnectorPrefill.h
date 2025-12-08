#pragma once

#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/core/Event.h"
#include "rtp_llm/cpp/engine_base/stream/GenerateStream.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorStreamStore.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorPrefillScheduler.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorPrefillWorker.h"
#include "rtp_llm/cpp/cache_new/TpBroadcastManager.h"
#include <grpc++/grpc++.h>
#include <memory>
#include <string>
#include <vector>

namespace rtp_llm {

class P2PConnectorPrefill: public KVCacheConnector {
public:
    P2PConnectorPrefill(const GptInitParameter&                  gpt_init_parameter,
                        DeviceBase*                              device_base,
                        const std::shared_ptr<KVCacheAllocator>& kv_cache_allocator);
    ~P2PConnectorPrefill();

public:
    bool init();

public:
    std::shared_ptr<AsyncContext> asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                            const std::shared_ptr<Meta>&              meta);
    std::shared_ptr<AsyncContext> asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource,
                                             const std::shared_ptr<Meta>&              meta);
    std::shared_ptr<AsyncContext> asyncWriteByLayer(int                                       layer_id,
                                                    const std::shared_ptr<KVCacheResourceV1>& resource,
                                                    const std::shared_ptr<Meta>&              meta);

    grpc::Status handleWrite(const std::string&                                   unique_key,
                             const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                             int64_t                                              deadline_ms);
    void         addStream(const std::string& unique_key, GenerateStreamPtr stream);

    std::shared_ptr<TPBroadcastService::Callback> makeCallback();

private:
    const GptInitParameter&                       gpt_init_parameter_;
    DeviceBase*                                   device_base_;
    std::shared_ptr<KVCacheAllocator>             kv_cache_allocator_;
    std::shared_ptr<P2PConnectorPrefillScheduler> scheduler_;
    std::shared_ptr<P2PConnectorPrefillWorker>    worker_;
    std::shared_ptr<PrefillConnectorStreamStore>  stream_store_;
};

}  // namespace rtp_llm
