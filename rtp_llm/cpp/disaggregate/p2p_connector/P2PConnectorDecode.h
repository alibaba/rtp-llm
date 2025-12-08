#pragma once

#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/devices/DeviceBase.h"
#include "rtp_llm/cpp/cache_new/KVCacheAllocator.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorDecodeScheduler.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorDecodeWorker.h"
#include "rtp_llm/cpp/cache_new/TpBroadcastManager.h"
#include <memory>

namespace rtp_llm {

class P2PConnectorDecodeMeta: public KVCacheConnector::Meta {
public:
    P2PConnectorDecodeMeta(int64_t            request_id,
                           const std::string& unique_key,
                           const std::string& prefill_ip,
                           uint32_t           prefill_port,
                           int64_t            deadline_ms):
        request_id_(request_id),
        unique_key_(unique_key),
        prefill_ip_(prefill_ip),
        prefill_port_(prefill_port),
        deadline_ms_(deadline_ms) {}
    ~P2PConnectorDecodeMeta() override = default;

public:
    int64_t requestId() const {
        return request_id_;
    }
    const std::string& uniqueKey() const {
        return unique_key_;
    }
    const std::string& prefillIp() const {
        return prefill_ip_;
    }
    uint32_t prefillPort() const {
        return prefill_port_;
    }
    int64_t deadlineMs() const {
        return deadline_ms_;
    }

private:
    int64_t     request_id_;
    std::string unique_key_;
    std::string prefill_ip_;
    uint32_t    prefill_port_;
    int64_t     deadline_ms_;
};

class P2PConnectorDecode: public KVCacheConnector {

public:
    P2PConnectorDecode(const GptInitParameter&                  gpt_init_parameter,
                       DeviceBase*                              device_base,
                       const std::shared_ptr<KVCacheAllocator>& kv_cache_allocator);
    virtual ~P2PConnectorDecode();

public:
    bool                                            init() override;
    std::shared_ptr<KVCacheConnector::AsyncContext> asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                              const std::shared_ptr<Meta>&              meta) override;
    std::shared_ptr<KVCacheConnector::AsyncContext> asyncWrite(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                               const std::shared_ptr<Meta>&              meta) override;
    std::shared_ptr<KVCacheConnector::AsyncContext> asyncWriteByLayer(
        int layer_id, const std::shared_ptr<KVCacheResourceV1>& resource, const std::shared_ptr<Meta>& meta) override;

    std::shared_ptr<TPBroadcastService::Callback> makeCallback();

private:
    const GptInitParameter&           gpt_init_parameter_;
    DeviceBase*                       device_base_;
    std::shared_ptr<KVCacheAllocator> kv_cache_allocator_;

    std::shared_ptr<P2PConnectorDecodeScheduler> scheduler_;
    std::shared_ptr<P2PConnectorDecodeWorker>    worker_;
};
}  // namespace rtp_llm
