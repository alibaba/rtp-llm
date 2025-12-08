#pragma once

#include "rtp_llm/cpp/cache_new/KVCacheConnector.h"
#include "rtp_llm/cpp/config/GptInitParameter.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/TPBroadcastClient.h"
#include "rtp_llm/cpp/disaggregate/p2p_connector/PrefillLoadClient.h"
#include <memory>
#include <string>

namespace rtp_llm {

class P2PConnectorDecodeAsyncContext: public KVCacheConnector::AsyncContext {
public:
    P2PConnectorDecodeAsyncContext(const std::shared_ptr<KVCacheResourceV1>&         resource,
                                   const std::shared_ptr<TPBroadcastClient::Result>& tp_sync_result,
                                   const std::shared_ptr<PrefillLoadClient::Result>& prefill_load_result,
                                   const std::shared_ptr<TPBroadcastClient>&         tp_broadcast_client);
    virtual ~P2PConnectorDecodeAsyncContext();

public:
    bool success() const override;
    void cancel() override;
    void waitDone() override;

private:
    std::shared_ptr<KVCacheResourceV1>         resource_;
    std::shared_ptr<TPBroadcastClient>         tp_broadcast_client_;
    std::shared_ptr<TPBroadcastClient::Result> tp_sync_result_;
    std::shared_ptr<PrefillLoadClient::Result> prefill_load_result_;
};

class P2PConnectorDecodeScheduler {
public:
    P2PConnectorDecodeScheduler(const GptInitParameter& gpt_init_parameter);
    ~P2PConnectorDecodeScheduler();

public:
    bool init();

    std::shared_ptr<P2PConnectorDecodeAsyncContext> asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                                              int64_t                                   request_id,
                                                              const std::string&                        unique_key,
                                                              const std::string&                        prefill_ip,
                                                              uint32_t                                  prefill_port,
                                                              int64_t                                   deadline_ms);

private:
    const GptInitParameter& gpt_init_parameter_;

    std::shared_ptr<TPBroadcastClient> tp_broadcast_client_;
    std::shared_ptr<PrefillLoadClient> prefill_load_client_;
};

}  // namespace rtp_llm
