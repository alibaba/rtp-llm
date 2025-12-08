#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorPrefillScheduler.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache_new/TpBroadcastManager.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBufferUtil.h"
#include <grpc++/grpc++.h>

namespace rtp_llm {

P2PConnectorPrefillScheduler::P2PConnectorPrefillScheduler(const GptInitParameter& gpt_init_parameter):
    gpt_init_parameter_(gpt_init_parameter) {}

P2PConnectorPrefillScheduler::~P2PConnectorPrefillScheduler() {}

bool P2PConnectorPrefillScheduler::init() {
    tp_broadcast_client_ = std::make_shared<TPBroadcastClient>(gpt_init_parameter_);
    if (!tp_broadcast_client_->init()) {
        RTP_LLM_LOG_ERROR("P2PConnectorPrefillScheduler init failed: tp_broadcast_client is null");
        return false;
    }
    RTP_LLM_LOG_INFO("P2PConnectorPrefillScheduler init success");
    return true;
}

grpc::Status
P2PConnectorPrefillScheduler::write(const std::shared_ptr<KVCacheResourceV1>&            resource,
                                    const std::string&                                   unique_key,
                                    int64_t                                              request_id,
                                    const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                                    int64_t                                              deadline_ms) {
    // 转换为 layer_cache_buffers
    auto layer_cache_buffers = LayerCacheBufferUtil::convert(*resource, 0);
    if (layer_cache_buffers.empty()) {
        RTP_LLM_LOG_WARNING("P2PConnectorPrefillScheduler write: layer_cache_buffers is empty");
        return grpc::Status(grpc::StatusCode::INTERNAL, "layer_cache_buffers is empty");
    }

    // 调用 broadcast
    auto result = tp_broadcast_client_->broadcast(
        request_id, layer_cache_buffers, decode_transfer_servers, unique_key, deadline_ms);
    if (!result) {
        RTP_LLM_LOG_WARNING("P2PConnectorPrefillScheduler write: broadcast failed");
        return grpc::Status(grpc::StatusCode::INTERNAL, "broadcast failed");
    }

    result->result->waitDone();
    RTP_LLM_LOG_INFO("P2PConnectorPrefillScheduler write: broadcast result wait done success: %d",
                     result->result->success());
    if (!result->success()) {
        RTP_LLM_LOG_WARNING("P2PConnectorPrefillScheduler write: broadcast result wait done failed");
        return grpc::Status(grpc::StatusCode::INTERNAL, "broadcast result wait done failed");
    }
    RTP_LLM_LOG_INFO("P2PConnectorPrefillScheduler write: broadcast result wait done success");
    return grpc::Status::OK;
}

}  // namespace rtp_llm
