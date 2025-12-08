#include "rtp_llm/cpp/disaggregate/p2p_connector/P2PConnectorDecodeScheduler.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/disaggregate/transfer/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/model_rpc/RPCPool.h"
#include <memory>

namespace rtp_llm {

P2PConnectorDecodeAsyncContext::P2PConnectorDecodeAsyncContext(
    const std::shared_ptr<KVCacheResourceV1>&         resource,
    const std::shared_ptr<TPBroadcastClient::Result>& tp_sync_result,
    const std::shared_ptr<PrefillLoadClient::Result>& prefill_load_result,
    const std::shared_ptr<TPBroadcastClient>&         tp_broadcast_client):
    resource_(resource),
    tp_sync_result_(tp_sync_result),
    prefill_load_result_(prefill_load_result),
    tp_broadcast_client_(tp_broadcast_client) {}

P2PConnectorDecodeAsyncContext::~P2PConnectorDecodeAsyncContext() {}

bool P2PConnectorDecodeAsyncContext::success() const {
    return tp_sync_result_->success() && prefill_load_result_->success();
}

void P2PConnectorDecodeAsyncContext::cancel() {
    if (tp_sync_result_ && tp_broadcast_client_) {
        tp_broadcast_client_->cancel(tp_sync_result_);
    }
    if (prefill_load_result_) {
        prefill_load_result_->cancel();
    }
}

void P2PConnectorDecodeAsyncContext::waitDone() {
    if (prefill_load_result_) {
        prefill_load_result_->waitDone();
    }
    if (tp_sync_result_ && tp_sync_result_->result) {
        tp_sync_result_->result->waitDone();
    }
}

P2PConnectorDecodeScheduler::P2PConnectorDecodeScheduler(const GptInitParameter& gpt_init_parameter):
    gpt_init_parameter_(gpt_init_parameter) {}

P2PConnectorDecodeScheduler::~P2PConnectorDecodeScheduler() {}

bool P2PConnectorDecodeScheduler::init() {
    tp_broadcast_client_ = std::make_shared<TPBroadcastClient>(gpt_init_parameter_);
    if (!tp_broadcast_client_) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecodeScheduler init failed: tp_broadcast_client is null");
        return false;
    }
    if (!tp_broadcast_client_->init()) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecodeScheduler init failed: tp_broadcast_client init failed");
        return false;
    }

    auto rpc_pool = std::make_shared<RPCPool>();
    if (!rpc_pool) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecodeScheduler init failed: rpc_pool is null");
        return false;
    }

    prefill_load_client_ = std::make_shared<PrefillLoadClient>(gpt_init_parameter_);
    if (!prefill_load_client_) {
        RTP_LLM_LOG_ERROR("P2PConnectorDecodeScheduler init failed: prefill_load_client is null");
        return false;
    }
    return true;
}

std::shared_ptr<P2PConnectorDecodeAsyncContext>
P2PConnectorDecodeScheduler::asyncRead(const std::shared_ptr<KVCacheResourceV1>& resource,
                                       int64_t                                   request_id,
                                       const std::string&                        unique_key,
                                       const std::string&                        prefill_ip,
                                       uint32_t                                  prefill_port,
                                       int64_t                                   deadline_ms) {
    if (!resource) {
        RTP_LLM_LOG_WARNING("P2PConnectorDecodeScheduler asyncRead: resource is null");
        return nullptr;
    }

    auto layer_cache_buffers = LayerCacheBufferUtil::convert(*resource, 0);
    if (layer_cache_buffers.empty()) {
        RTP_LLM_LOG_WARNING("P2PConnectorDecodeScheduler asyncRead: layer_cache_buffers is empty");
        return nullptr;
    }
    RTP_LLM_LOG_INFO("P2PConnectorDecodeScheduler asyncRead: layer_cache_buffers: %zu", layer_cache_buffers.size());

    // 失败概率大的先执行
    auto prefill_load_result =
        prefill_load_client_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms);
    if (!prefill_load_result) {
        RTP_LLM_LOG_WARNING("P2PConnectorDecodeScheduler asyncRead: load failed");
        return nullptr;
    }
    RTP_LLM_LOG_INFO("P2PConnectorDecodeScheduler asyncRead: prefill_load_result: %p", prefill_load_result.get());

    // 先执行 TP broadcast
    auto tp_sync_result = tp_broadcast_client_->broadcast(request_id, layer_cache_buffers, {}, unique_key, deadline_ms);
    if (!tp_sync_result) {
        prefill_load_result->cancel();
        prefill_load_result->waitDone();
        RTP_LLM_LOG_WARNING("P2PConnectorDecodeScheduler asyncRead: broadcast failed");
        return nullptr;
    }
    RTP_LLM_LOG_INFO("P2PConnectorDecodeScheduler asyncRead: tp_sync_result: %p", tp_sync_result.get());

    return std::make_shared<P2PConnectorDecodeAsyncContext>(
        resource, tp_sync_result, prefill_load_result, tp_broadcast_client_);
}

}  // namespace rtp_llm
