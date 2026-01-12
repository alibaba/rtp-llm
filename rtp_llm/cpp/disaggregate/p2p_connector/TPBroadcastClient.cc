#include "rtp_llm/cpp/disaggregate/p2p_connector/TPBroadcastClient.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

TPBroadcastClient::TPBroadcastClient(const std::vector<std::string>& worker_addrs): worker_addrs_(worker_addrs) {
    // TODO: extra_wait_time_ms_ 需要从配置中获取.
}

bool TPBroadcastClient::init() {
    rpc_pool_ = std::make_shared<RPCPool>();
    if (!rpc_pool_) {
        RTP_LLM_LOG_ERROR("TPBroadcastClient init failed: rpc_pool_ is null");
        return false;
    }
    tp_broadcast_manager_ = std::make_shared<TpBroadcastManager>(worker_addrs_);
    if (!tp_broadcast_manager_->init()) {
        RTP_LLM_LOG_ERROR("TPBroadcastClient init failed: tp_broadcast_manager_ init failed");
        return false;
    }

    RTP_LLM_LOG_INFO("TPBroadcastClient init success");
    return true;
}

std::shared_ptr<TPBroadcastClient::Result>
TPBroadcastClient::broadcast(int64_t                                               request_id,
                             const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
                             const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
                             const std::string&                                    unique_key,
                             int64_t                                               deadline_ms,
                             P2PConnectorBroadcastType                             type) {
    // 构建 BroadcastTpRequestPB
    std::vector<std::shared_ptr<BroadcastTpRequestPB>> requests;
    size_t                                             worker_num = tp_broadcast_manager_->workerNum();
    requests.reserve(worker_num);

    for (size_t i = 0; i < worker_num; ++i) {
        auto request = std::make_shared<BroadcastTpRequestPB>();
        genBroadcastRequest(
            *request, request_id, layer_cache_buffers, decode_transfer_servers, unique_key, deadline_ms, type);
        requests.push_back(request);
    }

    // 执行广播
    auto timeout_ms = deadline_ms - currentTimeMs() + extra_wait_time_ms_;
    if (timeout_ms <= 0) {
        RTP_LLM_LOG_WARNING("broadcast timeout_ms: %ld <= 0, deadline_ms: %ld current_time_ms: %ld",
                            timeout_ms,
                            deadline_ms,
                            currentTimeMs());
        return nullptr;
    }

    auto result = tp_broadcast_manager_->broadcast(requests, timeout_ms);
    if (!result) {
        RTP_LLM_LOG_WARNING("broadcast failed, cannot create broadcast result");
        return nullptr;
    }

    return std::make_shared<Result>(unique_key, result);
}

void TPBroadcastClient::genBroadcastRequest(
    BroadcastTpRequestPB&                                 request,
    int64_t                                               request_id,
    const std::vector<std::shared_ptr<LayerCacheBuffer>>& layer_cache_buffers,
    const std::vector<std::pair<std::string, uint32_t>>&  decode_transfer_servers,
    const std::string&                                    unique_key,
    int64_t                                               deadline_ms,
    P2PConnectorBroadcastType                             type) {
    auto p2p_request = request.mutable_p2p_request();

    // 设置 layer_blocks
    for (const auto& layer_cache_buffer : layer_cache_buffers) {
        auto layer_block = p2p_request->add_layer_blocks();
        layer_block->set_layer_id(layer_cache_buffer->getLayerId());
        for (const auto& [key, block_id] : layer_cache_buffer->blockIdMap()) {
            layer_block->add_cache_keys(key);
            layer_block->add_block_ids(block_id);
        }
    }

    // 设置 peer_workers
    for (const auto& [ip, port] : decode_transfer_servers) {
        auto tp_worker = p2p_request->add_peer_workers();
        tp_worker->set_ip(ip);
        tp_worker->set_cache_store_port(port);
    }

    p2p_request->set_unique_key(unique_key);
    p2p_request->set_request_id(request_id);
    p2p_request->set_is_cancel(false);
    p2p_request->set_deadline_ms(deadline_ms);
    p2p_request->set_type(type);
}

void TPBroadcastClient::cancel(const std::shared_ptr<Result>& result, int64_t timeout_ms) {
    if (!result) {
        RTP_LLM_LOG_WARNING("TPBroadcastClient cancel: result is null");
        return;
    }

    // 构建取消请求
    std::vector<std::shared_ptr<BroadcastTpRequestPB>> requests;
    size_t                                             worker_num = tp_broadcast_manager_->workerNum();
    requests.reserve(worker_num);

    for (size_t i = 0; i < worker_num; ++i) {
        auto request = std::make_shared<BroadcastTpRequestPB>();
        genCancelRequest(*request, result->uniqueKey());
        requests.push_back(request);
    }

    auto cancel_result = tp_broadcast_manager_->broadcast(requests, timeout_ms);
    if (!cancel_result) {
        // 大概率某个rank挂了.
        RTP_LLM_FAIL("TPBroadcastClient cancel broadcast failed, cannot create broadcast result");
    }

    // 等待所有 RANK 完成取消
    cancel_result->waitDone(timeout_ms + extra_wait_time_ms_);
    if (!cancel_result->success()) {
        // 大概率某个rank挂了, cancel 处理应该快速响应.
        RTP_LLM_FAIL("TPBroadcastClient cancel broadcast failed, not all ranks succeeded");
    }

    auto responses = cancel_result->responses();
    for (const auto& response : responses) {
        if (!response || !response->has_p2p_response() || !response->p2p_response().success()) {
            // 不知道啥情况, 先挂一下
            RTP_LLM_FAIL("TPBroadcastClient cancel broadcast failed, response is null or not success");
        }
    }
}

void TPBroadcastClient::genCancelRequest(BroadcastTpRequestPB& request, const std::string& unique_key) {
    auto p2p_request = request.mutable_p2p_request();
    p2p_request->set_unique_key(unique_key);
    p2p_request->set_is_cancel(true);
}

bool TPBroadcastClient::Result::success() const {
    if (!tp_broadcast_result_ || !tp_broadcast_result_->success()) {
        return false;
    }
    auto responses = tp_broadcast_result_->responses();
    for (const auto& response : responses) {
        if (!response || !response->has_p2p_response() || !response->p2p_response().success()) {
            return false;
        }
    }
    return true;
}

void TPBroadcastClient::Result::checkDone() {
    tp_broadcast_result_->waitDone(1);  // wait 1ms
    if (tp_broadcast_result_->done()) {
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
    }
}

void TPBroadcastClient::setExtraWaitTimeMs(int64_t extra_wait_time_ms) {
    extra_wait_time_ms_ = extra_wait_time_ms;
}

}  // namespace rtp_llm
