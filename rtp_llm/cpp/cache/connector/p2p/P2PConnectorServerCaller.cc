#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorServerCaller.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/StringUtil.h"
#include <grpc++/grpc++.h>
#include <chrono>

namespace rtp_llm {

P2PConnectorServerCaller::P2PConnectorServerCaller(const std::vector<std::string>& worker_addrs):
    worker_addrs_(worker_addrs) {
    rpc_pool_ = std::make_shared<RPCPool>();
    if (!rpc_pool_) {
        RTP_LLM_LOG_ERROR("P2PConnectorServerCaller init failed: rpc_pool is null");
        return;
    }

    // 解析 worker_addrs 并构建 tp_worker_infos_
    for (const auto& worker_addr : worker_addrs_) {
        auto ip_parts = autil::StringUtil::split(worker_addr, ":");
        if (ip_parts.size() != 3) {
            RTP_LLM_FAIL("P2PConnectorServerCaller: invalid worker addr format [%s], expected ip:cache_store_port",
                         worker_addr.c_str());
            continue;
        }
        TPWorkerInfoPB tp_worker;
        tp_worker.set_ip(ip_parts[0]);
        tp_worker.set_cache_store_port(autil::StringUtil::strToInt32WithDefault(ip_parts[1].c_str(), 0));
        tp_worker_infos_.push_back(tp_worker);
    }
}

std::shared_ptr<P2PConnectorServerCaller::Result>
P2PConnectorServerCaller::load(int64_t                   request_id,
                               const std::string&        prefill_ip,
                               uint32_t                  prefill_port,
                               const std::string&        unique_key,
                               int64_t                   deadline_ms,
                               const IGenerateStreamPtr& generate_stream) {
    if (!rpc_pool_) {
        RTP_LLM_LOG_WARNING("P2PConnectorServerCaller load failed: rpc_pool is null");
        return nullptr;
    }

    auto result = std::make_shared<Result>();
    if (!result) {
        RTP_LLM_LOG_WARNING("P2PConnectorServerCaller load failed: cannot create result");
        return nullptr;
    }

    // 构建服务器地址
    result->server_addr = prefill_ip + ":" + std::to_string(prefill_port);

    // 获取连接
    auto conn_status = rpc_pool_->getConnection(result->server_addr);
    if (!conn_status.ok()) {
        RTP_LLM_LOG_WARNING("P2PConnectorServerCaller load failed: getConnection failed, addr: %s",
                            result->server_addr.c_str());
        return nullptr;
    }

    result->stub = conn_status.value().stub;
    if (!result->stub) {
        RTP_LLM_LOG_WARNING("P2PConnectorServerCaller load failed: stub is null, addr: %s",
                            result->server_addr.c_str());
        return nullptr;
    }

    // 保存 request_id 用于 tag 检查
    result->request_id_ = request_id;

    // 构建请求消息
    result->request.set_unique_key(unique_key);
    result->request.set_deadline_ms(deadline_ms);

    // 设置 workers
    for (const auto& tp_worker : tp_worker_infos_) {
        auto tp_worker_info = result->request.add_workers();
        tp_worker_info->set_ip(tp_worker.ip());
        tp_worker_info->set_cache_store_port(tp_worker.cache_store_port());
    }

    // 创建 ClientContext 和 CompletionQueue
    result->client_context    = std::make_shared<grpc::ClientContext>();
    result->completion_queue_ = std::make_shared<grpc::CompletionQueue>();

    // 设置超时时间
    result->timeout_ms_ = deadline_ms > 0 ? (static_cast<int>(deadline_ms) - currentTimeMs()) : 30000;
    result->client_context->set_deadline(std::chrono::system_clock::now()
                                         + std::chrono::milliseconds(result->timeout_ms_));

    result->generate_stream_ = generate_stream;

    result->reader_ = result->stub->PrepareAsyncStartLoad(
        result->client_context.get(), result->request, result->completion_queue_.get());
    if (!result->reader_) {
        RTP_LLM_LOG_WARNING("P2PConnectorServerCaller load failed: PrepareAsyncStartLoad failed, addr: %s",
                            result->server_addr.c_str());
        return nullptr;
    }

    result->reader_->StartCall();
    result->reader_->Finish(
        &result->response, &result->status, reinterpret_cast<void*>(static_cast<intptr_t>(request_id)));

    RTP_LLM_LOG_DEBUG(
        "P2PConnectorServerCaller load started, unique_key: %s, addr: %s, deadline_ms: %lld, timeout ms: %d",
        unique_key.c_str(),
        result->server_addr.c_str(),
        deadline_ms,
        result->timeout_ms_);
    return result;
}

void P2PConnectorServerCaller::Result::checkDone() {
    if (!completion_queue_) {
        RTP_LLM_LOG_WARNING("P2PConnectorServerCaller::Result::waitDone: completion_queue is null");
        success_ = false;
        done_    = true;
        return;
    }

    void* got_tag = nullptr;
    bool  ok      = false;

    // 计算超时时间
    auto once_deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(10);

    // 等待异步操作完成
    auto next_status = completion_queue_->AsyncNext(&got_tag, &ok, once_deadline);

    if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
        // not finish
        return;
    }

    if (!ok) {
        done_               = true;
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
        RTP_LLM_LOG_WARNING("P2PConnectorServerCaller::Result::waitDone: async next failed, server_addr: %s",
                            server_addr.c_str());
        return;
    }

    // 检查 RPC 状态
    if (!status.ok()) {
        done_               = true;
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
        RTP_LLM_LOG_WARNING("P2PConnectorServerCaller::Result::waitDone: rpc error: %s, server_addr: %s",
                            status.error_message().c_str(),
                            server_addr.c_str());
        return;
    }

    // 检查响应是否成功
    if (!response.success()) {
        done_               = true;
        total_cost_time_us_ = currentTimeUs() - start_time_us_;
        RTP_LLM_LOG_WARNING("P2PConnectorServerCaller::Result::waitDone: response success is false, server_addr: %s",
                            server_addr.c_str());
        return;
    }
    RTP_LLM_LOG_DEBUG("P2PConnectorServerCaller::Result::waitDone: response success");

    // update generate stream with response
    if (generate_stream_) {
        // update complete token ids
        int32_t token_id = static_cast<int32_t>(response.first_generate_token_id());
        generate_stream_->appendTokenId(0, token_id);
        RTP_LLM_LOG_DEBUG("P2PConnectorServerCaller::Result::waitDone: append token id: %d", token_id);

        // update reuse info with prefill reuse info
        generate_stream_->setPrefillReuseLength(
            response.total_reuse_len(), response.local_reuse_len(), response.remote_reuse_len());

        // update propose info
        if (response.propose_token_ids_size() > 0) {
            std::vector<int> propose_tokens;
            propose_tokens.assign(response.propose_token_ids().begin(), response.propose_token_ids().end());
            generate_stream_->appendSPInfo(propose_tokens, response.propose_probs(), response.propose_hidden());
            RTP_LLM_LOG_DEBUG(
                "P2PConnectorServerCaller::Result::waitDone: append propose info, propose tokens size: %zu",
                propose_tokens.size());
        }

        // update context position ids
        if (response.position_ids_size() > 0) {
            std::vector<int32_t> position_ids;
            position_ids.assign(response.position_ids().begin(), response.position_ids().end());
            generate_stream_->setContextPositionIds(position_ids);
            RTP_LLM_LOG_DEBUG(
                "P2PConnectorServerCaller::Result::waitDone: append context position ids, position ids size: %zu",
                position_ids.size());
        }
    } else {
        RTP_LLM_LOG_WARNING("generate stream is null");
    }

    success_            = true;
    done_               = true;
    total_cost_time_us_ = currentTimeUs() - start_time_us_;
    return;
}

}  // namespace rtp_llm
