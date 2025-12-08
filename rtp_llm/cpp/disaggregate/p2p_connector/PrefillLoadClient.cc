#include "rtp_llm/cpp/disaggregate/p2p_connector/PrefillLoadClient.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/StringUtil.h"
#include <grpc++/grpc++.h>
#include <chrono>

namespace rtp_llm {

PrefillLoadClient::PrefillLoadClient(const GptInitParameter& gpt_init_parameter):
    gpt_init_parameter_(gpt_init_parameter) {
    rpc_pool_ = std::make_shared<RPCPool>();
    if (!rpc_pool_) {
        RTP_LLM_LOG_ERROR("PrefillLoadClient init failed: rpc_pool is null");
        return;
    }

    // 解析 worker_addrs 并构建 tp_worker_infos_
    for (const auto& worker_addr : gpt_init_parameter_.worker_addrs_) {
        auto ip_parts = autil::StringUtil::split(worker_addr, ":");
        if (ip_parts.size() != 3) {
            RTP_LLM_FAIL("PrefillLoadClient: invalid worker addr format [%s], expected ip:cache_store_port",
                         worker_addr.c_str());
        L:
            continue;
        }
        TPWorkerInfoPB tp_worker;
        tp_worker.set_ip(ip_parts[0]);
        tp_worker.set_cache_store_port(autil::StringUtil::strToInt32WithDefault(ip_parts[1].c_str(), 0));
        tp_worker_infos_.push_back(tp_worker);
    }
}

std::shared_ptr<PrefillLoadClient::Result> PrefillLoadClient::load(int64_t            request_id,
                                                                   const std::string& prefill_ip,
                                                                   uint32_t           prefill_port,
                                                                   const std::string& unique_key,
                                                                   int64_t            deadline_ms) {
    if (!rpc_pool_) {
        RTP_LLM_LOG_WARNING("PrefillLoadClient load failed: rpc_pool is null");
        return nullptr;
    }

    auto result = std::make_shared<Result>();
    if (!result) {
        RTP_LLM_LOG_WARNING("PrefillLoadClient load failed: cannot create result");
        return nullptr;
    }

    // 构建服务器地址
    result->server_addr = prefill_ip + ":" + std::to_string(prefill_port);

    // 获取连接
    auto conn_status = rpc_pool_->getConnection(result->server_addr);
    if (!conn_status.ok()) {
        RTP_LLM_LOG_WARNING("PrefillLoadClient load failed: getConnection failed, addr: %s",
                            result->server_addr.c_str());
        return nullptr;
    }

    result->stub = conn_status.value().stub;
    if (!result->stub) {
        RTP_LLM_LOG_WARNING("PrefillLoadClient load failed: stub is null, addr: %s", result->server_addr.c_str());
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

    RTP_LLM_LOG_INFO("PrefillLoadClient load: request workers: %zu", result->request.workers_size());
    for (const auto& worker : result->request.workers()) {
        RTP_LLM_LOG_INFO("PrefillLoadClient load: request worker ip: %s, cache_store_port: %d",
                         worker.ip().c_str(),
                         worker.cache_store_port());
    }

    // 创建 ClientContext 和 CompletionQueue
    result->client_context    = std::make_shared<grpc::ClientContext>();
    result->completion_queue_ = std::make_shared<grpc::CompletionQueue>();

    // 设置超时时间
    result->timeout_ms_ = deadline_ms > 0 ? (static_cast<int>(deadline_ms) - currentTimeMs()) : 30000;
    RTP_LLM_LOG_INFO("PrefillLoadClient load: request deadline_ms: %lld, current time ms: %lld, timeout ms: %d",
                     deadline_ms,
                     currentTimeMs(),
                     result->timeout_ms_);

    result->client_context->set_deadline(std::chrono::system_clock::now()
                                         + std::chrono::milliseconds(result->timeout_ms_));

    result->reader_ = result->stub->PrepareAsyncStartLoad(
        result->client_context.get(), result->request, result->completion_queue_.get());
    if (!result->reader_) {
        RTP_LLM_LOG_WARNING("PrefillLoadClient load failed: PrepareAsyncStartLoad failed, addr: %s",
                            result->server_addr.c_str());
        return nullptr;
    }

    result->reader_->StartCall();
    result->reader_->Finish(
        &result->response, &result->status, reinterpret_cast<void*>(static_cast<intptr_t>(request_id)));

    RTP_LLM_LOG_INFO(
        "PrefillLoadClient load started, unique_key: %s, addr: %s", unique_key.c_str(), result->server_addr.c_str());
    return result;
}

bool PrefillLoadClient::Result::waitDone() {
    if (!completion_queue_) {
        RTP_LLM_LOG_WARNING("PrefillLoadClient::Result::waitDone: completion_queue is null");
        success_ = false;
        return false;
    }

    void* got_tag = nullptr;
    bool  ok      = false;

    // 计算超时时间
    auto deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms_ > 0 ? timeout_ms_ : 30000);

    // 等待异步操作完成
    auto next_status = completion_queue_->AsyncNext(&got_tag, &ok, deadline);

    if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
        RTP_LLM_LOG_WARNING("PrefillLoadClient::Result::waitDone: timeout, server_addr: %s", server_addr.c_str());
        success_ = false;
        if (client_context) {
            client_context->TryCancel();
        }
        return false;
    }

    if (!ok) {
        RTP_LLM_LOG_WARNING("PrefillLoadClient::Result::waitDone: async next failed, server_addr: %s",
                            server_addr.c_str());
        success_ = false;
        return false;
    }

    // 检查 tag 是否匹配（应该是 request_id）
    if (got_tag != reinterpret_cast<void*>(static_cast<intptr_t>(request_id_))) {
        RTP_LLM_LOG_WARNING(
            "PrefillLoadClient::Result::waitDone: unexpected tag, expected: %ld, got: %p, server_addr: %s",
            request_id_,
            got_tag,
            server_addr.c_str());
        success_ = false;
        return false;
    }

    // 检查 RPC 状态
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("PrefillLoadClient::Result::waitDone: rpc error: %s, server_addr: %s",
                            status.error_message().c_str(),
                            server_addr.c_str());
        success_ = false;
        return false;
    }

    // 检查响应是否成功
    if (!response.success()) {
        RTP_LLM_LOG_WARNING("PrefillLoadClient::Result::waitDone: response success is false, server_addr: %s",
                            server_addr.c_str());
        success_ = false;
        return false;
    }

    success_ = true;
    RTP_LLM_LOG_DEBUG("PrefillLoadClient::Result::waitDone: success, server_addr: %s", server_addr.c_str());
    return true;
}

void PrefillLoadClient::Result::cancel() {
    if (client_context) {
        client_context->TryCancel();
    }
}

}  // namespace rtp_llm
