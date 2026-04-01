#include "rtp_llm/cpp/cache/connector/p2p/PrefillLoadCaller.h"

#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/model_rpc/RpcErrorCode.h"
#include "autil/StringUtil.h"
#include <grpc++/grpc++.h>
#include <chrono>
#include <limits>

namespace rtp_llm {

PrefillLoadCaller::PrefillLoadCaller(const std::vector<std::string>& worker_addrs): worker_addrs_(worker_addrs) {
    rpc_pool_ = std::make_shared<RPCPool>();
    if (!rpc_pool_) {
        RTP_LLM_LOG_ERROR("PrefillLoadCaller init failed: rpc_pool is null");
        return;
    }

    // worker_addrs 每项为 ip:cache_store_port:grpc_port（三段），解析 ip 与 cache_store_port 写入 tp_worker_infos_
    for (const auto& worker_addr : worker_addrs_) {
        auto ip_parts = autil::StringUtil::split(worker_addr, ":");
        if (ip_parts.size() != 3) {
            RTP_LLM_FAIL("PrefillLoadCaller: invalid worker addr format [%s], expected ip:cache_store_port:grpc_port",
                         worker_addr.c_str());
            continue;
        }
        TPWorkerInfoPB tp_worker;
        tp_worker.set_ip(ip_parts[0]);
        tp_worker.set_cache_store_port(autil::StringUtil::strToInt32WithDefault(ip_parts[1].c_str(), 0));
        tp_worker_infos_.push_back(tp_worker);
    }
}

std::shared_ptr<PrefillLoadCaller::Result> PrefillLoadCaller::load(int64_t                   request_id,
                                                                   const std::string&        prefill_ip,
                                                                   uint32_t                  prefill_port,
                                                                   const std::string&        unique_key,
                                                                   int64_t                   deadline_ms,
                                                                   const IGenerateStreamPtr& generate_stream) {
    if (!rpc_pool_) {
        RTP_LLM_LOG_WARNING("PrefillLoadCaller load failed: rpc_pool is null");
        return nullptr;
    }
    if (!generate_stream) {
        RTP_LLM_LOG_WARNING("PrefillLoadCaller load failed: generate_stream is null");
        return nullptr;
    }

    auto result         = std::make_shared<Result>();
    result->server_addr = prefill_ip + ":" + std::to_string(prefill_port);

    auto conn_status = rpc_pool_->getConnection(result->server_addr);
    if (!conn_status.ok()) {
        RTP_LLM_LOG_WARNING("PrefillLoadCaller load failed: getConnection failed, addr: %s",
                            result->server_addr.c_str());
        return nullptr;
    }

    result->stub = conn_status.value().stub;
    if (!result->stub) {
        RTP_LLM_LOG_WARNING("PrefillLoadCaller load failed: stub is null, addr: %s", result->server_addr.c_str());
        return nullptr;
    }

    result->request_id      = request_id;
    result->generate_stream = generate_stream;

    if (!buildAndStartAsyncRpc(result, unique_key, deadline_ms, request_id)) {
        return nullptr;
    }

    RTP_LLM_LOG_DEBUG("PrefillLoadCaller load started, unique_key: %s, addr: %s, deadline_ms: %lld, timeout ms: %d",
                      unique_key.c_str(),
                      result->server_addr.c_str(),
                      deadline_ms,
                      result->timeout_ms);
    return result;
}

bool PrefillLoadCaller::buildAndStartAsyncRpc(const std::shared_ptr<Result>& result,
                                              const std::string&             unique_key,
                                              int64_t                        deadline_ms,
                                              int64_t                        request_id) {
    result->request.set_unique_key(unique_key);
    result->request.set_deadline_ms(deadline_ms);

    for (const auto& tp_worker : tp_worker_infos_) {
        auto tp_worker_info = result->request.add_workers();
        tp_worker_info->set_ip(tp_worker.ip());
        tp_worker_info->set_cache_store_port(tp_worker.cache_store_port());
    }

    result->client_context   = std::make_shared<grpc::ClientContext>();
    result->completion_queue = std::make_shared<grpc::CompletionQueue>();

    const int64_t now_ms       = currentTimeMs();
    int64_t       remaining_ms = deadline_ms > 0 ? (deadline_ms - now_ms) : 30000;
    if (remaining_ms < 0) {
        remaining_ms = 0;
    }
    const int64_t max_int_ms = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (remaining_ms > max_int_ms) {
        remaining_ms = max_int_ms;
    }
    result->timeout_ms = static_cast<int>(remaining_ms);
    result->client_context->set_deadline(std::chrono::system_clock::now()
                                         + std::chrono::milliseconds(result->timeout_ms));

    result->reader = result->stub->PrepareAsyncStartLoad(
        result->client_context.get(), result->request, result->completion_queue.get());
    if (!result->reader) {
        RTP_LLM_LOG_WARNING("PrefillLoadCaller: PrepareAsyncStartLoad failed, addr: %s", result->server_addr.c_str());
        return false;
    }

    result->reader->StartCall();
    result->reader->Finish(
        &result->response, &result->status, reinterpret_cast<void*>(static_cast<intptr_t>(request_id)));
    return true;
}

void PrefillLoadCaller::Result::shutdownAndDrainCompletionQueue() {
    if (!completion_queue || completion_queue_shutdown_drained_) {
        return;
    }
    completion_queue->Shutdown();
    void* tag = nullptr;
    bool  ok  = false;
    while (completion_queue->Next(&tag, &ok)) {
        // Drain Finish / cancellation notifications; tag is the request_id we passed to Finish().
    }
    completion_queue_shutdown_drained_ = true;
}

void PrefillLoadCaller::Result::cancel() {
    if (done_) {
        return;
    }
    if (client_context) {
        client_context->TryCancel();
        RTP_LLM_LOG_DEBUG("PrefillLoadCaller::Result::cancel: cancelled grpc request, server_addr: %s",
                          server_addr.c_str());
    }
    shutdownAndDrainCompletionQueue();
    done_              = true;
    success_           = false;
    total_cost_time_us = currentTimeUs() - start_time_us;
}

bool PrefillLoadCaller::Result::pollCompletionQueue() {
    if (!completion_queue) {
        RTP_LLM_LOG_WARNING("PrefillLoadCaller::Result::pollCompletionQueue: completion_queue is null");
        error_code    = ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED;
        error_message = "completion_queue is null";
        return false;
    }

    void* got_tag       = nullptr;
    bool  ok            = false;
    auto  once_deadline = std::chrono::system_clock::now() + std::chrono::milliseconds(1);
    auto  next_status   = completion_queue->AsyncNext(&got_tag, &ok, once_deadline);

    if (next_status == grpc::CompletionQueue::NextStatus::TIMEOUT) {
        return true;  // not finished yet, caller should retry
    }

    done_              = true;
    total_cost_time_us = currentTimeUs() - start_time_us;

    if (!ok) {
        RTP_LLM_LOG_WARNING("PrefillLoadCaller::Result::pollCompletionQueue: async next failed, server_addr: %s",
                            server_addr.c_str());
        error_code    = ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED;
        error_message = "async next failed: " + status.error_message();
        return false;
    }
    if (!status.ok()) {
        RTP_LLM_LOG_WARNING("PrefillLoadCaller::Result::pollCompletionQueue: rpc error: %s, server_addr: %s",
                            status.error_message().c_str(),
                            server_addr.c_str());
        error_code    = ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED;
        error_message = status.error_message();
        return false;
    }
    if (response.error_code() != ErrorCodePB::NONE_ERROR) {
        RTP_LLM_LOG_WARNING(
            "PrefillLoadCaller::Result::pollCompletionQueue: response error_code is not NONE_ERROR, server_addr: %s",
            server_addr.c_str());
        error_code    = transRPCErrorCode(response.error_code());
        error_message = response.error_message();
        return false;
    }
    return true;  // RPC succeeded
}

void PrefillLoadCaller::Result::updateStreamFromResponse() {
    int32_t token_id = static_cast<int32_t>(response.first_generate_token_id());
    generate_stream->appendTokenId(0, token_id);
    RTP_LLM_LOG_DEBUG("PrefillLoadCaller::Result: append token id: %d", token_id);

    generate_stream->setPrefillReuseLength(response.total_reuse_len(),
                                           response.local_reuse_len(),
                                           response.remote_reuse_len(),
                                           response.memory_reuse_len());

    if (response.propose_token_ids_size() > 0) {
        std::vector<int> propose_tokens(response.propose_token_ids().begin(), response.propose_token_ids().end());
        generate_stream->appendSPInfo(propose_tokens, response.propose_probs(), response.propose_hidden());
        RTP_LLM_LOG_DEBUG("PrefillLoadCaller::Result: append propose info, size: %zu", propose_tokens.size());
    }

    if (response.position_ids_size() > 0) {
        std::vector<int32_t> position_ids(response.position_ids().begin(), response.position_ids().end());
        generate_stream->setContextPositionIds(position_ids);
        RTP_LLM_LOG_DEBUG("PrefillLoadCaller::Result: append context position ids, size: %zu", position_ids.size());
    }
}

void PrefillLoadCaller::Result::checkDone() {
    if (done_) {
        return;
    }
    bool poll_ok = pollCompletionQueue();
    if (!poll_ok) {
        // error already set in pollCompletionQueue
        return;
    }
    if (!done_) {
        return;  // TIMEOUT — not finished yet
    }

    RTP_LLM_LOG_DEBUG("PrefillLoadCaller::Result::checkDone: response success");
    updateStreamFromResponse();
    success_           = true;
    done_              = true;
    total_cost_time_us = currentTimeUs() - start_time_us;
}

}  // namespace rtp_llm
