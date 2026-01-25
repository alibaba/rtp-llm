#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorScheduler.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/model_rpc/proto/model_rpc_service.pb.h"
#include <memory>
#include <thread>
#include <chrono>

namespace rtp_llm {

P2PConnectorScheduler::P2PConnectorScheduler(const RuntimeConfig&                runtime_config,
                                             const CacheStoreConfig&             cache_store_config,
                                             const kmonitor::MetricsReporterPtr& metrics_reporter):
    runtime_config_(runtime_config), cache_store_config_(cache_store_config), metrics_reporter_(metrics_reporter) {}

P2PConnectorScheduler::~P2PConnectorScheduler() {
    if (checker_) {
        checker_->stop();
    }
}

void P2PConnectorScheduler::stopChecker() {
    if (checker_) {
        checker_->stop();
    }
}

bool P2PConnectorScheduler::init(const std::string& process_id, int64_t decode_polling_call_prefill_ms) {
    RTP_LLM_LOG_INFO("P2PConnectorScheduler init start");
    // init tp broadcast client
    tp_broadcast_client_ = std::make_shared<TPBroadcastClient>(runtime_config_.worker_grpc_addrs,
                                                               cache_store_config_.p2p_extra_wait_time_ms);
    if (!tp_broadcast_client_) {
        RTP_LLM_LOG_ERROR("P2PConnectorScheduler init failed: tp_broadcast_client is null");
        return false;
    }
    if (!tp_broadcast_client_->init()) {
        RTP_LLM_LOG_ERROR("P2PConnectorScheduler init failed: tp_broadcast_client init failed");
        return false;
    }

    // init server caller (for decode side to call prefill)
    server_caller_ = std::make_shared<P2PConnectorServerCaller>(runtime_config_.worker_addrs);
    if (!server_caller_) {
        RTP_LLM_LOG_ERROR("P2PConnectorScheduler init failed: server_caller is null");
        return false;
    }

    // init prefill server caller (for decode side to call prefill server)
    prefill_server_caller_ = std::make_shared<PrefillServerCaller>(process_id, decode_polling_call_prefill_ms);
    if (!prefill_server_caller_) {
        RTP_LLM_LOG_ERROR("P2PConnectorScheduler init failed: prefill_server_caller is null");
        return false;
    }

    // init checker for async read contexts
    checker_ = std::make_shared<P2PConnectorAsyncReadContextChecker>();
    if (!checker_->init(metrics_reporter_, tp_broadcast_client_)) {
        RTP_LLM_LOG_ERROR("P2PConnectorScheduler init failed: checker init failed");
        return false;
    }

    RTP_LLM_LOG_INFO("P2PConnectorScheduler init success");
    return true;
}

std::shared_ptr<P2PConnectorAsyncReadContext>
P2PConnectorScheduler::asyncRead(const KVCacheResourcePtr&  resource,
                                 int64_t                    request_id,
                                 const std::string&         unique_key,
                                 int64_t                    deadline_ms,
                                 const IGenerateStreamPtr&  generate_stream,
                                 const std::pair<int, int>& block_range) {
    if (!generate_stream) {
        RTP_LLM_LOG_WARNING("P2PConnectorScheduler asyncRead: generate_stream is null");
        return nullptr;
    }

    auto [prefill_ip, prefill_port] = generate_stream->getPrefillAddr();
    RTP_LLM_LOG_DEBUG("P2PConnectorScheduler asyncRead start, request_id: %ld, unique_key: %s",
                      request_id,
                      unique_key.c_str(),
                      prefill_port);
    if (prefill_ip.empty() || prefill_port == 0) {
        RTP_LLM_LOG_WARNING("P2PConnectorScheduler asyncRead: prefill_ip is empty or prefill_port is 0");
        generate_stream->setStop(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                                 "prefill_ip is empty or prefill_port is 0");
        return nullptr;
    }
    auto collector = std::make_shared<P2PConnectorClientSchedulerMetricsCollector>(metrics_reporter_);
    if (!resource) {
        RTP_LLM_LOG_WARNING("P2PConnectorScheduler asyncRead: resource is null");
        collector->success = false;
        return nullptr;
    }

    // convert resource to layer cache buffers
    auto layer_cache_buffers = LayerCacheBufferUtil::convert(*resource, 0, block_range.first, block_range.second);
    if (layer_cache_buffers.empty()) {
        RTP_LLM_LOG_WARNING("P2PConnectorScheduler asyncRead: layer_cache_buffers is empty");
        collector->success = false;
        return nullptr;
    }

    // 如果 needCallPrefill，先调用 prefill server
    auto prefill_context =
        callPrefillIfNeeded(generate_stream, prefill_ip, prefill_port, unique_key, deadline_ms, collector);
    if (collector->success == false) {
        return nullptr;
    }

    // call prefill server to trigger write (higher failure probability, execute first)
    auto server_call_result =
        server_caller_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, generate_stream);
    if (!server_call_result) {
        RTP_LLM_LOG_WARNING("P2PConnectorScheduler asyncRead: server_caller load failed");
        collector->success = false;
        generate_stream->setStop(ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED, "layer_cache_buffers is empty");
        return nullptr;
    }

    // broadcast to all TP workers
    auto tp_sync_result = tp_broadcast_client_->broadcast(
        request_id, layer_cache_buffers, {}, unique_key, deadline_ms, P2PConnectorBroadcastType::READ);
    if (!tp_sync_result) {
        collector->success = false;
        RTP_LLM_LOG_WARNING("P2PConnectorScheduler asyncRead: broadcast failed");
        generate_stream->setStop(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "broadcast failed");
        return nullptr;
    }

    // create async context and add to checker
    // 将 prefill_context 和 generate_stream 传递给 async_context，统一处理失败和 cancel 逻辑
    auto async_context = std::make_shared<P2PConnectorAsyncReadContext>(
        resource, tp_sync_result, server_call_result, collector, prefill_context, generate_stream);
    checker_->addContext(async_context);

    RTP_LLM_LOG_DEBUG(
        "P2PConnectorScheduler asyncRead end, request_id: %ld, unique_key: %s", request_id, unique_key.c_str());
    return async_context;
}

ErrorInfo
P2PConnectorScheduler::handleRead(const KVCacheResourcePtr&                            resource,
                                  const std::string&                                   unique_key,
                                  int64_t                                              request_id,
                                  const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                                  int64_t                                              deadline_ms,
                                  std::function<bool()>                                is_cancelled) {
    RTP_LLM_LOG_DEBUG(
        "P2PConnectorScheduler handleRead start, request_id: %ld, unique_key: %s, decode_transfer_servers_size: %zu",
        request_id,
        unique_key.c_str(),
        decode_transfer_servers.size());

    int64_t start_time_us      = currentTimeUs();
    auto    collector          = std::make_shared<P2PConnectorServerSchedulerMetricsCollector>();
    auto    report_metric_func = [start_time_us, collector, metrics_reporter = metrics_reporter_](bool success) {
        collector->total_cost_time_us = currentTimeUs() - start_time_us;
        collector->success            = success;
        if (metrics_reporter) {
            metrics_reporter->report<P2PConnectorMetrics, P2PConnectorServerSchedulerMetricsCollector>(nullptr,
                                                                                                       collector.get());
        }
    };

    // convert resource to layer cache buffers
    auto layer_cache_buffers = LayerCacheBufferUtil::convert(*resource, 0);
    if (layer_cache_buffers.empty()) {
        std::string error_msg =
            "P2PConnectorScheduler handleRead: layer_cache_buffers is empty, request_id: " + std::to_string(request_id);
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        report_metric_func(false);
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, error_msg);
    }

    // broadcast to all TP workers with decode transfer servers
    auto result = tp_broadcast_client_->broadcast(request_id,
                                                  layer_cache_buffers,
                                                  decode_transfer_servers,
                                                  unique_key,
                                                  deadline_ms,
                                                  P2PConnectorBroadcastType::HANDLE_READ);
    if (!result) {
        std::string error_msg =
            "P2PConnectorScheduler handleRead: broadcast failed, request_id: " + std::to_string(request_id);
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        report_metric_func(false);
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, error_msg);
    }

    // wait for broadcast to complete (sync call)
    std::shared_ptr<TPBroadcastClient::Result> cancel_result = nullptr;
    while (!result->done()) {
        result->checkDone();
        // check if request is cancelled by decode side
        if (!cancel_result && is_cancelled && is_cancelled()) {
            RTP_LLM_LOG_WARNING(
                "P2PConnectorScheduler handleRead: request cancelled by client, request_id: %ld, unique_key: %s",
                request_id,
                unique_key.c_str());
            // send cancel request to workers to cancel handleRead
            cancel_result = tp_broadcast_client_->cancel(unique_key, P2PConnectorBroadcastType::CANCEL_HANDLE_READ);
        }
        if (cancel_result && !cancel_result->done()) {
            cancel_result->checkDone();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    report_metric_func(!cancel_result && result->success());

    if (cancel_result) {
        std::string error_msg =
            "P2PConnectorScheduler handleRead: cancelled by client, request_id: " + std::to_string(request_id);
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED, error_msg);
    }

    if (!result->success()) {
        // 从 result 获取 error_code 和 error_msg
        ErrorCode   error_code = result->errorCode();
        std::string error_msg  = result->errorMessage();
        RTP_LLM_LOG_WARNING(
            "P2PConnectorScheduler handleRead: broadcast result failed, request_id: %ld, error_code: %s, error_msg: %s",
            request_id,
            ErrorCodeToString(error_code).c_str(),
            error_msg.c_str());
        return ErrorInfo(error_code, error_msg);
    }

    RTP_LLM_LOG_DEBUG(
        "P2PConnectorScheduler handleRead end, request_id: %ld, unique_key: %s", request_id, unique_key.c_str());
    return ErrorInfo::OkStatus();
}

std::shared_ptr<PrefillServerCallerContext> P2PConnectorScheduler::callPrefillIfNeeded(
    const IGenerateStreamPtr&                                           generate_stream,
    const std::string&                                                  prefill_ip,
    uint32_t                                                            prefill_port,
    const std::string&                                                  unique_key,
    int64_t                                                             deadline_ms,
    const std::shared_ptr<P2PConnectorClientSchedulerMetricsCollector>& collector) {
    // 如果不需要调用 prefill server，直接返回 nullptr
    if (!generate_stream->needCallPrefill()) {
        return nullptr;
    }

    // TODO: change to sync call submit with check
    if (!prefill_server_caller_) {
        RTP_LLM_LOG_WARNING("P2PConnectorScheduler callPrefillIfNeeded: prefill_server_caller is null");
        collector->success = false;
        generate_stream->setStop(ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED, "prefill_server_caller is null");
        return nullptr;
    }

    // 从 IGenerateStream 获取原始请求
    const GenerateInputPB* original_request = generate_stream->getOriginalRequest();
    if (!original_request) {
        RTP_LLM_LOG_WARNING("P2PConnectorScheduler callPrefillIfNeeded: original_request is null, cannot call prefill");
        collector->success = false;
        generate_stream->setStop(ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED, "original_request is null");
        return nullptr;
    }

    // 使用原始请求调用 prefill server
    auto prefill_context =
        prefill_server_caller_->callPrefill(original_request, prefill_ip, prefill_port, unique_key, deadline_ms * 1000);
    if (!prefill_context) {
        RTP_LLM_LOG_WARNING("P2PConnectorScheduler callPrefillIfNeeded: prefill_server_caller callPrefill failed");
        collector->success = false;
        generate_stream->setStop(ErrorCode::P2P_CONNECTOR_CALL_PREFILL_FAILED,
                                 "prefill_server_caller callPrefill failed");
        return nullptr;
    }

    RTP_LLM_LOG_DEBUG("P2PConnectorScheduler callPrefillIfNeeded: prefill call started, unique_key: %s",
                      unique_key.c_str());
    return prefill_context;
}

}  // namespace rtp_llm
