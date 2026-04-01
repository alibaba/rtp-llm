#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerPrefill.h"

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <thread>

namespace rtp_llm {

P2PConnectorSchedulerPrefill::P2PConnectorSchedulerPrefill(
    P2PConnectorSchedulerConfig                config,
    const kmonitor::MetricsReporterPtr&        metrics_reporter,
    const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client):
    config_(std::move(config)), metrics_reporter_(metrics_reporter), tp_broadcast_client_(tp_broadcast_client) {}

ErrorInfo
P2PConnectorSchedulerPrefill::sendKVCache(const KVCacheResourcePtr&                            resource,
                                          const std::string&                                   unique_key,
                                          int64_t                                              request_id,
                                          const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                                          int64_t                                              deadline_ms,
                                          std::function<bool()>                                is_cancelled) {
    RTP_LLM_LOG_DEBUG("sendKVCache start, request_id: %ld, unique_key: %s, decode_transfer_servers_size: %zu",
                      request_id,
                      unique_key.c_str(),
                      decode_transfer_servers.size());

    int64_t start_time_us      = currentTimeUs();
    auto    collector          = std::make_shared<PrefillSchedulerMetricsCollector>();
    auto    report_metric_func = [start_time_us, collector, metrics_reporter = metrics_reporter_](bool success) {
        collector->total_cost_time_us = currentTimeUs() - start_time_us;
        collector->success            = success;
        if (metrics_reporter) {
            metrics_reporter->report<P2PConnectorMetrics, PrefillSchedulerMetricsCollector>(nullptr, collector.get());
        }
    };

    auto layer_cache_buffers = LayerCacheBufferUtil::convert(*resource, 0);
    if (layer_cache_buffers.empty()) {
        std::string error_msg = "sendKVCache: layer_cache_buffers is empty, request_id: " + std::to_string(request_id);
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        report_metric_func(false);
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, error_msg);
    }

    auto result = tp_broadcast_client_->broadcast(request_id,
                                                  layer_cache_buffers,
                                                  decode_transfer_servers,
                                                  unique_key,
                                                  deadline_ms,
                                                  P2PConnectorBroadcastType::HANDLE_READ);
    if (!result) {
        std::string error_msg = "sendKVCache: broadcast failed, request_id: " + std::to_string(request_id);
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        report_metric_func(false);
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, error_msg);
    }

    bool deadline_exceeded = false;
    auto cancel_result     = waitForBroadcastCompletion(
        result, unique_key, request_id, deadline_ms, std::move(is_cancelled), &deadline_exceeded);
    report_metric_func(!cancel_result && !deadline_exceeded && result->success());

    if (deadline_exceeded) {
        std::string error_msg =
            "sendKVCache: broadcast wait exceeded deadline_ms, request_id: " + std::to_string(request_id);
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT, error_msg);
    }

    if (cancel_result) {
        std::string error_msg = "sendKVCache: cancelled by client, request_id: " + std::to_string(request_id);
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED, error_msg);
    }

    if (!result->success()) {
        RTP_LLM_LOG_WARNING("sendKVCache: broadcast result failed, request_id: %ld, error_code: %s, error_msg: %s",
                            request_id,
                            ErrorCodeToString(result->errorCode()).c_str(),
                            result->errorMessage().c_str());
        return ErrorInfo(result->errorCode(), result->errorMessage());
    }

    RTP_LLM_LOG_DEBUG("sendKVCache end, request_id: %ld, unique_key: %s", request_id, unique_key.c_str());
    return ErrorInfo::OkStatus();
}

std::shared_ptr<P2PBroadcastClient::Result>
P2PConnectorSchedulerPrefill::waitForBroadcastCompletion(const std::shared_ptr<P2PBroadcastClient::Result>& result,
                                                         const std::string&                                 unique_key,
                                                         int64_t                                            request_id,
                                                         int64_t                                            deadline_ms,
                                                         std::function<bool()> is_cancelled,
                                                         bool*                 deadline_exceeded_out) {

    std::shared_ptr<P2PBroadcastClient::Result> cancel_result = nullptr;
    int                                         sleep_ms      = 1;
    constexpr int                               kBackoffCapMs = 8;
    while (!result->done()) {
        result->checkDone();
        if (!cancel_result && is_cancelled && is_cancelled()) {
            RTP_LLM_LOG_WARNING("sendKVCache: request cancelled by client, request_id: %ld, unique_key: %s",
                                request_id,
                                unique_key.c_str());
            cancel_result = tp_broadcast_client_->cancel(unique_key, P2PConnectorBroadcastType::CANCEL_HANDLE_READ);
        }
        if (!cancel_result && currentTimeMs() >= deadline_ms) {
            RTP_LLM_LOG_WARNING(
                "sendKVCache: broadcast still pending past deadline_ms=%ld, cancelling, request_id: %ld, unique_key: %s",
                deadline_ms,
                request_id,
                unique_key.c_str());
            cancel_result = tp_broadcast_client_->cancel(unique_key, P2PConnectorBroadcastType::CANCEL_HANDLE_READ);
            if (deadline_exceeded_out) {
                *deadline_exceeded_out = true;
            }
        }
        if (cancel_result && !cancel_result->done()) {
            cancel_result->checkDone();
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        sleep_ms = std::min(sleep_ms * 2, kBackoffCapMs);
    }
    return cancel_result;
}

}  // namespace rtp_llm
