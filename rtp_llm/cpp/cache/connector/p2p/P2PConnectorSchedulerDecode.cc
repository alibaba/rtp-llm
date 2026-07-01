#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerDecode.h"

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include <algorithm>
#include <memory>
#include <optional>

namespace rtp_llm {

P2PConnectorSchedulerDecode::P2PConnectorSchedulerDecode(
    P2PConnectorSchedulerConfig                config,
    const kmonitor::MetricsReporterPtr&        metrics_reporter,
    const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client):
    config_(std::move(config)), metrics_reporter_(metrics_reporter), tp_broadcast_client_(tp_broadcast_client) {}

P2PConnectorSchedulerDecode::~P2PConnectorSchedulerDecode() {
    if (checker_) {
        checker_->stop();
    }
}

bool P2PConnectorSchedulerDecode::init(const std::string& process_id) {
    server_caller_ = std::make_shared<PrefillLoadCaller>(config_.worker_addrs);

    checker_ = std::make_shared<P2PConnectorAsyncReadContextChecker>();
    if (!checker_->init(metrics_reporter_, tp_broadcast_client_)) {
        RTP_LLM_LOG_ERROR("P2PConnectorSchedulerDecode init failed: checker init failed");
        return false;
    }

    return true;
}

void P2PConnectorSchedulerDecode::stopChecker() {
    if (checker_) {
        checker_->stop();
    }
}

P2PConnectorSchedulerDecode::AsyncReadResult P2PConnectorSchedulerDecode::asyncRead(
    const KVCacheResourcePtr& resource, const std::shared_ptr<Meta>& meta, const std::pair<int, int>& block_range) {
    if (!meta || !resource) {
        RTP_LLM_LOG_WARNING("asyncRead: meta or resource is null");
        return {nullptr, ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "meta or resource is null")};
    }

    // Extract routing from Meta::p2pRouting()
    auto routing = meta->p2pRouting();
    if (!routing.has_value()) {
        RTP_LLM_LOG_WARNING("asyncRead: meta->p2pRouting() returned nullopt");
        return {
            nullptr,
            ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "meta->p2pRouting() returned nullopt")};
    }

    const int64_t     request_id      = routing->request_id;
    const std::string unique_key      = routing->unique_key;
    // Clamp the per-transfer deadline to p2p_max_transfer_deadline_ms. The
    // routing deadline is the request's business deadline (often ~1h via
    // generate_config->timeout_ms), which is too long for a single P2P read:
    // a single hung RDMA transfer would block waitForBroadcastCompletion /
    // waitSendCallbacksWithTimeout for the full hour. See 5/26 incident.
    const int64_t     now_ms          = currentTimeMs();
    const int64_t     deadline_ms     = std::min(routing->deadline_ms, now_ms + config_.p2p_max_transfer_deadline_ms);
    const auto&       prefill_addr    = routing->prefill_addr;
    const int         prefill_tp_size = routing->prefill_tp_size;

    if (unique_key.empty()) {
        RTP_LLM_LOG_WARNING("asyncRead: unique_key is empty");
        return {nullptr, ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "unique_key is empty")};
    }

    if (prefill_addr.first.empty() || prefill_addr.second == 0) {
        RTP_LLM_LOG_WARNING("asyncRead: prefill_ip is empty or prefill_port is 0");
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                          "prefill_ip is empty or prefill_port is 0")};
    }

    auto collector = std::make_shared<DecodeSchedulerMetricsCollector>(metrics_reporter_);
    auto layer_cache_buffers =
        LayerCacheBufferUtil::convert(*resource, 0, config_.layer_attn_types, block_range.first, block_range.second);
    if (layer_cache_buffers.empty()) {
        RTP_LLM_LOG_WARNING("asyncRead: layer_cache_buffers is empty");
        collector->success = false;
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "layer_cache_buffers is empty")};
    }

    auto      generate_stream = meta->generateStream();
    ErrorInfo start_error;
    auto      async_calls = startAsyncReadCalls(request_id,
                                           prefill_addr.first,
                                           prefill_addr.second,
                                           unique_key,
                                           deadline_ms,
                                           layer_cache_buffers,
                                           generate_stream,
                                           collector,
                                           start_error,
                                           prefill_tp_size);
    if (!async_calls) {
        return {nullptr, start_error};
    }

    auto async_context = std::make_shared<P2PConnectorAsyncReadContext>(resource,
                                                                        async_calls->tp_sync_result,
                                                                        async_calls->server_call_result,
                                                                        collector,
                                                                        config_.p2p_transfer_not_done_resource_hold_ms);

    // [PD-DIAG] addContext takes async_contexts_mutex_; if the checker thread is busy
    // (or — before the P0-2 fix — blocked in a slow cancel()), this can serialize the
    // whole scheduler. We bracket it separately so that any residual asyncReadAfterMatch
    // slowness can be attributed to startAsyncReadCalls vs addContext without ambiguity.
    const int64_t add_context_start_us = currentTimeUs();
    checker_->addContext(async_context);
    const int64_t add_context_cost_us = currentTimeUs() - add_context_start_us;
    if (add_context_cost_us >= 100000) {
        RTP_LLM_LOG_WARNING("[PD-DIAG] P2PConnectorSchedulerDecode::asyncRead slow addContext, "
                            "unique_key=%s, cost_us=%ld",
                            unique_key.c_str(),
                            add_context_cost_us);
    }

    return {async_context, ErrorInfo::OkStatus()};
}

std::optional<P2PConnectorSchedulerDecode::AsyncReadCallResults> P2PConnectorSchedulerDecode::startAsyncReadCalls(
    int64_t                                                 request_id,
    const std::string&                                      prefill_ip,
    uint32_t                                                prefill_port,
    const std::string&                                      unique_key,
    int64_t                                                 deadline_ms,
    const std::vector<std::shared_ptr<LayerCacheBuffer>>&   layer_cache_buffers,
    GenerateStream*                                         generate_stream,
    const std::shared_ptr<DecodeSchedulerMetricsCollector>& collector,
    ErrorInfo&                                              out_error,
    int                                                     prefill_tp_size) {

    const int64_t entry_us = currentTimeUs();
    RTP_LLM_LOG_DEBUG("[PD-DIAG] startAsyncReadCalls entry, unique_key=%s, prefill=%s:%u, timestamp_us=%ld",
                     unique_key.c_str(),
                     prefill_ip.c_str(),
                     prefill_port,
                     entry_us);

    // [PD-DIAG] Sub-stage timing. server_caller_->load eventually hits
    // RpcPool::getConnection (a pool-wide mutex + potentially synchronous
    // gRPC channel reconnection). tp_broadcast_client_->broadcast does
    // per-worker gRPC AsyncExecuteFunction. Either can be the source of
    // 18-22s asyncReadAfterMatch stalls observed in production.
    const int64_t server_load_start_us = currentTimeUs();
    auto          server_call_result =
        server_caller_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, generate_stream);
    const int64_t server_load_cost_us = currentTimeUs() - server_load_start_us;
    if (server_load_cost_us >= 100000) {
        RTP_LLM_LOG_WARNING("[PD-DIAG] startAsyncReadCalls slow server_caller->load, "
                            "unique_key=%s, prefill=%s:%u, cost_us=%ld",
                            unique_key.c_str(),
                            prefill_ip.c_str(),
                            prefill_port,
                            server_load_cost_us);
    }
    if (!server_call_result) {
        RTP_LLM_LOG_WARNING("asyncRead: server_caller load failed, unique_key: %s", unique_key.c_str());
        collector->success = false;
        out_error          = ErrorInfo(ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED,
                              "server_caller load failed: failed to start async StartLoad RPC to prefill");
        return std::nullopt;
    }

    const int64_t broadcast_start_us = currentTimeUs();
    auto          tp_sync_result     = tp_broadcast_client_->broadcast(
        request_id, layer_cache_buffers, {}, unique_key, deadline_ms, P2PConnectorBroadcastType::READ, prefill_tp_size);
    const int64_t broadcast_cost_us = currentTimeUs() - broadcast_start_us;
    if (broadcast_cost_us >= 100000) {
        RTP_LLM_LOG_WARNING(
            "[PD-DIAG] startAsyncReadCalls slow tp_broadcast_client->broadcast, unique_key=%s, cost_us=%ld",
            unique_key.c_str(),
            broadcast_cost_us);
    }
    if (!tp_sync_result) {
        collector->success = false;
        RTP_LLM_LOG_WARNING("asyncRead: broadcast failed");
        out_error = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "broadcast failed");
        server_call_result->cancel();
        return std::nullopt;
    }

    const int64_t total_cost_us = currentTimeUs() - entry_us;
    if (total_cost_us >= 100000) {
        RTP_LLM_LOG_WARNING(
            "[PD-DIAG] startAsyncReadCalls slow total, unique_key=%s, total_us=%ld, server_load_us=%ld, broadcast_us=%ld",
            unique_key.c_str(),
            total_cost_us,
            server_load_cost_us,
            broadcast_cost_us);
    }
    return AsyncReadCallResults{server_call_result, tp_sync_result};
}

}  // namespace rtp_llm
