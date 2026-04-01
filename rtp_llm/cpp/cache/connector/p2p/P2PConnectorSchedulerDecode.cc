#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerDecode.h"

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
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

P2PConnectorSchedulerDecode::AsyncReadResult
P2PConnectorSchedulerDecode::asyncRead(const KVCacheResourcePtr&  resource,
                                       const IGenerateStreamPtr&  generate_stream,
                                       const std::pair<int, int>& block_range) {
    if (!generate_stream || !resource) {
        RTP_LLM_LOG_WARNING("asyncRead: generate_stream or resource is null");
        return {
            nullptr,
            ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "generate_stream or resource is null")};
    }

    const int64_t     request_id  = generate_stream->requestId();
    const std::string unique_key  = generate_stream->uniqueKey();
    const int64_t     deadline_ms = generate_stream->deadlineMs();

    auto [prefill_ip, prefill_port] = generate_stream->getPrefillAddr();
    if (prefill_ip.empty() || prefill_port == 0) {
        RTP_LLM_LOG_WARNING("asyncRead: prefill_ip is empty or prefill_port is 0");
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                          "prefill_ip is empty or prefill_port is 0")};
    }

    auto collector           = std::make_shared<DecodeSchedulerMetricsCollector>(metrics_reporter_);
    auto layer_cache_buffers = LayerCacheBufferUtil::convert(*resource, 0, block_range.first, block_range.second);
    if (layer_cache_buffers.empty()) {
        RTP_LLM_LOG_WARNING("asyncRead: layer_cache_buffers is empty");
        collector->success = false;
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "layer_cache_buffers is empty")};
    }

    ErrorInfo start_error;
    auto      async_calls = startAsyncReadCalls(request_id,
                                           prefill_ip,
                                           prefill_port,
                                           unique_key,
                                           deadline_ms,
                                           layer_cache_buffers,
                                           generate_stream,
                                           collector,
                                           start_error);
    if (!async_calls) {
        return {nullptr, start_error};
    }

    auto async_context = std::make_shared<P2PConnectorAsyncReadContext>(resource,
                                                                        async_calls->tp_sync_result,
                                                                        async_calls->server_call_result,
                                                                        collector,
                                                                        config_.p2p_transfer_not_done_resource_hold_ms);
    checker_->addContext(async_context);

    return {async_context, ErrorInfo::OkStatus()};
}

std::optional<P2PConnectorSchedulerDecode::AsyncReadCallResults> P2PConnectorSchedulerDecode::startAsyncReadCalls(
    int64_t                                                 request_id,
    const std::string&                                      prefill_ip,
    uint32_t                                                prefill_port,
    const std::string&                                      unique_key,
    int64_t                                                 deadline_ms,
    const std::vector<std::shared_ptr<LayerCacheBuffer>>&   layer_cache_buffers,
    const IGenerateStreamPtr&                               generate_stream,
    const std::shared_ptr<DecodeSchedulerMetricsCollector>& collector,
    ErrorInfo&                                              out_error) {

    int prefill_tp_size = generate_stream->getPrefillTpSize();

    auto server_call_result =
        server_caller_->load(request_id, prefill_ip, prefill_port, unique_key, deadline_ms, generate_stream);
    if (!server_call_result) {
        RTP_LLM_LOG_WARNING("asyncRead: server_caller load failed, unique_key: %s", unique_key.c_str());
        collector->success = false;
        out_error          = ErrorInfo(ErrorCode::P2P_CONNECTOR_LOAD_FROM_PREFILL_FAILED,
                              "server_caller load failed: failed to start async StartLoad RPC to prefill");
        return std::nullopt;
    }

    auto tp_sync_result = tp_broadcast_client_->broadcast(
        request_id, layer_cache_buffers, {}, unique_key, deadline_ms, P2PConnectorBroadcastType::READ, prefill_tp_size);
    if (!tp_sync_result) {
        collector->success = false;
        RTP_LLM_LOG_WARNING("asyncRead: broadcast failed");
        out_error = ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "broadcast failed");
        server_call_result->cancel();
        return std::nullopt;
    }
    return AsyncReadCallResults{server_call_result, tp_sync_result};
}

}  // namespace rtp_llm
