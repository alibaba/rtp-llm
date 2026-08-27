#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerPrefill.h"

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/RouteCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
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

std::shared_ptr<const PlanResult> P2PConnectorSchedulerPrefill::planFor(int decode_tp_size) {
    {
        std::lock_guard<std::mutex> lock(plan_cache_mutex_);
        auto                        it = plan_cache_.find(decode_tp_size);
        if (it != plan_cache_.end()) {
            return it->second;
        }
    }
    // 与 decode 侧同一个 planner 函数、镜像的参数。decode 的 cp_size 由部署级同配开关推导。
    const auto src_layout = ShardLayoutFactory::fromTopology(
        *config_.topology, config_.parallelism_config, RoleType::PREFILL);
    const auto dst_layout = ShardLayoutFactory::peerOf(src_layout,
                                                      decode_tp_size,
                                                      config_.parallelism_config.prefill_cp_config.kv_cache_sharded,
                                                      RoleType::DECODE);
    const auto tags   = ShardLayoutFactory::tagsOf(*config_.topology);
    auto       result = std::make_shared<const PlanResult>(KVCacheTransferPlanner::plan(src_layout, dst_layout, tags));

    std::lock_guard<std::mutex> lock(plan_cache_mutex_);
    auto [it, inserted] = plan_cache_.emplace(decode_tp_size, result);
    (void)inserted;
    return it->second;
}

P2PBroadcastClient::RankRoutes
P2PConnectorSchedulerPrefill::buildPrefillRankRoutes(const TransferPlan& plan, size_t worker_num) const {
    P2PBroadcastClient::RankRoutes rank_routes(worker_num);
    for (size_t worker_rank = 0; worker_rank < worker_num; ++worker_rank) {
        for (const auto* route : plan.forPrefillRank(static_cast<int>(worker_rank))) {
            TransferRoutePB pb;
            // peer_index 是 decode_transfer_servers 的下标，与 route->dst_rank 同一命名空间。
            RouteCodec::encodeForPrefill(*route, route->dst_rank, &pb);
            rank_routes[worker_rank].push_back(std::move(pb));
        }
    }
    return rank_routes;
}

ErrorInfo
P2PConnectorSchedulerPrefill::sendKVCache(const KVCacheResourcePtr&                            resource,
                                          const std::string&                                   unique_key,
                                          int64_t                                              request_id,
                                          const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                                          int64_t                                              deadline_ms,
                                          std::function<bool()>                                is_cancelled,
                                          bool                                                 no_transfer,
                                          int64_t                                              request_deadline_ms) {
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

    if (!no_transfer && !config_.topology) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                         "sendKVCache: cache topology is null");
    }
    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    if (!no_transfer) {
        layer_cache_buffers =
            LayerCacheBufferUtil::convert(*resource, *config_.topology, 0, -1, config_.cp_rank, config_.cp_size);
    }
    if (!no_transfer && layer_cache_buffers.empty()) {
        std::string error_msg = "sendKVCache: layer_cache_buffers is empty, request_id: " + std::to_string(request_id);
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        report_metric_func(false);
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, error_msg);
    }

    const auto broadcast_type = no_transfer ? P2PConnectorBroadcastType::HANDLE_READ_NO_TRANSFER :
                                              P2PConnectorBroadcastType::HANDLE_READ;

    // 编排层：与 decode 侧用同一个 planner 算镜像 plan，逐 prefill worker 下发它自己的 route。
    const size_t                   worker_num = config_.worker_grpc_addrs.size();
    P2PBroadcastClient::RankRoutes rank_routes;
    uint64_t                       plan_digest = 0;
    if (!no_transfer) {
        if (worker_num == 0) {
            report_metric_func(false);
            return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "worker list is empty");
        }
        auto plan = planFor(static_cast<int>(decode_transfer_servers.size()));
        if (!plan->ok()) {
            RTP_LLM_LOG_WARNING("sendKVCache: transfer plan failed, unique_key=%s, error=%s",
                                unique_key.c_str(),
                                plan->error.ToString().c_str());
            report_metric_func(false);
            return plan->error;
        }
        plan_digest = plan->plan.digest();
        rank_routes = buildPrefillRankRoutes(plan->plan, worker_num);
    }

    // broadcastPerRank 要求 buffer 数组与 worker 数等长；prefill 侧的 layer_blocks 不上线
    // （worker 用自身投影），故只传等长空壳。
    P2PBroadcastClient::RankLayerCacheBuffers rank_layer_cache_buffers(no_transfer ? 0 : worker_num);
    auto result = no_transfer ? tp_broadcast_client_->broadcast(request_id,
                                                               layer_cache_buffers,
                                                               decode_transfer_servers,
                                                               unique_key,
                                                               deadline_ms,
                                                               broadcast_type,
                                                               request_deadline_ms) :
                                tp_broadcast_client_->broadcastPerRank(request_id,
                                                                      rank_layer_cache_buffers,
                                                                      decode_transfer_servers,
                                                                      unique_key,
                                                                      deadline_ms,
                                                                      broadcast_type,
                                                                      request_deadline_ms,
                                                                      rank_routes,
                                                                      plan_digest);
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
            if (!cancel_result) {
                // Cancellation is already terminal for this StartLoad. Do not
                // keep waiting for the original HANDLE_READ merely because the
                // best-effort cancel RPC could not be created.
                cancel_result = std::make_shared<P2PBroadcastClient::Result>(unique_key);
            }
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
            if (!cancel_result) {
                // The transfer deadline remains authoritative even when the
                // cancel broadcast itself cannot be started.
                cancel_result = std::make_shared<P2PBroadcastClient::Result>(unique_key);
            }
        }
        if (cancel_result && !cancel_result->done()) {
            cancel_result->checkDone();
        }
        // Once cancel completed (either acked by all workers or its own gRPC
        // timeout fired ~1s later), there is nothing the scheduler can gain by
        // continuing to poll `result`: worker-side cleanup runs independently
        // off worker's own return_deadline_ms, and the upstream caller only
        // inspects cancel_result / deadline_exceeded_out for the final error
        // code. Without this break, a hung broadcast channel kept the loop
        // spinning until the client business deadline (~1h) — see 5/22 PD log
        // analysis, B-class 7 cases > 60s.
        if (cancel_result && cancel_result->done()) {
            RTP_LLM_LOG_WARNING("sendKVCache: cancel completed, exiting wait early "
                                "(deadline_exceeded=%d), request_id=%ld, unique_key=%s",
                                (deadline_exceeded_out && *deadline_exceeded_out) ? 1 : 0,
                                request_id,
                                unique_key.c_str());
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        sleep_ms = std::min(sleep_ms * 2, kBackoffCapMs);
    }
    return cancel_result;
}

}  // namespace rtp_llm
