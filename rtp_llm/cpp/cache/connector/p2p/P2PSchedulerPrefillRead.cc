#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillRead.h"

#include "rtp_llm/cpp/cache/connector/p2p/plan/PipelineLayoutCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/RouteCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <thread>

namespace rtp_llm {

P2PSchedulerPrefillRead::P2PSchedulerPrefillRead(
    P2PConnectorSchedulerConfig                config,
    const kmonitor::MetricsReporterPtr&        metrics_reporter,
    const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client):
    config_(std::move(config)), metrics_reporter_(metrics_reporter), tp_broadcast_client_(tp_broadcast_client) {}

std::shared_ptr<const PlanResult> P2PSchedulerPrefillRead::planFor(int                decode_tp_size,
                                                                   const std::string& unique_key,
                                                                   const std::string& decode_pipeline_layout) {
    const auto offset = KVCacheTransferPlanner::sourceReplicaOffset(unique_key, config_.parallelism_config.tp_size);
    const auto key    = std::make_tuple(decode_tp_size, offset, decode_pipeline_layout);
    {
        std::lock_guard<std::mutex> lock(plan_cache_mutex_);
        auto                        it = plan_cache_.find(key);
        if (it != plan_cache_.end()) {
            return it->second;
        }
    }
    std::shared_ptr<const PlanResult> result;
    if (config_.parallelism_config.pp_size > 1 || !decode_pipeline_layout.empty()) {
        result = std::make_shared<const PlanResult>(
            planP2PPipelineTransfer(config_.pipeline_layout, decode_pipeline_layout, unique_key));
    } else {
        auto src_layout = ShardLayoutFactory::fromTopology(
            *config_.topology, config_.parallelism_config, RoleType::PREFILL, config_.cache_key_tokens_per_block);
        src_layout.use_group_cache_transfer_policy = config_.use_group_cache_transfer_policy;
        src_layout.use_mla                         = config_.use_mla;
        const auto dst_layout =
            ShardLayoutFactory::peerOf(src_layout, decode_tp_size, src_layout.cpSize() > 1, RoleType::DECODE);
        result = std::make_shared<const PlanResult>(KVCacheTransferPlanner::plan(
            src_layout, dst_layout, ShardLayoutFactory::tagsOf(*config_.topology), offset));
    }

    if (!result->ok()) {
        return result;
    }
    std::lock_guard<std::mutex> lock(plan_cache_mutex_);
    const auto                  cached = plan_cache_.find(key);
    if (cached != plan_cache_.end()) {
        return cached->second;
    }
    // Bound the cache by the number of local replica offsets. StartLoad's
    // peer TP can vary across requests.
    const auto cache_limit = decode_pipeline_layout.empty() ? std::max<size_t>(1, config_.parallelism_config.tp_size) :
                                                              std::max<size_t>(64, config_.parallelism_config.tp_size);
    if (plan_cache_.size() >= cache_limit) {
        plan_cache_.erase(plan_cache_.begin());
    }
    auto [it, inserted] = plan_cache_.emplace(key, result);
    (void)inserted;
    return it->second;
}

ErrorResult<P2PSchedulerPrefillRead::PreparedPlan>
P2PSchedulerPrefillRead::preparePlan(int                decode_tp_size,
                                     uint64_t           decode_plan_digest,
                                     const std::string& unique_key,
                                     size_t             decode_worker_count,
                                     std::set<int>      active_route_ids,
                                     const std::string& decode_pipeline_layout) {
    const bool pipeline = config_.parallelism_config.pp_size > 1 || !decode_pipeline_layout.empty();
    P2PPipelineLayoutPB decode_layout;
    if (decode_tp_size < 0 || static_cast<size_t>(decode_tp_size) > decode_worker_count
        || (pipeline && (active_route_ids.empty() || !decode_layout.ParseFromString(decode_pipeline_layout)
                         || decode_layout.worker_count() != decode_worker_count))) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                         "invalid StartLoad Decode worker list or PP layout");
    }
    if (!config_.topology) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                         "StartLoad plan digest check: cache topology is null");
    }
    if (decode_tp_size <= 0) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                         "StartLoad plan digest check: decode worker list is empty");
    }
    auto plan = planFor(decode_tp_size, unique_key, decode_pipeline_layout);
    if (!plan->ok()) {
        return ErrorInfo(plan->error.code(), "StartLoad plan digest check: " + plan->error.ToString());
    }
    const uint64_t prefill_plan_digest = plan->plan.digest();
    if (decode_plan_digest != prefill_plan_digest) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                         "StartLoad plan digest mismatch: decode=" + std::to_string(decode_plan_digest)
                             + " prefill=" + std::to_string(prefill_plan_digest)
                             + " decode_tp_size=" + std::to_string(decode_tp_size));
    }
    const auto& pc         = config_.parallelism_config;
    const auto  worker_num = config_.worker_grpc_addrs.size();
    if (pc.tp_size <= 0 || pc.pp_size <= 0 || worker_num != static_cast<size_t>(pc.tp_size) * pc.pp_size) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                         "Prefill worker list does not match PP and TP sizes");
    }
    std::set<int> unmatched_route_ids = active_route_ids;
    for (const auto& route : plan->plan.routes) {
        if (route.src_rank < 0 || static_cast<size_t>(route.src_rank) >= worker_num || route.dst_rank < 0
            || static_cast<size_t>(route.dst_rank) >= decode_worker_count) {
            return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                             "PP transfer route refers to a missing worker");
        }
        unmatched_route_ids.erase(route.route_id);
    }
    if (!unmatched_route_ids.empty()) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                         "StartLoad: active route id " + std::to_string(*unmatched_route_ids.begin())
                             + " is absent from transfer plan");
    }
    return PreparedPlan{
        std::shared_ptr<const TransferPlan>(plan, &plan->plan), prefill_plan_digest, std::move(active_route_ids)};
}

P2PBroadcastClient::RankRoutes
P2PSchedulerPrefillRead::buildPrefillRankRoutes(const TransferPlan&  plan,
                                                     size_t               worker_num,
                                                     const std::set<int>& active_route_ids) const {
    P2PBroadcastClient::RankRoutes rank_routes(worker_num);
    for (size_t worker_rank = 0; worker_rank < worker_num; ++worker_rank) {
        for (const auto* route : plan.forPrefillRank(static_cast<int>(worker_rank))) {
            if (!active_route_ids.empty() && active_route_ids.count(route->route_id) == 0) {
                continue;
            }
            TransferRoutePB pb;
            // peer_index 是 decode_transfer_servers 的下标，与 route->dst_rank 同一命名空间。
            RouteCodec::encodeForPrefill(*route, route->dst_rank, &pb);
            rank_routes[worker_rank].push_back(std::move(pb));
        }
    }
    return rank_routes;
}

ErrorInfo
P2PSchedulerPrefillRead::sendKVCache(const std::string&                                   unique_key,
                                     int64_t                                              request_id,
                                     const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers,
                                     int64_t                                              deadline_ms,
                                     const PreparedPlan*                                  prepared_plan,
                                     std::function<bool()>                                is_cancelled,
                                     int64_t                                              request_deadline_ms,
                                     int64_t                                              communication_timeout_ms) {
    RTP_LLM_LOG_DEBUG("sendKVCache start, request_id: %ld, unique_key: %s, decode_transfer_servers_size: %zu",
                      request_id,
                      unique_key.c_str(),
                      decode_transfer_servers.size());

    int64_t start_time_us      = currentTimeUs();
    auto    collector          = std::make_shared<P2PConnectorMetricsCollector>();
    auto    report_metric_func = [start_time_us, collector, metrics_reporter = metrics_reporter_](bool success) {
        collector->prefill_scheduler_total_cost_time_us = currentTimeUs() - start_time_us;
        collector->prefill_scheduler_success            = success;
        if (metrics_reporter) {
            metrics_reporter->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, collector.get());
        }
    };

    const bool no_transfer = prepared_plan == nullptr;
    const auto broadcast_type = no_transfer ? P2PConnectorBroadcastType::HANDLE_READ_NO_TRANSFER :
                                              P2PConnectorBroadcastType::HANDLE_READ;

    P2PBroadcastClient::RankRoutes rank_routes;
    uint64_t                       plan_digest = 0;
    if (prepared_plan) {
        const auto plan_start_us = currentTimeUs();
        plan_digest              = prepared_plan->digest;
        rank_routes              = buildPrefillRankRoutes(
            *prepared_plan->plan, config_.worker_grpc_addrs.size(), prepared_plan->active_route_ids);
        collector->prefill_scheduler_plan_cost_time_us = currentTimeUs() - plan_start_us;
    }

    P2PBroadcastClient::BroadcastParams params;
    params.request_id               = request_id;
    params.unique_key               = unique_key;
    params.deadline_ms         = deadline_ms;
    params.request_deadline_ms = request_deadline_ms;
    params.type                = broadcast_type;
    params.communication_timeout_ms = communication_timeout_ms;
    params.peer_workers        = decode_transfer_servers;
    // no_transfer 时 rank_routes 为空 ⇒ 所有 worker 收到同一份不带传输计划的请求。
    params.routes      = std::move(rank_routes);
    params.plan_digest = plan_digest;

    const auto                                  broadcast_start_us = currentTimeUs();
    std::shared_ptr<P2PBroadcastClient::Result> result;
    try {
        result = tp_broadcast_client_->broadcast(std::move(params));
    } catch (const std::exception& error) {
        RTP_LLM_LOG_WARNING("sendKVCache: broadcast threw, unique_key=%s, error=%s", unique_key.c_str(), error.what());
    } catch (...) {
        RTP_LLM_LOG_WARNING("sendKVCache: broadcast threw, unique_key=%s", unique_key.c_str());
    }
    collector->prefill_scheduler_broadcast_submit_time_us = currentTimeUs() - broadcast_start_us;
    if (!result) {
        std::string error_msg = "sendKVCache: broadcast failed, request_id: " + std::to_string(request_id);
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        report_metric_func(false);
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, error_msg);
    }

    bool deadline_exceeded = false;
    const auto wait_start_us          = currentTimeUs();
    auto cancel_result = waitForBroadcastCompletion(result,
                                                    unique_key,
                                                    request_id,
                                                    deadline_ms,
                                                    request_deadline_ms,
                                                    std::move(is_cancelled),
                                                    &deadline_exceeded);
    collector->prefill_scheduler_broadcast_wait_time_us = currentTimeUs() - wait_start_us;
    report_metric_func(!cancel_result && !deadline_exceeded && result->success());

    if (deadline_exceeded) {
        std::string error_msg =
            "sendKVCache: broadcast wait exceeded deadline_ms, request_id: " + std::to_string(request_id);
        const auto observed_error = result->firstError().error;
        error_msg += " unique_key=" + unique_key + " observed_error_code="
                     + std::to_string(static_cast<int>(observed_error.code()))
                     + " observed_error=" + observed_error.ToString();
        RTP_LLM_LOG_WARNING("P2P Prefill terminal, request_id=%ld unique_key=%s error_code=%d "
                            "observed_error_code=%d observed_error_name=%s observed_error=%s",
                            request_id, unique_key.c_str(),
                            static_cast<int>(ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT),
                            static_cast<int>(observed_error.code()), ErrorCodeToString(observed_error.code()).c_str(),
                            observed_error.ToString().c_str());
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT, error_msg);
    }

    if (cancel_result) {
        std::string error_msg = "sendKVCache: cancelled by client, request_id: " + std::to_string(request_id);
        const auto observed_error = result->firstError().error;
        error_msg += " unique_key=" + unique_key + " observed_error_code="
                     + std::to_string(static_cast<int>(observed_error.code()))
                     + " observed_error=" + observed_error.ToString();
        RTP_LLM_LOG_WARNING("P2P Prefill terminal, request_id=%ld unique_key=%s error_code=%d "
                            "observed_error_code=%d observed_error_name=%s observed_error=%s",
                            request_id, unique_key.c_str(),
                            static_cast<int>(ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED),
                            static_cast<int>(observed_error.code()), ErrorCodeToString(observed_error.code()).c_str(),
                            observed_error.ToString().c_str());
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
P2PSchedulerPrefillRead::waitForBroadcastCompletion(const std::shared_ptr<P2PBroadcastClient::Result>& result,
                                                         const std::string&                                 unique_key,
                                                         int64_t                                            request_id,
                                                         int64_t                                            deadline_ms,
                                                         int64_t                                            request_deadline_ms,
                                                         std::function<bool()>                              is_cancelled,
                                                         bool*                                              deadline_exceeded_out) {

    std::shared_ptr<P2PBroadcastClient::Result> cancel_result = nullptr;
    int                                         sleep_ms      = 1;
    constexpr int                               kBackoffCapMs = 8;
    const auto                                  cancel_workers = [&] {
        try {
            return tp_broadcast_client_->cancel(unique_key,
                                                P2PConnectorBroadcastType::CANCEL_HANDLE_READ,
                                                request_deadline_ms,
                                                request_id,
                                                deadline_ms);
        } catch (const std::exception& error) {
            RTP_LLM_LOG_WARNING(
                "sendKVCache: cancel broadcast threw, unique_key=%s, error=%s", unique_key.c_str(), error.what());
        } catch (...) {
            RTP_LLM_LOG_WARNING("sendKVCache: cancel broadcast threw, unique_key=%s", unique_key.c_str());
        }
        return std::shared_ptr<P2PBroadcastClient::Result>{};
    };
    while (!result->done()) {
        result->checkDone();
        if (!cancel_result && is_cancelled && is_cancelled()) {
            RTP_LLM_LOG_WARNING("sendKVCache: request cancelled by client, request_id: %ld, unique_key: %s",
                                request_id,
                                unique_key.c_str());
            cancel_result = cancel_workers();
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
            cancel_result = cancel_workers();
            if (deadline_exceeded_out) {
                *deadline_exceeded_out = true;
            }
            if (!cancel_result) {
                // The transfer deadline remains authoritative even when the
                // cancel broadcast itself cannot be started.
                cancel_result = std::make_shared<P2PBroadcastClient::Result>(unique_key);
            }
        }
        // Cancellation is safety cleanup; do not add its RPC budget to load waiting.
        if (cancel_result) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        sleep_ms = std::min(sleep_ms * 2, kBackoffCapMs);
    }

    // The HANDLE_READ RPC and the transfer share the same deadline. gRPC may
    // therefore publish DEADLINE_EXCEEDED immediately before this polling
    // loop observes the clock crossing deadline_ms. In that ordering the
    // loop exits through result->done() and would otherwise skip the
    // best-effort CANCEL_HANDLE_READ cleanup entirely.
    if (!cancel_result && !result->success() && currentTimeMs() >= deadline_ms) {
        RTP_LLM_LOG_WARNING(
            "sendKVCache: broadcast completed unsuccessfully at deadline_ms=%ld, cancelling, request_id: %ld, "
            "unique_key: %s",
            deadline_ms,
            request_id,
            unique_key.c_str());
        cancel_result = cancel_workers();
        if (deadline_exceeded_out) {
            *deadline_exceeded_out = true;
        }
        if (!cancel_result) {
            cancel_result = std::make_shared<P2PBroadcastClient::Result>(unique_key);
        }
    }
    return cancel_result;
}

}  // namespace rtp_llm
