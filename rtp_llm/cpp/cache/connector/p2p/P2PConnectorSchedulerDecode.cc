#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorSchedulerDecode.h"

#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/RouteCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "autil/LockFreeThreadPool.h"
#include <algorithm>
#include <memory>
#include <optional>

namespace rtp_llm {
namespace {
// Keep kickoff bounded: connection acquisition/reconnection may block, while
// completion polling remains isolated on P2PConnectorAsyncReadContextChecker.
constexpr size_t kAsyncReadThreadCount = 4;
constexpr size_t kAsyncReadQueueSize   = 1024;
}  // namespace

P2PConnectorSchedulerDecode::P2PConnectorSchedulerDecode(
    P2PConnectorSchedulerConfig                config,
    const kmonitor::MetricsReporterPtr&        metrics_reporter,
    const std::shared_ptr<P2PBroadcastClient>& tp_broadcast_client):
    config_(std::move(config)), metrics_reporter_(metrics_reporter), tp_broadcast_client_(tp_broadcast_client) {}

P2PConnectorSchedulerDecode::~P2PConnectorSchedulerDecode() {
    if (async_read_pool_) {
        async_read_pool_->stop(autil::ThreadPool::STOP_AFTER_QUEUE_EMPTY);
        async_read_pool_->join();
        async_read_pool_.reset();
    }
    if (checker_) {
        checker_->stop();
    }
}

bool P2PConnectorSchedulerDecode::init(const std::string& process_id) {
    server_caller_ = std::make_shared<PrefillLoadCaller>(config_.worker_addrs);

    auto async_read_pool = std::make_shared<autil::LockFreeThreadPool>(
        kAsyncReadThreadCount, kAsyncReadQueueSize, nullptr, "P2PAsyncReadKickoff");
    if (!async_read_pool->start()) {
        RTP_LLM_LOG_ERROR("P2PConnectorSchedulerDecode init failed: async read pool start failed");
        return false;
    }
    async_read_pool_ = std::move(async_read_pool);

    checker_ = std::make_shared<P2PConnectorAsyncReadContextChecker>();
    if (!checker_->init(metrics_reporter_, tp_broadcast_client_)) {
        RTP_LLM_LOG_ERROR("P2PConnectorSchedulerDecode init failed: checker init failed");
        async_read_pool_->stop();
        async_read_pool_.reset();
        return false;
    }

    return true;
}

ErrorInfo P2PConnectorSchedulerDecode::checkPeerCpLayout(int prefill_tp_size, int prefill_cp_size) const {
    const auto& cp_cfg = config_.parallelism_config.prefill_cp_config;

    // kv_cache_sharded 是部署级同配开关（OpaqueKVCacheSpec::fixedRegionCpSize 的 DECODE
    // 分支已依赖此前提），因此本端可以推出对端应有的 CP 片数。
    const int derived = cp_cfg.kv_cache_sharded ? prefill_tp_size : 1;
    if (prefill_cp_size != derived) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                         "peer CP layout drift: reported prefill_cp_size=" + std::to_string(prefill_cp_size)
                             + " but local config derives " + std::to_string(derived)
                             + " (kv_cache_sharded=" + std::to_string(cp_cfg.kv_cache_sharded ? 1 : 0)
                             + ", prefill_tp_size=" + std::to_string(prefill_tp_size) + ")");
    }
    // 配置里显式写了 prefill_cp_size 时，它也必须与上报值一致。
    if (cp_cfg.prefill_cp_size > 0 && static_cast<int>(cp_cfg.prefill_cp_size) != prefill_cp_size) {
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                         "peer CP layout drift: reported prefill_cp_size=" + std::to_string(prefill_cp_size)
                             + " but local prefill_cp_config.prefill_cp_size="
                             + std::to_string(cp_cfg.prefill_cp_size));
    }
    return ErrorInfo::OkStatus();
}

std::shared_ptr<const PlanResult> P2PConnectorSchedulerDecode::planFor(int prefill_tp_size, int prefill_cp_size) {
    const auto key = std::make_pair(prefill_tp_size, prefill_cp_size);
    {
        std::lock_guard<std::mutex> lock(plan_cache_mutex_);
        auto                        it = plan_cache_.find(key);
        if (it != plan_cache_.end()) {
            return it->second;
        }
    }

    // 本端布局从真实 CacheTopology + ParallelismConfig 取；对端布局由本端推导
    // （ShardLayoutFactory::peerOf），因此 GetPeerInfo / StartLoad 无需新增字段。
    const auto dst_layout = ShardLayoutFactory::fromTopology(
        *config_.topology, config_.parallelism_config, RoleType::DECODE);
    const auto src_layout = ShardLayoutFactory::peerOf(dst_layout,
                                                       prefill_tp_size,
                                                       config_.parallelism_config.prefill_cp_config.kv_cache_sharded,
                                                       RoleType::PREFILL);
    const auto tags = ShardLayoutFactory::tagsOf(*config_.topology);

    auto result = std::make_shared<const PlanResult>(KVCacheTransferPlanner::plan(src_layout, dst_layout, tags));

    std::lock_guard<std::mutex> lock(plan_cache_mutex_);
    auto [it, inserted] = plan_cache_.emplace(key, result);
    (void)inserted;
    return it->second;
}

P2PBroadcastClient::RankRoutes
P2PConnectorSchedulerDecode::buildDecodeRankRoutes(const TransferPlan&        plan,
                                                   KVCacheResource&           resource,
                                                   const std::pair<int, int>& block_range,
                                                   size_t                     worker_num) const {
    P2PBroadcastClient::RankRoutes rank_routes(worker_num);

    // logical_count 必须是**全序列**的 cache_keys 数量，不是 block_range 窗口长度：
    // prefill 侧不知道 decode 的 block_range（prefix 部分命中的结果），两侧若用不同的
    // count，include_final_key 与 tail_count 会算出不同的键，破坏键集包含契约。
    const size_t logical_count = resource.cacheKeys().size();
    const size_t window_begin  = static_cast<size_t>(std::max(0, block_range.first));
    const size_t window_end    = block_range.second > 0 ?
                                     std::min(logical_count, window_begin + static_cast<size_t>(block_range.second)) :
                                     logical_count;

    for (size_t worker_rank = 0; worker_rank < worker_num; ++worker_rank) {
        for (const auto* route : plan.forDecodeRank(static_cast<int>(worker_rank))) {
            auto positions = KVCacheTransferPlanner::resolveKeys(route->src_keys, logical_count);
            // block_range 窗口只在 decode 侧叠加在 resolveKeys 结果之上。
            positions.erase(std::remove_if(positions.begin(),
                                           positions.end(),
                                           [&](size_t pos) { return pos < window_begin || pos >= window_end; }),
                            positions.end());
            if (positions.empty()) {
                // 该 route 在本请求长度 / 窗口下解析为空是预期行为（两侧规则相同故一致判空）。
                continue;
            }
            auto layer_buffers = LayerCacheBufferUtil::convertTagForRoute(resource,
                                                                         *config_.topology,
                                                                         route->cache_tag,
                                                                         positions,
                                                                         config_.cp_rank,
                                                                         config_.cp_size);
            if (layer_buffers.empty()) {
                continue;
            }
            TransferRoutePB pb;
            RouteCodec::encodeForDecode(*route, &pb);
            for (const auto& buffer : layer_buffers) {
                auto* layer_block = pb.add_layer_blocks();
                layer_block->set_layer_id(buffer->getLayerId());
                layer_block->set_cache_tag(buffer->cacheTag());
                for (const auto& [key, block_id] : buffer->blockIdMap()) {
                    layer_block->add_cache_keys(key);
                    layer_block->add_block_ids(block_id);
                }
            }
            rank_routes[worker_rank].push_back(std::move(pb));
        }
    }
    return rank_routes;
}

void P2PConnectorSchedulerDecode::stopChecker() {
    if (checker_) {
        checker_->stop();
    }
}

void P2PConnectorSchedulerDecode::cancel(const std::shared_ptr<P2PConnectorAsyncReadContext>& context) {
    if (context) {
        context->cancel(tp_broadcast_client_);
    }
}

P2PConnectorSchedulerDecode::AsyncReadResult P2PConnectorSchedulerDecode::asyncRead(
    const KVCacheResourcePtr& resource,
    const std::shared_ptr<Meta>& meta,
    const std::pair<int, int>& block_range,
    bool no_transfer) {
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

    const int64_t     request_id          = routing->request_id;
    const std::string unique_key          = routing->unique_key;
    const int64_t     request_deadline_ms = routing->deadline_ms;
    const int64_t     now_ms              = currentTimeMs();
    if (request_deadline_ms <= 0 || now_ms >= request_deadline_ms) {
        RTP_LLM_LOG_WARNING("asyncRead: request deadline expired, unique_key: %s", unique_key.c_str());
        return {nullptr, ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "P2P request deadline expired")};
    }
    // StartLoad carries both deadlines: deadline_ms bounds this physical
    // transfer, while request_deadline_ms only bounds request-level terminal
    // state such as Prefill resource tombstones.
    const int64_t     transfer_deadline_ms =
        std::min(request_deadline_ms, now_ms + config_.p2p_max_transfer_deadline_ms);
    const auto& prefill_addr    = routing->prefill_addr;
    const int   prefill_tp_size = routing->prefill_tp_size;
    const int   prefill_cp_size = routing->prefill_cp_size;

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
    if (!no_transfer && !config_.topology) {
        collector->success = false;
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "cache topology is null")};
    }
    if (!no_transfer) {
        // 漂移断言（设计文档 §3.2.4）：这是「对端 layout 由本端推导」唯一的保护，
        // 把配置漂移从传输期的疑难杂症变成入口处的确定性报错。
        const auto drift = checkPeerCpLayout(prefill_tp_size, prefill_cp_size);
        if (drift.hasError()) {
            RTP_LLM_LOG_WARNING("asyncRead: %s", drift.ToString().c_str());
            collector->success = false;
            return {nullptr, drift};
        }
    }
    // 编排层：计算传输计划并投影成每个 worker 的 route 列表。route 现在**驱动执行**，
    // §2.1 那条「两侧 CP 必须相等」的硬拒因此删除 —— 非对称 CP 由 planner 的白名单
    // （dst.cpSize() ∈ {1, src.cpSize()}）判定，不再由这里一刀切。
    P2PBroadcastClient::RankRoutes rank_routes;
    uint64_t                       plan_digest = 0;
    const size_t                   worker_num  = config_.worker_grpc_addrs.size();
    if (!no_transfer) {
        if (worker_num == 0) {
            collector->success = false;
            return {nullptr,
                    ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED, "worker list is empty")};
        }
        auto plan = planFor(prefill_tp_size, prefill_cp_size);
        if (!plan->ok()) {
            RTP_LLM_LOG_WARNING("asyncRead: transfer plan failed, unique_key=%s, error=%s",
                                unique_key.c_str(),
                                plan->error.ToString().c_str());
            collector->success = false;
            return {nullptr, plan->error};
        }
        plan_digest = plan->plan.digest();
        rank_routes = buildDecodeRankRoutes(plan->plan, *resource, block_range, worker_num);

        const bool all_routes_empty =
            std::all_of(rank_routes.begin(), rank_routes.end(), [](const auto& routes) { return routes.empty(); });
        if (all_routes_empty) {
            RTP_LLM_LOG_WARNING("asyncRead: transfer plan resolved to no routes for any worker, unique_key=%s",
                                unique_key.c_str());
            collector->success = false;
            return {nullptr,
                    ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                              "transfer plan resolved to no routes")};
        }
    }
    // layer_blocks 已随 route 逐条下发（TransferRoutePB::layer_blocks），此处只留一个
    // 与 worker 数等长的空壳以满足 broadcastPerRank 的形状校验。旧的 per-rank 投影
    // （worker_rank % cp_size）连同 LayerCacheBufferUtil::convert 的调用一并退场。
    P2PBroadcastClient::RankLayerCacheBuffers rank_layer_cache_buffers(no_transfer ? 0 : worker_num);

    auto async_context = std::make_shared<P2PConnectorAsyncReadContext>(resource,
                                                                        unique_key,
                                                                        collector,
                                                                        config_.p2p_transfer_not_done_resource_hold_ms,
                                                                        no_transfer,
                                                                        request_deadline_ms);

    auto submit_result = async_read_pool_->pushTask(
        [this,
         request_id,
         prefill_ip = prefill_addr.first,
         prefill_port = prefill_addr.second,
         unique_key,
         request_deadline_ms,
         transfer_deadline_ms,
         rank_layer_cache_buffers = std::move(rank_layer_cache_buffers),
         rank_routes = std::move(rank_routes),
         plan_digest,
         collector,
         async_context,
         prefill_tp_size,
         no_transfer]() mutable {
            if (!async_context->beginKickoff()) {
                return;
            }

            ErrorInfo start_error;
            auto      async_calls = startAsyncReadCalls(request_id,
                                                        prefill_ip,
                                                        prefill_port,
                                                        unique_key,
                                                        request_deadline_ms,
                                                        transfer_deadline_ms,
                                                        rank_layer_cache_buffers,
                                                        collector,
                                                        start_error,
                                                        prefill_tp_size,
                                                        no_transfer,
                                                        rank_routes,
                                                        plan_digest);
            if (!async_calls) {
                async_context->markStartFailed(start_error);
                return;
            }

            const bool cancel_requested =
                async_context->setCallResults(async_calls->tp_sync_result, async_calls->server_call_result);
            if (cancel_requested) {
                async_context->cancel(tp_broadcast_client_);
            }
        },
        false);
    if (submit_result != autil::ThreadPoolBase::ERROR_NONE) {
        collector->success = false;
        return {nullptr,
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                          "failed to submit P2P async read kickoff task")};
    }

    // addContext only publishes the pending context. The potentially slow RPC
    // kickoff runs in async_read_pool_ and never holds async_contexts_mutex_.
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
    int64_t                                                 request_deadline_ms,
    int64_t                                                 transfer_deadline_ms,
    const P2PBroadcastClient::RankLayerCacheBuffers&        rank_layer_cache_buffers,
    const std::shared_ptr<DecodeSchedulerMetricsCollector>& collector,
    ErrorInfo&                                              out_error,
    int                                                     prefill_tp_size,
    bool                                                    no_transfer,
    const P2PBroadcastClient::RankRoutes&                   rank_routes,
    uint64_t                                                plan_digest) {

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
    auto server_call_result = server_caller_->load(request_id,
                                                   prefill_ip,
                                                   prefill_port,
                                                   unique_key,
                                                   request_deadline_ms,
                                                   transfer_deadline_ms,
                                                   no_transfer);
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
    std::shared_ptr<P2PBroadcastClient::Result> tp_sync_result;
    if (no_transfer) {
        tp_sync_result = std::make_shared<P2PBroadcastClient::Result>(unique_key);
    } else {
        tp_sync_result = tp_broadcast_client_->broadcastPerRank(request_id,
                                                                rank_layer_cache_buffers,
                                                                {},
                                                                unique_key,
                                                                transfer_deadline_ms,
                                                                P2PConnectorBroadcastType::READ,
                                                                request_deadline_ms,
                                                                rank_routes,
                                                                plan_digest);
    }
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
