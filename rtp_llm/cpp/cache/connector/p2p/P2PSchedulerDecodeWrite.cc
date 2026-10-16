#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.h"

#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/RouteCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include <algorithm>
#include <limits>
#include <set>

namespace rtp_llm {

P2PSchedulerDecodeWrite::P2PSchedulerDecodeWrite(P2PConnectorSchedulerConfig         config,
                                                 std::shared_ptr<P2PBroadcastClient> client,
                                                 P2PWorkerDecodeWrite&               worker,
                                                 kmonitor::MetricsReporterPtr        metrics_reporter):
    config_(std::move(config)),
    client_(std::move(client)),
    worker_(worker),
    metrics_reporter_(std::move(metrics_reporter)) {}

P2PSchedulerDecodeWrite::~P2PSchedulerDecodeWrite() {
    stop();
}

void P2PSchedulerDecodeWrite::stop() {
    checker_.cancelAll();
    if (async_write_pool_) {
        async_write_pool_->stop();
    }
    checker_.stop();
}

bool P2PSchedulerDecodeWrite::init() {
    if (async_write_pool_ || !client_ || !config_.topology || config_.p2p_writeback_timeout_ms <= 0
        || config_.load_cache_timeout_ms <= 0 || config_.p2p_lease_query_timeout_ms <= 0
        || config_.p2p_cancel_broadcast_timeout_ms <= 0 || config_.parallelism_config.tp_size <= 0
        || config_.worker_grpc_addrs.size() != static_cast<size_t>(config_.parallelism_config.tp_size)) {
        return false;
    }
    auto async_write_pool = std::make_shared<autil::LockFreeThreadPool>(
        kP2PDecodeKickoffThreadCount, kP2PDecodeKickoffQueueSize, nullptr, "P2PAsyncWriteKickoff");
    if (!async_write_pool->start() || !checker_.init()) {
        return false;
    }
    async_write_pool_ = std::move(async_write_pool);
    return true;
}

std::shared_ptr<const PlanResult> P2PSchedulerDecodeWrite::planFor(int prefill_tp_size) {
    {
        std::lock_guard<std::mutex> lock(plan_cache_mutex_);
        auto                        it = plan_cache_.find(prefill_tp_size);
        if (it != plan_cache_.end()) {
            return it->second;
        }
    }
    PlanResult plan{
        {},
        ShardLayoutFactory::validateWritebackLayout(*config_.topology, config_.parallelism_config, prefill_tp_size)};
    if (plan.ok()) {
        const auto src_layout =
            ShardLayoutFactory::fromTopology(*config_.topology, config_.parallelism_config, RoleType::DECODE);
        const auto dst_layout = ShardLayoutFactory::peerOf(src_layout, prefill_tp_size, false, RoleType::PREFILL);
        const auto tags       = ShardLayoutFactory::tagsOf(*config_.topology);
        plan                  = KVCacheTransferPlanner::plan(src_layout, dst_layout, tags);
        for (const auto& route : plan.plan.routes) {
            if (route.src_partition.count != 1 || route.dst_partition.count != 1
                || route.src_slice.mode != CpBlockSliceMode::NONE || route.dst_slice.mode != CpBlockSliceMode::NONE) {
                plan = PlanResult{{}, ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback requires whole-block routes")};
                break;
            }
        }
    }
    auto                        result = std::make_shared<const PlanResult>(std::move(plan));
    std::lock_guard<std::mutex> lock(plan_cache_mutex_);
    auto [it, inserted] = plan_cache_.emplace(prefill_tp_size, result);
    (void)inserted;
    return it->second;
}

ErrorInfo P2PSchedulerDecodeWrite::buildDecodeRankRoutes(const TransferPlan&             plan,
                                                         KVCacheResource&                resource,
                                                         size_t                          start_block,
                                                         size_t                          block_count,
                                                         P2PBroadcastClient::RankRoutes& routes,
                                                         int64_t*                        planned_bytes) const {
    if (planned_bytes) {
        *planned_bytes = 0;
    }
    if (start_block > resource.cacheKeys().size() || block_count > resource.cacheKeys().size() - start_block) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback range exceeds source keys");
    }
    for (const auto& route : plan.routes) {
        if (route.src_rank < 0 || static_cast<size_t>(route.src_rank) >= routes.size()) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback source rank out of range");
        }
        auto positions = KVCacheTransferPlanner::resolveKeys(route.src_keys, resource.cacheKeys().size());
        positions.erase(
            std::remove_if(positions.begin(),
                           positions.end(),
                           [&](size_t pos) { return pos < start_block || pos >= start_block + block_count; }),
            positions.end());
        if (positions.empty()) {
            continue;
        }
        const auto& group    = config_.topology->group(route.cache_tag);
        const auto  group_id = config_.topology->groupIdForTag(group.tag);
        if (group_id >= resource.groupNums()) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback source group missing");
        }
        for (const auto pos : positions) {
            if (pos >= resource.blocks(group_id).size() || resource.blocks(group_id)[pos] <= 0) {
                return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback source block missing");
            }
        }
        const auto buffers =
            LayerCacheBufferUtil::convertTagForRoute(resource, *config_.topology, group.tag, positions, 0, 1);
        if (!buffers.ok()) {
            return buffers.status();
        }
        if (buffers.value().size() != group.layer_ids.size()) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback source layer projection incomplete");
        }
        TransferRoutePB pb;
        RouteCodec::encodeForPrefill(route, route.dst_rank, &pb);
        for (const auto& buffer : buffers.value()) {
            if (!buffer || buffer->blockIdMap().size() != positions.size()) {
                return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback source key projection incomplete");
            }
            auto* layer = pb.add_layer_blocks();
            layer->set_layer_id(buffer->getLayerId());
            layer->set_cache_tag(buffer->cacheTag());
            for (const auto& [key, id] : buffer->blockIdMap()) {
                layer->add_cache_keys(key);
                layer->add_block_ids(id);
            }
        }
        routes[route.src_rank].push_back(std::move(pb));
        if (planned_bytes) {
            *planned_bytes += route.src_bytes * positions.size() * group.layer_ids.size();
        }
    }
    return ErrorInfo::OkStatus();
}

std::shared_ptr<P2PConnectorAsyncWriteContext> P2PSchedulerDecodeWrite::asyncWrite(KVCacheResourcePtr resource,
                                                                                   std::vector<int>   token_ids,
                                                                                   int                input_length,
                                                                                   size_t kv_ready_token_count,
                                                                                   Meta::P2PRoutingContext routing) {
    if (!async_write_pool_ || !resource || routing.unique_key.empty()) {
        return nullptr;
    }
    const auto timeout = std::min<int64_t>(
        {config_.p2p_writeback_timeout_ms, config_.load_cache_timeout_ms, std::numeric_limits<int>::max()});
    const auto budget_deadline = currentTimeMs() + timeout;
    const auto deadline = routing.deadline_ms > 0 ? std::min(budget_deadline, routing.deadline_ms) : budget_deadline;
    auto       source_group    = worker_.holdSourceResource(routing.unique_key, deadline, std::move(resource));
    if (!source_group) {
        return nullptr;
    }
    auto context = std::make_shared<P2PConnectorAsyncWriteContext>(
        routing.unique_key,
        deadline,
        client_,
        config_.p2p_cancel_broadcast_timeout_ms,
        config_.p2p_lease_query_timeout_ms,
        [source_group]() {
            source_group->cancel();
            source_group->releaseSourceResource();
        },
        metrics_reporter_);
    if (!checker_.addContext(context)) {
        context->finishWithoutTransfer(
            ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback checker stopped or duplicate key"));
        return context;
    }
    const auto accepted = async_write_pool_->pushTask(
        [this,
         context,
         source_group,
         token_ids = std::move(token_ids),
         input_length,
         kv_ready_token_count,
         routing = std::move(routing),
         deadline]() {
            startAsyncWriteCalls(
                context, source_group, token_ids, input_length, kv_ready_token_count, routing, deadline);
        },
        false,
        false);
    if (accepted != autil::ThreadPoolBase::ERROR_NONE) {
        context->finishWithoutTransfer(ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback kickoff queue unavailable"));
    }
    return context;
}

void P2PSchedulerDecodeWrite::cancel(const std::shared_ptr<P2PConnectorAsyncWriteContext>& context) {
    if (context) {
        context->cancel();
    }
}

void P2PSchedulerDecodeWrite::startAsyncWriteCalls(const std::shared_ptr<P2PConnectorAsyncWriteContext>& context,
                                                   const std::shared_ptr<p2p_internal::WriteTaskGroup>&  source_group,
                                                   const std::vector<int>&                               token_ids,
                                                   int                                                   input_length,
                                                   size_t                         kv_ready_token_count,
                                                   const Meta::P2PRoutingContext& routing,
                                                   int64_t                        deadline_ms) {
    if (!context->beginKickoff()) {
        return;
    }
    const auto invalid = [&](const std::string& message) {
        context->finishWithoutTransfer(ErrorInfo(ErrorCode::INVALID_PARAMS, message));
    };
    try {
        if (input_length < 0 || static_cast<size_t>(input_length) > kv_ready_token_count
            || kv_ready_token_count > token_ids.size()
            || token_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max()) || routing.prefill_cp_size != 1
            || formatGrpcHostPort(routing.prefill_addr.first, routing.prefill_addr.second).empty()) {
            return invalid("invalid writeback input or peer endpoint");
        }

        // step 1. generate transfer plan, including several routes from decode tp to prefill tp,
        // each route contains the src/dst rank, src/dest keys/tp/cp configurations.
        const auto planned = planFor(routing.prefill_tp_size);
        if (!planned->ok()) {
            return context->finishWithoutTransfer(planned->error);
        }
        const auto      block_size = config_.topology->groups().front().seq_size_per_block;
        const auto      blocks     = kv_ready_token_count / block_size;
        auto            owner      = source_group->sourceResource();
        if (!owner) {
            return context->finishWithoutTransfer(ErrorInfo(ErrorCode::CANCELLED, "writeback source was released"));
        }
        KVCacheResource resource = *owner;
        if (resource.cacheKeysAreCpCanonical() || resource.cacheKeys().size() < blocks) {
            return invalid("writeback source keys do not cover ready KV");
        }
        resource.cacheKeys().resize(blocks);
        resource.setLastBlockAligned(true);
        if (calculateCacheKeys(token_ids.data(), blocks * block_size, block_size) != resource.cacheKeys()
            || std::set<int64_t>(resource.cacheKeys().begin(), resource.cacheKeys().end()).size() != blocks) {
            return invalid("writeback source key hash mismatch");
        }
        if (blocks == 0) {
            return context->finishWithoutTransfer();
        }
        P2PConnectorStartWriteRequestPB request;
        request.set_request_id(routing.request_id);
        request.set_unique_key(routing.unique_key);
        request.set_input_length(input_length);
        request.set_decode_tp_size(config_.parallelism_config.tp_size);
        request.set_layout_digest(
            ShardLayoutFactory::writebackLayoutDigest(*config_.topology, config_.parallelism_config.tp_size));
        for (const auto key : resource.cacheKeys()) {
            request.add_cache_keys(key);
        }
        for (const auto token : token_ids) {
            request.add_token_ids(token);
        }

        P2PBroadcastClient::RankRoutes routes(config_.worker_grpc_addrs.size());
        int64_t                        planned_bytes   = 0;
        const size_t                   candidate_start = static_cast<size_t>(input_length) / block_size;
        const auto                     error           = buildDecodeRankRoutes(
            planned->plan, resource, candidate_start, blocks - candidate_start, routes, &planned_bytes);
        if (error.hasError()) {
            return context->finishWithoutTransfer(error);
        }
        request.set_plan_digest(planned->plan.digest());
        for (size_t rank = 0; rank < routes.size(); ++rank) {
            auto* source = request.add_decode_workers();
            source->set_worker_addr(config_.worker_grpc_addrs[rank]);
            for (auto& route : routes[rank]) {
                source->add_routes()->Swap(&route);
            }
        }
        context->setPlannedBytes(planned_bytes);
        if (!source_group->beginSourceTransfer(deadline_ms)) {
            return context->finishWithoutTransfer(ErrorInfo(ErrorCode::CANCELLED, "writeback cancelled before start"));
        }
        const auto prefill_addr = formatGrpcHostPort(routing.prefill_addr.first, routing.prefill_addr.second);
        if (!context->beginPeerCall([this,
                                     prefill_addr,
                                     key  = routing.unique_key,
                                     weak = std::weak_ptr<P2PConnectorAsyncWriteContext>(context)]() {
                const auto owner = weak.lock();
                if (!owner) {
                    return std::shared_ptr<DecodeWriteHelper::CancelResult>{};
                }
                const auto deadline  = owner->deadlineMs();
                const auto remaining = deadline + config_.p2p_lease_query_timeout_ms - currentTimeMs();
                return server_caller_.cancel(
                    prefill_addr, key, deadline, std::min(remaining, config_.p2p_cancel_broadcast_timeout_ms));
            })) {
            return context->finishWithoutTransfer(context->errorInfo());
        }
        context->setPeerCallResult(server_caller_.write(
            routing.prefill_addr.first, routing.prefill_addr.second, std::move(request), deadline_ms));
    } catch (const std::exception& e) {
        context->cancel(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, e.what()));
        context->finishWithoutTransfer(context->errorInfo());
    } catch (...) {
        context->cancel(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback kickoff failed"));
        context->finishWithoutTransfer(context->errorInfo());
    }
}

}  // namespace rtp_llm
