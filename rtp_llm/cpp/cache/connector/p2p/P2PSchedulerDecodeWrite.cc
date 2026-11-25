#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.h"

#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>

namespace rtp_llm {
namespace {
constexpr size_t kAsyncWriteThreadCount = 4;
constexpr size_t kAsyncWriteQueueSize   = 1024;
}  // namespace

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

void P2PSchedulerDecodeWrite::stop(int64_t steady_deadline_ms) {
    const auto deadline = writebackShutdownDeadline(steady_deadline_ms);
    checker_.cancelAll();
    if (async_write_pool_) {
        async_write_pool_->stop();
    }
    checker_.stop(deadline);
}

bool P2PSchedulerDecodeWrite::init() {
    if (async_write_pool_ || !client_ || !config_.topology
        || config_.load_cache_timeout_ms <= 0 || config_.p2p_lease_query_timeout_ms <= 0
        || config_.p2p_cancel_broadcast_timeout_ms <= 0 || config_.parallelism_config.tp_size <= 0
        || config_.worker_grpc_addrs.size()
               != static_cast<size_t>(config_.parallelism_config.tp_size * config_.parallelism_config.pp_size)) {
        return false;
    }
    auto async_write_pool = std::make_shared<autil::LockFreeThreadPool>(
        kAsyncWriteThreadCount, kAsyncWriteQueueSize, nullptr, "P2PAsyncWriteKickoff");
    if (!async_write_pool->start() || !checker_.init(config_, metrics_reporter_)) {
        return false;
    }
    async_write_pool_ = std::move(async_write_pool);
    return true;
}

std::shared_ptr<const PlanResult>
P2PSchedulerDecodeWrite::planFor(int tp, int cp, const std::string& layout, const std::string& unique_key) {
    return planner_.planFor(tp, cp, layout, 0, unique_key);
}

ErrorInfo P2PSchedulerDecodeWrite::buildDecodeRankRoutes(const TransferPlan&             plan,
                                                         KVCacheResource&                resource,
                                                         size_t                          start_block,
                                                         size_t                          block_count,
                                                         P2PBroadcastClient::RankRoutes& routes,
                                                         int64_t*                        planned_bytes) const {
    return planner_.buildRankRoutes(plan, resource, start_block, block_count, 1, routes, planned_bytes);
}

std::optional<P2PSchedulerDecodeWrite::WriteRange> P2PSchedulerDecodeWrite::writeRange(
    int input_length, size_t kv_ready_token_count, const Meta::P2PRoutingContext& routing) const {
    const int peer_cp = routing.prefill_cp_size;
    if (input_length < 0 || static_cast<size_t>(input_length) > kv_ready_token_count
        || routing.prefill_tp_size <= 0 || (peer_cp != 1 && peer_cp != routing.prefill_tp_size)
        || config_.cache_key_tokens_per_block == 0) {
        return std::nullopt;
    }
    // P reuses whole CP units, so both ends round down to one; the unit straddling the prompt is resent whole.
    const size_t block_size = config_.cache_key_tokens_per_block;
    const size_t cp_size    = peer_cp;
    return WriteRange{static_cast<size_t>(input_length) / block_size / cp_size * cp_size,
                      kv_ready_token_count / block_size / cp_size * cp_size};
}

ErrorInfo P2PSchedulerDecodeWrite::selectSourceBlocks(KVCacheResource&               resource,
                                                      int                            input_length,
                                                      size_t                         kv_ready_token_count,
                                                      const Meta::P2PRoutingContext& routing) {
    const auto range = writeRange(input_length, kv_ready_token_count, routing);
    if (!range || !config_.topology || resource.cacheKeysAreCpCanonical()
        || resource.groupNums() != config_.topology->groups().size()) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "invalid writeback source snapshot");
    }
    const bool use_positions = resource.cacheKeys().empty();
    if (!use_positions && range->end_block > resource.cacheKeys().size()) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback source keys do not cover ready KV");
    }
    const auto planned =
        planFor(routing.prefill_tp_size, routing.prefill_cp_size, routing.prefill_pipeline_layout, routing.unique_key);
    if (!planned->ok()) {
        return planned->error;
    }
    resource.cacheKeys().resize(range->end_block);
    if (use_positions) {
        // The projection only needs unique logical labels. Rebuild real keys on the kickoff worker.
        std::iota(resource.cacheKeys().begin(), resource.cacheKeys().end(), int64_t{0});
    }
    // All layers of a group share one block table, so the positions sent per tag decide which blocks to keep.
    std::map<std::string, std::set<size_t>> sent_positions;
    for (const auto& route : planned->plan.routes) {
        auto& positions = sent_positions[route.cache_tag];
        for (const auto pos : KVCacheTransferPlanner::resolveKeys(route.src_keys, range->end_block)) {
            if (pos >= range->start_block && pos < range->end_block) {
                positions.insert(pos);
            }
        }
    }
    for (const auto& group : config_.topology->groups()) {
        std::set<BlockIdxType> retained;
        const auto             sent = sent_positions.find(group.tag);
        if (sent != sent_positions.end() && !sent->second.empty()) {
            const std::vector<size_t> positions(sent->second.begin(), sent->second.end());
            const auto                buffer = LayerCacheBufferUtil::convertLayerTagForRoute(resource,
                                                                              group,
                                                                              /*layer_id=*/0,
                                                                              positions,
                                                                              0,
                                                                              1,
                                                                              config_.cache_key_tokens_per_block,
                                                                              /*shared_pp_block_table=*/true);
            if (!buffer.ok()) {
                return buffer.status();
            }
            for (const auto& [key, block_id] : buffer.value()->blockIdMap()) {
                retained.insert(block_id);
            }
        }
        auto block_ids = resource.blocks(group.tag);
        for (auto& block_id : block_ids) {
            if (!retained.count(block_id)) {
                block_id = NULL_BLOCK_IDX;
            }
        }
        while (!block_ids.empty() && isNullBlockIdx(block_ids.back())) {
            block_ids.pop_back();
        }
        resource.mutableBlockIds(group.tag).assign(std::move(block_ids));
    }
    if (use_positions) {
        resource.cacheKeys().clear();
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
    const auto timeout =
        std::min<int64_t>(config_.load_cache_timeout_ms, std::numeric_limits<int>::max());
    const auto deadline     = currentTimeMs() + timeout;
    auto       source_group = worker_.holdSourceResource(routing.unique_key, deadline, std::move(resource));
    if (!source_group) {
        return nullptr;
    }
    const auto release_source = [source_group]() {
        source_group->cancel();
        source_group->releaseSourceResource();
    };
    std::shared_ptr<P2PConnectorAsyncWriteContext> context;
    try {
        context = std::make_shared<P2PConnectorAsyncWriteContext>(routing.unique_key,
                                                                  deadline,
                                                                  client_,
                                                                  config_.p2p_cancel_broadcast_timeout_ms,
                                                                  config_.p2p_lease_query_timeout_ms,
                                                                  release_source,
                                                                  metrics_reporter_,
                                                                  source_group->sourceResource());
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
             routing = std::move(routing)]() {
                startAsyncWriteCalls(context, source_group, token_ids, input_length, kv_ready_token_count, routing);
            },
            false,
            false);
        if (accepted != autil::ThreadPoolBase::ERROR_NONE) {
            context->finishWithoutTransfer(
                ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback kickoff queue unavailable"));
        }
    } catch (...) {
        // Without a queued kickoff nothing else would ever release the held source.
        if (context) {
            context->finishWithoutTransfer(
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback submission failed"));
        } else {
            release_source();
        }
        throw;
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
                                                   const Meta::P2PRoutingContext& routing) {
    if (!context->beginKickoff()) {
        return;
    }
    const auto invalid = [&](const std::string& message) {
        context->finishWithoutTransfer(ErrorInfo(ErrorCode::INVALID_PARAMS, message));
    };
    try {
        const auto range = writeRange(input_length, kv_ready_token_count, routing);
        if (!range || kv_ready_token_count > token_ids.size()
            || token_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max())
            || formatGrpcHostPort(routing.prefill_addr.first, routing.prefill_addr.second).empty()) {
            return invalid("invalid writeback input or peer endpoint");
        }

        // step 1. generate transfer plan, including several routes from decode tp to prefill tp,
        // each route contains the src/dst rank, src/dest keys/tp/cp configurations.
        const auto                     plan_start_us = currentTimeUs();
        const auto                     planned       = planFor(
            routing.prefill_tp_size, routing.prefill_cp_size, routing.prefill_pipeline_layout, routing.unique_key);
        WritebackStageMetricsCollector plan_metrics;
        plan_metrics.stage        = "plan";
        plan_metrics.cost_time_us = currentTimeUs() - plan_start_us;
        plan_metrics.error        = planned->error;
        reportWritebackMetrics(metrics_reporter_, plan_metrics);
        if (!planned->ok()) {
            return context->finishWithoutTransfer(planned->error);
        }
        const auto block_size      = config_.cache_key_tokens_per_block;
        const auto blocks          = range->end_block;
        const auto candidate_start = range->start_block;
        if (blocks <= candidate_start) {
            return context->finishWithoutTransfer();
        }
        auto owner = source_group->sourceResource();
        if (!owner) {
            return context->finishWithoutTransfer(ErrorInfo(ErrorCode::CANCELLED, "writeback source was released"));
        }
        KVCacheResource resource = *owner;
        auto keys = calculateCacheKeys(token_ids.data(), blocks * block_size, block_size, config_.cache_key_hash_seed);
        if (resource.cacheKeysAreCpCanonical()
            || (!resource.cacheKeys().empty()
                && (resource.cacheKeys().size() < blocks
                    || !std::equal(keys.begin(), keys.end(), resource.cacheKeys().begin())))
            || std::set<int64_t>(keys.begin(), keys.end()).size() != blocks) {
            return invalid("writeback source key hash mismatch");
        }
        resource.setCacheKeys(std::move(keys));
        resource.setLastBlockAligned(true);

        // STARTWRITE RPC
        P2PConnectorStartWriteBackRequestPB request;
        request.set_request_id(routing.request_id);
        request.set_unique_key(routing.unique_key);
        request.set_input_length(input_length);
        request.set_decode_tp_size(config_.parallelism_config.tp_size);
        request.set_writeback_kv_step(routing.writeback_kv_step);
        if (!routing.prefill_pipeline_layout.empty()) {
            request.set_decode_pipeline_layout(config_.pipeline_layout);
        }
        for (const auto key : resource.cacheKeys()) {
            request.add_cache_keys(key);
        }
        const size_t ready_tokens = blocks * block_size;
        request.mutable_token_ids()->Reserve(static_cast<int>(ready_tokens));
        for (size_t index = 0; index < ready_tokens; ++index) {
            request.add_token_ids(token_ids[index]);
        }

        // step 2. generate the actual routes for each decode worker to prefill worker, including the block ids and
        // layer ids.
        P2PBroadcastClient::RankRoutes routes(config_.worker_grpc_addrs.size());
        int64_t                        planned_bytes = 0;
        const auto                     error         = buildDecodeRankRoutes(
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
        if (!source_group->beginSourceTransfer(context->deadlineMs())) {
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
                const auto deadline = owner->deadlineMs();
                return server_caller_.cancel(
                    prefill_addr,
                    key,
                    deadline,
                    std::min(config_.p2p_lease_query_timeout_ms, config_.p2p_cancel_broadcast_timeout_ms));
            })) {
            return context->finishWithoutTransfer(context->errorInfo());
        }

        // step 3. call the prefill worker to start the writeback, which will trigger the prefill worker to broadcast
        // the request to all decode workers.
        context->setPeerCallResult(server_caller_.write(
            routing.prefill_addr.first, routing.prefill_addr.second, std::move(request), context->deadlineMs()));
    } catch (const std::exception& e) {
        context->cancel(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, e.what()));
        context->finishWithoutTransfer(context->errorInfo());
    } catch (...) {
        context->cancel(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback kickoff failed"));
        context->finishWithoutTransfer(context->errorInfo());
    }
}

}  // namespace rtp_llm
