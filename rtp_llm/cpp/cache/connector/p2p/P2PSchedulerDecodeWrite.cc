#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerDecodeWrite.h"

#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
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
                                                 kmonitor::MetricsReporterPtr        metrics_reporter):
    config_(std::move(config)), client_(std::move(client)), metrics_reporter_(std::move(metrics_reporter)) {}

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
        || config_.p2p_max_transfer_deadline_ms <= 0
        || config_.p2p_cancel_broadcast_timeout_ms <= 0 || config_.parallelism_config.tp_size <= 0
        || config_.worker_grpc_addrs.size() != static_cast<size_t>(config_.parallelism_config.tp_size)) {
        return false;
    }
    auto async_write_pool = std::make_shared<autil::LockFreeThreadPool>(
        kP2PDecodeKickoffThreadCount, kP2PDecodeKickoffQueueSize, nullptr, "P2PAsyncWriteKickoff");
    if (!async_write_pool->start() || !checker_.init(config_.p2p_resource_store_timeout_check_interval_ms)) {
        return false;
    }
    async_write_pool_ = std::move(async_write_pool);
    return true;
}

std::shared_ptr<const PlanResult> P2PSchedulerDecodeWrite::planFor(int prefill_tp_size) {
    {
        std::lock_guard<std::mutex> lock(plan_cache_mutex_);
        auto it = plan_cache_.find(prefill_tp_size);
        if (it != plan_cache_.end()) {
            return it->second;
        }
    }
    PlanResult plan{{}, ShardLayoutFactory::validateWritebackLayout(
                            *config_.topology, config_.parallelism_config, prefill_tp_size)};
    if (plan.ok()) {
        const auto src_layout = ShardLayoutFactory::fromTopology(
            *config_.topology, config_.parallelism_config, RoleType::DECODE);
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
    auto result = std::make_shared<const PlanResult>(std::move(plan));
    std::lock_guard<std::mutex> lock(plan_cache_mutex_);
    auto [it, inserted] = plan_cache_.emplace(prefill_tp_size, result);
    (void)inserted;
    return it->second;
}

ErrorInfo P2PSchedulerDecodeWrite::buildDecodeRankRoutes(const TransferPlan&              plan,
                                                         KVCacheResource&                 resource,
                                                         size_t                           start_block,
                                                         size_t                           block_count,
                                                         P2PBroadcastClient::RankRoutes& routes,
                                                         int64_t*                         planned_bytes) const {
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
        positions.erase(std::remove_if(positions.begin(),
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
        if (buffers.size() != group.layer_ids.size()) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback source layer projection incomplete");
        }
        TransferRoutePB pb;
        RouteCodec::encodeForSender(route, route.dst_rank, &pb);
        for (const auto& buffer : buffers) {
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
                                                                              size_t             kv_ready_token_count,
                                                                              Meta::P2PRoutingContext routing) {
    if (!async_write_pool_ || !resource || routing.unique_key.empty()) {
        return nullptr;
    }
    const auto timeout = std::min<int64_t>(
        {config_.p2p_writeback_timeout_ms, config_.p2p_max_transfer_deadline_ms, std::numeric_limits<int>::max()});
    const auto deadline = currentTimeMs() + timeout;
    auto       context  = std::make_shared<P2PConnectorAsyncWriteContext>(std::move(resource),
                                                                   routing.unique_key,
                                                                   deadline,
                                                                   WRITE,
                                                                   client_,
                                                                   config_.p2p_cancel_broadcast_timeout_ms,
                                                                   P2PConnectorAsyncWriteContext::Settle{},
                                                                   std::function<void()>{},
                                                                   metrics_reporter_);
    if (!checker_.addContext(context)) {
        context->finishWithoutTransfer(
            ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback checker stopped or duplicate key"));
        return context;
    }
    const auto accepted = async_write_pool_->pushTask(
        [this,
         context,
         token_ids = std::move(token_ids),
         input_length,
         kv_ready_token_count,
         routing = std::move(routing),
         deadline]() { startAsyncWriteCalls(context, token_ids, input_length, kv_ready_token_count, routing, deadline); },
        false,
        false);
    if (accepted != autil::ThreadPoolBase::ERROR_NONE) {
        context->finishWithoutTransfer(ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback kickoff queue unavailable"));
    }
    return context;
}

void P2PSchedulerDecodeWrite::startAsyncWriteCalls(
    const std::shared_ptr<P2PConnectorAsyncWriteContext>& context,
    const std::vector<int>&                               token_ids,
    int                                                   input_length,
    size_t                                                kv_ready_token_count,
    const Meta::P2PRoutingContext&                        routing,
    int64_t                                               deadline_ms) {
    if (!context->beginKickoff()) {
        return;
    }
    bool       worker_start_attempted = false;
    const auto invalid                = [&](const std::string& message) {
        context->finishWithoutTransfer(ErrorInfo(ErrorCode::INVALID_PARAMS, message));
    };
    try {
        if (input_length < 0 || static_cast<size_t>(input_length) > kv_ready_token_count
            || kv_ready_token_count > token_ids.size()
            || token_ids.size() > static_cast<size_t>(std::numeric_limits<int>::max())
            || routing.prefill_cp_size != 1
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
        KVCacheResource resource   = *context->resource();
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
        request.set_deadline_ms(deadline_ms);
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

        // step 2. initiate the StartWrite RPC to the prefill peer, which will return the accepted start block and count, as well as the transfer addresses of the prefill workers.
        auto result = server_caller_.write(routing.prefill_addr.first, routing.prefill_addr.second, request);
        if (!result) {
            return context->finishWithoutTransfer(ErrorInfo(
                ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "StartWrite submission failed"));
        }
        while (!result->done() && !context->done()) {
            result->waitDone(1);
        }
        if (context->done()) {
            return context->finishWithoutTransfer(context->errorInfo());
        }
        if (!result->success() || result->responses().size() != 1) {
            return context->finishWithoutTransfer(
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "StartWrite RPC failed"));
        }
        const auto response = result->responses().front();
        if (response.error_code() != ErrorCodePB::NONE_ERROR) {
            return context->finishWithoutTransfer(ErrorInfo(
                transRPCErrorCode(static_cast<ErrorCodePB>(response.error_code())), response.error_message()));
        }
        const auto start = response.accepted_start_block();
        const auto count = response.accepted_block_count();
        if (start < 0 || count < 0 || static_cast<size_t>(start) > blocks
            || static_cast<size_t>(count) != blocks - start || static_cast<size_t>(start) < input_length / block_size
            || response.plan_digest() != planned->plan.digest()) {
            return invalid("StartWrite returned an invalid range or plan");
        }
        if (count == 0) {
            return context->finishWithoutTransfer();
        }
        if (response.prefill_worker_transfer_addrs_size() != routing.prefill_tp_size) {
            return invalid("StartWrite returned an invalid worker count");
        }

        // step 3. build the rank routes and broadcast the transfer plan to each decode worker, which will then send the data to the corresponding prefill worker.
        std::vector<std::pair<std::string, uint32_t>> peers;
        for (const auto& address : response.prefill_worker_transfer_addrs()) {
            std::string host;
            int32_t     port = 0;
            if (!parseGrpcHostPort(address, host, port)) {
                return invalid("StartWrite returned an invalid transfer endpoint");
            }
            peers.emplace_back(std::move(host), port);
        }
        P2PBroadcastClient::RankRoutes routes(config_.worker_grpc_addrs.size());
        int64_t                        planned_bytes = 0;
        const auto                     error         = buildDecodeRankRoutes(
            planned->plan, resource, start, count, routes, &planned_bytes);
        if (error.hasError()) {
            return context->finishWithoutTransfer(error);
        }
        if (context->done() || currentTimeMs() >= deadline_ms) {
            return context->finishWithoutTransfer(
                ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback expired before send"));
        }
        worker_start_attempted = true;
        context->setPlannedBytes(planned_bytes);
        context->setCallResults(client_->broadcastPerRank(routing.request_id,
                                                          P2PBroadcastClient::RankLayerCacheBuffers(routes.size()),
                                                          peers,
                                                          routing.unique_key,
                                                          deadline_ms,
                                                          WRITE,
                                                          deadline_ms,
                                                          routes,
                                                          planned->plan.digest()));
    } catch (const std::exception& e) {
        if (worker_start_attempted) {
            context->setCallResults(nullptr);
        } else {
            context->finishWithoutTransfer(
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, e.what()));
        }
    } catch (...) {
        if (worker_start_attempted) {
            context->setCallResults(nullptr);
        } else {
            context->finishWithoutTransfer(
                ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback kickoff failed"));
        }
    }
}

}  // namespace rtp_llm
