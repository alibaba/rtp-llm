#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.h"

#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/RouteCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include <algorithm>
#include <set>
#include <thread>

namespace rtp_llm {

P2PSchedulerPrefillWrite::P2PSchedulerPrefillWrite(P2PConnectorSchedulerConfig         config,
                                                   KVCacheAllocatorPtr                 allocator,
                                                   const kmonitor::MetricsReporterPtr& metrics_reporter,
                                                   std::shared_ptr<P2PBroadcastClient> client):
    config_(std::move(config)),
    allocator_(std::move(allocator)),
    client_(std::move(client)),
    metrics_reporter_(metrics_reporter) {}

P2PSchedulerPrefillWrite::~P2PSchedulerPrefillWrite() {
    stop();
}

void P2PSchedulerPrefillWrite::stop() {
    checker_.stop();
}

bool P2PSchedulerPrefillWrite::init() {
    if (initialized_ || !allocator_ || !client_ || !config_.topology || config_.p2p_cancel_broadcast_timeout_ms <= 0
        || config_.p2p_max_transfer_deadline_ms <= 0
        || config_.p2p_worker_addrs.size() != static_cast<size_t>(config_.parallelism_config.tp_size)
        || config_.worker_grpc_addrs.size() != config_.p2p_worker_addrs.size()) {
        return false;
    }
    for (const auto& address : config_.p2p_worker_addrs) {
        WorkerAddrParts parts;
        if (!parseWorkerAddr(address, &parts)) {
            return false;
        }
        transfer_addrs_.push_back(formatGrpcHostPort(parts.host, parts.cache_store_port));
    }
    initialized_ = checker_.init(config_.p2p_resource_store_timeout_check_interval_ms);
    return initialized_;
}

std::shared_ptr<const PlanResult> P2PSchedulerPrefillWrite::planFor(int decode_tp_size) {
    {
        std::lock_guard<std::mutex> lock(plan_cache_mutex_);
        auto it = plan_cache_.find(decode_tp_size);
        if (it != plan_cache_.end()) {
            return it->second;
        }
    }
    PlanResult plan{{}, ShardLayoutFactory::validateWritebackLayout(
                            *config_.topology, config_.parallelism_config, decode_tp_size)};
    if (plan.ok()) {
        const auto dst_layout = ShardLayoutFactory::fromTopology(
            *config_.topology, config_.parallelism_config, RoleType::PREFILL);
        const auto src_layout = ShardLayoutFactory::peerOf(dst_layout, decode_tp_size, false, RoleType::DECODE);
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
    auto [it, inserted] = plan_cache_.emplace(decode_tp_size, result);
    (void)inserted;
    return it->second;
}

ErrorInfo P2PSchedulerPrefillWrite::buildPrefillRankRoutes(const TransferPlan&              plan,
                                                           KVCacheResource&                 resource,
                                                           size_t                           start_block,
                                                           size_t                           block_count,
                                                           P2PBroadcastClient::RankRoutes& routes,
                                                           int64_t*                         planned_bytes) const {
    if (planned_bytes) {
        *planned_bytes = 0;
    }
    if (start_block > resource.cacheKeys().size() || block_count > resource.cacheKeys().size() - start_block) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback range exceeds destination keys");
    }
    for (const auto& route : plan.routes) {
        if (route.dst_rank < 0 || static_cast<size_t>(route.dst_rank) >= routes.size()) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback destination rank out of range");
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
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback destination group missing");
        }
        for (const auto pos : positions) {
            if (pos >= resource.blocks(group_id).size() || resource.blocks(group_id)[pos] <= 0) {
                return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback destination block missing");
            }
        }
        const auto buffers =
            LayerCacheBufferUtil::convertTagForRoute(resource, *config_.topology, group.tag, positions, 0, 1);
        if (buffers.size() != group.layer_ids.size()) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback destination layer projection incomplete");
        }
        TransferRoutePB pb;
        RouteCodec::encodeForReceiver(route, &pb);
        for (const auto& buffer : buffers) {
            if (!buffer || buffer->blockIdMap().size() != positions.size()) {
                return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback destination key projection incomplete");
            }
            auto* layer = pb.add_layer_blocks();
            layer->set_layer_id(buffer->getLayerId());
            layer->set_cache_tag(buffer->cacheTag());
            for (const auto& [key, id] : buffer->blockIdMap()) {
                layer->add_cache_keys(key);
                layer->add_block_ids(id);
            }
        }
        routes[route.dst_rank].push_back(std::move(pb));
        if (planned_bytes) {
            *planned_bytes += route.dst_bytes * positions.size() * group.layer_ids.size();
        }
    }
    return ErrorInfo::OkStatus();
}

void P2PSchedulerPrefillWrite::handleWrite(const P2PConnectorStartWriteRequestPB& request,
                                           P2PConnectorStartWriteResponsePB&      response,
                                           std::function<bool()>                  is_cancelled) {
    response.Clear();
    const auto start_us                 = currentTimeUs();
    bool       metrics_owned_by_context = false;
    const auto report                   = [&](const ErrorInfo& error, bool no_transfer) {
        if (metrics_reporter_ && !metrics_owned_by_context) {
            WriteSchedulerMetricsCollector metrics;
            metrics.prefill            = true;
            metrics.error              = error;
            metrics.no_transfer        = no_transfer;
            metrics.total_cost_time_us = currentTimeUs() - start_us;
            metrics_reporter_->report<P2PConnectorMetrics, WriteSchedulerMetricsCollector>(nullptr, &metrics);
        }
    };
    const auto reject = [&](const ErrorInfo& error) {
        report(error, false);
        response.Clear();
        response.set_error_code(transErrorCodeToRPC(error.code()));
        response.set_error_message(error.ToString());
    };
    const auto invalid = [&](const std::string& message) { reject(ErrorInfo(ErrorCode::INVALID_PARAMS, message)); };
    const auto now_ms  = currentTimeMs();
    if (!initialized_) {
        return invalid("writeback receiver unavailable");
    }
    if (request.unique_key().empty() || request.cache_keys().empty() || request.input_length() < 0
        || request.input_length() > request.token_ids_size() || request.deadline_ms() <= now_ms
        || request.deadline_ms() - now_ms > config_.p2p_max_transfer_deadline_ms) {
        return invalid("invalid writeback request or deadline");
    }
    if (is_cancelled && is_cancelled()) {
        return reject(ErrorInfo(ErrorCode::CANCELLED, "StartWrite cancelled"));
    }
    const auto planned = planFor(request.decode_tp_size());
    if (!planned->ok()) {
        return reject(planned->error);
    }
    if (request.layout_digest()
        != ShardLayoutFactory::writebackLayoutDigest(*config_.topology, config_.parallelism_config.tp_size)) {
        return invalid("writeback layout mismatch");
    }

    // step1: validate request and calculate the accepted start_block and block_count
    const size_t block_size    = config_.topology->groups().front().seq_size_per_block;
    const size_t blocks        = request.cache_keys_size();
    const size_t prompt_blocks = request.input_length() / block_size;
    if (blocks > request.token_ids().size() / block_size || prompt_blocks > blocks) {
        return invalid("writeback keys exceed the declared complete KV range");
    }
    const CacheKeysType keys(request.cache_keys().begin(), request.cache_keys().end());
    if (std::set<int64_t>(keys.begin(), keys.end()).size() != keys.size()
        || calculateCacheKeys(request.token_ids().data(), blocks * block_size, block_size) != keys) {
        return invalid("writeback cache key hash mismatch");
    }
    if (currentTimeMs() >= request.deadline_ms()) {
        return reject(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback expired during admission"));
    }
    if (is_cancelled && is_cancelled()) {
        return reject(ErrorInfo(ErrorCode::CANCELLED, "StartWrite cancelled"));
    }

    // step2: admit the writeback and prepare the resource and load context
    KVCacheResourcePtr                resource;
    std::shared_ptr<LoadAsyncContext> load_context;
    size_t                            start = 0;
    const auto                        admission_error =
        allocator_->admitWriteBackDecodeCache(keys, prompt_blocks, resource, start, load_context);
    if (admission_error.hasError()) {
        return reject(admission_error);
    }
    if (currentTimeMs() >= request.deadline_ms()) {
        return reject(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback expired during admission"));
    }
    if (is_cancelled && is_cancelled()) {
        return reject(ErrorInfo(ErrorCode::CANCELLED, "StartWrite cancelled"));
    }
    const auto count = blocks - start;
    if (count == 0 && !load_context) {
        response.set_accepted_start_block(start);
        response.set_plan_digest(planned->plan.digest());
        report(ErrorInfo::OkStatus(), true);
        return;
    }

    // step3: build the transfer routes
    P2PBroadcastClient::RankRoutes routes(transfer_addrs_.size());
    int64_t                        planned_bytes = 0;
    if (count > 0) {
        const auto route_error = buildPrefillRankRoutes(planned->plan, *resource, start, count, routes, &planned_bytes);
        if (route_error.hasError()) {
            return reject(route_error);
        }
    }

    // step4: register the resource and kickoff the transfer
    auto context = std::make_shared<P2PConnectorAsyncWriteContext>(
        std::move(resource),
        request.unique_key(),
        request.deadline_ms(),
        HANDLE_WRITE,
        client_,
        config_.p2p_cancel_broadcast_timeout_ms,
        [allocator = allocator_, start, deadline = request.deadline_ms()](const KVCacheResourcePtr& owner) {
            return allocator->commitWriteBackDecodeCache(*owner, start, deadline);
        },
        std::function<void()>{},
        metrics_reporter_,
        load_context);
    metrics_owned_by_context = true;
    context->setPlannedBytes(planned_bytes);
    if (!checker_.addContext(context)) {
        context->finishWithoutTransfer(ErrorInfo(ErrorCode::CANCELLED, "writeback receiver stopping"));
        return reject(context->errorInfo());
    }
    if (!context->beginKickoff()) {
        return reject(context->errorInfo());
    }

    // step5. submit the local load if needed
    if (load_context && !load_context->commit()) {
        const ErrorInfo error(ErrorCode::CACHE_STORE_STORE_FAILED, "writeback local load submission failed");
        context->finishWithoutTransfer(error);
        return reject(error);
    }
    if (count == 0) {
        context->finishWithoutTransfer();
        if (context->done() && !context->success()) {
            return reject(context->errorInfo());
        }
        response.set_accepted_start_block(start);
        response.set_plan_digest(planned->plan.digest());
        return;
    }
    // step6. submit the transfer plan to the workers and wait for registration to complete
    try {
        context->setCallResults(client_->broadcastPerRank(request.request_id(),
                                                          P2PBroadcastClient::RankLayerCacheBuffers(routes.size()),
                                                          {},
                                                          request.unique_key(),
                                                          request.deadline_ms(),
                                                          HANDLE_WRITE,
                                                          request.deadline_ms(),
                                                          routes,
                                                          planned->plan.digest()));
    } catch (...) {
        context->setCallResults(nullptr);
    }
    while (!context->registrationDone() && !context->done()) {
        if (is_cancelled && is_cancelled()) {
            context->cancel();
            break;
        }
        context->checkDone();
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if ((is_cancelled && is_cancelled()) || currentTimeMs() >= request.deadline_ms()) {
        context->cancel();
    }
    if (!context->registrationSucceeded() || (context->done() && !context->success())) {
        return reject(context->errorInfo());
    }
    response.set_accepted_start_block(start);
    response.set_accepted_block_count(count);
    response.set_plan_digest(planned->plan.digest());
    for (const auto& address : transfer_addrs_) {
        response.add_prefill_worker_transfer_addrs(address);
    }
}

}  // namespace rtp_llm
