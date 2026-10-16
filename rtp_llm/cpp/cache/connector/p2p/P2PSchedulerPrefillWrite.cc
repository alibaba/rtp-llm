#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.h"

#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/KVCacheTransferPlanner.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/RouteCodec.h"
#include "rtp_llm/cpp/cache/connector/p2p/plan/ShardLayoutFactory.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include <algorithm>
#include <limits>
#include <set>
#include <thread>
#include <future>
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

namespace rtp_llm {
namespace {
constexpr int64_t kControlInitialIntervalMs = 10;
constexpr int64_t kControlMaxIntervalMs     = 100;
constexpr int64_t kLeaseQueryTimeoutMs      = 500;

template<typename Call>
auto submitWriteRpc(const std::shared_ptr<autil::ThreadPool>& pool, Call call, const std::function<void()>& notify) {
    using Result = decltype(call());
    auto task    = std::make_shared<std::packaged_task<Result()>>(std::move(call));
    auto result  = task->get_future();
    if (!pool
        || pool->pushTask(
               [task, notify] {
                   (*task)();
                   notify();
               },
               false,
               false)
               != autil::ThreadPoolBase::ERROR_NONE)
        return std::future<Result>{};
    return result;
}
}  // namespace

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
    std::vector<std::shared_ptr<PrefillWriteRequest>> requests;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        stopping_ = true;
        for (const auto& [key, state] : requests_)
            requests.push_back(state);
    }
    for (const auto& state : requests)
        state->fail(ErrorInfo(ErrorCode::CANCELLED, "writeback receiver stopping"));
    {
        std::unique_lock<std::mutex> lock(requests_mutex_);
        requests_cv_.wait(lock, [this] {
            return std::none_of(requests_.begin(), requests_.end(), [](const auto& item) {
                return item.second->resourceHoldPending();
            });
        });
    }
    if (control_pool_)
        control_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
}

bool P2PSchedulerPrefillWrite::init() {
    if (initialized_ || !allocator_ || !client_ || !config_.topology || config_.p2p_cancel_broadcast_timeout_ms <= 0
        || config_.p2p_writeback_timeout_ms <= 0 || config_.load_cache_timeout_ms <= 0
        || config_.p2p_lease_query_timeout_ms <= 0
        || config_.worker_addrs.size() != static_cast<size_t>(config_.parallelism_config.tp_size)
        || config_.worker_grpc_addrs.size() != config_.worker_addrs.size()) {
        return false;
    }
    for (const auto& address : config_.worker_addrs) {
        p2p_internal::WorkerAddrParts parts;
        if (!p2p_internal::parseWorkerAddr(address, &parts)) {
            return false;
        }
        transfer_addrs_.push_back(formatGrpcHostPort(parts.host, parts.cache_store_port));
    }
    control_pool_ = std::make_shared<autil::ThreadPool>(4, 1024, nullptr, "P2PWriteControl");
    initialized_  = control_pool_->start();
    return initialized_;
}

std::shared_ptr<P2PBroadcastClient> P2PSchedulerPrefillWrite::sourceClient(const std::vector<std::string>& addresses) {
    std::lock_guard<std::mutex> lock(source_clients_mutex_);
    const auto                  it = source_clients_.find(addresses);
    if (it != source_clients_.end())
        return it->second;
    auto client = std::make_shared<P2PBroadcastClient>(addresses, config_.p2p_cancel_broadcast_timeout_ms);
    if (!client->init())
        return nullptr;
    source_clients_.emplace(addresses, client);
    return client;
}

std::shared_ptr<const PlanResult> P2PSchedulerPrefillWrite::planFor(int decode_tp_size) {
    {
        std::lock_guard<std::mutex> lock(plan_cache_mutex_);
        auto                        it = plan_cache_.find(decode_tp_size);
        if (it != plan_cache_.end()) {
            return it->second;
        }
    }
    const auto& pc = config_.parallelism_config;
    if (pc.tp_size <= 0 || decode_tp_size != pc.tp_size || pc.prefill_cp_config.is_enabled()
        || pc.prefill_cp_config.kv_cache_sharded) {
        return std::make_shared<const PlanResult>(
            PlanResult{{}, ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback requires symmetric TP and disabled CP")});
    }
    const auto               dst_layout = ShardLayoutFactory::fromTopology(*config_.topology, pc, RoleType::PREFILL);
    std::vector<std::string> tags;
    for (const auto& group : config_.topology->groups()) {
        if (group.policy.enable_prefix_reuse) {
            tags.push_back(group.tag);
        }
    }
    const auto src_layout = ShardLayoutFactory::peerOf(dst_layout, decode_tp_size, false, RoleType::DECODE);
    auto       result = std::make_shared<const PlanResult>(KVCacheTransferPlanner::plan(src_layout, dst_layout, tags));
    std::lock_guard<std::mutex> lock(plan_cache_mutex_);
    auto [it, inserted] = plan_cache_.emplace(decode_tp_size, result);
    (void)inserted;
    return it->second;
}

ErrorInfo P2PSchedulerPrefillWrite::buildPrefillRankRoutes(const TransferPlan&             plan,
                                                           KVCacheResource&                resource,
                                                           size_t                          start_block,
                                                           size_t                          block_count,
                                                           P2PBroadcastClient::RankRoutes& routes,
                                                           int64_t*                        planned_bytes) const {
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
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback destination group missing");
        }
        for (const auto pos : positions) {
            if (pos >= resource.blocks(group_id).size() || resource.blocks(group_id)[pos] <= 0) {
                return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback destination block missing");
            }
        }
        const auto buffers =
            LayerCacheBufferUtil::convertTagForRoute(resource, *config_.topology, group.tag, positions, 0, 1);
        if (!buffers.ok()) {
            return buffers.status();
        }
        if (buffers.value().size() != group.layer_ids.size()) {
            return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback destination layer projection incomplete");
        }
        TransferRoutePB pb;
        RouteCodec::encodeForDecode(route, &pb);
        for (const auto& buffer : buffers.value()) {
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

void P2PSchedulerPrefillWrite::PrefillWriteRequest::fail(const ErrorInfo& failure) {
    std::lock_guard<std::mutex> lock(mutex);
    if (stopped)
        return;
    if (error.ok())
        error = failure;
    cancelled = true;
    cv.notify_all();
}

ErrorInfo P2PSchedulerPrefillWrite::PrefillWriteRequest::errorInfo() const {
    std::lock_guard<std::mutex> lock(mutex);
    return error;
}

bool P2PSchedulerPrefillWrite::PrefillWriteRequest::resourceHoldPending() const {
    std::lock_guard<std::mutex> lock(mutex);
    return !stopped;
}

int64_t P2PSchedulerPrefillWrite::holdUntilMs(int64_t deadline_ms) const {
    return config_.p2p_lease_query_timeout_ms > std::numeric_limits<int64_t>::max() - deadline_ms ?
               std::numeric_limits<int64_t>::max() :
               deadline_ms + config_.p2p_lease_query_timeout_ms;
}

void P2PSchedulerPrefillWrite::pruneRequestsLocked(int64_t now_ms) {
    for (auto it = request_expirations_.begin(); it != request_expirations_.end() && it->first <= now_ms;) {
        const auto state = requests_.find(it->second);
        if (state == requests_.end() || !state->second->resourceHoldPending()) {
            if (state != requests_.end())
                requests_.erase(state);
            it = request_expirations_.erase(it);
        } else {
            ++it;
        }
    }
}

std::shared_ptr<P2PSchedulerPrefillWrite::PrefillWriteRequest>
P2PSchedulerPrefillWrite::findRequest(const std::string& key) const {
    std::lock_guard<std::mutex> lock(requests_mutex_);
    const auto                  it = requests_.find(key);
    return it == requests_.end() ? nullptr : it->second;
}

ErrorInfo P2PSchedulerPrefillWrite::workerRegistered(const P2PConnectorBroadcastTpRequestPB& request) {
    const auto state = findRequest(request.unique_key());
    if (!state)
        return ErrorInfo(ErrorCode::CANCELLED, "unknown write registration key");
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->cancelled || state->stopped || currentTimeMs() >= state->deadline_ms)
        return ErrorInfo(ErrorCode::CANCELLED, "write registration is no longer accepted");
    if (request.worker_rank() < 0 || static_cast<size_t>(request.worker_rank()) >= state->registered_workers.size()
        || request.plan_digest() != state->plan_digest)
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "write registration rank or plan mismatch");
    state->registered_workers[request.worker_rank()] = true;
    state->cv.notify_all();
    return ErrorInfo::OkStatus();
}

void P2PSchedulerPrefillWrite::cancelWrite(const P2PConnectorCancelWriteRequestPB& request,
                                           P2PConnectorStartWriteResponsePB&       response) {
    response.Clear();
    if (request.unique_key().empty() || request.deadline_ms() <= 0) {
        response.set_error_code(transErrorCodeToRPC(ErrorCode::INVALID_PARAMS));
        response.set_error_message("invalid CancelWrite key or deadline");
        return;
    }
    std::shared_ptr<PrefillWriteRequest> state;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        pruneRequestsLocked(currentTimeMs());
        auto it = requests_.find(request.unique_key());
        if (it == requests_.end() && !stopping_) {
            state            = std::make_shared<PrefillWriteRequest>(request.deadline_ms());
            state->error     = ErrorInfo(ErrorCode::CANCELLED, "writeback cancelled before StartWrite");
            state->cancelled = true;
            state->stopped   = true;
            requests_.emplace(request.unique_key(), state);
            request_expirations_.emplace(holdUntilMs(state->deadline_ms), request.unique_key());
        } else if (it != requests_.end()) {
            state = it->second;
        }
    }
    if (state) {
        state->fail(ErrorInfo(ErrorCode::CANCELLED, "writeback cancelled"));
        response.set_stopped(!state->resourceHoldPending());
        const auto error = state->errorInfo();
        response.set_error_code(transErrorCodeToRPC(error.code()));
        response.set_error_message(error.ok() ? "" : error.ToString());
    } else {
        response.set_stopped(true);
        response.set_error_code(transErrorCodeToRPC(ErrorCode::CANCELLED));
    }
}

size_t P2PSchedulerPrefillWrite::inflightContextCount() const {
    std::lock_guard<std::mutex> lock(requests_mutex_);
    return std::count_if(
        requests_.begin(), requests_.end(), [](const auto& item) { return item.second->resourceHoldPending(); });
}

void P2PSchedulerPrefillWrite::handleWrite(const P2PConnectorStartWriteRequestPB& request,
                                           P2PConnectorStartWriteResponsePB&      response,
                                           std::function<bool()>                  is_cancelled) {
    response.Clear();
    const auto start_us                 = currentTimeUs();
    const auto report                   = [&](const ErrorInfo& error, bool no_transfer) {
        if (metrics_reporter_) {
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
        const auto existing = findRequest(request.unique_key());
        response.set_stopped(!existing || !existing->resourceHoldPending());
    };
    const auto invalid = [&](const std::string& message) { reject(ErrorInfo(ErrorCode::INVALID_PARAMS, message)); };
    const auto now_ms  = currentTimeMs();
    if (!initialized_) {
        return invalid("writeback receiver unavailable");
    }
    if (request.unique_key().empty() || request.cache_keys().empty() || request.input_length() < 0
        || request.input_length() > request.token_ids_size() || request.deadline_ms() <= 0 || request.timeout_ms() <= 0
        || request.timeout_ms() > std::numeric_limits<int>::max()) {
        return invalid("invalid writeback request or timeout");
    }
    const int64_t local_deadline =
        now_ms + std::min({request.timeout_ms(), config_.p2p_writeback_timeout_ms, config_.load_cache_timeout_ms});
    const int64_t deadline_ms =
        request.deadline_ms() > 0 ? std::min(local_deadline, request.deadline_ms()) : local_deadline;
    if (findRequest(request.unique_key())) {
        return reject(ErrorInfo(ErrorCode::CANCELLED, "duplicate or cancelled StartWrite key"));
    }
    if (is_cancelled && is_cancelled()) {
        return reject(ErrorInfo(ErrorCode::CANCELLED, "StartWrite cancelled"));
    }
    const auto planned = planFor(request.decode_tp_size());
    if (!planned->ok()) {
        return reject(planned->error);
    }
    if (request.plan_digest() != planned->plan.digest() || request.decode_workers_size() != request.decode_tp_size()) {
        return invalid("writeback source workers or plan mismatch");
    }
    std::vector<std::string> source_addrs;
    for (const auto& source : request.decode_workers()) {
        std::string host;
        int32_t     port = 0;
        if (!parseGrpcHostPort(source.worker_addr(), host, port))
            return invalid("writeback source worker address is invalid");
        source_addrs.push_back(source.worker_addr());
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
    if (currentTimeMs() >= deadline_ms) {
        return reject(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback expired during admission"));
    }
    if (is_cancelled && is_cancelled()) {
        return reject(ErrorInfo(ErrorCode::CANCELLED, "StartWrite cancelled"));
    }

    const auto& unique_key = request.unique_key();
    auto        state      = std::make_shared<PrefillWriteRequest>(deadline_ms);
    bool        inserted   = false;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        pruneRequestsLocked(currentTimeMs());
        if (!stopping_ && requests_.emplace(unique_key, state).second) {
            request_expirations_.emplace(holdUntilMs(deadline_ms), unique_key);
            inserted = true;
        }
    }
    if (!inserted)
        return reject(ErrorInfo(ErrorCode::CANCELLED, "duplicate, cancelled or stopped StartWrite key"));

    WriteSchedulerMetricsCollector metrics;
    metrics.prefill   = true;
    const auto notify = [weak = std::weak_ptr<PrefillWriteRequest>(state)] {
        if (const auto current = weak.lock())
            current->cv.notify_all();
    };
    const auto pause = [&] {
        std::unique_lock<std::mutex> lock(state->mutex);
        state->cv.wait_for(lock, std::chrono::milliseconds(kControlInitialIntervalMs));
    };
    KVCacheResourcePtr                                       resource;
    std::shared_ptr<LoadAsyncContext>                        load_context;
    std::future<std::shared_ptr<P2PBroadcastClient::Result>> start_submission;
    std::shared_ptr<P2PBroadcastClient::Result>              receive_result;
    std::future<std::shared_ptr<P2PBroadcastClient::Result>> send_submission;
    std::shared_ptr<P2PBroadcastClient::Result>              send_result;
    bool                                                     receive_started = false;
    bool                                                     receive_stopped = true;
    size_t                                                   start           = 0;
    size_t                                                   count           = 0;
    const auto                                               observe         = [&] {
        if (start_submission.valid()
            && start_submission.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            try {
                receive_result = start_submission.get();
            } catch (...) {}
            if (receive_result)
                receive_result->setDoneCallback(notify);
            else
                state->fail(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                                      "writeback receive broadcast failed"));
        }
        if (receive_result) {
            const auto first = receive_result->firstError();
            if (first.error.hasError())
                state->fail(first.error);
            if (receive_result->done()) {
                if (receive_result->success()) {
                    receive_stopped = true;
                } else {
                    state->fail(ErrorInfo(receive_result->errorCode(), receive_result->errorMessage()));
                }
            }
        }
        if (send_submission.valid()
            && send_submission.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            try {
                send_result = send_submission.get();
            } catch (...) {}
            if (send_result)
                send_result->setDoneCallback(notify);
            else
                state->fail(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                                      "writeback sender broadcast failed"));
        }
        if (send_result) {
            const auto first = send_result->firstError();
            if (first.error.hasError())
                state->fail(first.error);
            if (send_result->done() && !send_result->success())
                state->fail(ErrorInfo(send_result->errorCode(), send_result->errorMessage()));
        }
        if (load_context && load_context->done() && !load_context->success())
            state->fail(ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "writeback local load failed"));
        if (is_cancelled && is_cancelled())
            state->fail(ErrorInfo(ErrorCode::CANCELLED, "StartWrite cancelled"));
        if (currentTimeMs() >= deadline_ms)
            state->fail(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback transfer deadline exceeded"));
    };
    const auto waitFor = [&](const auto& ready) {
        for (;;) {
            observe();
            if (state->errorInfo().hasError())
                return false;
            if (ready())
                return true;
            pause();
        }
    };
    const auto run = [&]() -> ErrorInfo {
        observe();
        if (state->errorInfo().hasError())
            return state->errorInfo();
        const auto error = allocator_->admitWriteBackDecodeCache(keys, prompt_blocks, resource, start, load_context);
        if (error.hasError())
            return error;
        if (load_context)
            load_context->onDone([notify](ErrorInfo) { notify(); });
        observe();
        if (state->errorInfo().hasError())
            return state->errorInfo();
        count = blocks - start;
        if (count == 0 && !load_context)
            return ErrorInfo::OkStatus();
        P2PBroadcastClient::RankRoutes routes(transfer_addrs_.size());
        if (count > 0) {
            const auto error =
                buildPrefillRankRoutes(planned->plan, *resource, start, count, routes, &metrics.planned_bytes);
            if (error.hasError())
                return error;
        }
        if (load_context && !load_context->commit())
            return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "writeback local load submission failed");
        if (count > 0) {
            P2PBroadcastClient::BroadcastParams params;
            params.request_id          = request.request_id();
            params.unique_key          = unique_key;
            params.deadline_ms         = deadline_ms;
            params.request_deadline_ms = deadline_ms;
            params.type                = HANDLE_WRITE;
            params.routes              = std::move(routes);
            params.plan_digest         = planned->plan.digest();
            {
                std::lock_guard<std::mutex> lock(state->mutex);
                if (state->cancelled)
                    return state->error;
                state->registered_workers.assign(config_.worker_grpc_addrs.size(), false);
                state->plan_digest = planned->plan.digest();
            }
            start_submission = submitWriteRpc(
                control_pool_,
                [client = client_, params = std::move(params)]() mutable {
                    return client->broadcast(std::move(params));
                },
                notify);
            if (!start_submission.valid())
                return ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback receive submission queue unavailable");
            receive_started   = true;
            receive_stopped   = false;
            metrics.submitted = true;
            if (!waitFor([&] {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    return std::all_of(state->registered_workers.begin(),
                                       state->registered_workers.end(),
                                       [](bool registered) { return registered; });
                }))
                return state->errorInfo();
            P2PBroadcastClient::BroadcastParams send;
            send.request_id          = request.request_id();
            send.unique_key          = unique_key;
            send.deadline_ms         = deadline_ms;
            send.request_deadline_ms = deadline_ms;
            send.type                = WRITE;
            send.plan_digest         = planned->plan.digest();
            for (const auto& address : transfer_addrs_) {
                std::string host;
                int32_t     port = 0;
                parseGrpcHostPort(address, host, port);
                send.peer_workers.emplace_back(std::move(host), port);
            }
            for (const auto& source : request.decode_workers())
                send.routes.emplace_back(source.routes().begin(), source.routes().end());
            send_submission = submitWriteRpc(
                control_pool_,
                [this, addresses = source_addrs, send = std::move(send)]() mutable {
                    const auto client = sourceClient(addresses);
                    return client ? client->broadcast(std::move(send)) : nullptr;
                },
                notify);
            if (!send_submission.valid())
                return ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback sender submission queue unavailable");
        }
        if (!waitFor([&] {
                return receive_stopped && (count == 0 || (send_result && send_result->done() && send_result->success()))
                       && (!load_context || load_context->done());
            }))
            return state->errorInfo();
        std::lock_guard<std::mutex> lock(state->mutex);
        if (state->cancelled)
            return state->error;
        return allocator_->commitWriteBackDecodeCache(*resource, start, deadline_ms);
    };
    try {
        const auto error = run();
        if (error.hasError())
            state->fail(error);
    } catch (const std::exception& e) {
        state->fail(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, e.what()));
    } catch (...) {
        state->fail(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "writeback receive failed"));
    }

    // Keep targets through physical stop, including any committed local load.
    std::future<std::shared_ptr<P2PBroadcastClient::TpBroadcastResult>> control_submission;
    std::shared_ptr<P2PBroadcastClient::TpBroadcastResult>              control_result;
    bool                                                                cancel_confirmed = false;
    int64_t                                                             next_control_ms  = 0;
    int64_t                                                             interval_ms      = kControlInitialIntervalMs;
    const auto                                                          hold_until       = holdUntilMs(deadline_ms);
    try {
        while (!receive_stopped || (load_context && !load_context->done())) {
            observe();
            if (load_context)
                load_context->abortPending();
            if (control_submission.valid()
                && control_submission.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
                try {
                    control_result = control_submission.get();
                } catch (...) {}
                if (control_result)
                    control_result->setProgressCallback(notify);
                else {
                    next_control_ms = currentTimeMs() + interval_ms;
                    interval_ms = std::min(interval_ms * 2, kControlMaxIntervalMs);
                }
            }
            if (control_result && control_result->done()) {
                const auto status = P2PBroadcastClient::leaseStatus(control_result);
                if (status.success) {
                    cancel_confirmed = true;
                    receive_stopped  = receive_stopped || status.allStopped();
                }
                control_result.reset();
                next_control_ms = currentTimeMs() + interval_ms;
                interval_ms     = std::min(interval_ms * 2, kControlMaxIntervalMs);
            }
            if (receive_stopped && (!load_context || load_context->done()))
                break;
            if (currentTimeMs() >= hold_until) {
                RTP_LLM_LOG_ERROR("writeback receive stop unconfirmed at D+query timeout, unique_key=%s",
                                  unique_key.c_str());
                std::abort();
            }
            if (receive_started && !receive_stopped && !control_submission.valid() && !control_result
                && currentTimeMs() >= next_control_ms) {
                const auto timeout_limit =
                    cancel_confirmed ? kLeaseQueryTimeoutMs : config_.p2p_cancel_broadcast_timeout_ms;
                control_submission = submitWriteRpc(
                    control_pool_,
                    [client = client_, key = unique_key, deadline_ms, hold_until, timeout_limit, cancel_confirmed] {
                        const auto timeout = std::min(hold_until - currentTimeMs(), timeout_limit);
                        if (timeout <= 0)
                            return std::shared_ptr<P2PBroadcastClient::TpBroadcastResult>{};
                        return cancel_confirmed ?
                                   client->queryLeaseStatusAsync(key, HANDLE_WRITE, timeout) :
                                   client->controlWriteAsync(key, CANCEL_HANDLE_WRITE, deadline_ms, timeout);
                    },
                    notify);
                if (!control_submission.valid()) {
                    next_control_ms = currentTimeMs() + interval_ms;
                    interval_ms = std::min(interval_ms * 2, kControlMaxIntervalMs);
                }
            }
            pause();
        }
    } catch (...) {
        RTP_LLM_LOG_ERROR("writeback receive cleanup failed before stop confirmation, unique_key=%s",
                          unique_key.c_str());
        std::abort();
    }
    load_context.reset();
    ErrorInfo error;
    {
        std::lock_guard<std::mutex> requests_lock(requests_mutex_);
        std::lock_guard<std::mutex> lock(state->mutex);
        resource.reset();
        state->stopped = true;
        error          = state->error;
        state->cv.notify_all();
    }
    requests_cv_.notify_all();
    response.set_accepted_start_block(start);
    response.set_accepted_block_count(count);
    response.set_plan_digest(planned->plan.digest());
    response.set_stopped(true);
    response.set_error_code(transErrorCodeToRPC(error.code()));
    response.set_error_message(error.ok() ? "" : error.ToString());
    metrics.no_transfer        = !receive_started && error.ok();
    metrics.error              = error;
    metrics.total_cost_time_us = currentTimeUs() - start_us;
    metrics.hold_time_us       = metrics.total_cost_time_us;
    if (metrics_reporter_)
        metrics_reporter_->report<P2PConnectorMetrics, WriteSchedulerMetricsCollector>(nullptr, &metrics);
}

}  // namespace rtp_llm
