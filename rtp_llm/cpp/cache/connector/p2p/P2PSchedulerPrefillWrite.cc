#include "rtp_llm/cpp/cache/connector/p2p/P2PSchedulerPrefillWrite.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PWritebackResourceUtil.h"
#include "rtp_llm/cpp/cache/CPSlotMapper.h"

#include "rtp_llm/cpp/cache/KVCacheHashUtil.h"
#include "rtp_llm/cpp/cache/block_tree_cache/BlockTreeCache.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/utils/GrpcAddressUtil.h"
#include <algorithm>
#include <limits>
#include <set>
#include <tuple>
#include <thread>
#include <future>
#include "rtp_llm/cpp/cache/block_tree_cache/load/LoadAsyncContext.h"

namespace rtp_llm {
namespace {
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

P2PSchedulerPrefillWrite::P2PSchedulerPrefillWrite(P2PConnectorSchedulerConfig              config,
                                                   std::shared_ptr<CoordinatorCacheManager> coordinator_manager,
                                                   const kmonitor::MetricsReporterPtr&      metrics_reporter,
                                                   std::shared_ptr<P2PBroadcastClient>      client):
    config_(std::move(config)),
    coordinator_manager_(std::move(coordinator_manager)),
    client_(std::move(client)),
    metrics_reporter_(metrics_reporter) {}

P2PSchedulerPrefillWrite::~P2PSchedulerPrefillWrite() {
    stop();
}

void P2PSchedulerPrefillWrite::stop(int64_t steady_deadline_ms) {
    const auto deadline = std::chrono::steady_clock::time_point(
        std::chrono::milliseconds(writebackShutdownDeadline(steady_deadline_ms)));
    std::vector<std::shared_ptr<PrefillWriteRequest>> requests;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        if (stopped_) {
            return;
        }
        stopping_ = true;
        for (const auto& [key, state] : requests_)
            requests.push_back(state);
    }
    for (const auto& state : requests)
        state->cancel(ErrorInfo(ErrorCode::CANCELLED, "writeback receiver stopping"));
    {
        std::unique_lock<std::mutex> lock(requests_mutex_);
        const auto drained = [this] { return active_write_handlers_ == 0 && pending_drains_.empty(); };
        if (!requests_cv_.wait_until(lock, deadline, drained)) {
            RTP_LLM_LOG_ERROR(
                "writeback shutdown timed out on Prefill with %zu active handler(s) and %zu pending drain(s)",
                active_write_handlers_, pending_drains_.size());
            drain_aborted_.store(true, std::memory_order_release);
            for (const auto& [key, state] : requests_) {
                state->notifyProgress();
            }
            requests_cv_.notify_all();
            requests_cv_.wait(lock, drained);
        }
        stopped_ = true;
    }
    requests_cv_.notify_all();
    if (drain_thread_.joinable())
        drain_thread_.join();
    if (control_pool_)
        control_pool_->stop(autil::ThreadPoolBase::STOP_AFTER_QUEUE_EMPTY);
    if (initialized_) {
        checkRetainedResources();
    }
}

bool P2PSchedulerPrefillWrite::init() {
    if (initialized_ || !coordinator_manager_ || !client_ || !config_.topology
        || config_.p2p_cancel_broadcast_timeout_ms <= 0
        || config_.load_cache_timeout_ms <= 0 || config_.p2p_lease_query_timeout_ms <= 0
        || !validWritebackMaxKvRatio(config_.p2p_writeback_max_kv_ratio)
        || !retained_kv_.init(config_, true, metrics_reporter_)
        || config_.worker_addrs.size()
               != static_cast<size_t>(config_.parallelism_config.tp_size * config_.parallelism_config.pp_size)
        || config_.worker_grpc_addrs.size() != config_.worker_addrs.size()) {
        return false;
    }
    for (const auto& address : config_.worker_addrs) {
        WorkerAddrParts parts;
        if (!parseWorkerAddr(address, &parts)) {
            return false;
        }
        transfer_addrs_.push_back(formatGrpcHostPort(parts.host, parts.cache_store_port));
    }
    control_pool_ = std::make_shared<autil::ThreadPool>(4, 1024, nullptr, "P2PWriteControl");
    initialized_  = control_pool_->start();
    if (initialized_) {
        drain_thread_ = std::thread(&P2PSchedulerPrefillWrite::runDrainLoop, this);
    }
    return initialized_;
}

std::shared_ptr<const PlanResult>
P2PSchedulerPrefillWrite::planFor(int tp, const std::string& layout, int workers, const std::string& unique_key) {
    return planner_.planFor(tp, 1, layout, workers, unique_key);
}

ErrorInfo P2PSchedulerPrefillWrite::buildPrefillRankRoutes(const TransferPlan&             plan,
                                                           KVCacheResource&                resource,
                                                           size_t                          start_block,
                                                           size_t                          block_count,
                                                           P2PBroadcastClient::RankRoutes& routes,
                                                           int64_t*                        planned_bytes) const {
    const auto& cp = coordinator_manager_->cpSlotMapper();
    return planner_.buildRankRoutes(
        plan, resource, start_block, block_count, cp ? cp->cpSize() : 1, routes, planned_bytes);
}

void P2PSchedulerPrefillWrite::PrefillWriteRequest::fail(const ErrorInfo& failure) {
    std::lock_guard<std::mutex> lock(mutex);
    failLocked(failure);
}

void P2PSchedulerPrefillWrite::PrefillWriteRequest::cancel(const ErrorInfo& failure) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!committing)
        failLocked(failure);
}

ErrorInfo P2PSchedulerPrefillWrite::PrefillWriteRequest::beginCommit() {
    std::lock_guard<std::mutex> lock(mutex);
    if (cancelled)
        return error;
    // Publication owns the outcome once it wins the race with cancellation.
    committing = true;
    return ErrorInfo::OkStatus();
}

void P2PSchedulerPrefillWrite::PrefillWriteRequest::failLocked(const ErrorInfo& failure) {
    if (stopped)
        return;
    if (error.ok())
        error = failure;
    cancelled = true;
    ++progress_seq;
    cv.notify_all();
}

ErrorInfo P2PSchedulerPrefillWrite::PrefillWriteRequest::errorInfo() const {
    std::lock_guard<std::mutex> lock(mutex);
    return error;
}

bool P2PSchedulerPrefillWrite::PrefillWriteRequest::isStopped() const {
    std::lock_guard<std::mutex> lock(mutex);
    return stopped;
}

int64_t P2PSchedulerPrefillWrite::holdUntilMs(int64_t deadline_ms) const {
    const auto ttl_ms = std::max<int64_t>(1, config_.p2p_cancelled_keys_ttl_ms);
    return ttl_ms > std::numeric_limits<int64_t>::max() - deadline_ms ? std::numeric_limits<int64_t>::max() :
                                                                        deadline_ms + ttl_ms;
}

void P2PSchedulerPrefillWrite::checkRetainedResources(bool force) {
    constexpr int64_t kRetainedCheckIntervalMs = 1000;
    const int64_t     now_ms                   = P2PNotification::nowMs();
    if (!force) {
        int64_t last_ms = last_retained_check_ms_.load(std::memory_order_relaxed);
        if (last_ms > 0 && now_ms - last_ms < kRetainedCheckIntervalMs) {
            return;
        }
        if (!last_retained_check_ms_.compare_exchange_strong(last_ms, now_ms, std::memory_order_acq_rel)) {
            return;
        }
    } else {
        last_retained_check_ms_.store(now_ms, std::memory_order_release);
    }
    std::vector<KVCacheResourcePtr> retained;
    WritebackStatusMetricsCollector status;
    const auto                      now_us = currentTimeUs();
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        for (const auto& [key, state] : requests_) {
            std::lock_guard<std::mutex> state_lock(state->mutex);
            if (!state->stopped) {
                ++status.inflight_context_count;
                status.oldest_inflight_ms =
                    std::max(status.oldest_inflight_ms, std::max<int64_t>(0, now_us - state->start_time_us) / 1000);
            }
        }
        for (const auto& [key, resource] : retained_resources_) {
            retained.push_back(resource);
            const auto state = requests_.find(key);
            if (state != requests_.end()) {
                std::lock_guard<std::mutex> state_lock(state->second->mutex);
                const auto                  hold_start_us = state->second->resource_start_time_us;
                if (hold_start_us > 0) {
                    status.oldest_kv_hold_ms =
                        std::max(status.oldest_kv_hold_ms, std::max<int64_t>(0, now_us - hold_start_us) / 1000);
                }
            }
        }
    }
    retained_kv_.check(retained, status);
}

void P2PSchedulerPrefillWrite::pruneRequestsLocked(int64_t now_ms) {
    while (!request_expirations_.empty() && request_expirations_.begin()->first <= now_ms) {
        const auto it = request_expirations_.begin();
        completed_requests_.erase(it->second);
        request_expirations_.erase(it);
    }
}

std::shared_ptr<P2PSchedulerPrefillWrite::PrefillWriteRequest>
P2PSchedulerPrefillWrite::findRequest(const std::string& key) const {
    std::lock_guard<std::mutex> lock(requests_mutex_);
    return findRequestLocked(key);
}

std::shared_ptr<P2PSchedulerPrefillWrite::PrefillWriteRequest>
P2PSchedulerPrefillWrite::findRequestLocked(const std::string& key) const {
    if (const auto active = requests_.find(key); active != requests_.end()) {
        return active->second;
    }
    const auto completed = completed_requests_.find(key);
    return completed == completed_requests_.end() ? nullptr : completed->second;
}

ErrorInfo P2PSchedulerPrefillWrite::workerRegistered(const P2PConnectorBroadcastTpRequestPB& request) {
    const auto state = findRequest(request.unique_key());
    if (!state)
        return ErrorInfo(ErrorCode::CANCELLED, "unknown write registration key");
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->cancelled || state->committing || state->stopped || P2PNotification::nowMs() >= state->stop_at_ms)
        return ErrorInfo(ErrorCode::CANCELLED, "write registration is no longer accepted");
    if (request.worker_rank() < 0 || static_cast<size_t>(request.worker_rank()) >= state->registered_workers.size()
        || request.plan_digest() != state->plan_digest)
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "write registration rank or plan mismatch");
    state->registered_workers[request.worker_rank()] = true;
    ++state->progress_seq;
    state->cv.notify_all();
    return ErrorInfo::OkStatus();
}

void P2PSchedulerPrefillWrite::cancelWrite(const P2PConnectorCancelWriteBackRequestPB& request,
                                           P2PConnectorStartWriteBackResponsePB&       response) {
    response.Clear();
    if (request.unique_key().empty() || request.deadline_ms() <= 0) {
        response.set_error_code(transErrorCodeToRPC(ErrorCode::INVALID_PARAMS));
        response.set_error_message("invalid CancelWriteBack key or deadline");
        return;
    }
    std::shared_ptr<PrefillWriteRequest> state;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        pruneRequestsLocked(P2PNotification::nowMs());
        state = findRequestLocked(request.unique_key());
        if (!state && !stopping_) {
            if (requests_.size() + completed_requests_.size() >= kMaxWritebackRequestEntries) {
                response.set_stopped(false);
                response.set_error_code(transErrorCodeToRPC(ErrorCode::OUTPUT_QUEUE_FULL));
                response.set_error_message("writeback request table full");
                return;
            }
            const auto stop_at_ms =
                p2p_internal::writebackSteadyDeadline(currentTimeMs() + config_.load_cache_timeout_ms);
            state        = std::make_shared<PrefillWriteRequest>(stop_at_ms);
            state->error = ErrorInfo(ErrorCode::CANCELLED, "writeback cancelled before StartWriteBack");
            state->cancelled = true;
            state->stopped   = true;
            const auto expiry = request_expirations_.emplace(holdUntilMs(state->stop_at_ms), request.unique_key());
            try {
                completed_requests_.emplace(request.unique_key(), state);
            } catch (...) {
                request_expirations_.erase(expiry);
                throw;
            }
        }
    }
    if (state) {
        state->cancel(ErrorInfo(ErrorCode::CANCELLED, "writeback cancelled"));
        response.set_stopped(state->isStopped());
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
        requests_.begin(), requests_.end(), [](const auto& item) { return !item.second->isStopped(); });
}

ErrorInfo P2PSchedulerPrefillWrite::selectSendRoutes(const P2PConnectorStartWriteBackRequestPB& request,
                                                     const P2PBroadcastClient::RankRoutes&      receive_routes,
                                                     P2PBroadcastClient::RankRoutes&            send_routes) const {
    using RouteLayerKey = std::tuple<int32_t, uint32_t, int64_t>;
    std::set<RouteLayerKey> receive_keys;
    for (const auto& rank_routes : receive_routes) {
        for (const auto& route : rank_routes) {
            for (const auto& layer : route.layer_blocks()) {
                for (const auto key : layer.cache_keys()) {
                    receive_keys.emplace(route.route_id(), layer.layer_id(), key);
                }
            }
        }
    }
    auto uncovered_keys = receive_keys;
    send_routes.clear();
    for (const auto& worker : request.decode_workers()) {
        auto& rank_routes = send_routes.emplace_back();
        for (const auto& source : worker.routes()) {
            TransferRoutePB route = source;
            route.clear_layer_blocks();
            for (const auto& source_layer : source.layer_blocks()) {
                if (source_layer.cache_keys_size() != source_layer.block_ids_size()) {
                    return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback source key/block count mismatch");
                }
                LayerCacheBlockPB layer = source_layer;
                layer.clear_cache_keys();
                layer.clear_block_ids();
                for (int i = 0; i < source_layer.cache_keys_size(); ++i) {
                    const RouteLayerKey key{source.route_id(), source_layer.layer_id(), source_layer.cache_keys(i)};
                    if (receive_keys.count(key)) {
                        layer.add_cache_keys(source_layer.cache_keys(i));
                        layer.add_block_ids(source_layer.block_ids(i));
                        uncovered_keys.erase(key);
                    }
                }
                if (layer.cache_keys_size() > 0) {
                    route.add_layer_blocks()->Swap(&layer);
                }
            }
            if (route.layer_blocks_size() > 0) {
                rank_routes.push_back(std::move(route));
            }
        }
    }
    if (!uncovered_keys.empty()) {
        return ErrorInfo(ErrorCode::INVALID_PARAMS, "writeback source routes do not cover receiver keys");
    }
    return ErrorInfo::OkStatus();
}

uint64_t P2PSchedulerPrefillWrite::PrefillWriteRequest::progressSeq() const {
    std::lock_guard<std::mutex> lock(mutex);
    return progress_seq;
}

void P2PSchedulerPrefillWrite::PrefillWriteRequest::notifyProgress() {
    std::lock_guard<std::mutex> lock(mutex);
    ++progress_seq;
    cv.notify_all();
}

void P2PSchedulerPrefillWrite::PrefillWriteRequest::waitForProgress(uint64_t observed_seq) {
    std::unique_lock<std::mutex> lock(mutex);
    cv.wait_for(lock, std::chrono::milliseconds(kControlInitialIntervalMs), [this, observed_seq] {
        return progress_seq != observed_seq;
    });
}

bool P2PSchedulerPrefillWrite::WriteTransfer::targetStopped() const {
    return receive_stopped && (!load_context || load_context->done());
}

void P2PSchedulerPrefillWrite::WriteTransfer::observe(PrefillWriteRequest& request,
                                                      const std::function<void()>& notify) {
    const auto observe_direction = [&](auto& submission, auto& result, const char* stage) {
        if (submission.valid()
            && submission.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            std::string failure;
            result = p2p_internal::callForResult([&] { return submission.get(); }, failure);
            if (!result) {
                request.fail(ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                                       std::string(stage) + " broadcast failed: " + failure));
                return;
            }
            result->setDoneCallback(notify);
        }
        if (result) {
            const auto first = result->firstError();
            if (first.error.hasError()) {
                request.fail(first.error);
            }
            if (result->done() && !result->success()) {
                request.fail(ErrorInfo(result->errorCode(), result->errorMessage()));
            }
        }
    };
    observe_direction(receive_submission, receive_result, "writeback receive");
    if (receive_result && receive_result->done() && receive_result->success()) {
        receive_stopped = true;
    }
    observe_direction(send_submission, send_result, "writeback send");
    if (load_context && load_context->done() && !load_context->success()) {
        request.fail(ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "writeback local load failed"));
    }
}

void P2PSchedulerPrefillWrite::runDrainLoop() {
    std::unique_lock<std::mutex> lock(requests_mutex_);
    while (!stopped_) {
        if (pending_drains_.empty()) {
            requests_cv_.wait_for(lock, std::chrono::seconds(1), [this] { return stopped_ || !pending_drains_.empty(); });
        } else {
            requests_cv_.wait_for(lock, std::chrono::milliseconds(kControlInitialIntervalMs),
                                  [this] { return stopped_ || drain_aborted_.load(); });
        }
        if (stopped_)
            break;
        std::vector<std::shared_ptr<WriteDrain>> drains;
        for (const auto& [key, drain] : pending_drains_)
            drains.push_back(drain);
        lock.unlock();
        for (const auto& drain : drains) {
            if (!drain_aborted_.load())
                advanceDrain(*drain);
            if (drain->transfer.targetStopped() || drain_aborted_.load())
                finishWrite(*drain);
        }
        if (metrics_reporter_ || !drains.empty())
            checkRetainedResources(false);
        drains.clear();
        lock.lock();
    }
}

void P2PSchedulerPrefillWrite::advanceDrain(WriteDrain& drain) {
    auto& transfer = drain.transfer;
    const auto notify = [weak = std::weak_ptr<PrefillWriteRequest>(drain.state)] {
        if (const auto state = weak.lock())
            state->notifyProgress();
    };
    try {
        transfer.observe(*drain.state, notify);
        if (transfer.load_context)
            transfer.load_context->abortPending();
        if (drain.control_submission.valid()
            && drain.control_submission.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            std::string failure;
            drain.control_result = p2p_internal::callForResult([&] { return drain.control_submission.get(); }, failure);
            if (drain.control_result)
                drain.control_result->setProgressCallback(notify);
            else {
                drain.cancel_confirmed = false;
                RTP_LLM_LOG_WARNING("writeback receive stop control failed, unique_key=%s: %s",
                                    drain.unique_key.c_str(),
                                    failure.c_str());
                drain.next_control_ms = P2PNotification::nowMs() + drain.interval_ms;
                drain.interval_ms     = std::min(drain.interval_ms * 2, kControlMaxIntervalMs);
            }
        }
        if (drain.control_result && drain.control_result->done()) {
            const auto status = P2PBroadcastClient::leaseStatus(drain.control_result);
            drain.cancel_confirmed  = status.success;
            if (status.success) {
                transfer.receive_stopped = transfer.receive_stopped || status.allStopped();
            }
            drain.control_result.reset();
            drain.next_control_ms = P2PNotification::nowMs() + drain.interval_ms;
            drain.interval_ms     = std::min(drain.interval_ms * 2, kControlMaxIntervalMs);
        }
        if (transfer.targetStopped() || drain_aborted_.load(std::memory_order_acquire))
            return;
        if (transfer.receive_started && !transfer.receive_stopped && !drain.control_submission.valid() && !drain.control_result
            && P2PNotification::nowMs() >= drain.next_control_ms) {
            const auto timeout_limit =
                std::min(config_.p2p_lease_query_timeout_ms,
                         drain.cancel_confirmed ? kLeaseQueryTimeoutMs : config_.p2p_cancel_broadcast_timeout_ms);
            drain.control_submission = submitWriteRpc(
                control_pool_,
                [client = client_, key = drain.unique_key, deadline_ms = drain.deadline_ms, timeout_limit,
                 cancel_confirmed = drain.cancel_confirmed] {
                    return cancel_confirmed ?
                               client->queryLeaseStatusAsync(key, HANDLE_WRITE, timeout_limit) :
                               client->controlWriteAsync(key, CANCEL_HANDLE_WRITE, deadline_ms, timeout_limit);
                },
                notify);
            if (!drain.control_submission.valid()) {
                drain.cancel_confirmed = false;
                drain.next_control_ms  = P2PNotification::nowMs() + drain.interval_ms;
                drain.interval_ms      = std::min(drain.interval_ms * 2, kControlMaxIntervalMs);
            }
        }
    } catch (const std::exception& e) {
        drain.cancel_confirmed = false;
        RTP_LLM_LOG_ERROR("writeback receive cleanup iteration failed, retaining target KV, unique_key=%s: %s",
                          drain.unique_key.c_str(),
                          e.what());
        drain.next_control_ms = P2PNotification::nowMs() + drain.interval_ms;
        drain.interval_ms     = std::min(drain.interval_ms * 2, kControlMaxIntervalMs);
    } catch (...) {
        drain.cancel_confirmed = false;
        RTP_LLM_LOG_ERROR("writeback receive cleanup iteration failed, retaining target KV, unique_key=%s",
                          drain.unique_key.c_str());
        drain.next_control_ms = P2PNotification::nowMs() + drain.interval_ms;
        drain.interval_ms     = std::min(drain.interval_ms * 2, kControlMaxIntervalMs);
    }
}

P2PSchedulerPrefillWrite::ActiveWriteHandler::ActiveWriteHandler(P2PSchedulerPrefillWrite& scheduler,
                                                                const std::string& unique_key) {
    std::lock_guard<std::mutex> lock(scheduler.requests_mutex_);
    if (scheduler.stopping_) {
        const auto existing = scheduler.findRequestLocked(unique_key);
        existing_request_stopped_ = !existing || existing->isStopped();
        return;
    }
    scheduler_ = &scheduler;
    ++scheduler_->active_write_handlers_;
}

P2PSchedulerPrefillWrite::ActiveWriteHandler::~ActiveWriteHandler() {
    if (scheduler_) {
        std::lock_guard<std::mutex> lock(scheduler_->requests_mutex_);
        --scheduler_->active_write_handlers_;
        scheduler_->requests_cv_.notify_all();
    }
}

void P2PSchedulerPrefillWrite::handleWrite(const P2PConnectorStartWriteBackRequestPB& request,
                                           P2PConnectorStartWriteBackResponsePB&      response,
                                           std::function<bool()>                      is_cancelled) {
    const ActiveWriteHandler handler(*this, request.unique_key());
    response.Clear();
    if (!handler.admitted()) {
        response.set_error_code(transErrorCodeToRPC(ErrorCode::CANCELLED));
        response.set_error_message("writeback receiver stopping");
        response.set_stopped(handler.existingRequestStopped());
        return;
    }
    const auto                     start_us = currentTimeUs();
    WriteSchedulerMetricsCollector metrics;
    metrics.prefill   = true;
    const auto report = [&] {
        metrics.total_cost_time_us = currentTimeUs() - start_us;
        reportWritebackMetrics(metrics_reporter_, metrics);
    };
    const auto report_stage =
        [&](const char* stage, int64_t stage_start_us, const ErrorInfo& error, int64_t transferred_bytes = -1) {
            WritebackStageMetricsCollector collector;
            collector.prefill           = true;
            collector.stage             = stage;
            collector.error             = error;
            collector.cost_time_us      = currentTimeUs() - stage_start_us;
            collector.transferred_bytes = transferred_bytes;
            reportWritebackMetrics(metrics_reporter_, collector);
        };
    const auto reject = [&](const ErrorInfo& error) {
        response.Clear();
        response.set_error_code(transErrorCodeToRPC(error.code()));
        response.set_error_message(error.ToString());
        const auto existing = findRequest(request.unique_key());
        response.set_stopped(!existing || existing->isStopped());
        metrics.error = error;
        RTP_LLM_LOG_INFO("writeback_rejected request_id=%ld unique_key=%s error=%s",
                         request.request_id(),
                         request.unique_key().c_str(),
                         error.ToString().c_str());
        report();
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
    const int64_t deadline_ms = now_ms + std::min(request.timeout_ms(), config_.load_cache_timeout_ms);
    const auto stop_at_ms = p2p_internal::writebackSteadyDeadline(deadline_ms);
    bool early_duplicate = false;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        pruneRequestsLocked(P2PNotification::nowMs());
        early_duplicate = findRequestLocked(request.unique_key()) != nullptr;
    }
    if (early_duplicate) {
        return reject(ErrorInfo(ErrorCode::CANCELLED, "duplicate or cancelled StartWriteBack key"));
    }
    if (is_cancelled && is_cancelled()) {
        return reject(ErrorInfo(ErrorCode::CANCELLED, "StartWriteBack cancelled"));
    }
    const auto plan_start_us = currentTimeUs();
    const auto planned = planFor(request.decode_tp_size(),
                                 request.decode_pipeline_layout(),
                                 request.decode_workers_size(),
                                 request.unique_key());
    report_stage("plan", plan_start_us, planned->error);
    if (!planned->ok()) {
        return reject(planned->error);
    }
    if (request.plan_digest() != planned->plan.digest() || request.decode_workers_size() == 0) {
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

    // step 1: validate canonical key alignment and token hash against declared complete KV range
    const size_t block_size = config_.cache_key_tokens_per_block;
    const size_t blocks     = request.cache_keys_size();
    const auto&  pc        = config_.parallelism_config;
    const auto&  cp_config = pc.prefill_cp_config;
    const size_t cp_size   = cp_config.is_enabled() && cp_config.kv_cache_sharded ? pc.tp_size : 1;
    const auto&  cp_mapper = coordinator_manager_->cpSlotMapper();
    if (cp_size == 0 || block_size == 0 || blocks % cp_size != 0 || block_size != coordinator_manager_->seqSizePerBlock()
        || cp_size != static_cast<size_t>(cp_mapper ? cp_mapper->cpSize() : 1)) {
        return invalid("writeback allocator layout or canonical key alignment mismatch");
    }
    const size_t prompt_blocks = request.input_length() / block_size / cp_size * cp_size;
    if (blocks > request.token_ids().size() / block_size || prompt_blocks > blocks) {
        return invalid("writeback keys exceed the declared complete KV range");
    }
    const CacheKeysType keys(request.cache_keys().begin(), request.cache_keys().end());
    if (std::set<int64_t>(keys.begin(), keys.end()).size() != keys.size()
        || calculateCacheKeys(request.token_ids().data(), blocks * block_size, block_size, config_.cache_key_hash_seed)
               != keys) {
        return invalid("writeback cache key hash mismatch");
    }
    if (P2PNotification::nowMs() >= stop_at_ms) {
        return reject(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback expired during admission"));
    }
    if (is_cancelled && is_cancelled()) {
        return reject(ErrorInfo(ErrorCode::CANCELLED, "StartWriteBack cancelled"));
    }

    // step 2: register request state so concurrent cancels and shutdown barriers observe it
    const auto& unique_key = request.unique_key();
    auto        state      = std::make_shared<PrefillWriteRequest>(stop_at_ms);
    bool        inserted   = false;
    bool        table_full = false;
    {
        std::lock_guard<std::mutex> lock(requests_mutex_);
        pruneRequestsLocked(P2PNotification::nowMs());
        if (!stopping_ && !findRequestLocked(unique_key)) {
            if (requests_.size() + completed_requests_.size() >= kMaxWritebackRequestEntries) {
                table_full = true;
            } else {
                const auto expiry = request_expirations_.emplace(holdUntilMs(state->stop_at_ms), unique_key);
                try {
                    inserted = requests_.emplace(unique_key, state).second;
                } catch (...) {
                    request_expirations_.erase(expiry);
                    throw;
                }
            }
        }
    }
    if (table_full)
        return reject(ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback request table full"));
    if (!inserted)
        return reject(ErrorInfo(ErrorCode::CANCELLED, "duplicate, cancelled or stopped StartWriteBack key"));

    const auto notify = [weak = std::weak_ptr<PrefillWriteRequest>(state)] {
        if (const auto current = weak.lock())
            current->notifyProgress();
    };
    const auto transferDeadline = [&] {
        return currentTimeMs() + std::max<int64_t>(0, stop_at_ms - P2PNotification::nowMs());
    };
    WriteTransfer transfer;
    size_t                                                   start             = 0;
    size_t                                                   count             = 0;
    int64_t                                                  hold_start_us     = 0;
    int64_t                                                  transfer_start_us = 0;
    bool                                                     transfer_reported = false;
    const auto                                               finish_transfer   = [&](const ErrorInfo& error) {
        if (transfer_start_us > 0 && !transfer_reported) {
            transfer_reported = true;
            report_stage("transfer", transfer_start_us, error, metrics.planned_bytes);
        }
    };
    const auto observe = [&] {
        transfer.observe(*state, notify);
        if (is_cancelled && is_cancelled())
            state->fail(ErrorInfo(ErrorCode::CANCELLED, "StartWriteBack cancelled"));
        if (P2PNotification::nowMs() >= stop_at_ms)
            state->fail(ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "writeback transfer deadline exceeded"));
    };
    const auto waitFor = [&](const auto& ready) {
        for (;;) {
            const auto seq = state->progressSeq();
            observe();
            if (state->errorInfo().hasError())
                return false;
            if (ready())
                return true;
            state->waitForProgress(seq);
        }
    };
    const auto run = [&]() -> ErrorInfo {
        observe();
        if (state->errorInfo().hasError())
            return state->errorInfo();
        // step 3: admit target KV blocks (and optional HOST/DISK prefix load) atomically under malloc_mutex_
        const auto admission_start_us = currentTimeUs();
        const auto error              = coordinator_manager_->admitWriteBackDecodeCache(keys,
                                                                           prompt_blocks,
                                                                           transfer.resource,
                                                                           start,
                                                                           transfer.load_context,
                                                                           config_.p2p_writeback_require_device_prefix,
                                                                           config_.p2p_writeback_max_kv_ratio);
        if (transfer.resource) {
            hold_start_us = currentTimeUs();
            std::lock_guard<std::mutex> lock(state->mutex);
            state->resource_start_time_us = hold_start_us;
        }
        report_stage("admission", admission_start_us, error);
        if (error.hasError())
            return error;
        if (transfer.load_context)
            transfer.load_context->onDone([notify](ErrorInfo) { notify(); });
        observe();
        if (state->errorInfo().hasError())
            return state->errorInfo();
        count = blocks - start;
        if (count == 0 && !transfer.load_context) {
            metrics.committed_token_count = 0;
            return ErrorInfo::OkStatus();
        }
        P2PBroadcastClient::RankRoutes routes(transfer_addrs_.size());
        P2PBroadcastClient::RankRoutes send_routes;
        if (count > 0) {
            const auto error =
                buildPrefillRankRoutes(planned->plan, *transfer.resource, start, count, routes, &metrics.planned_bytes);
            if (error.hasError())
                return error;
            const auto send_error = selectSendRoutes(request, routes, send_routes);
            if (send_error.hasError())
                return send_error;
        }
        transfer_start_us = currentTimeUs();
        if (transfer.load_context && !transfer.load_context->commit())
            return ErrorInfo(ErrorCode::CACHE_STORE_STORE_FAILED, "writeback local load submission failed");
        if (count > 0) {
            // step 4 (Phase 1): broadcast HANDLE_WRITE to all Prefill workers and wait for HANDLE_WRITE_READY
            P2PBroadcastClient::BroadcastParams params;
            params.request_id          = request.request_id();
            params.unique_key          = unique_key;
            params.deadline_ms         = transferDeadline();
            params.request_deadline_ms = params.deadline_ms;
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
            transfer.receive_submission = submitWriteRpc(
                control_pool_,
                [client = client_, params = std::move(params)]() mutable {
                    return client->broadcast(std::move(params));
                },
                notify);
            if (!transfer.receive_submission.valid())
                return ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback receive submission queue unavailable");
            transfer.receive_started   = true;
            transfer.receive_stopped   = false;
            metrics.submitted = true;
            if (!waitFor([&] {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    return std::all_of(state->registered_workers.begin(),
                                       state->registered_workers.end(),
                                       [](bool registered) { return registered; });
                }))
                return state->errorInfo();
            // step 4 (Phase 2): broadcast WRITE to all Decode source workers with routes and writeback_kv_step
            P2PBroadcastClient::BroadcastParams send;
            send.request_id          = request.request_id();
            send.unique_key          = unique_key;
            send.deadline_ms         = transferDeadline();
            send.request_deadline_ms = send.deadline_ms;
            send.type                = WRITE;
            send.writeback_kv_step   = request.writeback_kv_step();
            send.plan_digest         = planned->plan.digest();
            for (const auto& address : transfer_addrs_) {
                std::string host;
                int32_t     port = 0;
                parseGrpcHostPort(address, host, port);
                send.peer_workers.emplace_back(std::move(host), port);
            }
            send.routes     = std::move(send_routes);
            transfer.send_submission = submitWriteRpc(
                control_pool_,
                [this, addresses = source_addrs, send = std::move(send)]() mutable {
                    return P2PBroadcastClient::broadcastToAddresses(source_rpc_pool_, addresses, std::move(send));
                },
                notify);
            if (!transfer.send_submission.valid())
                return ErrorInfo(ErrorCode::OUTPUT_QUEUE_FULL, "writeback sender submission queue unavailable");
        }
        if (!waitFor([&] {
                return transfer.targetStopped()
                       && (count == 0
                           || (transfer.send_result && transfer.send_result->done() && transfer.send_result->success()));
            }))
            return state->errorInfo();
        finish_transfer(ErrorInfo::OkStatus());
        if (is_cancelled && is_cancelled()) {
            state->cancel(ErrorInfo(ErrorCode::CANCELLED, "StartWriteBack cancelled"));
        }
        // step 5: arbitrate commit vs cancellation and publish transferred suffix into BlockTreeCache
        if (const auto error = state->beginCommit(); error.hasError()) {
            return error;
        }
        const auto commit_start_us = currentTimeUs();
        const auto commit_error =
            coordinator_manager_->commitWriteBackDecodeCache(*transfer.resource, start, transferDeadline());
        report_stage("commit", commit_start_us, commit_error);
        if (commit_error.ok()) {
            metrics.committed_token_count = count * block_size;
        }
        return commit_error;
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
    finish_transfer(state->errorInfo());
    WriteDrain drain{unique_key, state, std::move(transfer), std::move(metrics),
                     deadline_ms, start_us, hold_start_us, currentTimeUs()};
    if (!drain.transfer.targetStopped()) {
        // A local load that never submitted can be aborted and released before replying.
        advanceDrain(drain);
    }
    if (drain.transfer.targetStopped()) {
        finishWrite(drain);
    } else {
        // Cleanup outlives the RPC. CancelWriteBack can continue to query this request until all receivers stop.
        auto pending = std::make_shared<WriteDrain>(std::move(drain));
        {
            std::lock_guard<std::mutex> lock(requests_mutex_);
            if (pending->transfer.resource)
                retained_resources_.emplace(unique_key, pending->transfer.resource);
            pending_drains_.emplace(unique_key, std::move(pending));
        }
        requests_cv_.notify_all();
        checkRetainedResources(false);
        for (;;) {
            const auto seq = state->progressSeq();
            if (state->isStopped() || drain_aborted_.load() || P2PNotification::nowMs() >= stop_at_ms
                || (is_cancelled && is_cancelled())) {
                break;
            }
            state->waitForProgress(seq);
        }
    }
    const auto error = state->errorInfo();
    response.set_accepted_block_count(error.ok() ? count : 0);
    response.set_stopped(state->isStopped());
    response.set_error_code(transErrorCodeToRPC(error.code()));
    response.set_error_message(error.ok() ? "" : error.ToString());
}

void P2PSchedulerPrefillWrite::finishWrite(WriteDrain& drain) {
    auto&       transfer   = drain.transfer;
    const auto& unique_key = drain.unique_key;
    const auto& state      = drain.state;
    const bool target_stopped = transfer.targetStopped();
    bool       became_empty   = false;
    if (target_stopped) {
        transfer.load_context.reset();
        {
            std::lock_guard<std::mutex> lock(requests_mutex_);
            const bool erased = retained_resources_.erase(unique_key) > 0;
            became_empty      = erased && retained_resources_.empty();
        }
        transfer.resource.reset();
    } else {
        // Shutdown timed out before physical stop was confirmed: keep target KV pinned until local receiver
        // teardown and until any in-flight local HOST/DISK load drains.
        if (transfer.load_context && transfer.resource) {
            transfer.load_context->onDone([resource = transfer.resource](ErrorInfo) {});
        }
        std::lock_guard<std::mutex> lock(requests_mutex_);
        if (transfer.resource) {
            retained_resources_.try_emplace(unique_key, std::move(transfer.resource));
        }
        if (transfer.load_context) {
            retained_load_contexts_.push_back(std::move(transfer.load_context));
        }
    }
    const auto released_us = currentTimeUs();
    ErrorInfo  error;
    {
        std::lock_guard<std::mutex> requests_lock(requests_mutex_);
        std::lock_guard<std::mutex> lock(state->mutex);
        pending_drains_.erase(unique_key);
        state->stopped = target_stopped;
        error          = state->error;
        if (target_stopped) {
            auto completed = requests_.extract(unique_key);
            if (P2PNotification::nowMs() < holdUntilMs(state->stop_at_ms)) {
                completed_requests_.insert(std::move(completed));
            }
        }
        ++state->progress_seq;
        state->cv.notify_all();
    }
    checkRetainedResources(/*force=*/became_empty);
    requests_cv_.notify_all();
    auto& metrics = drain.metrics;
    metrics.no_transfer        = !transfer.receive_started && error.ok();
    metrics.error              = error;
    metrics.hold_time_us       = drain.hold_start_us > 0 ? released_us - drain.hold_start_us : 0;
    metrics.drain_time_us      = released_us - drain.drain_start_us;
    metrics.total_cost_time_us = released_us - drain.start_us;
    reportWritebackMetrics(metrics_reporter_, metrics);
}

}  // namespace rtp_llm
