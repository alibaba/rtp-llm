#include "rtp_llm/cpp/model_rpc/PrefillBatchRpcServer.h"

#include "rtp_llm/cpp/config/ConfigModules.h"
#include "rtp_llm/cpp/utils/AtomicUtil.h"
#include "rtp_llm/cpp/utils/ProfilingScope.h"
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

using namespace std;
namespace rtp_llm {

// Dedicated managed executor for idle priority-cancel cleanup. Tasks submitted
// here never wait for a prepare/Fetch operation to exit; operation owners only
// submit after they have exited. This prevents Cancel storms from consuming
// the finite prepare pool with waiters.
class PriorityCancelExecutor {
public:
    explicit PriorityCancelExecutor(size_t worker_count) {
        worker_count = std::max<size_t>(worker_count, 1);
        workers_.reserve(worker_count);
        for (size_t i = 0; i < worker_count; ++i) {
            workers_.emplace_back([this] { run(); });
        }
    }

    ~PriorityCancelExecutor() {
        stop();
    }

    bool submit(std::function<void()> task) {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_) {
                return false;
            }
            tasks_.push_back(std::move(task));
        }
        cv_.notify_one();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            if (stopping_) {
                return;
            }
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        workers_.clear();
    }

private:
    void run() {
        while (true) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
                if (stopping_ && tasks_.empty()) {
                    return;
                }
                task = std::move(tasks_.front());
                tasks_.pop_front();
            }
            try {
                task();
            } catch (const std::exception& e) {
                RTP_LLM_LOG_ERROR("priority-cancel finalizer failed: %s", e.what());
            } catch (...) {
                RTP_LLM_LOG_ERROR("priority-cancel finalizer failed with unknown exception");
            }
        }
    }

    std::mutex                        mu_;
    std::condition_variable           cv_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::thread>          workers_;
    bool                              stopping_{false};
};

namespace {

grpc::Status statusFromErrorInfo(const ErrorInfo& error_info) {
    if (!error_info.hasError()) {
        return grpc::Status::OK;
    }
    const auto     error_msg       = error_info.ToString();
    auto           grpc_error_code = transErrorCodeToGrpc(error_info.code());
    ErrorDetailsPB error_details;
    error_details.set_error_code(static_cast<int>(error_info.code()));
    error_details.set_error_message(error_msg);
    std::string error_details_serialized;
    if (error_details.SerializeToString(&error_details_serialized)) {
        return grpc::Status(grpc_error_code, error_msg, error_details_serialized);
    }
    RTP_LLM_LOG_WARNING(
        "statusFromErrorInfo error details serialize to string failed, error code [%s], error message [%s]",
        ErrorCodeToString(error_info.code()).c_str(),
        error_msg.c_str());
    return grpc::Status(grpc_error_code, error_msg);
}

void addBatchSuccess(EnqueueBatchResponsePB* response, int64_t request_id) {
    auto* success = response->add_successes();
    success->set_request_id(request_id);
}

void addBatchError(EnqueueBatchResponsePB* response, int64_t request_id, int64_t code, const std::string& msg) {
    auto* error = response->add_errors();
    error->set_request_id(request_id);
    auto* error_info = error->mutable_error_info();
    error_info->set_error_code(code);
    error_info->set_error_message(msg);
}

int64_t batchErrorCode(const grpc::Status& status) {
    // AutoTPM 8429 is carried in gRPC details because RESOURCE_EXHAUSTED is
    // only its transport projection. Preserve the domain code when adapting
    // the status into EnqueueBatchErrorPB.
    ErrorDetailsPB details;
    if (!status.error_details().empty() && details.ParseFromString(status.error_details())
        && details.error_code() == static_cast<int64_t>(ErrorCode::PRIORITY_PREEMPTED)) {
        return details.error_code();
    }
    return status.error_code();
}

// Both cancel-before-register fences and recently completed request ids only
// need to cover the rolling dispatch/reconciliation window. Keeping the two
// registries on the same bounded lifetime also prevents unbounded id history.
constexpr int64_t kPriorityCancelRegistryTtlMs = 10 * 60 * 1000;

}  // namespace

void DeferredPrefillContext::cancel(const grpc::Status& status) {
    if (!context) {
        return;
    }
    if (!context->tryMarkOtherTerminal()) {
        return;
    }
    context->error_status = status;
    context->cancel_state->store(true);
    context->tryCancelDownstream();
    auto stream = context->getStream();
    if (stream && !stream->hasError() && stream->getStatus() != StreamState::FINISHED) {
        stream->reportError(status.error_code() == grpc::StatusCode::CANCELLED ? ErrorCode::CANCELLED :
                                                                                 ErrorCode::UNKNOWN_ERROR,
                            status.error_message());
    }
}

bool DeferredPrefillContext::finishOperation() {
    std::lock_guard<std::mutex> lock(operation_mu_);
    if (!operation_active_) {
        return false;
    }
    operation_active_ = false;
    if (priority_finalize_requested_ && !priority_finalize_claimed_) {
        priority_finalize_claimed_ = true;
        return true;
    }
    return false;
}

DeferredPrefillContext::StartOperationResult DeferredPrefillContext::tryStartOperation() {
    std::lock_guard<std::mutex> lock(operation_mu_);
    if (operation_active_) {
        return {};
    }
    if (priority_finalize_requested_) {
        if (!priority_finalize_claimed_) {
            priority_finalize_claimed_ = true;
            return {/*started=*/false, /*priority_finalizer_claimed=*/true};
        }
        return {};
    }
    if (!context || context->terminalCause() != PrefillTerminalCause::ACTIVE || priority_finalize_claimed_) {
        return {};
    }
    operation_active_ = true;
    return {/*started=*/true, /*priority_finalizer_claimed=*/false};
}

bool DeferredPrefillContext::requestPriorityFinalization() {
    std::lock_guard<std::mutex> lock(operation_mu_);
    priority_finalize_requested_ = true;
    if (!operation_active_ && !priority_finalize_claimed_) {
        priority_finalize_claimed_ = true;
        return true;
    }
    return false;
}

grpc::Status DeferredPrefillContextMap::registerActive(int64_t                                        request_id,
                                                       const std::shared_ptr<DeferredPrefillContext>& deferred) {
    if (!deferred || !deferred->context) {
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "active prefill context is missing");
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (stopping_) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill batch server is shutting down");
    }
    const int64_t now_ms = autil::TimeUtility::currentTimeInMilliSeconds();
    sweepPriorityPreemptionTombstones(now_ms);
    sweepRecentlySeenRequests(now_ms);
    if (priority_preemption_tombstones_.find(request_id) != priority_preemption_tombstones_.end()) {
        return statusFromErrorInfo(ErrorInfo(ErrorCode::PRIORITY_PREEMPTED, "preempted by a higher-priority request"));
    }
    auto active = active_contexts_.find(request_id);
    if (active != active_contexts_.end()) {
        auto current = active->second.lock();
        if (current) {
            return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "request already exists in active context map");
        }
        active_contexts_.erase(active);
    }
    active_contexts_[request_id] = deferred;
    rememberRecentlySeenRequest(request_id, now_ms);
    return grpc::Status::OK;
}

grpc::Status DeferredPrefillContextMap::store(int64_t                                        request_id,
                                              const std::shared_ptr<DeferredPrefillContext>& deferred) {
    std::lock_guard<std::mutex> lock(mu_);
    if (stopping_) {
        return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill batch server is shutting down");
    }
    if (!contexts_.emplace(request_id, deferred).second) {
        return grpc::Status(grpc::StatusCode::ALREADY_EXISTS, "request already exists in deferred context map");
    }
    return grpc::Status::OK;
}

grpc::Status DeferredPrefillContextMap::armTtl(int64_t                                        request_id,
                                               const std::shared_ptr<DeferredPrefillContext>& deferred,
                                               std::chrono::milliseconds                      ttl) {
    auto alarm = std::make_shared<grpc::Alarm>();
    auto weak  = weak_from_this();
    ttl        = std::max(ttl, std::chrono::milliseconds(1));
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stopping_) {
            return grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill batch server is shutting down");
        }
        auto it = contexts_.find(request_id);
        if (it == contexts_.end() || it->second.get() != deferred.get()) {
            return grpc::Status(grpc::StatusCode::FAILED_PRECONDITION,
                                "request context is missing from deferred context map");
        }
        deferred->ttl_alarm = alarm;
        // Arm before returning success. cancelAll() cannot remove this context
        // until Set() has completed, so Cancel() never races an unset Alarm.
        alarm->experimental().Set(std::chrono::system_clock::now() + ttl,
                                  [weak, request_id, expected = deferred.get()](bool ok) {
                                      if (ok) {
                                          if (auto contexts = weak.lock()) {
                                              contexts->expire(request_id, expected);
                                          }
                                      }
                                  });
    }
    return grpc::Status::OK;
}

grpc::Status DeferredPrefillContextMap::take(int64_t request_id, std::shared_ptr<DeferredPrefillContext>& deferred) {
    deferred.reset();
    {
        std::lock_guard<std::mutex> lock(mu_);
        sweepPriorityPreemptionTombstones(autil::TimeUtility::currentTimeInMilliSeconds());
        if (priority_preemption_tombstones_.find(request_id) != priority_preemption_tombstones_.end()) {
            return statusFromErrorInfo(
                ErrorInfo(ErrorCode::PRIORITY_PREEMPTED, "preempted by a higher-priority request"));
        }
        auto it = contexts_.find(request_id);
        if (it == contexts_.end()) {
            return grpc::Status(grpc::StatusCode::NOT_FOUND,
                                "request [" + std::to_string(request_id) + "] not found in deferred context map");
        }
        auto start_result = it->second->tryStartOperation();
        if (!start_result.started) {
            return it->second->context && it->second->context->isPriorityPreempted() ?
                       statusFromErrorInfo(
                           ErrorInfo(ErrorCode::PRIORITY_PREEMPTED, "preempted by a higher-priority request")) :
                       grpc::Status(grpc::StatusCode::FAILED_PRECONDITION, "request context is already terminal");
        }
        deferred = std::move(it->second);
        contexts_.erase(it);
    }
    if (deferred->ttl_alarm) {
        deferred->ttl_alarm->Cancel();
    }
    return grpc::Status::OK;
}

std::shared_ptr<DeferredPrefillContext> DeferredPrefillContextMap::remove(int64_t                       request_id,
                                                                          const DeferredPrefillContext* expected) {
    std::shared_ptr<DeferredPrefillContext> deferred;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto                        it = contexts_.find(request_id);
        if (it == contexts_.end() || it->second.get() != expected) {
            return nullptr;
        }
        deferred = std::move(it->second);
        contexts_.erase(it);
        if (deferred->context) {
            deferred->context->tryMarkOtherTerminal();
        }
        auto active = active_contexts_.find(request_id);
        if (active != active_contexts_.end()) {
            auto current = active->second.lock();
            if (!current || current.get() == expected) {
                active_contexts_.erase(active);
            }
        }
        rememberRecentlySeenRequest(request_id, autil::TimeUtility::currentTimeInMilliSeconds());
    }
    if (deferred->ttl_alarm) {
        deferred->ttl_alarm->Cancel();
    }
    return deferred;
}

PriorityCancelResult DeferredPrefillContextMap::cancelByPriorityPreemption(
    int64_t request_id, std::shared_ptr<DeferredPrefillContext>& deferred, bool* newly_installed) {
    if (newly_installed) {
        *newly_installed = false;
    }
    std::shared_ptr<grpc::Alarm> alarm;
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (stopping_) {
            return PriorityCancelResult::NOT_FOUND;
        }
        const int64_t now_ms = autil::TimeUtility::currentTimeInMilliSeconds();
        sweepPriorityPreemptionTombstones(now_ms);
        sweepRecentlySeenRequests(now_ms);
        auto tombstone = priority_preemption_tombstones_.find(request_id);
        if (tombstone != priority_preemption_tombstones_.end()) {
            deferred.reset();
            return tombstone->second.kind == PriorityPreemptionTombstoneKind::ACTIVE_CANCEL ?
                       PriorityCancelResult::ACCEPTED :
                       PriorityCancelResult::TOMBSTONED;
        }
        auto it = active_contexts_.find(request_id);
        if (it == active_contexts_.end()) {
            if (recently_seen_requests_.find(request_id) != recently_seen_requests_.end()) {
                deferred.reset();
                return PriorityCancelResult::NOT_FOUND;
            }
            installPriorityPreemptionTombstone(request_id, now_ms, PriorityPreemptionTombstoneKind::ABSENT_FENCE);
            deferred.reset();
            return PriorityCancelResult::TOMBSTONED;
        }
        deferred = it->second.lock();
        if (!deferred) {
            active_contexts_.erase(it);
            // The active entry itself proves registerActive previously won,
            // even if its weak owner disappeared after the seen TTL elapsed.
            // Refresh that evidence instead of misclassifying it as an
            // enqueue that never reached this Engine.
            rememberRecentlySeenRequest(request_id, now_ms);
            return PriorityCancelResult::NOT_FOUND;
        }

        // The terminal-cause CAS below is the first-cause linearization point.
        // Holding the registry lock also prevents Fetch/finish from moving the
        // context while the priority latch and tombstone are installed.
        const auto preempt_result = deferred->context->requestPriorityPreempt();
        if (preempt_result == PriorityPreemptionRequestResult::REJECTED) {
            deferred.reset();
            return PriorityCancelResult::NOT_FOUND;
        }
        if (newly_installed) {
            *newly_installed = preempt_result == PriorityPreemptionRequestResult::INSTALLED;
        }
        installPriorityPreemptionTombstone(request_id, now_ms, PriorityPreemptionTombstoneKind::ACTIVE_CANCEL);
        auto fetchable = contexts_.find(request_id);
        if (fetchable != contexts_.end() && fetchable->second.get() == deferred.get()) {
            contexts_.erase(fetchable);
            alarm = deferred->ttl_alarm;
        }
    }
    if (alarm) {
        alarm->Cancel();
    }
    return PriorityCancelResult::ACCEPTED;
}

void DeferredPrefillContextMap::installPriorityPreemptionTombstone(int64_t                         request_id,
                                                                   int64_t                         now_ms,
                                                                   PriorityPreemptionTombstoneKind kind) {
    const int64_t expires_at_ms                 = now_ms + kPriorityCancelRegistryTtlMs;
    priority_preemption_tombstones_[request_id] = PriorityPreemptionTombstone{expires_at_ms, kind};
    priority_preemption_tombstone_expiries_.emplace_back(expires_at_ms, request_id);
}

void DeferredPrefillContextMap::sweepPriorityPreemptionTombstones(int64_t now_ms) {
    while (!priority_preemption_tombstone_expiries_.empty()
           && priority_preemption_tombstone_expiries_.front().first <= now_ms) {
        const auto [expires_at_ms, request_id] = priority_preemption_tombstone_expiries_.front();
        priority_preemption_tombstone_expiries_.pop_front();
        auto tombstone = priority_preemption_tombstones_.find(request_id);
        if (tombstone != priority_preemption_tombstones_.end() && tombstone->second.expires_at_ms == expires_at_ms) {
            priority_preemption_tombstones_.erase(tombstone);
        }
    }
}

void DeferredPrefillContextMap::rememberRecentlySeenRequest(int64_t request_id, int64_t now_ms) {
    const int64_t expires_at_ms         = now_ms + kPriorityCancelRegistryTtlMs;
    recently_seen_requests_[request_id] = expires_at_ms;
    recently_seen_request_expiries_.emplace_back(expires_at_ms, request_id);
}

void DeferredPrefillContextMap::sweepRecentlySeenRequests(int64_t now_ms) {
    while (!recently_seen_request_expiries_.empty() && recently_seen_request_expiries_.front().first <= now_ms) {
        const auto [expires_at_ms, request_id] = recently_seen_request_expiries_.front();
        recently_seen_request_expiries_.pop_front();
        auto seen = recently_seen_requests_.find(request_id);
        if (seen != recently_seen_requests_.end() && seen->second == expires_at_ms) {
            recently_seen_requests_.erase(seen);
        }
    }
}

void DeferredPrefillContextMap::publishPriorityPreemptionCanceled(int64_t                       request_id,
                                                                  const DeferredPrefillContext* expected) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        tombstone = priority_preemption_tombstones_.find(request_id);
    if (tombstone != priority_preemption_tombstones_.end()
        && tombstone->second.kind == PriorityPreemptionTombstoneKind::ACTIVE_CANCEL) {
        // Typed CANCELED is now observable in WorkerStatus. Retain the same
        // expiry but downgrade the weak active ACK to an absent-request fence.
        tombstone->second.kind = PriorityPreemptionTombstoneKind::ABSENT_FENCE;
    }
    auto active = active_contexts_.find(request_id);
    if (active == active_contexts_.end()) {
        return;
    }
    auto current = active->second.lock();
    if (!current || current.get() == expected) {
        active_contexts_.erase(active);
    }
}

void DeferredPrefillContextMap::finish(int64_t request_id, const DeferredPrefillContext* expected) {
    std::lock_guard<std::mutex> lock(mu_);
    auto                        it = active_contexts_.find(request_id);
    if (it == active_contexts_.end()) {
        return;
    }
    auto current = it->second.lock();
    if (!current || current.get() == expected) {
        if (current && current->context) {
            current->context->tryMarkOtherTerminal();
        }
        active_contexts_.erase(it);
        rememberRecentlySeenRequest(request_id, autil::TimeUtility::currentTimeInMilliSeconds());
    }
}

void DeferredPrefillContextMap::expire(int64_t request_id, const DeferredPrefillContext* expected) {
    std::shared_ptr<DeferredPrefillContext> deferred;
    {
        std::lock_guard<std::mutex> lock(mu_);
        auto                        it = contexts_.find(request_id);
        if (it == contexts_.end() || it->second.get() != expected) {
            return;
        }
        deferred = std::move(it->second);
        contexts_.erase(it);
        if (deferred->context) {
            deferred->context->tryMarkOtherTerminal();
        }
        active_contexts_.erase(request_id);
        rememberRecentlySeenRequest(request_id, autil::TimeUtility::currentTimeInMilliSeconds());
    }
    deferred->cancel(grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, "FetchResponse context TTL expired"));
}

void DeferredPrefillContextMap::stopAccepting() {
    std::lock_guard<std::mutex> lock(mu_);
    stopping_ = true;
}

void DeferredPrefillContextMap::cancelAll(const grpc::Status& status) {
    std::vector<std::shared_ptr<DeferredPrefillContext>> deferred_contexts;
    {
        std::lock_guard<std::mutex> lock(mu_);
        stopping_ = true;
        deferred_contexts.reserve(contexts_.size() + active_contexts_.size());
        std::unordered_set<const DeferredPrefillContext*> collected_contexts;
        auto                                              collect_context = [&deferred_contexts,
                                &collected_contexts](const std::shared_ptr<DeferredPrefillContext>& deferred) {
            if (!deferred || !collected_contexts.insert(deferred.get()).second) {
                return;
            }
            if (deferred->context) {
                deferred->context->tryMarkOtherTerminal();
            }
            deferred_contexts.push_back(deferred);
        };
        for (auto& entry : contexts_) {
            collect_context(entry.second);
        }
        for (auto& entry : active_contexts_) {
            if (auto active = entry.second.lock()) {
                collect_context(active);
            }
        }
        contexts_.clear();
        active_contexts_.clear();
    }
    for (auto& deferred : deferred_contexts) {
        if (deferred->ttl_alarm) {
            deferred->ttl_alarm->Cancel();
        }
        deferred->cancel(status);
    }
}

PriorityCancelResult PrefillBatchRpcServer::onCancelRequest(int64_t request_id) {
    std::shared_ptr<DeferredPrefillContext> deferred;
    const auto result = deferred_contexts_->cancelByPriorityPreemption(request_id, deferred);
    // Active Stage 2/4 operations own their exit and submit finalization only
    // after quiescing. An idle Stage 3 context is submitted immediately. No
    // waiter is ever placed on the prepare pool.
    if (result == PriorityCancelResult::ACCEPTED && deferred && deferred->requestPriorityFinalization()) {
        schedulePriorityFinalization(request_id, deferred);
    }
    return result;
}

void PrefillBatchRpcServer::finishSlotOperation(int64_t                                        request_id,
                                                const std::shared_ptr<DeferredPrefillContext>& deferred) {
    if (deferred && deferred->finishOperation()) {
        schedulePriorityFinalization(request_id, deferred);
    }
}

void PrefillBatchRpcServer::finalizePriorityPreemption(int64_t                                 request_id,
                                                       std::shared_ptr<DeferredPrefillContext> deferred) {
    if (!deferred->context->finalizePriorityPreemption()) {
        // The scheduler still owns the local stream. Retry in a later executor
        // turn rather than occupying a worker in an unbounded polling loop.
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        schedulePriorityFinalization(request_id, std::move(deferred));
        return;
    }
    deferred_contexts_->publishPriorityPreemptionCanceled(request_id, deferred.get());
}

void PrefillBatchRpcServer::schedulePriorityFinalization(int64_t                                 request_id,
                                                         std::shared_ptr<DeferredPrefillContext> deferred) {
    if (!priority_cancel_executor_ || !priority_cancel_executor_->submit([this, request_id, deferred] {
            finalizePriorityPreemption(request_id, deferred);
        })) {
        RTP_LLM_LOG_WARNING("request [%ld] priority-preemption finalizer executor is stopping", request_id);
    }
}

size_t DeferredPrefillContextMap::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return contexts_.size();
}

PrefillBatchRpcServer::PrefillBatchRpcServer() = default;

PrefillBatchRpcServer::~PrefillBatchRpcServer() {
    beginShutdown();
    cancelPendingRequests();
    if (prepare_resource_worker_pool_) {
        prepare_resource_worker_pool_->stop();
        prepare_resource_worker_pool_.reset();
    }
    if (priority_cancel_executor_) {
        priority_cancel_executor_->stop();
        priority_cancel_executor_.reset();
    }
}

grpc::Status PrefillBatchRpcServer::init(const EngineInitParams&                                maga_init_params,
                                         py::object                                             mm_process_engine,
                                         std::unique_ptr<rtp_llm::ProposeModelEngineInitParams> propose_params) {
    auto ret = PrefillRpcServer::init(maga_init_params, mm_process_engine, std::move(propose_params));
    if (!ret.ok()) {
        return ret;
    }
    initThreadPools();
    return grpc::Status::OK;
}

// ---------------------------------------------------------------------------
// Batch infrastructure: short-lived resource preparation only
// ---------------------------------------------------------------------------

void PrefillBatchRpcServer::beginShutdown() {
    if (!stopping_.exchange(true)) {
        deferred_contexts_->stopAccepting();
    }
}

void PrefillBatchRpcServer::cancelPendingRequests() {
    deferred_contexts_->cancelAll(grpc::Status(grpc::StatusCode::UNAVAILABLE, "Prefill batch server is shutting down"));
}

void PrefillBatchRpcServer::initThreadPools() {
    const auto&   pd_sep_config     = maga_init_params_.pd_sep_config;
    const int64_t concurrency_limit = std::max<int64_t>(1, maga_init_params_.concurrency_config.concurrency_limit);
    const int64_t configured_prepare_size       = pd_sep_config.prefill_prepare_resource_pool_size > 0 ?
                                                      pd_sep_config.prefill_prepare_resource_pool_size :
                                                      concurrency_limit * 2;
    const int64_t prepare_resource_threads      = std::max<int64_t>(128, configured_prepare_size);
    const int64_t prepare_resource_queue        = prepare_resource_threads;
    const size_t  prepare_resource_thread_count = static_cast<size_t>(prepare_resource_threads);
    const size_t  prepare_resource_queue_size   = static_cast<size_t>(prepare_resource_queue);

    prepare_resource_worker_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        prepare_resource_thread_count, prepare_resource_queue_size, nullptr, "PrefillPrepareResource");
    RTP_LLM_CHECK_WITH_INFO(prepare_resource_worker_pool_->start(),
                            "PrefillRpcServer prepare-resource thread pool start failed");
    RTP_LLM_LOG_INFO("PrefillRpcServer prepare-resource pool started: threads=%ld queue=%ld "
                     "(configured=%ld concurrency_limit=%ld)",
                     prepare_resource_threads,
                     prepare_resource_queue,
                     pd_sep_config.prefill_prepare_resource_pool_size,
                     concurrency_limit);

    const size_t cancel_workers = static_cast<size_t>(std::min<int64_t>(16, std::max<int64_t>(2, concurrency_limit)));
    priority_cancel_executor_   = std::make_unique<PriorityCancelExecutor>(cancel_workers);
}

// ---------------------------------------------------------------------------
// EnqueueBatch — single-DP adapter for EnqueueGroup
// ---------------------------------------------------------------------------

grpc::Status PrefillBatchRpcServer::EnqueueBatch(grpc::ServerContext*         context,
                                                 const EnqueueBatchRequestPB* request,
                                                 EnqueueBatchResponsePB*      response) {
    RTP_LLM_PROFILE_FUNCTION();
    const auto& parallelism_config = maga_init_params_.parallelism_config;
    RTP_LLM_CHECK_WITH_INFO(parallelism_config.dp_size == 1,
                            "EnqueueBatch only supports single-DP mode, dp_size=%ld",
                            parallelism_config.dp_size);

    const int             local_dp_rank = static_cast<int>(parallelism_config.dp_rank);
    EnqueueGroupRequestPB group_request;
    group_request.set_batch_id(request->batch_id());
    group_request.set_dp_rank(local_dp_rank);
    group_request.set_fetch_attach_timeout_ms(request->fetch_attach_timeout_ms());
    response->set_batch_id(request->batch_id());

    int                         input_count          = 0;
    bool                        duplicate_request_id = false;
    std::unordered_set<int64_t> seen_request_ids;
    for (const auto& dp_slot : request->dp_slots()) {
        for (const auto& external_input : dp_slot.requests()) {
            ++input_count;
            if (external_input.has_input() && !seen_request_ids.insert(external_input.input().request_id()).second) {
                duplicate_request_id = true;
            }
        }
    }
    if (duplicate_request_id) {
        for (const auto& dp_slot : request->dp_slots()) {
            for (const auto& external_input : dp_slot.requests()) {
                if (external_input.has_input()) {
                    addBatchError(response,
                                  external_input.input().request_id(),
                                  grpc::StatusCode::ALREADY_EXISTS,
                                  "duplicate request_id in EnqueueBatch");
                } else {
                    addBatchError(response,
                                  /*request_id=*/0,
                                  grpc::StatusCode::INVALID_ARGUMENT,
                                  "EnqueueBatch external request missing input");
                }
            }
        }
        return grpc::Status::OK;
    }

    for (const auto& dp_slot : request->dp_slots()) {
        for (const auto& external_input : dp_slot.requests()) {
            if (dp_slot.dp_rank() != local_dp_rank) {
                addBatchError(response,
                              external_input.has_input() ? external_input.input().request_id() : 0,
                              grpc::StatusCode::INVALID_ARGUMENT,
                              "EnqueueBatch dp_rank mismatch, request dp_rank " + std::to_string(dp_slot.dp_rank())
                                  + ", local dp_rank " + std::to_string(local_dp_rank));
                continue;
            }
            auto* group_input = group_request.add_requests();
            if (external_input.has_input()) {
                group_input->mutable_input()->CopyFrom(external_input.input());
            }
        }
    }

    EnqueueBatchResponsePB group_response;
    auto                   status = EnqueueGroup(context, &group_request, &group_response);
    response->mutable_successes()->MergeFrom(group_response.successes());
    response->mutable_errors()->MergeFrom(group_response.errors());
    RTP_LLM_CHECK_WITH_INFO(response->successes_size() + response->errors_size() == input_count,
                            "EnqueueBatch result size mismatch: request=%d response=%d",
                            input_count,
                            response->successes_size() + response->errors_size());
    return status;
}

// ---------------------------------------------------------------------------
// EnqueueGroup — mirrors GenerateStreamCall: linear top level over named phases
// ---------------------------------------------------------------------------

grpc::Status PrefillBatchRpcServer::EnqueueGroup(grpc::ServerContext* /*context*/,
                                                 const EnqueueGroupRequestPB* request,
                                                 EnqueueBatchResponsePB*      response) {
    RTP_LLM_PROFILE_FUNCTION();
    response->set_batch_id(request->batch_id());
    if (stopping_.load()) {
        for (const auto& item : request->requests()) {
            addBatchError(response,
                          item.has_input() ? item.input().request_id() : 0,
                          grpc::StatusCode::UNAVAILABLE,
                          "Prefill batch server is shutting down");
        }
        return grpc::Status::OK;
    }

    std::vector<BatchSlot> slots;
    auto                   status = admitGroup(request, response, slots);
    if (status.ok() && !slots.empty()) {
        status = acceptGroup(std::move(slots), response);
    }
    const int response_size = response->successes_size() + response->errors_size();
    RTP_LLM_CHECK_WITH_INFO(response_size == request->requests_size(),
                            "EnqueueGroup result size mismatch: request=%d response=%d",
                            request->requests_size(),
                            response_size);
    return status;
}

grpc::Status PrefillBatchRpcServer::admitGroup(const EnqueueGroupRequestPB* request,
                                               EnqueueBatchResponsePB*      response,
                                               std::vector<BatchSlot>&      slots) {
    std::vector<const GenerateInputPB*> all_inputs;
    all_inputs.reserve(request->requests_size());
    std::unordered_set<int64_t> seen_request_ids;
    bool                        duplicate_request_id = false;
    for (const auto& dp_input : request->requests()) {
        if (!dp_input.has_input()) {
            addBatchError(response,
                          /*request_id=*/0,
                          grpc::StatusCode::INVALID_ARGUMENT,
                          "EnqueueGroup request missing input");
            continue;
        }
        all_inputs.push_back(&dp_input.input());
        if (!seen_request_ids.insert(dp_input.input().request_id()).second) {
            duplicate_request_id = true;
        }
    }

    response->mutable_successes()->Reserve(static_cast<int>(all_inputs.size()));
    response->mutable_errors()->Reserve(static_cast<int>(all_inputs.size()));

    auto add_error_for_all = [&](int64_t code, const std::string& message) {
        for (const auto* input : all_inputs) {
            addBatchError(response, input->request_id(), code, message);
        }
    };

    const int local_dp_rank = static_cast<int>(maga_init_params_.parallelism_config.dp_rank);
    if (request->dp_rank() != local_dp_rank) {
        add_error_for_all(grpc::StatusCode::INVALID_ARGUMENT,
                          "EnqueueGroup dp_rank mismatch, request dp_rank " + std::to_string(request->dp_rank())
                              + ", local dp_rank " + std::to_string(local_dp_rank));
        return grpc::Status::OK;
    }
    if (duplicate_request_id) {
        add_error_for_all(grpc::StatusCode::ALREADY_EXISTS, "duplicate request_id in EnqueueGroup");
        return grpc::Status::OK;
    }

    slots.reserve(all_inputs.size());
    const int group_size = static_cast<int>(all_inputs.size());
    for (const auto* input : all_inputs) {
        auto input_copy = std::make_shared<GenerateInputPB>(*input);
        // Worker status derives batch_id from stream metadata; the batch RPC envelope is authoritative.
        input_copy->set_group_size(group_size);
        input_copy->mutable_group_id()->set_value(request->batch_id());

        BatchSlot slot;
        slot.input                   = std::move(input_copy);
        slot.fetch_attach_timeout_ms = request->fetch_attach_timeout_ms();
        slots.push_back(std::move(slot));
    }

    return grpc::Status::OK;
}

grpc::Status PrefillBatchRpcServer::acceptGroup(std::vector<BatchSlot> slots, EnqueueBatchResponsePB* response) {
    buildSlotContexts(slots);
    auto prepare_results = prepareGroup(slots);

    std::vector<ReadySlot> ready_slots;
    ready_slots.reserve(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        auto& slot            = slots[i];
        auto& result          = prepare_results[i];
        auto  request_id      = slot.input->request_id();
        auto& prefill_context = *slot.deferred->context;
        if (!result.prepared) {
            if (result.stage_status.ok()) {
                result.stage_status = grpc::Status(grpc::StatusCode::INTERNAL, "prepareAllocateResource failed");
            }
            // Finish under the same mutex used by Cancel, then select the
            // terminal status. If Cancel won, its 8429 latch must override
            // every prepare error/exception; if finish won, Cancel returns
            // NOT_FOUND.
            deferred_contexts_->finish(request_id, slot.deferred.get());
            result.stage_status = preferPriorityPreemption(prefill_context, result.stage_status);
            addBatchError(
                response, request_id, batchErrorCode(result.stage_status), result.stage_status.error_message());
            continue;
        }

        // PREPARE ownership ended in its own future. Claim a new group-phase
        // operation before touching the context again. A priority finalizer
        // that won the gap owns the context and this slot must not continue.
        const auto start_result = slot.deferred->tryStartOperation();
        if (start_result.priority_finalizer_claimed) {
            schedulePriorityFinalization(request_id, slot.deferred);
        }
        if (!start_result.started) {
            deferred_contexts_->finish(request_id, slot.deferred.get());
            auto terminal_status =
                prefill_context.isPriorityPreempted() ?
                    preferPriorityPreemption(prefill_context, grpc::Status::OK) :
                    grpc::Status(grpc::StatusCode::UNAVAILABLE, "request became terminal before group admission");
            addBatchError(response, request_id, batchErrorCode(terminal_status), terminal_status.error_message());
            continue;
        }

        // Frontend sends FetchResponse only after the master has received this
        // request's success ACK and set enqueued_by_master. The supported
        // frontend therefore cannot Fetch in this interval. Storing first
        // atomically prevents concurrent batches from enqueueing the same ID.
        auto deferred = storeSlot(slot, response);
        if (deferred) {
            ready_slots.push_back(ReadySlot{&slot, std::move(deferred)});
        }
    }

    auto enqueue_status = enqueueGroupStreams(ready_slots, response);
    if (!enqueue_status.ok()) {
        for (auto& ready_slot : ready_slots) {
            rejectSlot(ready_slot, enqueue_status, response);
        }
        return grpc::Status::OK;
    }

    for (auto& ready_slot : ready_slots) {
        publishSlot(ready_slot, response);
    }
    return grpc::Status::OK;
}

void PrefillBatchRpcServer::buildSlotContexts(std::vector<BatchSlot>& slots) {
    for (auto& slot : slots) {
        RPCContext rpc_ctx{slot.input.get(), nullptr};
        auto       pfx_ctx = std::make_unique<PrefillGenerateContext>(
            &this->resource(),
            rpc_ctx,
            slot.input->generate_config().timeout_ms(),
            /*server_context=*/nullptr,
            metrics_reporter_,
            meta_,
            maga_init_params_.pd_sep_config.prefill_stop_stream_wait_timeout_ms);
        pfx_ctx->onflight_requests      = onflight_requests_;
        pfx_ctx->loading_cache_requests = loading_cache_requests_;
        auto guard                      = std::make_shared<AtomicGuard>(onflight_requests_);
        auto deferred                   = std::make_shared<DeferredPrefillContext>();
        deferred->context               = std::move(pfx_ctx);
        deferred->input                 = slot.input;
        deferred->request_guard         = std::move(guard);
        slot.deferred                   = std::move(deferred);
        slot.registration_status        = deferred_contexts_->registerActive(slot.input->request_id(), slot.deferred);
    }
}

std::vector<PrefillBatchRpcServer::PrepareResult> PrefillBatchRpcServer::prepareGroup(std::vector<BatchSlot>& slots) {
    const auto max_retry_times      = maga_init_params_.pd_sep_config.prefill_retry_times;
    const auto max_retry_timeout_ms = maga_init_params_.pd_sep_config.prefill_retry_timeout_ms;

    std::vector<PrepareResult>                       results(slots.size());
    std::vector<autil::ThreadPoolBase::Future<void>> prepare_futures;
    prepare_futures.reserve(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        auto* slot   = &slots[i];
        auto* result = &results[i];
        if (!slot->registration_status.ok()) {
            result->stage_status = slot->registration_status;
            slot->deferred->context->tryMarkOtherTerminal();
            finishSlotOperation(slot->input->request_id(), slot->deferred);
            continue;
        }
        try {
            auto future =
                prepare_resource_worker_pool_->async([this, slot, result, max_retry_times, max_retry_timeout_ms] {
                    auto& prefill_context = *slot->deferred->context;
                    try {
                        int64_t begin_time_us = currentTimeUs();
                        auto    stage         = prefill_context.stat_info.saveStage();
                        for (int attempt = 0; attempt <= max_retry_times; ++attempt) {
                            if (prefill_context.isPriorityPreempted()) {
                                result->stage_status = preferPriorityPreemption(prefill_context, grpc::Status::OK);
                                break;
                            }
                            prefill_context.reset();
                            prefill_context.stat_info.restoreStage(stage);
                            prefill_context.retry_times++;
                            prepareAllocateResource(prefill_context);
                            if (prefill_context.isPriorityPreempted()) {
                                result->stage_status =
                                    preferPriorityPreemption(prefill_context, prefill_context.error_status);
                                break;
                            }
                            if (prefill_context.ok()) {
                                result->prepared = true;
                                break;
                            }
                            auto cost_time_us                  = currentTimeUs() - begin_time_us;
                            prefill_context.retry_cost_time_ms = cost_time_us / 1000;
                            if (max_retry_timeout_ms > 0 && cost_time_us >= max_retry_timeout_ms * 1000) {
                                break;
                            }
                            usleep(1000);
                        }
                        if (!result->prepared && result->stage_status.ok()) {
                            result->stage_status = prefill_context.error_status.ok() ?
                                                       statusFromErrorInfo(prefill_context.error_info) :
                                                       prefill_context.error_status;
                            if (result->stage_status.ok()) {
                                result->stage_status =
                                    grpc::Status(grpc::StatusCode::INTERNAL, "prepareAllocateResource failed");
                            }
                        }
                    } catch (const std::exception& e) {
                        result->stage_status = grpc::Status(
                            grpc::StatusCode::INTERNAL, "prepareAllocateResource exception: " + std::string(e.what()));
                    } catch (...) {
                        result->stage_status =
                            grpc::Status(grpc::StatusCode::INTERNAL, "prepareAllocateResource unknown exception");
                    }
                    if (!result->prepared && !prefill_context.tryMarkOtherTerminal()) {
                        result->stage_status = preferPriorityPreemption(prefill_context, result->stage_status);
                    }
                    // PREPARE is owned per slot. A canceled slot can now hand
                    // itself to the finalizer without waiting for sibling
                    // futures or the remaining group phases.
                    finishSlotOperation(slot->input->request_id(), slot->deferred);
                });
            prepare_futures.emplace_back(std::move(future));
        } catch (const std::exception& e) {
            result->stage_status =
                grpc::Status(grpc::StatusCode::INTERNAL, "submit prepare task exception: " + std::string(e.what()));
            slot->deferred->context->tryMarkOtherTerminal();
            finishSlotOperation(slot->input->request_id(), slot->deferred);
        } catch (...) {
            result->stage_status = grpc::Status(grpc::StatusCode::INTERNAL, "submit prepare task unknown exception");
            slot->deferred->context->tryMarkOtherTerminal();
            finishSlotOperation(slot->input->request_id(), slot->deferred);
        }
    }
    for (auto& future : prepare_futures) {
        future.get();
    }
    return results;
}

grpc::Status PrefillBatchRpcServer::enqueueGroupStreams(std::vector<ReadySlot>& ready_slots,
                                                        EnqueueBatchResponsePB* response) {
    if (ready_slots.empty()) {
        return grpc::Status::OK;
    }
    // Context-local R1 checkpoint. Active registration precedes prepare, so
    // an accepted priority Cancel is already latched here; no global
    // scheduler intent or full-stream scan is needed.
    std::vector<ReadySlot> live_slots;
    live_slots.reserve(ready_slots.size());
    for (auto& ready_slot : ready_slots) {
        auto& prefill_context = *ready_slot.deferred->context;
        if (prefill_context.isPriorityPreempted()) {
            rejectSlot(ready_slot, preferPriorityPreemption(prefill_context, grpc::Status::OK), response);
            continue;
        }
        live_slots.push_back(std::move(ready_slot));
    }
    ready_slots = std::move(live_slots);
    if (ready_slots.empty()) {
        return grpc::Status::OK;
    }
    std::vector<std::shared_ptr<GenerateInput>> generate_inputs;
    generate_inputs.reserve(ready_slots.size());
    for (auto& ready_slot : ready_slots) {
        auto& prefill_context = *ready_slot.deferred->context;
        prefill_context.stat_info.nextStage();
        generate_inputs.push_back(prefill_context.generate_input);
    }

    std::vector<bool>              enqueue_successes;
    std::vector<GenerateStreamPtr> streams;
    const auto                     mark_all_other_terminal = [&ready_slots] {
        for (auto& ready_slot : ready_slots) {
            if (ready_slot.deferred && ready_slot.deferred->context) {
                ready_slot.deferred->context->tryMarkOtherTerminal();
            }
        }
    };
    try {
        std::tie(enqueue_successes, streams) = engine_->enqueueMultiple(generate_inputs);
    } catch (const std::exception& e) {
        mark_all_other_terminal();
        return grpc::Status(grpc::StatusCode::INTERNAL, "enqueueMultiple exception: " + std::string(e.what()));
    } catch (...) {
        mark_all_other_terminal();
        return grpc::Status(grpc::StatusCode::INTERNAL, "enqueueMultiple unknown exception");
    }

    if (enqueue_successes.size() != generate_inputs.size() || streams.size() != generate_inputs.size()) {
        mark_all_other_terminal();
        return grpc::Status(grpc::StatusCode::INTERNAL,
                            "enqueueMultiple result size mismatch: input=" + std::to_string(generate_inputs.size())
                                + " status=" + std::to_string(enqueue_successes.size())
                                + " stream=" + std::to_string(streams.size()));
    }

    std::vector<ReadySlot> admitted_slots;
    admitted_slots.reserve(ready_slots.size());
    for (size_t i = 0; i < ready_slots.size(); ++i) {
        auto& stream     = streams[i];
        auto& ready_slot = ready_slots[i];
        if (!stream) {
            mark_all_other_terminal();
            return grpc::Status(grpc::StatusCode::INTERNAL, "enqueueMultiple returned null stream");
        }
        if (stream->streamId() != ready_slot.slot->input->request_id()) {
            mark_all_other_terminal();
            return grpc::Status(grpc::StatusCode::INTERNAL,
                                "enqueueMultiple result order mismatch: expected request_id="
                                    + std::to_string(ready_slot.slot->input->request_id())
                                    + " actual=" + std::to_string(stream->streamId()));
        }
        ready_slot.deferred->context->setStream(stream);
        ready_slot.deferred->context->setLocalStreamSchedulerOwned(enqueue_successes[i]);
        if (!enqueue_successes[i]) {
            // The scheduler rejection and priority Cancel arbitrate through
            // the same terminal-cause CAS. Whichever wins determines the
            // outward error; processing order below cannot rewrite it.
            ready_slot.deferred->context->tryMarkOtherTerminal();
            auto status = statusFromErrorInfo(stream->statusInfo());
            if (status.ok()) {
                status = grpc::Status(grpc::StatusCode::INTERNAL, "scheduler rejected request");
            }
            rejectSlot(ready_slot, status, response);
            continue;
        }
        if (ready_slot.deferred->context->isPriorityPreempted()) {
            rejectSlot(ready_slot, preferPriorityPreemption(*ready_slot.deferred->context, grpc::Status::OK), response);
            continue;
        }
        admitted_slots.push_back(std::move(ready_slot));
    }
    ready_slots = std::move(admitted_slots);
    return grpc::Status::OK;
}

std::shared_ptr<DeferredPrefillContext> PrefillBatchRpcServer::storeSlot(BatchSlot&              slot,
                                                                         EnqueueBatchResponsePB* response) {
    const auto request_id = slot.input->request_id();
    auto       deferred   = slot.deferred;

    const auto store_status = deferred_contexts_->store(request_id, deferred);
    if (!store_status.ok()) {
        deferred_contexts_->finish(request_id, deferred.get());
        const auto outward_status = preferPriorityPreemption(*deferred->context, store_status);
        if (!deferred->context->isPriorityPreempted()) {
            deferred->cancel(store_status);
        }
        addBatchError(response, request_id, batchErrorCode(outward_status), outward_status.error_message());
        finishSlotOperation(request_id, deferred);
        return nullptr;
    }
    return deferred;
}

void PrefillBatchRpcServer::publishSlot(ReadySlot& ready_slot, EnqueueBatchResponsePB* response) {
    auto&       slot       = *ready_slot.slot;
    const auto  request_id = slot.input->request_id();
    const auto& deferred   = ready_slot.deferred;
    if (deferred->context->isPriorityPreempted()) {
        rejectSlot(ready_slot, preferPriorityPreemption(*deferred->context, grpc::Status::OK), response);
        return;
    }
    constexpr int64_t kDefaultContextTtlMs = 10 * 60 * 1000;
    int64_t           ttl_ms = slot.fetch_attach_timeout_ms > 0 ? slot.fetch_attach_timeout_ms : kDefaultContextTtlMs;
    if (deferred->context->request_timeout_ms > 0) {
        const int64_t elapsed_ms   = (currentTimeUs() - deferred->context->request_begin_time_us) / 1000;
        const int64_t remaining_ms = deferred->context->request_timeout_ms - elapsed_ms;
        if (remaining_ms <= 0) {
            rejectSlot(ready_slot,
                       grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED,
                                    "request deadline expired before FetchResponse became available"),
                       response);
            return;
        }
        ttl_ms = std::min(ttl_ms, remaining_ms);
    }
    auto publish_status = deferred_contexts_->armTtl(request_id, deferred, std::chrono::milliseconds(ttl_ms));
    if (!publish_status.ok()) {
        rejectSlot(ready_slot, publish_status, response);
        return;
    }
    addBatchSuccess(response, request_id);
    finishSlotOperation(request_id, deferred);
}

void PrefillBatchRpcServer::rejectSlot(ReadySlot&              ready_slot,
                                       const grpc::Status&     status,
                                       EnqueueBatchResponsePB* response) {
    auto&   slot       = *ready_slot.slot;
    int64_t request_id = slot.input->request_id();
    auto    deferred   = deferred_contexts_->remove(request_id, ready_slot.deferred.get());
    if (!deferred) {
        deferred = ready_slot.deferred;
    }
    const auto outward_status =
        deferred && deferred->context ? preferPriorityPreemption(*deferred->context, status) : status;
    if (deferred && deferred->context && !deferred->context->isPriorityPreempted()
        && !deferred->context->cancel_state->load()) {
        deferred->cancel(outward_status);
    }
    addBatchError(response, request_id, batchErrorCode(outward_status), outward_status.error_message());
    ready_slot.deferred.reset();
    slot.deferred.reset();
    finishSlotOperation(request_id, deferred);
}

// ---------------------------------------------------------------------------
// FetchResponse — atomically take the stored context and continue its Decode stream
// ---------------------------------------------------------------------------

grpc::Status PrefillBatchRpcServer::FetchResponse(grpc::ServerContext*                   context,
                                                  const FetchRequestPB*                  request,
                                                  grpc::ServerWriter<GenerateOutputsPB>* writer) {
    RTP_LLM_PROFILE_FUNCTION();
    const int64_t                           request_id = request->request_id();
    std::shared_ptr<DeferredPrefillContext> deferred;
    const auto                              take_status = deferred_contexts_->take(request_id, deferred);
    if (!take_status.ok()) {
        return take_status;
    }

    auto& prefill_context              = *deferred->context;
    prefill_context.server_context     = context;
    prefill_context.rpc_context.writer = writer;
    grpc::Status status                = grpc::Status::OK;
    try {
        status = finishStream(prefill_context);
    } catch (const std::exception& e) {
        status =
            grpc::Status(grpc::StatusCode::INTERNAL,
                         "request [" + prefill_context.request_key + "] finishStream exception [" + e.what() + "]");
    } catch (...) {
        status = grpc::Status(grpc::StatusCode::INTERNAL, "finishStream unknown exception");
    }
    // Linearize completion against Cancel: a Cancel that wins this map lock
    // has already latched its reason; a later Cancel no longer targets a
    // completed FetchResponse.
    deferred_contexts_->finish(request_id, deferred.get());
    status = preferPriorityPreemption(prefill_context, status);
    if (!status.ok()) {
        prefill_context.error_status = status;
    }
    if (deferred->finishOperation()) {
        schedulePriorityFinalization(request_id, deferred);
    }
    return status;
}

}  // namespace rtp_llm
