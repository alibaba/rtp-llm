#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerDecodeRead.h"

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PKeyUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <thread>

namespace rtp_llm {

P2PWorkerDecodeRead::P2PWorkerDecodeRead(P2PConnectorWorkerConfig                    config,
                                                   const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                                                   const kmonitor::MetricsReporterPtr&         metrics_reporter,
                                                   const transfer::IKVCacheReceiverPtr&        receiver):
    config_(std::move(config)),
    layer_block_converter_(layer_block_converter),
    metrics_reporter_(metrics_reporter),
    receiver_(receiver) {
    completion_callback_state_        = std::make_shared<CompletionCallbackState>();
    completion_callback_state_->owner = this;
    pending_cancel_expiry_thread_     = std::thread(&P2PWorkerDecodeRead::runPendingCancelExpiryLoop, this);
}

P2PWorkerDecodeRead::~P2PWorkerDecodeRead() {
    {
        std::lock_guard<std::mutex> lock(completion_callback_state_->mutex);
        completion_callback_state_->owner = nullptr;
    }
    {
        std::lock_guard<std::mutex> lock(read_tasks_mutex_);
        stopping_ = true;
        ++pending_cancel_generation_;
    }
    pending_cancel_cv_.notify_one();
    if (pending_cancel_expiry_thread_.joinable()) {
        pending_cancel_expiry_thread_.join();
    }
}

ErrorInfo P2PWorkerDecodeRead::buildRecvTasks(const P2PWorkerRoutePlan&             worker_plan,
                                                   const std::string&                    unique_key,
                                                   int64_t                               deadline_ms,
                                                   const std::shared_ptr<ReadTaskGroup>& task_group,
                                                   int&                                  total_block_count) {
    auto fail_registration = [&](const std::string& message) {
        cleanupRecvTaskStore(task_group, /*cancel_pending_tasks=*/true);
        RTP_LLM_LOG_WARNING("%s", message.c_str());
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, message);
    };
    struct PreparedReceive {
        transfer::RecvRequest request;
        std::string           context;
    };
    std::vector<PreparedReceive> prepared;
    for (const auto& route : worker_plan.routes) {
        if (route.layer_buffers.empty()) {
            return fail_registration("read: route=" + std::to_string(route.route_id) + " tag=" + route.cache_tag
                                     + " has no layer buffers, unique_key=" + unique_key);
        }
        const size_t payload_bytes =
            config_.topology ? config_.topology->group(route.cache_tag).spec->k_block_payload_bytes() : 0;

        for (const auto& layer_cache_buffer : route.layer_buffers) {
            if (!layer_cache_buffer) {
                return fail_registration("read: route=" + std::to_string(route.route_id) + " tag=" + route.cache_tag
                                         + " has a null layer buffer, unique_key=" + unique_key);
            }
            const int layer_id = layer_cache_buffer->getLayerId();
            if (route.cache_tag != layer_cache_buffer->cacheTag()) {
                return fail_registration("read: route=" + std::to_string(route.route_id) + " tag=" + route.cache_tag
                                         + " layer=" + std::to_string(layer_id)
                                         + ": layer buffer tag does not match route");
            }
            if (layer_cache_buffer->blockIdMap().empty()) {
                return fail_registration("read: route=" + std::to_string(route.route_id) + " layer="
                                         + std::to_string(layer_id) + " tag=" + layer_cache_buffer->cacheTag()
                                         + " has no cache keys, unique_key=" + unique_key);
            }

            // partition / slice 均来自 route（本侧那一半），worker 不再自行推导。
            auto key_block_infos = LayerCacheBufferUtil::buildKeyBlockInfosSliced(layer_block_converter_,
                                                                                  layer_cache_buffer,
                                                                                  route.partition.count,
                                                                                  route.partition.id,
                                                                                  route.slice,
                                                                                  payload_bytes);
            if (!key_block_infos.ok() || key_block_infos.value().empty()
                || key_block_infos.value().size() != layer_cache_buffer->blockIdMap().size()) {
                const std::string conversion_message =
                    !key_block_infos.ok() ? key_block_infos.status().ToString() :
                                            "converted key count=" + std::to_string(key_block_infos.value().size())
                                                + " differs from source key count="
                                                + std::to_string(layer_cache_buffer->blockIdMap().size());
                return fail_registration("read: route=" + std::to_string(route.route_id) + " layer="
                                         + std::to_string(layer_id) + " tag=" + layer_cache_buffer->cacheTag()
                                         + " task registration failed, unique_key=" + unique_key + ": "
                                         + conversion_message);
            }

            // key 由编排层签发的 route_id + plan digest 命名 —— 两侧不做任何独立推导。
            const std::string partition_layer_key = P2PKeyUtil::makeRouteLayerKey(
                unique_key, layer_id, layer_cache_buffer->cacheTag(), route.route_id, worker_plan.plan_digest);

            transfer::RecvRequest recv_req;
            recv_req.unique_key  = partition_layer_key;
            recv_req.block_info  = std::move(key_block_infos.value());
            recv_req.deadline_ms = deadline_ms;

            task_group->task_buffer_keys.emplace(partition_layer_key, layer_cache_buffer->bufferKey());
            task_group->pending_buffer_tasks.emplace(layer_cache_buffer->bufferKey(), 0);
            prepared.push_back({std::move(recv_req),
                                "read: layer=" + std::to_string(layer_id) + " tag=" + route.cache_tag
                                    + " route=" + std::to_string(route.route_id) + " unique_key=" + unique_key});
        }
    }
    // No receiver task is registered until every route/layer has valid metadata.
    for (const auto& receive : prepared) {
        task_group->task_start_time_us.emplace(receive.request.unique_key, 0);
        ++task_group->pending_buffer_tasks.at(task_group->task_buffer_keys.at(receive.request.unique_key));
    }
    for (const auto& receive : prepared) {
        const int64_t recv_start_us = currentTimeUs();
        auto task = receiver_->recv(receive.request);
        if (!task) {
            return fail_registration(receive.context + ": create recv task failed");
        }
        task_group->lease->onTransferStarted();
        task_group->partition_keys.push_back(receive.request.unique_key);
        task_group->tasks.push_back(task);
        task_group->task_start_time_us.at(receive.request.unique_key) = recv_start_us;
        registerTaskCompletionCallback(task, unique_key, task_group);
        total_block_count += static_cast<int>(receive.request.block_info.size());
    }
    return ErrorInfo::OkStatus();
}

void P2PWorkerDecodeRead::cleanupRecvTaskStore(const std::shared_ptr<ReadTaskGroup>& task_group,
                                                    bool                                   cancel_pending_tasks) const {
    if (!task_group) {
        return;
    }
    const size_t cleanup_count = std::min(task_group->partition_keys.size(), task_group->tasks.size());
    for (size_t i = 0; i < cleanup_count; ++i) {
        const auto& task = task_group->tasks[i];
        if (cancel_pending_tasks && task && !task->done()) {
            task->cancel();
        }
        receiver_->stealTask(task_group->partition_keys[i]);
    }
}

P2PWorkerDecodeRead::ReadWaitOutcome
P2PWorkerDecodeRead::waitRecvTasksWithReadDeadlinePolicy(const std::shared_ptr<ReadTaskGroup>& task_group,
                                                              int64_t                               deadline_ms,
                                                              int64_t                               request_id,
                                                              const std::string&                    unique_key) const {
    const auto all_done = [&]() {
        return std::all_of(
            task_group->tasks.begin(), task_group->tasks.end(), [](const auto& task) { return task->done(); });
    };
    {
        std::unique_lock<std::mutex> lock(task_group->completion_mutex);
        task_group->completion_cv.wait_until(
            lock, std::chrono::system_clock::time_point(std::chrono::milliseconds(deadline_ms)), [&]() {
                return task_group->cancelled.load() || all_done()
                       || task_group->first_error.snapshot().error.hasError();
            });
        if (task_group->cancelled.load()) {
            return ReadWaitOutcome::Cancelled;
        }
        if (all_done()) {
            return ReadWaitOutcome::AllDone;
        }
        if (task_group->first_error.snapshot().error.hasError())
            return ReadWaitOutcome::Failed;
    }
    task_group->first_error.record(ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE,
                                             "Decode recv transfer deadline exceeded key=" + unique_key
                                                 + " deadline_ms=" + std::to_string(deadline_ms)));
    // Cancel outside completion_mutex: cancel() may synchronously invoke the
    // completion callback, which takes that mutex to prevent a lost wakeup.
    for (const auto& key : task_group->partition_keys) {
        receiver_->stealTask(key);
    }
            task_group->lease->seal();
            for (const auto& task : task_group->tasks) {
                if (task && !task->done()) {
                    task->cancel();
                }
            }
            RTP_LLM_LOG_WARNING("read: deadline reached with pending transfers; recv tasks stolen, lease sealed, "
                                "and unfinished tasks cancelled, request_id=%ld, unique_key=%s, deadline_ms=%ld",
                                request_id,
                                unique_key.c_str(),
                                deadline_ms);
            return ReadWaitOutcome::ReturnDeadlineIncomplete;
}

void P2PWorkerDecodeRead::reportReadMetrics(P2PConnectorMetricsCollector&         collector,
                                                 bool                                  success,
                                                 int64_t                               read_start_time_us,
                                                 const std::shared_ptr<ReadTaskGroup>& task_group) const {
    if (!metrics_reporter_) {
        return;
    }
    collector.decode_worker_success            = success;
    collector.decode_worker_total_cost_time_us = currentTimeUs() - read_start_time_us;
    const int64_t first_done_us                = task_group->first_layer_done_time_us.load();
    if (first_done_us >= 0) {
        collector.decode_worker_first_layer_wait_time_us = first_done_us - read_start_time_us;
    }
    metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, &collector);
}

ErrorInfo P2PWorkerDecodeRead::read(int64_t                   request_id,
                                         const std::string&        unique_key,
                                         int64_t                   deadline_ms,
                                         const P2PWorkerRoutePlan& worker_plan) {
    RTP_LLM_LOG_DEBUG("read start, request_id: %ld, unique_key: %s, deadline_ms: %ld, routes: %zu, plan_digest: %llu",
                      request_id,
                      unique_key.c_str(),
                      deadline_ms,
                      worker_plan.routes.size(),
                      static_cast<unsigned long long>(worker_plan.plan_digest));

    // routes 为空 = 编排层判定本 worker 无任务。
    if (worker_plan.empty()) {
        return ErrorInfo::OkStatus();
    }

    const int64_t read_start_time_us               = currentTimeUs();
    auto          task_group                       = std::make_shared<ReadTaskGroup>();
    task_group->lease                              = std::make_shared<DecodeTargetWriteLease>();
    int                          total_block_count = 0;
    P2PConnectorMetricsCollector collector;
    {
        std::lock_guard<std::mutex> lock(read_tasks_mutex_);
        auto                        pending_cancel = pending_cancel_keys_.find(unique_key);
        if (pending_cancel != pending_cancel_keys_.end()) {
            const bool still_valid = currentTimeMs() <= pending_cancel->second;
            pending_cancel_keys_.erase(pending_cancel);
            schedulePendingCancelExpiryLocked();
            if (still_valid) {
                reportReadMetrics(collector, false, read_start_time_us, task_group);
                return ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED,
                                 "read cancelled before recv registration");
            }
        }
        building_read_keys_.insert(unique_key);
    }

    const auto prepare_start_us = currentTimeUs();
    ErrorInfo  build_result     = buildRecvTasks(worker_plan, unique_key, deadline_ms, task_group, total_block_count);
    collector.decode_worker_prepare_time_us   = currentTimeUs() - prepare_start_us;
    collector.decode_worker_total_block_count = total_block_count;
    if (build_result.hasError()) {
        reportReadMetrics(collector, false, read_start_time_us, task_group);
        const bool has_inflight_task = std::any_of(
            task_group->tasks.begin(), task_group->tasks.end(), [](const auto& task) { return task && !task->done(); });
        {
            std::lock_guard<std::mutex> lock(read_tasks_mutex_);
            building_read_keys_.erase(unique_key);
            if (pending_cancel_keys_.erase(unique_key) > 0) {
                schedulePendingCancelExpiryLocked();
            }
            if (has_inflight_task) {
                task_group->lease->seal();
                std::lock_guard<std::mutex> lease_lock(lease_map_mutex_);
                auto& lease_entry      = lease_map_[unique_key] = LeaseMapEntry{task_group};
                if (lease_entry.task_group->lease->isStopped()) {
                    lease_map_.erase(unique_key);
                }
            }
        }
        if (has_inflight_task) {
            return ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_READ_TRANSFER_NOT_DONE,
                             build_result.ToString() + "; registered recv task is still stopping");
        }
        return build_result;
    }

    bool pending_cancel = false;
    {
        std::lock_guard<std::mutex> lock(read_tasks_mutex_);
        building_read_keys_.erase(unique_key);
        read_tasks_[unique_key] = task_group;
        pending_cancel          = pending_cancel_keys_.erase(unique_key) > 0;
        if (pending_cancel) {
            schedulePendingCancelExpiryLocked();
        }
        // Publish the lease before registration stops being visible. A lease
        // query after CANCEL_READ acknowledgement must always observe one of them.
        std::lock_guard<std::mutex> lease_lock(lease_map_mutex_);
        lease_map_[unique_key] = LeaseMapEntry{task_group};
    }

    if (pending_cancel) {
        task_group->first_error.record(
            ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED, "Decode recv cancelled key=" + unique_key));
        task_group->cancelled.store(true);
        for (const auto& task : task_group->tasks) {
            if (task) {
                task->cancel();
            }
        }
    }

    const auto            recv_wait_start_us = currentTimeUs();
    const ReadWaitOutcome outcome =
        waitRecvTasksWithReadDeadlinePolicy(task_group, deadline_ms, request_id, unique_key);
    collector.decode_worker_recv_wait_time_us = currentTimeUs() - recv_wait_start_us;

    {
        std::lock_guard<std::mutex> lock(read_tasks_mutex_);
        building_read_keys_.erase(unique_key);
        if (pending_cancel_keys_.erase(unique_key) > 0) {
            schedulePendingCancelExpiryLocked();
        }
        read_tasks_.erase(unique_key);
    }

    if (outcome == ReadWaitOutcome::ReturnDeadlineIncomplete) {
        // The deadline path has sealed the lease. Completions may have arrived before seal().
        onRecvTaskDone(unique_key, task_group);
        int done_count = 0;
        for (const auto& task : task_group->tasks) {
            if (task->done()) {
                ++done_count;
            }
        }
        reportReadMetrics(collector, false, read_start_time_us, task_group);
        const auto        first = task_group->first_error.snapshot().error;
        const std::string msg   = first.ToString();
        RTP_LLM_LOG_WARNING("read failed, request_id: %ld, unique_key: %s, %s, done_tasks=%d/%zu",
                            request_id,
                            unique_key.c_str(),
                            msg.c_str(),
                            done_count,
                            task_group->tasks.size());
        return first;
    }

    // Seal the lease — no more recv tasks will be created.
    task_group->lease->seal();
    cleanupRecvTaskStore(task_group, /*cancel_pending_tasks=*/outcome == ReadWaitOutcome::Failed);

    // Recheck after sealing: all completion callbacks may have already run.
    onRecvTaskDone(unique_key, task_group);
    // Cancelled path: TRANSFERRING tasks remain in lease_map_ until their completion
    // callbacks report physical completion, preventing premature block release.

    auto recv_result = aggregateRecvTaskResults(task_group);

    reportReadMetrics(collector, recv_result.success, read_start_time_us, task_group);

    if (!recv_result.success) {
        RTP_LLM_LOG_WARNING("read failed, request_id: %ld, unique_key: %s, error_code: %s, error_msg: %s",
                            request_id,
                            unique_key.c_str(),
                            ErrorCodeToString(recv_result.error_code).c_str(),
                            recv_result.error_msg.c_str());
        return ErrorInfo(recv_result.error_code, recv_result.error_msg);
    }

    RTP_LLM_LOG_DEBUG("read end, request_id: %ld, unique_key: %s, success: true", request_id, unique_key.c_str());
    return ErrorInfo::OkStatus();
}

P2PWorkerDecodeRead::RecvResultInfo
P2PWorkerDecodeRead::aggregateRecvTaskResults(const std::shared_ptr<ReadTaskGroup>& task_group) const {
    RecvResultInfo result;
    const auto     first = task_group->first_error.snapshot().error;
    if (first.hasError())
        return {false, first.code(), first.ToString()};
    for (const auto& task : task_group->tasks) {
        if (!task->success()) {
            result.success = false;
            if (result.error_code == ErrorCode::NONE_ERROR) {
                result.error_code = transfer::toErrorCode(task->errorCode());
                result.error_msg  = task->errorMessage();
            }
        }
    }
    if (result.success && task_group->cancelled.load()) {
        result.success    = false;
        result.error_code = ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED;
        result.error_msg  = "read cancelled";
        return result;
    }
    if (!result.success && result.error_code == ErrorCode::NONE_ERROR) {
        result.error_code = ErrorCode::P2P_CONNECTOR_WORKER_READ_FAILED;
    }
    return result;
}

bool P2PWorkerDecodeRead::cancelRead(const std::string& unique_key, int64_t request_deadline_ms) {
    RTP_LLM_LOG_DEBUG("cancelRead start, unique_key: %s", unique_key.c_str());
    std::shared_ptr<ReadTaskGroup> task_group;
    {
        std::lock_guard<std::mutex> lock(read_tasks_mutex_);
        auto                        it = read_tasks_.find(unique_key);
        if (it == read_tasks_.end()) {
            const int64_t now_ms = currentTimeMs();
            const int64_t ttl_ms = std::max<int64_t>(1, config_.p2p_cancelled_keys_ttl_ms);
            const int64_t fallback_deadline = now_ms > std::numeric_limits<int64_t>::max() - ttl_ms ?
                                                  std::numeric_limits<int64_t>::max() :
                                                  now_ms + ttl_ms;
            const bool has_finite_request_deadline =
                request_deadline_ms > 0 && request_deadline_ms != std::numeric_limits<int64_t>::max();
            pending_cancel_keys_[unique_key] = has_finite_request_deadline ?
                                                   std::max(request_deadline_ms, fallback_deadline) :
                                                   fallback_deadline;
            schedulePendingCancelExpiryLocked();
            RTP_LLM_LOG_INFO("cancelRead: queued pending cancel, unique_key: %s, during_registration: %d",
                             unique_key.c_str(),
                             building_read_keys_.count(unique_key) > 0);
            return true;
        }
        task_group = it->second;
    }

    {
        std::lock_guard<std::mutex> lock(task_group->completion_mutex);
        task_group->first_error.record(
            ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_READ_CANCELLED, "Decode recv cancelled key=" + unique_key));
        task_group->cancelled.store(true);
    }
    task_group->completion_cv.notify_all();
    for (const auto& task : task_group->tasks) {
        task->cancel();
    }
    RTP_LLM_LOG_DEBUG("cancelRead success, unique_key: %s", unique_key.c_str());
    return true;
}

void P2PWorkerDecodeRead::registerTaskCompletionCallback(
    const transfer::IKVCacheRecvTaskPtr& task,
    const std::string&                    unique_key,
    const std::shared_ptr<ReadTaskGroup>& task_group) {
    const auto callback_state = completion_callback_state_;
    const std::weak_ptr<ReadTaskGroup> weak_task_group = task_group;
    const std::string task_key = task_group->partition_keys.empty() ? unique_key : task_group->partition_keys.back();
    std::weak_ptr<transfer::IKVCacheRecvTask> weak_task = task;
    const auto                                start_it  = task_group->task_start_time_us.find(task_key);
    const int64_t recv_start_us = start_it == task_group->task_start_time_us.end() ? -1 : start_it->second;
    task->setDoneCallback([callback_state,
                           unique_key,
                           weak_task_group,
                           weak_task,
                           task_key,
                           reporter = metrics_reporter_,
                           recv_start_us]() {
        const auto completed_us = currentTimeUs();
        const auto completed    = weak_task.lock();
        // This signal owns no worker pointer. It is safe even after worker
        // teardown, and must also run before lease_map_ registration completes.
        if (auto group = weak_task_group.lock()) {
            if (completed && !completed->success()) {
                group->first_error.record(ErrorInfo(transfer::toErrorCode(completed->errorCode()),
                                                    "Decode recv key=" + task_key + " transfer_code="
                                                        + std::to_string(static_cast<int>(completed->errorCode()))
                                                        + ": " + completed->errorMessage()));
            }
            std::lock_guard<std::mutex> lock(group->completion_mutex);
            if (completed && completed->success()) {
                const auto key = group->task_buffer_keys.find(task_key);
                if (key != group->task_buffer_keys.end()) {
                    auto& remaining = group->pending_buffer_tasks.at(key->second);
                    if (remaining > 0 && --remaining == 0) {
                        int64_t unset = -1;
                        group->first_layer_done_time_us.compare_exchange_strong(unset, completed_us);
                    }
                }
            }
            // Each task invokes this callback exactly once, including when it
            // finishes before callback registration or lease-map publication.
            group->lease->onTransferFinished();
            group->completion_cv.notify_all();
        }
        {
            std::lock_guard<std::mutex> lock(callback_state->mutex);
            if (callback_state->owner) {
                callback_state->owner->onRecvTaskDone(unique_key, weak_task_group);
            }
        }
        if (completed && reporter && recv_start_us >= 0) {
            P2PConnectorMetricsCollector collector;
            collector.decode_worker_success           = completed->success();
            collector.decode_worker_recv_task_time_us = completed_us - recv_start_us;
            reporter->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, &collector);
        }
    });
}

void P2PWorkerDecodeRead::onRecvTaskDone(const std::string& unique_key,
                                               const std::weak_ptr<ReadTaskGroup>& weak_task_group) {
    const auto task_group = weak_task_group.lock();
    if (!task_group) {
        return;
    }
    std::lock_guard<std::mutex> lock(lease_map_mutex_);
    auto                        it = lease_map_.find(unique_key);
    if (it == lease_map_.end() || it->second.task_group != task_group) {
        return;
    }
    if (it->second.task_group->lease->isStopped()) {
        lease_map_.erase(it);
    }
}

void P2PWorkerDecodeRead::schedulePendingCancelExpiryLocked() {
    ++pending_cancel_generation_;
    pending_cancel_cv_.notify_one();
}

void P2PWorkerDecodeRead::runPendingCancelExpiryLoop() {
    std::unique_lock<std::mutex> lock(read_tasks_mutex_);
    while (!stopping_) {
        const int64_t now_ms = currentTimeMs();
        for (auto it = pending_cancel_keys_.begin(); it != pending_cancel_keys_.end();) {
            if (it->second < now_ms) {
                it = pending_cancel_keys_.erase(it);
            } else {
                ++it;
            }
        }

        const uint64_t observed_generation = pending_cancel_generation_;
        int64_t        next_expiry_ms      = std::numeric_limits<int64_t>::max();
        for (const auto& [unique_key, expiry_ms] : pending_cancel_keys_) {
            (void)unique_key;
            if (expiry_ms < std::numeric_limits<int64_t>::max()) {
                next_expiry_ms = std::min(next_expiry_ms, expiry_ms + 1);
            }
        }

        if (next_expiry_ms == std::numeric_limits<int64_t>::max()) {
            pending_cancel_cv_.wait(lock, [this, observed_generation]() {
                return stopping_ || pending_cancel_generation_ != observed_generation;
            });
        } else {
            pending_cancel_cv_.wait_until(
                lock,
                std::chrono::system_clock::time_point(std::chrono::milliseconds(next_expiry_ms)),
                [this, observed_generation]() {
                    return stopping_ || pending_cancel_generation_ != observed_generation;
                });
        }
    }
}

bool P2PWorkerDecodeRead::queryLeaseStatus(
    const std::string& unique_key, bool& sealed, int& started_ops, int& finished_ops, bool& stopped) {
    // Keep the registration state and lease-map transition in one observation.
    // The read path acquires these locks in the same order when publishing a lease.
    std::lock_guard<std::mutex> read_lock(read_tasks_mutex_);
    if (building_read_keys_.count(unique_key) > 0) {
        sealed       = false;
        started_ops  = 0;
        finished_ops = 0;
        stopped      = false;
        return true;
    }

    std::lock_guard<std::mutex> lease_lock(lease_map_mutex_);
    auto                        it = lease_map_.find(unique_key);
    if (it == lease_map_.end()) {
        // Lease not in map — either never created or already cleaned up after all ops finished.
        // Treat as stopped (safe to free).
        sealed       = true;
        started_ops  = 0;
        finished_ops = 0;
        stopped      = true;
        return false;
    }

    const LeaseMapEntry&          entry      = it->second;
    const auto&                   task_group = entry.task_group;
    const DecodeTargetWriteLease& lease      = *task_group->lease;

    sealed       = lease.isSealed();
    started_ops  = lease.startedOps();
    finished_ops = lease.finishedOps();
    stopped      = lease.isStopped();

    // Lazily remove from map once fully stopped.
    if (stopped) {
        lease_map_.erase(it);
    }

    return true;
}

std::shared_ptr<DecodeTargetWriteLease> P2PWorkerDecodeRead::leaseFor(const std::string& unique_key) const {
    std::lock_guard<std::mutex> lease_lock(lease_map_mutex_);
    auto                        it = lease_map_.find(unique_key);
    if (it == lease_map_.end() || !it->second.task_group) {
        return nullptr;
    }
    return it->second.task_group->lease;
}

}  // namespace rtp_llm
