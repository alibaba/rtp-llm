#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerPrefill.h"

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PKeyUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include "autil/LockFreeThreadPool.h"
#include <algorithm>
#include <chrono>
#include <exception>
#include <limits>
#include <set>
#include <thread>

namespace rtp_llm {

namespace {

constexpr size_t kSenderPoolThreadCount                = 4;
constexpr size_t kSenderPoolQueueSize                  = 10000;
constexpr int    kMaxOutstandingAsyncSendTasksPerRequest = static_cast<int>(kSenderPoolThreadCount * 2);

int64_t addWithSaturation(int64_t base_ms, int64_t delta_ms) {
    if (delta_ms <= 0) {
        return base_ms;
    }
    if (base_ms > std::numeric_limits<int64_t>::max() - delta_ms) {
        return std::numeric_limits<int64_t>::max();
    }
    return base_ms + delta_ms;
}

}  // namespace

P2PConnectorWorkerPrefill::P2PConnectorWorkerPrefill(P2PConnectorWorkerConfig                    config,
                                                     const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                                                     const kmonitor::MetricsReporterPtr&         metrics_reporter,
                                                     const transfer::IKVCacheSenderPtr&          sender):
    config_(std::move(config)),
    layer_block_converter_(layer_block_converter),
    metrics_reporter_(metrics_reporter),
    sender_(sender),
    // Note: config_ is already initialized (declared before asymmetric_tp_util_ in the class),
    // so reading config_.tp_size/tp_rank here is safe.
    asymmetric_tp_util_(std::make_shared<AsymmetricTpUtil>(config_.tp_size, config_.tp_rank)),
    computed_buffers_(std::make_shared<ComputedLayerCacheBufferStore>()) {}

P2PConnectorWorkerPrefill::~P2PConnectorWorkerPrefill() {
    if (cleanup_thread_) {
        cleanup_thread_->stop();
    }
    if (async_sender_pool_) {
        async_sender_pool_->stop();
        async_sender_pool_.reset();
    }
}

bool P2PConnectorWorkerPrefill::init(int64_t store_wait_timeout_ms) {
    store_wait_context_checker_ = std::make_shared<StoreWaitContextChecker>(metrics_reporter_, computed_buffers_);

    cleanup_thread_ = autil::LoopThread::createLoopThread(
        std::bind(&P2PConnectorWorkerPrefill::loopCheckProc, this), 1000, "P2PConnectorWorkerCleanupThread");
    if (!cleanup_thread_) {
        RTP_LLM_LOG_ERROR("init failed: cleanup_thread is null");
        return false;
    }

    // OPT-A2: dedicated pool for sender_->send so the dispatcher thread does
    // not block on the synchronous cuda copy + sync inside TcpKVCacheSender.
    // See PrefillRpcServerNew2.h / P2PConnectorWorkerPrefill.h comments and
    // the OPT-0 analysis for full rationale.
    auto             sender_pool            = std::make_shared<autil::LockFreeThreadPool>(
        kSenderPoolThreadCount, kSenderPoolQueueSize, nullptr, "P2PWorkerAsyncSender");
    if (!sender_pool->start()) {
        RTP_LLM_LOG_ERROR("init failed: async_sender_pool start failed");
        return false;
    }
    async_sender_pool_ = std::move(sender_pool);

    store_wait_timeout_ms_ = store_wait_timeout_ms;
    return true;
}

bool P2PConnectorWorkerPrefill::AsyncSendTaskState::takeForStart(
    transfer::SendRequestPtr*          send_request_out,
    std::shared_ptr<LayerCacheBuffer>* buffer_keepalive_out) {
    std::lock_guard<std::mutex> lock(mutex);
    if (released || started || !send_request || !buffer_keepalive) {
        return false;
    }
    started = true;
    if (send_request_out) {
        *send_request_out = std::move(send_request);
    }
    if (buffer_keepalive_out) {
        *buffer_keepalive_out = std::move(buffer_keepalive);
    }
    return true;
}

bool P2PConnectorWorkerPrefill::AsyncSendTaskState::releaseIfNotStarted() {
    std::lock_guard<std::mutex> lock(mutex);
    if (released || started) {
        return false;
    }
    released = true;
    send_request.reset();
    buffer_keepalive.reset();
    return true;
}

void P2PConnectorWorkerPrefill::registerAsyncSendTask(const std::string&                       unique_key,
                                                      const std::shared_ptr<AsyncSendTaskState>& task_state) {
    std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
    auto                        it = handle_cancel_flags_.find(unique_key);
    if (it == handle_cancel_flags_.end()) {
        return;
    }
    it->second.async_send_tasks.emplace_back(task_state);
}

int P2PConnectorWorkerPrefill::releaseNotStartedTaskStates(
    const std::vector<std::shared_ptr<AsyncSendTaskState>>& task_states) {
    int released_count = 0;
    for (const auto& task_state : task_states) {
        if (task_state && task_state->releaseIfNotStarted()) {
            ++released_count;
        }
    }
    return released_count;
}

int P2PConnectorWorkerPrefill::releasePendingAsyncSendTasks(const std::string&               unique_key,
                                                            std::shared_ptr<SendTransferResult>* transfer_result_out) {
    std::vector<std::shared_ptr<AsyncSendTaskState>> task_states;
    {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        auto                        it = handle_cancel_flags_.find(unique_key);
        if (it == handle_cancel_flags_.end()) {
            return 0;
        }
        if (transfer_result_out) {
            if (auto transfer_result = it->second.transfer_result.lock()) {
                *transfer_result_out = std::move(transfer_result);
            }
        }
        std::vector<std::weak_ptr<AsyncSendTaskState>> alive_tasks;
        alive_tasks.reserve(it->second.async_send_tasks.size());
        task_states.reserve(it->second.async_send_tasks.size());
        for (const auto& weak_task : it->second.async_send_tasks) {
            if (auto task_state = weak_task.lock()) {
                task_states.emplace_back(task_state);
                alive_tasks.emplace_back(task_state);
            }
        }
        it->second.async_send_tasks.swap(alive_tasks);
    }
    return releaseNotStartedTaskStates(task_states);
}

bool P2PConnectorWorkerPrefill::writeByLayer(int                           layer_id,
                                             const KVCacheResourcePtr&     resource,
                                             int64_t                       request_id,
                                             std::shared_ptr<torch::Event> event,
                                             int64_t                       request_deadline_ms) {
    auto collector = std::make_shared<PrefillWorkerStoreMetricsCollector>();

    auto layer_cache_buffer = LayerCacheBufferUtil::convertLayer(*resource, 0, layer_id, 0, -1);
    if (!layer_cache_buffer) {
        RTP_LLM_LOG_ERROR(
            "writeByLayer failed: layer_cache_buffer is null, request_id=%ld, layer_id=%d", request_id, layer_id);
        if (metrics_reporter_) {
            collector->success = false;
            metrics_reporter_->report<P2PConnectorMetrics, PrefillWorkerStoreMetricsCollector>(nullptr,
                                                                                               collector.get());
        }
        return false;
    }
    layer_cache_buffer->setKVCacheResource(resource);
    collector->total_block_count = layer_cache_buffer->blockIdMap().size();

    // Per-layer computed buffers should not outlive the stream-store contract.
    // We still give decode the usual store_wait_timeout slack when the request
    // deadline is near/past, but the final lifetime is capped by the same
    // prefill_resource_hold_ms used by P2PConnectorResourceStore.
    const int64_t now_ms            = currentTimeMs();
    const int64_t fallback_deadline = addWithSaturation(now_ms, store_wait_timeout_ms_);
    const int64_t hold_cap_deadline = addWithSaturation(now_ms, config_.p2p_prefill_resource_hold_ms);
    int64_t       base_deadline;
    if (request_deadline_ms == std::numeric_limits<int64_t>::max() || request_deadline_ms <= now_ms) {
        base_deadline = fallback_deadline;
    } else {
        // Honor the larger of "request deadline" and "fallback" first so a
        // tiny remaining business deadline does not immediately drop the
        // layer, then clamp the result to the prefill hold window below.
        base_deadline = std::max(request_deadline_ms, fallback_deadline);
    }
    const int64_t deadline_ms = std::min(base_deadline, hold_cap_deadline);
    store_wait_context_checker_->addContext(
        StoreWaitContext(request_id, std::move(event), layer_cache_buffer, deadline_ms, collector));
    if (layer_id == 0) {
        RTP_LLM_LOG_DEBUG(
            "writeByLayer [P2P Prefill]: queued request_id=%ld, layer_id=%d, blocks=%zu, deadline_ms=%ld "
            "(request=%ld, fallback=%ld, hold_cap=%ld)",
            request_id,
            layer_id,
            layer_cache_buffer->blockIdMap().size(),
            deadline_ms,
            request_deadline_ms,
            fallback_deadline,
            hold_cap_deadline);
    }
    return true;
}

void P2PConnectorWorkerPrefill::loopCheckProc() {
    store_wait_context_checker_->checkOnce();
    computed_buffers_->checkTimeout();

    if (metrics_reporter_) {
        auto collector = std::make_shared<PrefillWorkerStatusMetricsCollector>();
        collector->wait_store_event_count =
            store_wait_context_checker_ ? store_wait_context_checker_->getContextCount() : 0;
        collector->task_count             = 0;
        collector->computed_request_count = computed_buffers_->getBuffersCount();
        metrics_reporter_->report<P2PConnectorMetrics, PrefillWorkerStatusMetricsCollector>(nullptr, collector.get());
    }
}

int P2PConnectorWorkerPrefill::dispatchPendingLayerTransfers(
    const std::shared_ptr<ComputedLayerCacheBuffer>& computed_buffer,
    const std::vector<AsymmetricTPContext>&          tp_partition_ctxs,
    const std::string&                               unique_key,
    int64_t                                          return_deadline_ms,
    const std::shared_ptr<std::atomic<bool>>&        cancel_flag,
    const std::shared_ptr<SendTransferResult>&       transfer_result,
    std::set<int>&                                   sent_layer_ids,
    int                                              total_transfers) {
    int sent_count = 0;
    while (sent_count < total_transfers && !cancel_flag->load() && currentTimeMs() < return_deadline_ms) {
        std::set<int> need_layer_ids;
        for (int lid = 0; lid < static_cast<int>(config_.layer_all_num); ++lid) {
            if (!sent_layer_ids.count(lid)) {
                need_layer_ids.insert(lid);
            }
        }
        if (need_layer_ids.empty()) {
            break;
        }

        auto [total_layer_num, ready_layer_buffers] = computed_buffer->getBuffers(need_layer_ids);
        for (const auto& layer_cache_buffer : ready_layer_buffers) {
            int layer_id = layer_cache_buffer->getLayerId();
            if (sent_layer_ids.count(layer_id)) {
                continue;
            }
            sent_layer_ids.insert(layer_id);
            sent_count += sendLayerToPartitions(
                layer_cache_buffer,
                tp_partition_ctxs,
                unique_key,
                return_deadline_ms,
                sent_count,
                kMaxOutstandingAsyncSendTasksPerRequest,
                cancel_flag,
                transfer_result);
        }

        if (ready_layer_buffers.empty()) {
            computed_buffer->waitChange(total_layer_num, 50);
        }
    }
    return sent_count;
}

int P2PConnectorWorkerPrefill::sendLayerToPartitions(const std::shared_ptr<LayerCacheBuffer>&   layer_cache_buffer,
                                                     const std::vector<AsymmetricTPContext>&    tp_partition_ctxs,
                                                     const std::string&                         unique_key,
                                                     int64_t                                    transfer_deadline_ms,
                                                     int                                        scheduled_transfer_count,
                                                     int                                        max_outstanding_tasks,
                                                     const std::shared_ptr<std::atomic<bool>>&  cancel_flag,
                                                     const std::shared_ptr<SendTransferResult>& transfer_result) {
    int       count    = 0;
    const int layer_id = layer_cache_buffer->getLayerId();

    // Reusable result callback for both success / error paths. We need the same
    // callback shape whether the send dispatch happens inline (when async pool
    // push fails) or async on the worker pool.
    auto make_send_done_cb = [transfer_result](const std::string& partition_layer_key) {
        return [transfer_result, partition_layer_key](transfer::TransferErrorCode transfer_ec,
                                                      const std::string&          cb_error_msg) {
            RTP_LLM_LOG_DEBUG("send done, partition_layer_key: %s, success: %d",
                              partition_layer_key.c_str(),
                              transfer_ec == transfer::TransferErrorCode::OK);
            if (transfer_ec != transfer::TransferErrorCode::OK) {
                std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
                if (transfer_result->all_success.exchange(false)) {
                    transfer_result->error_code = transfer::toErrorCode(transfer_ec);
                    transfer_result->error_msg  = cb_error_msg;
                }
            }
            transfer_result->done_count.fetch_add(1);
            {
                std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
                transfer_result->result_cv.notify_one();
            }
        };
    };

    for (const auto& partition_ctx : tp_partition_ctxs) {
        if (!waitForAsyncSendSlot(transfer_result,
                                  scheduled_transfer_count + count,
                                  max_outstanding_tasks,
                                  transfer_deadline_ms,
                                  cancel_flag)) {
            return count;
        }

        auto key_block_infos = LayerCacheBufferUtil::buildKeyBlockInfos(layer_block_converter_,
                                                                        layer_cache_buffer,
                                                                        partition_ctx.local_partition_count,
                                                                        partition_ctx.local_partition_id);

        std::string partition_layer_key =
            P2PKeyUtil::makePartitionLayerKey(unique_key, layer_id, partition_ctx.remote_partition_id);

        transfer::SendRequest send_req;
        send_req.ip          = partition_ctx.decode_ip;
        send_req.port        = partition_ctx.decode_port;
        send_req.unique_key  = partition_layer_key;
        send_req.block_info  = std::move(key_block_infos);
        send_req.deadline_ms = transfer_deadline_ms;

        ++count;

        // OPT-A2: push sender_->send onto the dedicated pool. The dispatcher
        // (this thread) returns immediately after enqueuing; the cuda copy +
        // sync inside TcpKVCacheSender::makeTransferRequest now runs on the
        // pool worker thread. The completion callback is unchanged (still
        // invoked by arpc IO threads when the response arrives).
        //
        // send_req is wrapped in shared_ptr so both the async task and the
        // inline fallback path can reference it without a stale move.
        auto                              done_cb         = make_send_done_cb(partition_layer_key);
        auto                              send_req_shared = std::make_shared<transfer::SendRequest>(std::move(send_req));
        auto task_state                   = std::make_shared<AsyncSendTaskState>();
        task_state->send_request          = send_req_shared;
        task_state->buffer_keepalive      = layer_cache_buffer;
        registerAsyncSendTask(unique_key, task_state);

        auto task = [sender = sender_, task_state, done_cb, cancel_flag, partition_layer_key]() mutable {
            transfer::SendRequestPtr      send_req_local;
            std::shared_ptr<LayerCacheBuffer> buffer_keepalive_local;
            if (!task_state->takeForStart(&send_req_local, &buffer_keepalive_local)) {
                done_cb(transfer::TransferErrorCode::CANCELLED, "send cancelled before async task started");
                return;
            }
            if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) {
                done_cb(transfer::TransferErrorCode::CANCELLED, "send cancelled before async task started");
                return;
            }
            // [HANG-DIAG/OPT-A2] Measure inside-pool cost so we can compare
            // against dispatchPendingLayerTransfers.dispatch_us. If A2 works
            // as designed, dispatch_us drops to ms-level while this inside-
            // pool cost keeps the seconds-level cuda sync wait — confirming
            // the cuda sync is the dominant blocker (per OPT-0).
            const int64_t pool_task_start_us = currentTimeUs();
            try {
                sender->send(*send_req_local, done_cb);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("P2PConnectorWorkerPrefill async send threw, partition_layer_key=%s, error=%s",
                                    partition_layer_key.c_str(),
                                    e.what());
                done_cb(transfer::TransferErrorCode::UNKNOWN, e.what());
                return;
            } catch (...) {
                RTP_LLM_LOG_WARNING("P2PConnectorWorkerPrefill async send threw unknown exception, "
                                    "partition_layer_key=%s",
                                    partition_layer_key.c_str());
                done_cb(transfer::TransferErrorCode::UNKNOWN, "unknown async send exception");
                return;
            }
            const int64_t pool_task_cost_us = currentTimeUs() - pool_task_start_us;
            if (pool_task_cost_us >= 100000) {
                RTP_LLM_LOG_WARNING(
                    "[HANG-DIAG/OPT-A2] async_sender_pool task slow, "
                    "partition_layer_key=%s, pool_task_cost_us=%ld "
                    "(blame: TcpKVCacheSender::send sync prefix = cuda copy + sync + getChannel)",
                    partition_layer_key.c_str(),
                    pool_task_cost_us);
            }
            // buffer_keepalive_local keeps layer_cache_buffer alive only until
            // sender->send() has built the transport request. Late-cancelled
            // tasks that never start release this ref via AsyncSendTaskState.
            (void)buffer_keepalive_local;
        };
        auto async_task = task;

        if (!async_sender_pool_
            || async_sender_pool_->pushTask(std::move(async_task)) != autil::ThreadPoolBase::ERROR_NONE) {
            // Pool full or not initialized: fall back to inline send so we
            // never lose a callback. WARN so we can see this in production.
            RTP_LLM_LOG_WARNING(
                "[HANG-DIAG/OPT-A2] async_sender_pool pushTask failed, fallback to inline send, "
                "partition_layer_key=%s",
                partition_layer_key.c_str());
            task();
        }
    }
    return count;
}

bool P2PConnectorWorkerPrefill::waitForAsyncSendSlot(
    const std::shared_ptr<SendTransferResult>& transfer_result,
    int                                        scheduled_transfer_count,
    int                                        max_outstanding_tasks,
    int64_t                                    return_deadline_ms,
    const std::shared_ptr<std::atomic<bool>>&  cancel_flag) const {
    if (max_outstanding_tasks <= 0) {
        return true;
    }

    std::unique_lock<std::mutex> lock(transfer_result->result_mutex);
    while (scheduled_transfer_count - transfer_result->done_count.load(std::memory_order_relaxed)
           >= max_outstanding_tasks) {
        if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) {
            return false;
        }
        const int64_t now = currentTimeMs();
        if (now >= return_deadline_ms) {
            return false;
        }
        transfer_result->result_cv.wait_for(
            lock,
            std::chrono::milliseconds(return_deadline_ms - now),
            [&transfer_result, scheduled_transfer_count, max_outstanding_tasks, &cancel_flag]() {
                return scheduled_transfer_count - transfer_result->done_count.load(std::memory_order_relaxed)
                           < max_outstanding_tasks
                       || (cancel_flag && cancel_flag->load(std::memory_order_relaxed));
            });
    }
    return !(cancel_flag && cancel_flag->load(std::memory_order_relaxed));
}

bool P2PConnectorWorkerPrefill::waitSendCallbacksWithTimeout(const std::shared_ptr<SendTransferResult>& transfer_result,
                                                             int     sent_transfer_count,
                                                             int64_t return_deadline_ms,
                                                             const std::shared_ptr<std::atomic<bool>>& cancel_flag) const {
    const int64_t                rdma_cap_ms = config_.transfer_backend_config.rdma_transfer_wait_timeout_ms;
    std::unique_lock<std::mutex> lock(transfer_result->result_mutex);
    while (transfer_result->done_count.load(std::memory_order_relaxed) < sent_transfer_count) {
        // Honor cancel_flag here too — without this, a CANCEL_HANDLE_READ RPC
        // arriving while we wait for RDMA send callbacks has nowhere to land,
        // and we'd block until return_deadline_ms (≈ business deadline, up to 1h).
        // determineSendResult() will see cancel_flag.load() and return
        // P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED instead of TIMEOUT.
        if (cancel_flag && cancel_flag->load()) {
            RTP_LLM_LOG_WARNING(
                "waitSendCallbacksWithTimeout cancelled, done_count: %ld, expected: %d, return_deadline_ms: %ld",
                transfer_result->done_count.load(std::memory_order_relaxed),
                sent_transfer_count,
                return_deadline_ms);
            return false;
        }
        const int64_t now = currentTimeMs();
        if (now >= return_deadline_ms) {
            RTP_LLM_LOG_WARNING(
                "waitSendCallbacksWithTimeout timeout, done_count: %ld, expected: %d, return_deadline_ms: %ld",
                transfer_result->done_count.load(std::memory_order_relaxed),
                sent_transfer_count,
                return_deadline_ms);
            return false;
        }
        const int64_t remaining_return_ms = return_deadline_ms - now;
        const int64_t wait_ms             = std::min(remaining_return_ms, rdma_cap_ms);
        if (wait_ms <= 0) {
            return false;
        }
        const bool ready = transfer_result->result_cv.wait_for(
            lock,
            std::chrono::milliseconds(wait_ms),
            [&transfer_result, sent_transfer_count, &cancel_flag]() {
                // Wake up early when cancel_flag flips so we don't have to wait
                // out the full rdma_cap_ms slice before re-checking it. This
                // requires the cancelSend code path to call result_cv.notify_one()
                // after setting cancel_flag — see below.
                return transfer_result->done_count.load(std::memory_order_relaxed) >= sent_transfer_count
                       || (cancel_flag && cancel_flag->load());
            });
        if (cancel_flag && cancel_flag->load()) {
            // Loop will re-check and return false on next iteration; exit early
            // here too in case the cv predicate fired due to cancel.
            continue;
        }
        if (ready) {
            return true;
        }
    }
    return true;
}

ErrorInfo
P2PConnectorWorkerPrefill::sendKVCache(int64_t                                              request_id,
                                       const std::string&                                   unique_key,
                                       int64_t                                              deadline_ms,
                                       const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers) {
    // For MLA, KV cache is identical across all TP ranks. Only the primary rank
    // within each decode-target group needs to send. In NP1D mode (prefill_tp > decode_tp),
    // multiple prefill ranks map to the same decode server; only partition_id=0 rank sends.
    // This matches the old DecodeRpcServer behavior where partition_count=1 was used.
    if (config_.is_mla && !decode_transfer_servers.empty()
        && config_.tp_size > static_cast<int64_t>(decode_transfer_servers.size())) {
        int local_partition_count = static_cast<int>(config_.tp_size / decode_transfer_servers.size());
        int local_partition_id    = static_cast<int>(config_.tp_rank % local_partition_count);
        if (local_partition_id != 0) {
            RTP_LLM_LOG_DEBUG(
                "sendKVCache [P2P]: skip for MLA non-primary rank, request_id=%ld, unique_key=%s, tp_rank=%ld",
                request_id,
                unique_key.c_str(),
                config_.tp_rank);
            computed_buffers_->removeBuffer(request_id);
            return ErrorInfo::OkStatus();
        }
    }

    // D（deadline_ms）为 RPC 语义截止；return_deadline_ms = D - return_before，与 decode recv_req.deadline_ms 对齐。
    const int64_t return_before_ms   = config_.p2p_read_return_before_deadline_ms;
    const int64_t return_deadline_ms = deadline_ms - return_before_ms;
    RTP_LLM_LOG_DEBUG(
        "sendKVCache [P2P]: start request_id=%ld, unique_key=%s, deadline_ms=%ld, return_deadline_ms=%ld, decode_servers=%zu",
        request_id,
        unique_key.c_str(),
        deadline_ms,
        return_deadline_ms,
        decode_transfer_servers.size());
    const int64_t start_time_us = currentTimeUs();
    auto          collector     = std::make_shared<PrefillWorkerSendMetricsCollector>();

    // 不对称TP
    auto tp_partition_ctxs = asymmetric_tp_util_->handleAsymmetricTP(decode_transfer_servers);
    if (tp_partition_ctxs.empty()) {
        const std::string error_msg = "sendKVCache: tp_partition_ctxs is empty, unique_key: " + unique_key;
        RTP_LLM_LOG_ERROR("%s", error_msg.c_str());
        if (metrics_reporter_) {
            collector->success = false;
            metrics_reporter_->report<P2PConnectorMetrics, PrefillWorkerSendMetricsCollector>(nullptr, collector.get());
        }
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_WORKER_ASYMMETRIC_TP_FAILED, error_msg);
    }

    // 计算总传输量
    const int total_transfers = static_cast<int>(config_.layer_all_num) * static_cast<int>(tp_partition_ctxs.size());
    auto      transfer_result = std::make_shared<SendTransferResult>();

    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
    {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        // transfer_result is held in this stack frame; weak_ptr is fine because
        // we only dereference it from cancelSend() while this frame is alive.
        handle_cancel_flags_[unique_key] = {cancel_flag, std::weak_ptr<SendTransferResult>(transfer_result)};
    }

    // [PD-DIAG] Three-stage timing for sendKVCache: addBuffer / dispatch / waitCallbacks.
    // Each can independently account for the dominant cost depending on the failure mode:
    //  - addBuffer slow → ComputedLayerCacheBufferStore mutex contention
    //  - dispatch slow → sender_->send synchronous blocking or layer cache scan slow
    //  - waitCallbacks slow → RDMA callback never arrives (the actual stuck RPC scenario)
    const int64_t add_buffer_start_us = currentTimeUs();
    auto computed_layer_cache_buffer    = computed_buffers_->addBuffer(request_id, nullptr, deadline_ms);
    const int64_t add_buffer_cost_us = currentTimeUs() - add_buffer_start_us;
    collector->first_layer_wait_time_us = currentTimeUs() - start_time_us;

    if (!computed_layer_cache_buffer) {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        handle_cancel_flags_.erase(unique_key);
        const std::string error_msg =
            "sendKVCache: computed layers already expired and removed, unique_key: " + unique_key;
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        if (metrics_reporter_) {
            collector->success            = false;
            collector->total_cost_time_us = currentTimeUs() - start_time_us;
            metrics_reporter_->report<P2PConnectorMetrics, PrefillWorkerSendMetricsCollector>(nullptr, collector.get());
        }
        // GENERATE_TIMEOUT (603) signals "request's business deadline already
        // expired by the time prefill could serve the StartLoad". Decode side
        // should surface this as a timeout to the client rather than retry.
        // Distinct from P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT (8316), which
        // means the in-flight RDMA transfer itself missed its deadline.
        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, error_msg);
    }

    const int64_t dispatch_start_us = currentTimeUs();
    std::set<int> sent_layer_ids;
    const int     sent_transfer_count  = dispatchPendingLayerTransfers(computed_layer_cache_buffer,
                                                                  tp_partition_ctxs,
                                                                  unique_key,
                                                                  return_deadline_ms,
                                                                  cancel_flag,
                                                                  transfer_result,
                                                                  sent_layer_ids,
                                                                  total_transfers);
    const int64_t dispatch_cost_us = currentTimeUs() - dispatch_start_us;
    collector->last_layer_wait_time_us = currentTimeUs() - start_time_us;

    // NOTE: do NOT erase handle_cancel_flags_[unique_key] here. The wait below
    // is the dominant phase (RDMA callbacks may take seconds to minutes), and
    // erasing the cancel_flag now would make any late-arriving CANCEL_HANDLE_READ
    // RPC silently miss its target — see Bug ① in the analysis doc.
    // We erase only after waitSendCallbacksWithTimeout returns.

    const int64_t wait_cb_start_us = currentTimeUs();
    const bool    all_callbacks_received =
        waitSendCallbacksWithTimeout(transfer_result, sent_transfer_count, return_deadline_ms, cancel_flag);
    const int64_t wait_cb_cost_us = currentTimeUs() - wait_cb_start_us;
    const bool    timeout_cancelled_pending_tasks =
        !all_callbacks_received && !cancel_flag->load(std::memory_order_relaxed);

    if (timeout_cancelled_pending_tasks) {
        cancel_flag->store(true, std::memory_order_relaxed);
        std::shared_ptr<SendTransferResult> wake_result = transfer_result;
        const int released_pending_task_count = releasePendingAsyncSendTasks(unique_key, &wake_result);
        {
            std::lock_guard<std::mutex> lk(wake_result->result_mutex);
            wake_result->result_cv.notify_all();
        }
        RTP_LLM_LOG_WARNING("sendKVCache timeout released %d queued async sender tasks, request_id: %ld, unique_key: %s",
                            released_pending_task_count,
                            request_id,
                            unique_key.c_str());
    }

    if (!all_callbacks_received) {
        RTP_LLM_LOG_WARNING(
            "sendKVCache transfer callback wait ended before return_deadline_ms or rdma cap, request_id: %ld, unique_key: %s, cancelled: %d",
            request_id,
            unique_key.c_str(),
            cancel_flag->load());
    }

    const int64_t total_send_cost_us = currentTimeUs() - start_time_us;
    if (total_send_cost_us >= 100 * 1000) {
        RTP_LLM_LOG_WARNING(
            "[PD-DIAG] sendKVCache slow phases, request_id=%ld, unique_key=%s, "
            "total_us=%ld, add_buffer_us=%ld, dispatch_us=%ld, wait_callbacks_us=%ld, "
            "sent=%d/%d, all_cb_received=%d, cancelled=%d",
            request_id,
            unique_key.c_str(),
            total_send_cost_us,
            add_buffer_cost_us,
            dispatch_cost_us,
            wait_cb_cost_us,
            sent_transfer_count,
            total_transfers,
            all_callbacks_received ? 1 : 0,
            cancel_flag->load() ? 1 : 0);
    }

    {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        handle_cancel_flags_.erase(unique_key);
    }

    // Always remove the computed buffer entry. This is safe because the caller (handleRead)
    // holds a whole-request KVCacheResourcePtr in resource_entry, which keeps all blocks
    // allocated via connector_ref_counter until handleRead returns. The per-layer refs here
    // are redundant for block lifetime safety.
    // This also marks the request_id as removed, preventing late-arriving layers from
    // StoreWaitContextChecker from creating orphan entries that pin blocks (LACK MEM).
    computed_buffers_->removeBuffer(request_id);

    auto send_result = determineSendResult(transfer_result,
                                           cancel_flag,
                                           timeout_cancelled_pending_tasks,
                                           all_callbacks_received,
                                           sent_transfer_count,
                                           total_transfers,
                                           return_deadline_ms,
                                           unique_key);

    if (metrics_reporter_) {
        collector->success            = send_result.success;
        collector->total_cost_time_us = currentTimeUs() - start_time_us;
        metrics_reporter_->report<P2PConnectorMetrics, PrefillWorkerSendMetricsCollector>(nullptr, collector.get());
    }

    if (!send_result.success) {
        RTP_LLM_LOG_WARNING("sendKVCache failed, request_id: %ld, unique_key: %s, error_code: %s, error_msg: %s",
                            request_id,
                            unique_key.c_str(),
                            ErrorCodeToString(send_result.error_code).c_str(),
                            send_result.error_msg.c_str());
        return ErrorInfo(send_result.error_code, send_result.error_msg);
    }

    RTP_LLM_LOG_DEBUG("sendKVCache [P2P]: done request_id=%ld, unique_key=%s, sent=%d/%d, cost_us=%ld",
                      request_id,
                      unique_key.c_str(),
                      sent_transfer_count,
                      total_transfers,
                      currentTimeUs() - start_time_us);
    return ErrorInfo::OkStatus();
}

P2PConnectorWorkerPrefill::SendResultInfo
P2PConnectorWorkerPrefill::determineSendResult(const std::shared_ptr<SendTransferResult>& transfer_result,
                                               const std::shared_ptr<std::atomic<bool>>&  cancel_flag,
                                               bool                                       timeout_cancelled_pending_tasks,
                                               bool                                       all_callbacks_received,
                                               int                                        sent_transfer_count,
                                               int                                        total_transfers,
                                               int64_t                                    return_deadline_ms,
                                               const std::string&                         unique_key) const {

    if (timeout_cancelled_pending_tasks) {
        return {false,
                ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT,
                "sendKVCache: transfer callback wait timeout, unique_key: " + unique_key};
    }
    if (cancel_flag->load()) {
        return {false,
                ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED,
                "sendKVCache cancelled, unique_key: " + unique_key};
    }
    if (!all_callbacks_received) {
        return {false,
                ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT,
                "sendKVCache: transfer callback wait timeout, unique_key: " + unique_key};
    }
    if (currentTimeMs() >= return_deadline_ms && sent_transfer_count < total_transfers) {
        return {false,
                ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT,
                "sendKVCache timeout before all transfers dispatched (return_deadline), unique_key: " + unique_key};
    }
    if (!transfer_result->all_success.load()) {
        std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
        return {false, transfer_result->error_code, transfer_result->error_msg};
    }
    return {};
}

bool P2PConnectorWorkerPrefill::cancelSend(const std::string& unique_key) {
    RTP_LLM_LOG_DEBUG("cancelSend start, unique_key: %s", unique_key.c_str());
    std::shared_ptr<std::atomic<bool>> cancel_flag;
    std::shared_ptr<SendTransferResult> transfer_result;
    {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        auto                        it = handle_cancel_flags_.find(unique_key);
        if (it == handle_cancel_flags_.end()) {
            RTP_LLM_LOG_INFO("cancelSend: unique_key not found: %s (best-effort)", unique_key.c_str());
            return true;
        }
        cancel_flag     = it->second.cancel_flag;
        transfer_result = it->second.transfer_result.lock();
    }
    cancel_flag->store(true, std::memory_order_relaxed);
    const int released_pending_task_count = releasePendingAsyncSendTasks(unique_key, &transfer_result);
    // Wake up waitSendCallbacksWithTimeout immediately so it sees the flag,
    // instead of letting it sit in cv.wait_for for up to rdma_transfer_wait_timeout_ms
    // (180s default) before re-checking.
    if (transfer_result) {
        std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
        transfer_result->result_cv.notify_all();
    }
    RTP_LLM_LOG_INFO("cancelSend success, unique_key: %s, released_pending_tasks: %d, notified_cv: %d",
                     unique_key.c_str(),
                     released_pending_task_count,
                     transfer_result ? 1 : 0);
    return true;
}

}  // namespace rtp_llm
