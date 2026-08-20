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
#include <set>
#include <thread>
#include <utility>

namespace rtp_llm {

namespace {
constexpr size_t kSenderPoolThreadCount                  = 4;
constexpr size_t kSenderPoolQueueSize                    = 10000;
constexpr int    kMaxOutstandingAsyncSendTasksPerRequest = static_cast<int>(kSenderPoolThreadCount * 2);

class ComputedBufferCleanupGuard {
public:
    ComputedBufferCleanupGuard(ComputedLayerCacheBufferStore* store, int64_t request_id):
        store_(store), request_id_(request_id) {}
    ~ComputedBufferCleanupGuard() {
        if (store_) {
            store_->removeBuffer(request_id_);
        }
    }
private:
    ComputedLayerCacheBufferStore* store_;
    int64_t                        request_id_;
};
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

    auto sender_pool = std::make_shared<autil::LockFreeThreadPool>(
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
    transfer::SendRequestPtr* send_request_out, std::shared_ptr<LayerCacheBuffer>* buffer_keepalive_out) {
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

void P2PConnectorWorkerPrefill::registerAsyncSendTask(
    const std::string& unique_key, const std::shared_ptr<AsyncSendTaskState>& task_state) {
    std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
    auto                        it = handle_cancel_flags_.find(unique_key);
    if (it != handle_cancel_flags_.end()) {
        it->second.async_send_tasks.emplace_back(task_state);
    }
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

int P2PConnectorWorkerPrefill::releasePendingAsyncSendTasks(
    const std::string& unique_key, std::shared_ptr<SendTransferResult>* transfer_result_out) {
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
        for (const auto& weak_task : it->second.async_send_tasks) {
            if (auto task = weak_task.lock()) {
                task_states.emplace_back(std::move(task));
            }
        }
    }
    return releaseNotStartedTaskStates(task_states);
}

bool P2PConnectorWorkerPrefill::writeByLayer(int                           layer_id,
                                             const KVCacheResourcePtr&     resource,
                                             int64_t                       request_id,
                                             std::shared_ptr<c10::Event>   event,
                                             int64_t                       request_deadline_ms) {
    auto collector = std::make_shared<PrefillWorkerStoreMetricsCollector>();

    RTP_LLM_CHECK_WITH_INFO(resource != nullptr, "writeByLayer requires a cache resource");
    RTP_LLM_CHECK_WITH_INFO(resource->layerNum() == static_cast<int>(config_.layer_all_num),
                            "P2P cache resource layer count mismatch: resource=%d configured=%u",
                            resource->layerNum(),
                            config_.layer_all_num);
    const int64_t now_ms            = currentTimeMs();
    const int64_t fallback_deadline = now_ms + store_wait_timeout_ms_;
    const int64_t hold_cap_deadline = now_ms + config_.p2p_prefill_resource_hold_ms;
    const int64_t base_deadline     = request_deadline_ms > now_ms ?
                                          std::max(request_deadline_ms, fallback_deadline) :
                                          fallback_deadline;
    const int64_t deadline_ms       = std::min(base_deadline, hold_cap_deadline);
    auto          computed_buffer = computed_buffers_->addBuffer(request_id, nullptr, deadline_ms);
    if (!computed_buffer) {
        RTP_LLM_LOG_DEBUG("writeByLayer rejected late buffer for released request_id=%ld, layer_id=%d",
                          request_id,
                          layer_id);
        return false;
    }
    if (!computed_buffer->expectedBufferCount().has_value()) {
        // The resource carries the immutable global cache topology even though
        // this per-layer callback only has physical blocks for the tag that was
        // just produced.  Derive completion from that topology, not from the
        // current block-id snapshot.  With a hybrid model the first callback is
        // commonly a FULL-attention layer; counting only tags that are already
        // transferable would publish (for Qwen3.5) 16 instead of the eventual
        // 69 target + DSpARK buffers and let StartLoad release the request while
        // later linear/aux layers are still being produced.
        size_t expected_buffer_count = 0;
        for (int expected_layer = 0; expected_layer < resource->layerNum(); ++expected_layer) {
            expected_buffer_count += resource->groupTagsForLayer(expected_layer).size();
        }
        computed_buffer->setExpectedBufferCount(expected_buffer_count);
    }

    std::vector<std::shared_ptr<LayerCacheBuffer>> layer_cache_buffers;
    for (const auto& cache_tag : resource->groupTagsForLayer(layer_id)) {
        auto layer_cache_buffer = LayerCacheBufferUtil::convertLayer(
            *resource, 0, layer_id, cache_tag, 0, -1, config_.cp_rank, config_.cp_size);
        if (layer_cache_buffer) {
            collector->total_block_count += layer_cache_buffer->blockIdMap().size();
            layer_cache_buffers.push_back(std::move(layer_cache_buffer));
        }
    }
    for (const auto& layer_cache_buffer : layer_cache_buffers) {
        layer_cache_buffer->setKVCacheResource(resource);
    }
    if (layer_cache_buffers.empty()) {
        RTP_LLM_LOG_DEBUG("writeByLayer has no transferable blocks, request_id=%ld, layer_id=%d", request_id, layer_id);
        return true;
    }

    for (const auto& layer_cache_buffer : layer_cache_buffers) {
        store_wait_context_checker_->addContext(
            StoreWaitContext(request_id, event, layer_cache_buffer, deadline_ms, collector));
    }
    if (layer_id == 0) {
        RTP_LLM_LOG_INFO("writeByLayer [P2P Prefill]: queued request_id=%ld, layer_id=%d, groups=%zu, blocks=%ld",
                         request_id,
                         layer_id,
                         layer_cache_buffers.size(),
                         collector->total_block_count);
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
    std::set<std::pair<int, std::string>>&           sent_layer_groups,
    int&                                             total_transfers,
    int64_t&                                         first_layer_ready_time_us) {
    int       sent_count    = 0;
    const int partition_num = static_cast<int>(tp_partition_ctxs.size());
    bool      dispatch_complete{false};

    while (!cancel_flag->load() && currentTimeMs() < return_deadline_ms) {
        std::set<int> need_layer_ids;
        for (int lid = 0; lid < static_cast<int>(config_.layer_all_num); ++lid) {
            need_layer_ids.insert(lid);
        }

        auto [total_layer_group_num, ready_layer_buffers] = computed_buffer->getBuffers(need_layer_ids);
        bool sent_any                                     = false;

        for (const auto& layer_cache_buffer : ready_layer_buffers) {
            const int   layer_id    = layer_cache_buffer->getLayerId();
            const auto& cache_tag   = layer_cache_buffer->cacheTag();
            const auto  layer_group = std::make_pair(layer_id, cache_tag);
            if (sent_layer_groups.count(layer_group)) {
                continue;
            }
            if (first_layer_ready_time_us == 0) {
                first_layer_ready_time_us = currentTimeUs();
            }
            sent_layer_groups.insert(layer_group);
            sent_count += sendLayerToPartitions(
                layer_cache_buffer,
                tp_partition_ctxs,
                unique_key,
                return_deadline_ms,
                kMaxOutstandingAsyncSendTasksPerRequest,
                cancel_flag,
                transfer_result);
            sent_any = true;
        }

        const auto expected_buffer_count = computed_buffer->expectedBufferCount();
        if (expected_buffer_count.has_value()) {
            total_transfers = static_cast<int>(*expected_buffer_count) * partition_num;
        }
        if (expected_buffer_count.has_value() && static_cast<size_t>(total_layer_group_num) == *expected_buffer_count
            && sent_layer_groups.size() == *expected_buffer_count) {
            dispatch_complete = true;
            break;
        }
        if (!sent_any) {
            computed_buffer->waitChange(total_layer_group_num, 50);
        }
    }

    if (!dispatch_complete) {
        total_transfers = std::max(total_transfers, sent_count + 1);
    }
    return sent_count;
}

int P2PConnectorWorkerPrefill::sendLayerToPartitions(const std::shared_ptr<LayerCacheBuffer>&   layer_cache_buffer,
                                                     const std::vector<AsymmetricTPContext>&    tp_partition_ctxs,
                                                     const std::string&                         unique_key,
                                                     int64_t                                    transfer_deadline_ms,
                                                     int                                        max_outstanding_tasks,
                                                     const std::shared_ptr<std::atomic<bool>>&  cancel_flag,
                                                     const std::shared_ptr<SendTransferResult>& transfer_result) {
    int               count     = 0;
    const int         layer_id  = layer_cache_buffer->getLayerId();
    const std::string cache_tag = layer_cache_buffer->cacheTag();

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
            transfer_result->done_count.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
            transfer_result->result_cv.notify_all();
        };
    };

    for (const auto& partition_ctx : tp_partition_ctxs) {
        if (!waitForAsyncSendSlot(
                transfer_result, max_outstanding_tasks, transfer_deadline_ms, cancel_flag, unique_key)) {
            return count;
        }
        auto key_block_infos = LayerCacheBufferUtil::buildKeyBlockInfos(layer_block_converter_,
                                                                        layer_cache_buffer,
                                                                        partition_ctx.local_partition_count,
                                                                        partition_ctx.local_partition_id);

        std::string partition_layer_key =
            P2PKeyUtil::makePartitionLayerTagKey(unique_key, layer_id, cache_tag, partition_ctx.remote_partition_id);

        transfer::SendRequest send_req;
        send_req.ip          = partition_ctx.decode_ip;
        send_req.port        = partition_ctx.decode_port;
        send_req.unique_key  = partition_layer_key;
        send_req.block_info  = std::move(key_block_infos);
        send_req.deadline_ms = transfer_deadline_ms;

        ++count;
        auto done_cb         = make_send_done_cb(partition_layer_key);
        auto send_req_shared = std::make_shared<transfer::SendRequest>(std::move(send_req));
        auto task_state      = std::make_shared<AsyncSendTaskState>();
        task_state->send_request     = send_req_shared;
        task_state->buffer_keepalive = layer_cache_buffer;
        transfer_result->async_send_task_count.fetch_add(1, std::memory_order_relaxed);
        registerAsyncSendTask(unique_key, task_state);

        auto task_with_slot = [sender = sender_,
                               task_state,
                               done_cb,
                               cancel_flag,
                               transfer_result]() mutable {
            struct SlotGuard {
                explicit SlotGuard(std::shared_ptr<SendTransferResult> result): result(std::move(result)) {}
                ~SlotGuard() {
                    result->async_send_task_count.fetch_sub(1, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lk(result->result_mutex);
                    result->result_cv.notify_all();
                }
                std::shared_ptr<SendTransferResult> result;
            } guard(transfer_result);

            transfer::SendRequestPtr          request;
            std::shared_ptr<LayerCacheBuffer> buffer_keepalive;
            if (!task_state->takeForStart(&request, &buffer_keepalive)
                || (cancel_flag && cancel_flag->load(std::memory_order_relaxed))) {
                done_cb(transfer::TransferErrorCode::CANCELLED, "send cancelled before async task started");
                return;
            }
            sender->send(*request, done_cb);
            (void)buffer_keepalive;
        };
        auto async_task = task_with_slot;
        if (!async_sender_pool_
            || async_sender_pool_->pushTask(std::move(async_task)) != autil::ThreadPoolBase::ERROR_NONE) {
            // This path only protects pool overload; it is not a DSpARK eager
            // execution path.  The same transport operation is performed.
            RTP_LLM_LOG_WARNING("P2P async sender pool full; dispatching inline for key=%s",
                                partition_layer_key.c_str());
            task_with_slot();
        }
    }
    return count;
}

bool P2PConnectorWorkerPrefill::waitForAsyncSendSlot(
    const std::shared_ptr<SendTransferResult>& transfer_result,
    int                                        max_outstanding_tasks,
    int64_t                                    return_deadline_ms,
    const std::shared_ptr<std::atomic<bool>>&  cancel_flag,
    const std::string&                         unique_key) const {
    if (max_outstanding_tasks <= 0) {
        return true;
    }
    std::unique_lock<std::mutex> lock(transfer_result->result_mutex);
    const int64_t wait_start_ms = currentTimeMs();
    int64_t       next_log_ms   = wait_start_ms + 1000;
    while (transfer_result->async_send_task_count.load(std::memory_order_relaxed) >= max_outstanding_tasks) {
        if ((cancel_flag && cancel_flag->load(std::memory_order_relaxed)) || currentTimeMs() >= return_deadline_ms) {
            return false;
        }
        transfer_result->result_cv.wait_for(lock, std::chrono::milliseconds(1));
        const int64_t now_ms = currentTimeMs();
        if (now_ms >= next_log_ms) {
            RTP_LLM_LOG_WARNING(
                "[PD-P2P] waiting async sender slot unique_key=%s wait_ms=%ld outstanding=%d limit=%d done=%d",
                unique_key.c_str(),
                now_ms - wait_start_ms,
                transfer_result->async_send_task_count.load(std::memory_order_relaxed),
                max_outstanding_tasks,
                transfer_result->done_count.load(std::memory_order_relaxed));
            next_log_ms = now_ms + 5000;
        }
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
        if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) {
            return false;
        }
        const int64_t now = currentTimeMs();
        if (now >= return_deadline_ms) {
            RTP_LLM_LOG_WARNING(
                "waitSendCallbacksWithTimeout timeout, done_count: %ld, expected: %d, active_or_queued: %d, return_deadline_ms: %ld",
                transfer_result->done_count.load(std::memory_order_relaxed),
                sent_transfer_count,
                transfer_result->async_send_task_count.load(std::memory_order_relaxed),
                return_deadline_ms);
            return false;
        }
        const int64_t remaining_return_ms = return_deadline_ms - now;
        const int64_t wait_ms             = std::min(remaining_return_ms, rdma_cap_ms);
        if (wait_ms <= 0) {
            return false;
        }
        const bool ready = transfer_result->result_cv.wait_for(
            lock, std::chrono::milliseconds(wait_ms), [&transfer_result, sent_transfer_count, &cancel_flag]() {
                return transfer_result->done_count.load(std::memory_order_relaxed) >= sent_transfer_count
                       || (cancel_flag && cancel_flag->load(std::memory_order_relaxed));
            });
        if (ready && !(cancel_flag && cancel_flag->load(std::memory_order_relaxed))) {
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
    ComputedBufferCleanupGuard computed_buffer_cleanup(computed_buffers_.get(), request_id);
    // D（deadline_ms）为 RPC 语义截止；return_deadline_ms = D - return_before，与 decode recv_req.deadline_ms 对齐。
    const int64_t return_before_ms   = config_.p2p_read_return_before_deadline_ms;
    const int64_t return_deadline_ms = deadline_ms - return_before_ms;
    RTP_LLM_LOG_INFO(
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

    int  total_transfers = 0;
    auto transfer_result = std::make_shared<SendTransferResult>();

    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
    {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        handle_cancel_flags_[unique_key] =
            HandleCancelEntry{cancel_flag, std::weak_ptr<SendTransferResult>(transfer_result), {}};
    }

    auto computed_layer_cache_buffer = computed_buffers_->addBuffer(request_id, nullptr, deadline_ms);
    if (!computed_layer_cache_buffer) {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        handle_cancel_flags_.erase(unique_key);
        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT,
                         "sendKVCache: computed layers already released, unique_key: " + unique_key);
    }

    std::set<std::pair<int, std::string>> sent_layer_groups;
    int64_t                               first_layer_ready_time_us = 0;
    const int                             sent_transfer_count = dispatchPendingLayerTransfers(computed_layer_cache_buffer,
                                                                 tp_partition_ctxs,
                                                                 unique_key,
                                                                 return_deadline_ms,
                                                                 cancel_flag,
                                                                 transfer_result,
                                                                 sent_layer_groups,
                                                                 total_transfers,
                                                                 first_layer_ready_time_us);
    const int64_t all_layers_dispatched_time_us = currentTimeUs();
    collector->first_layer_wait_time_us =
        (first_layer_ready_time_us > 0 ? first_layer_ready_time_us : all_layers_dispatched_time_us) - start_time_us;
    collector->last_layer_wait_time_us = all_layers_dispatched_time_us - start_time_us;

    const bool all_callbacks_received =
        waitSendCallbacksWithTimeout(transfer_result, sent_transfer_count, return_deadline_ms, cancel_flag);
    const bool timeout_cancelled_pending_tasks =
        !all_callbacks_received && !cancel_flag->load(std::memory_order_relaxed);
    if (timeout_cancelled_pending_tasks) {
        cancel_flag->store(true, std::memory_order_relaxed);
        std::shared_ptr<SendTransferResult> wake_result = transfer_result;
        releasePendingAsyncSendTasks(unique_key, &wake_result);
        std::lock_guard<std::mutex> lk(wake_result->result_mutex);
        wake_result->result_cv.notify_all();
    }
    if (!all_callbacks_received) {
        RTP_LLM_LOG_WARNING(
            "sendKVCache transfer callback wait ended before return_deadline_ms or rdma cap, request_id: %ld, unique_key: %s, dispatched=%d, total=%d, done=%d, active_or_queued=%d, layer_groups=%zu",
            request_id,
            unique_key.c_str(),
            sent_transfer_count,
            total_transfers,
            transfer_result->done_count.load(std::memory_order_relaxed),
            transfer_result->async_send_task_count.load(std::memory_order_relaxed),
            sent_layer_groups.size());
    }

    {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        handle_cancel_flags_.erase(unique_key);
    }

    auto send_result = determineSendResult(transfer_result,
                                           cancel_flag,
                                           timeout_cancelled_pending_tasks,
                                           all_callbacks_received,
                                           sent_transfer_count,
                                           total_transfers,
                                           return_deadline_ms,
                                           unique_key);

    const int64_t done_time_us = currentTimeUs();
    if (metrics_reporter_) {
        collector->success            = send_result.success;
        collector->total_cost_time_us = done_time_us - start_time_us;
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

    RTP_LLM_LOG_INFO(
        "sendKVCache [P2P]: done request_id=%ld, unique_key=%s, sent=%d/%d, first_layer_wait_us=%ld, "
        "all_layers_dispatched_us=%ld, callback_drain_us=%ld, cost_us=%ld",
        request_id,
        unique_key.c_str(),
        sent_transfer_count,
        total_transfers,
        collector->first_layer_wait_time_us,
        all_layers_dispatched_time_us - start_time_us,
        done_time_us - all_layers_dispatched_time_us,
        done_time_us - start_time_us);
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
    releasePendingAsyncSendTasks(unique_key, &transfer_result);
    if (transfer_result) {
        std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
        transfer_result->result_cv.notify_all();
    }
    return true;
}

}  // namespace rtp_llm
