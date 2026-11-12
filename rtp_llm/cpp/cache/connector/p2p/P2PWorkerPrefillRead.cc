#include "rtp_llm/cpp/cache/connector/p2p/P2PWorkerPrefillRead.h"

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

std::string describeRoutes(const P2PWorkerRoutePlan& worker_plan) {
    std::string result = "[";
    for (size_t i = 0; i < worker_plan.routes.size(); ++i) {
        if (i != 0) {
            result += ",";
        }
        const auto& route = worker_plan.routes[i];
        result += std::to_string(route.route_id) + ":" + route.cache_tag;
    }
    return result + "]";
}

void buildExpectedBufferMetadata(const CacheTopology& topology,
                                 std::set<std::string>* expected_keys,
                                 std::vector<std::string>* expected_tags) {
    expected_keys->clear();
    expected_tags->clear();
    for (const auto& layer : topology.layers()) {
        for (const auto& tag : layer.group_tags) {
            if (expected_keys->insert(std::to_string(layer.layer_id) + ":" + tag).second) {
                expected_tags->push_back(tag);
            }
        }
    }
}

}  // namespace

P2PWorkerPrefillRead::P2PWorkerPrefillRead(P2PConnectorWorkerConfig                    config,
                                                     const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                                                     const kmonitor::MetricsReporterPtr&         metrics_reporter,
                                                     const transfer::IKVCacheSenderPtr&          sender):
    config_(std::move(config)),
    layer_block_converter_(layer_block_converter),
    metrics_reporter_(metrics_reporter),
    sender_(sender),
    computed_buffers_(std::make_shared<ComputedLayerCacheBufferStore>()) {}

P2PWorkerPrefillRead::~P2PWorkerPrefillRead() {
    store_wait_stopping_.store(true);
    computed_buffers_->notification()->notify();
    if (store_wait_check_thread_.joinable()) {
        store_wait_check_thread_.join();
    }
    if (async_sender_pool_) {
        async_sender_pool_->stop();
        async_sender_pool_.reset();
    }
}

bool P2PWorkerPrefillRead::init() {
    if (config_.p2p_prefill_sender_thread_count <= 0 || config_.p2p_prefill_sender_queue_size <= 0) {
        RTP_LLM_LOG_ERROR("init failed: p2p_prefill_sender_thread_count=%d and p2p_prefill_sender_queue_size=%d "
                          "must both be positive",
                          config_.p2p_prefill_sender_thread_count,
                          config_.p2p_prefill_sender_queue_size);
        return false;
    }
    store_wait_context_checker_ = std::make_shared<StoreWaitContextChecker>(metrics_reporter_, computed_buffers_);

    if (config_.topology) {
        buildExpectedBufferMetadata(*config_.topology, &expected_buffer_keys_, &expected_buffer_tags_);
    }

    store_wait_check_thread_ = std::thread([this] {
        const auto notification = computed_buffers_->notification();
        while (!store_wait_stopping_.load()) {
            const auto generation = notification->generation();
            loopCheckProc();
            auto next = computed_buffers_->nextTimeoutMs();
            if (store_wait_context_checker_->getContextCount() > 0) {
                // torch::Event has no safe host notification here. Only genuinely
                // pending GPU events need this fallback; ready layers bypass it.
                next = std::min(next, currentTimeMs() + 1);
            }
            if (!store_wait_stopping_.load()) {
                // Buffer deadlines remain wire/wall timestamps. Convert only
                // this local wait duration; never compare them to steady epochs.
                if (next != std::numeric_limits<int64_t>::max()) {
                    const auto remaining = std::max<int64_t>(0, next - currentTimeMs());
                    const auto now = P2PNotification::nowMs();
                    next = now + std::min(remaining, std::numeric_limits<int64_t>::max() - now);
                }
                notification->waitUntil(generation, next);
            }
        }
    });

    // OPT-A2: dedicated pool for sender_->send so the dispatcher thread does
    // not block on the synchronous cuda copy + sync inside TcpKVCacheSender.
    // See PrefillRpcServer.h / P2PWorkerPrefillRead.h comments and
    // the OPT-0 analysis for full rationale.
    auto             sender_pool            = std::make_shared<autil::LockFreeThreadPool>(
        config_.p2p_prefill_sender_thread_count, config_.p2p_prefill_sender_queue_size, nullptr, "P2PWorkerAsyncSender");
    if (!sender_pool->start()) {
        RTP_LLM_LOG_ERROR("init failed: async_sender_pool start failed");
        return false;
    }
    async_sender_pool_ = std::move(sender_pool);

    return true;
}

bool P2PWorkerPrefillRead::AsyncSendTaskState::takeForStart(
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

bool P2PWorkerPrefillRead::AsyncSendTaskState::releaseIfNotStarted() {
    std::lock_guard<std::mutex> lock(mutex);
    if (released || started) {
        return false;
    }
    released = true;
    send_request.reset();
    buffer_keepalive.reset();
    return true;
}

void P2PWorkerPrefillRead::registerAsyncSendTask(const std::string&                       unique_key,
                                                      const std::shared_ptr<AsyncSendTaskState>& task_state) {
    std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
    auto                        it = handle_cancel_flags_.find(unique_key);
    if (it == handle_cancel_flags_.end()) {
        return;
    }
    it->second.async_send_tasks.emplace_back(task_state);
}

int P2PWorkerPrefillRead::releaseNotStartedTaskStates(
    const std::vector<std::shared_ptr<AsyncSendTaskState>>& task_states) {
    int released_count = 0;
    for (const auto& task_state : task_states) {
        if (task_state && task_state->releaseIfNotStarted()) {
            ++released_count;
        }
    }
    return released_count;
}

int P2PWorkerPrefillRead::releasePendingAsyncSendTasks(const std::string&               unique_key,
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

bool P2PWorkerPrefillRead::rejectLayer(int64_t request_id, int64_t request_deadline_ms, const ErrorInfo& error) {
    RTP_LLM_LOG_ERROR("P2P Prefill layer publication failed, request_id=%ld rank=%d error_code=%d error_name=%s error=%s",
                      request_id,
                      static_cast<int>(config_.tp_rank),
                      static_cast<int>(error.code()),
                      ErrorCodeToString(error.code()).c_str(),
                      error.ToString().c_str());
    const auto local_horizon_ms =
        computed_buffers_->requestHorizon(request_id, request_deadline_ms).value_or(request_deadline_ms);
    if (computed_buffers_->registerRequestHorizon(request_id, local_horizon_ms, request_deadline_ms)) {
        auto buffer = computed_buffers_->addBuffer(request_id, nullptr, request_deadline_ms);
        if (buffer) {
            buffer->setError(error);
        }
    }
    return false;
}

bool P2PWorkerPrefillRead::writeByLayerTag(int                                  layer_id,
                                                const std::string&                   tag,
                                                const KVCacheResourcePtr&            resource,
                                                int64_t                              request_id,
                                                const std::shared_ptr<torch::Event>& event,
                                                int64_t                              request_deadline_ms,
                                                int                                  chunk_id) {
    if (!resource || resource->cacheKeys().size() != resource->blocksForLayer(layer_id, tag).size()) {
        return rejectLayer(request_id,
                           request_deadline_ms,
                           ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                                     "layer=" + std::to_string(layer_id) + " tag=" + tag
                                         + ": missing resource or mismatched key/block counts"));
    }

    auto        layer_cache_buffer = std::make_shared<LayerCacheBuffer>(layer_id, tag, chunk_id);
    if (config_.tp_rank == 0) {
        layer_cache_buffer->setKVCacheResource(resource);
    }
    const auto& cache_keys         = resource->cacheKeys();
    const auto& block_ids          = resource->blocksForLayer(layer_id, tag);
    for (size_t i = 0; i < cache_keys.size(); ++i) {
        if (isNullBlockIdx(block_ids[i]) || block_ids[i] < 0 || layer_cache_buffer->blockIdMap().count(cache_keys[i])) {
            return rejectLayer(request_id,
                               request_deadline_ms,
                               ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                                         "layer=" + std::to_string(layer_id) + " tag=" + tag
                                             + " route=<unresolved> cache_key=" + std::to_string(cache_keys[i])
                                             + " block_id=" + std::to_string(block_ids[i])
                                             + ": null/invalid block or duplicate key"));
        }
        layer_cache_buffer->addBlockId(cache_keys[i], block_ids[i]);
    }
    auto collector = std::make_shared<P2PConnectorMetricsCollector>();
    // prefill_worker_store_start_time_us 是该场景的触发字段，创建时即写入。
    collector->prefill_worker_store_start_time_us = currentTimeUs();
    collector->prefill_worker_store_total_block_count = layer_cache_buffer->blockIdMap().size();

    const int64_t now_ms = currentTimeMs();
    // StartLoad may already have installed a receiver-local transfer horizon.
    // Late layer callbacks carry the unchanged source identity, not a new clock.
    const auto local_horizon_ms =
        computed_buffers_->requestHorizon(request_id, request_deadline_ms).value_or(request_deadline_ms);
    if (request_deadline_ms <= 0 || request_deadline_ms == std::numeric_limits<int64_t>::max()
        || local_horizon_ms <= now_ms) {
        RTP_LLM_LOG_WARNING("writeByLayerTag [P2P Prefill]: drop layer after request deadline, "
                            "request_id=%ld, layer_id=%d, request_deadline_ms=%ld",
                            request_id,
                            layer_id,
                            request_deadline_ms);
        // Expiration is a request outcome, not an executor dispatch failure.
        // Invalid/missing deadlines still reject the call.
        return request_deadline_ms > 0 && request_deadline_ms != std::numeric_limits<int64_t>::max();
    }
    const auto request_horizon =
        computed_buffers_->registerRequestHorizon(request_id, local_horizon_ms, request_deadline_ms);
    if (!request_horizon.has_value()) {
        RTP_LLM_LOG_DEBUG("writeByLayerTag [P2P Prefill]: ignore layer after transfer completed, "
                          "request_id=%ld, layer_id=%d",
                          request_id,
                          layer_id);
        return true;
    }
    const int64_t deadline_ms = *request_horizon;
    using PublicationStatus = ComputedLayerCacheBufferStore::PublicationStatus;
    const auto status = computed_buffers_->registerLayerPublication(request_id, request_deadline_ms, *layer_cache_buffer);
    if (status == PublicationStatus::Inactive) {
        return true;
    }
    if (status == PublicationStatus::Duplicate) {
        return rejectLayer(request_id,
                           request_deadline_ms,
                           ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_STREAM_RESOURCE_FAILED,
                                     "duplicate layer publication: layer=" + std::to_string(layer_id)
                                         + " tag=" + tag + " chunk_id=" + std::to_string(chunk_id)));
    }
    store_wait_context_checker_->addContext(
        StoreWaitContext(request_id, event, layer_cache_buffer, deadline_ms, collector, request_deadline_ms));
    if (layer_id == 0) {
        RTP_LLM_LOG_DEBUG("writeByLayerTag [P2P Prefill]: queued request_id=%ld, layer_id=%d, blocks=%zu, deadline_ms=%ld "
                          "(request=%ld)",
                          request_id,
                          layer_id,
                          collector->prefill_worker_store_total_block_count,
                          deadline_ms,
                          request_deadline_ms);
    }
    return true;
}

void P2PWorkerPrefillRead::loopCheckProc() {
    store_wait_context_checker_->checkOnce();
    computed_buffers_->checkTimeout();

    if (metrics_reporter_) {
        auto collector = std::make_shared<P2PConnectorMetricsCollector>();
        collector->prefill_worker_wait_store_event_count =
            store_wait_context_checker_ ? store_wait_context_checker_->getContextCount() : 0;
        collector->prefill_worker_task_count             = 0;
        collector->prefill_worker_computed_request_count = computed_buffers_->getBuffersCount();
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, collector.get());
    }
}

int P2PWorkerPrefillRead::dispatchPendingLayerTransfers(
    const std::shared_ptr<ComputedLayerCacheBuffer>& computed_buffer,
    const P2PWorkerRoutePlan&                        worker_plan,
    const std::string&                               unique_key,
    int64_t                                          return_deadline_ms,
    const std::shared_ptr<std::atomic<bool>>&        cancel_flag,
    const std::shared_ptr<SendTransferResult>&       transfer_result,
    const std::set<std::string>&                     expected_buffer_keys) {
    int sent_count = 0;
    while (transfer_result->all_success.load() && !cancel_flag->load() && currentTimeMs() < return_deadline_ms) {
        const auto sequence = computed_buffer->changeSequence();
        if (computed_buffer->sealedAndEmpty()) {
            break;
        }
        const auto error = computed_buffer->error();
        if (error.hasError()) {
            std::lock_guard<std::mutex> lock(transfer_result->result_mutex);
            if (transfer_result->all_success.exchange(false)) {
                transfer_result->error_code = error.code();
                transfer_result->error_msg = error.ToString();
            }
            break;
        }

        bool dispatched = false;
        for (const auto& buffer_key : expected_buffer_keys) {
            if (!transfer_result->all_success.load() || cancel_flag->load()) {
                break;
            }
            {
                std::lock_guard<std::mutex> lock(transfer_result->result_mutex);
                if (transfer_result->in_flight_by_layer[buffer_key] > 0) {
                    continue;
                }
            }
            auto buffer = computed_buffer->takeBuffer(buffer_key);
            if (!buffer) {
                continue;
            }
            if (buffer->blockIdMap().empty()) {
                continue;
            }
            const auto count = sendLayerToPartitions(
                buffer, worker_plan, unique_key, return_deadline_ms, cancel_flag, transfer_result);
            if (count > 0) {
                const auto ready_us = currentTimeUs();
                if (transfer_result->first_buffer_ready_us < 0) {
                    transfer_result->first_buffer_ready_us = ready_us;
                }
                transfer_result->last_buffer_ready_us = ready_us;
                sent_count += count;
            }
            dispatched = true;
        }
        if (!dispatched) {
            computed_buffer->waitChange(sequence, 50);
        }
    }
    return sent_count;
}

int P2PWorkerPrefillRead::sendLayerToPartitions(const std::shared_ptr<LayerCacheBuffer>&   layer_cache_buffer,
                                                     const P2PWorkerRoutePlan&                  worker_plan,
                                                     const std::string&                         unique_key,
                                                     int64_t                                    transfer_deadline_ms,
                                                     const std::shared_ptr<std::atomic<bool>>&  cancel_flag,
                                                     const std::shared_ptr<SendTransferResult>& transfer_result) {
    int       count    = 0;
    const int layer_id = layer_cache_buffer->getLayerId();
    const auto buffer_key = layer_cache_buffer->bufferKey();
    const int  chunk_id   = layer_cache_buffer->chunkId();

    // Account for both transport completion and a task rejected before enqueue.
    auto make_send_done_cb = [transfer_result, unique_key, buffer_key, chunk_id](
                                 const std::string& route_layer_key, const std::string& peer) {
        return [transfer_result, route_layer_key, unique_key, buffer_key, chunk_id, peer](
                   transfer::TransferErrorCode transfer_ec, const std::string& cb_error_msg) {
            RTP_LLM_LOG_DEBUG("send done, route_layer_key=%s chunk_id=%d success=%d",
                              route_layer_key.c_str(),
                              chunk_id,
                              transfer_ec == transfer::TransferErrorCode::OK);
            if (transfer_ec != transfer::TransferErrorCode::OK) {
                std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
                if (transfer_result->all_success.exchange(false)) {
                    transfer_result->error_code = transfer::toErrorCode(transfer_ec);
                    transfer_result->error_msg  = "Prefill send key=" + route_layer_key + " chunk_id="
                                                 + std::to_string(chunk_id) + " transfer_code="
                                                 + std::to_string(static_cast<int>(transfer_ec)) + ": " + cb_error_msg;
                }
            }
            {
                std::lock_guard<std::mutex> lock(transfer_result->result_mutex);
                --transfer_result->in_flight_by_layer[buffer_key];
                transfer_result->done_count.fetch_add(1);
                transfer_result->result_cv.notify_all();
            }
            if (auto computed_buffer = transfer_result->computed_buffer.lock()) {
                computed_buffer->notifyChange();
            }
            if (transfer_ec != transfer::TransferErrorCode::OK) {
                RTP_LLM_LOG_WARNING("P2P Prefill send callback failed, unique_key=%s route_layer_key=%s chunk_id=%d "
                                    "peer=%s transfer_code=%d transfer_name=%s error=%s",
                                    unique_key.c_str(),
                                    route_layer_key.c_str(),
                                    chunk_id,
                                    peer.c_str(),
                                    static_cast<int>(transfer_ec),
                                    transfer::transferErrorCodeToString(transfer_ec),
                                    cb_error_msg.c_str());
            }
        };
    };
    auto mark_dispatch_failure = [transfer_result](ErrorCode error_code, const std::string& error_msg) {
        std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
        if (!transfer_result->dispatch_failed.load()) {
            if (transfer_result->all_success.exchange(false)) {
                transfer_result->error_code = error_code;
                transfer_result->error_msg  = error_msg;
            }
            transfer_result->dispatch_failed.store(true);
            transfer_result->result_cv.notify_all();
        }
    };

    const size_t payload_bytes =
        config_.topology ? config_.topology->group(layer_cache_buffer->cacheTag()).spec->k_block_payload_bytes() : 0;

    for (const auto& route : worker_plan.routes) {
        // route 与 layer 无关：只发本 buffer 所属 tag 的那些 route。
        if (route.cache_tag != layer_cache_buffer->cacheTag()) {
            continue;
        }

        // partition / slice 均来自 route（本侧那一半）。§2.3 的修正就在这里生效：
        // NP1D 下 planner 给源端的是 {1,0}（整块），不再对本地 block 二次切分。
        auto key_block_infos = LayerCacheBufferUtil::buildKeyBlockInfosSliced(layer_block_converter_,
                                                                              layer_cache_buffer,
                                                                              route.partition.count,
                                                                              route.partition.id,
                                                                              route.slice,
                                                                              payload_bytes);
        if (!key_block_infos.ok() || key_block_infos.value().empty()
            || key_block_infos.value().size() != layer_cache_buffer->blockIdMap().size()) {
            const std::string conversion_message =
                !key_block_infos.ok() ?
                    key_block_infos.status().ToString() :
                    "converted key count=" + std::to_string(key_block_infos.value().size())
                        + " differs from source key count=" + std::to_string(layer_cache_buffer->blockIdMap().size());
            mark_dispatch_failure(!key_block_infos.ok() ? key_block_infos.status().code() :
                                      ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED,
                                  "sendKVCache: route=" + std::to_string(route.route_id) + " layer="
                                      + std::to_string(layer_id) + " tag=" + layer_cache_buffer->cacheTag()
                                      + " chunk_id=" + std::to_string(chunk_id)
                                      + " task registration failed, unique_key=" + unique_key + ": "
                                      + conversion_message);
            return count;
        }

        if (transfer_result->dispatch_failed.load() || (cancel_flag && cancel_flag->load(std::memory_order_relaxed))
            || currentTimeMs() >= transfer_deadline_ms) {
            return count;
        }

        std::string route_layer_key = P2PKeyUtil::makeRouteLayerKey(
            unique_key, layer_id, layer_cache_buffer->cacheTag(), route.route_id, worker_plan.plan_digest);

        transfer::SendRequest send_req;
        send_req.ip          = route.dst_ip;
        send_req.port        = route.dst_port;
        send_req.unique_key  = route_layer_key;
        send_req.block_info  = std::move(key_block_infos.value());
        send_req.deadline_ms = transfer_deadline_ms;
        // The Decode transfer budget owns this window, not Prefill's local configuration.
        send_req.timeout_ms = std::min<int64_t>(std::numeric_limits<int32_t>::max(),
                                                std::max<int64_t>(1, transfer_deadline_ms - currentTimeMs()));

        {
            std::lock_guard<std::mutex> lock(transfer_result->result_mutex);
            ++transfer_result->in_flight_by_layer[buffer_key];
        }
        ++count;

        // OPT-A2: push sender_->send onto the dedicated pool. The dispatcher
        // (this thread) returns immediately after enqueuing; the cuda copy +
        // sync inside TcpKVCacheSender::makeTransferRequest now runs on the
        // pool worker thread. The completion callback is unchanged (still
        // invoked by arpc IO threads when the response arrives).
        //
        // Task state owns the request until execution or cancellation releases it.
        auto done_cb = make_send_done_cb(route_layer_key, route.dst_ip + ":" + std::to_string(route.dst_port));
        auto                              send_req_shared = std::make_shared<transfer::SendRequest>(std::move(send_req));
        auto task_state                   = std::make_shared<AsyncSendTaskState>();
        task_state->send_request          = send_req_shared;
        task_state->buffer_keepalive      = layer_cache_buffer;
        registerAsyncSendTask(unique_key, task_state);

        const int64_t enqueue_time_us = currentTimeUs();
        auto          task            = [sender = sender_,
                     task_state,
                     done_cb,
                     cancel_flag,
                     route_layer_key,
                     chunk_id,
                     reporter = metrics_reporter_,
                     enqueue_time_us]() mutable {
            P2PConnectorMetricsCollector submit_collector;
            submit_collector.prefill_worker_write_sender_queue_time_us = currentTimeUs() - enqueue_time_us;
            if (reporter) {
                reporter->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, &submit_collector);
            }
            submit_collector.prefill_worker_write_sender_queue_time_us = -1;
            transfer::SendRequestPtr          send_req_local;
            std::shared_ptr<LayerCacheBuffer> buffer_keepalive_local;
            if (!task_state->takeForStart(&send_req_local, &buffer_keepalive_local)) {
                done_cb(transfer::TransferErrorCode::CANCELLED, "send cancelled before async task started");
                return;
            }
            if (cancel_flag && cancel_flag->load(std::memory_order_relaxed)) {
                done_cb(transfer::TransferErrorCode::CANCELLED, "send cancelled before async task started");
                return;
            }
            // Measure inside-pool cost; the sync prefix of sender->send()
            // (cuda copy + sync) can take seconds under contention.
            const int64_t pool_task_start_us = currentTimeUs();
            auto measured_done_cb = [done_cb, reporter, pool_task_start_us, buffer_keepalive_local](
                                        transfer::TransferErrorCode ec, const std::string& message) {
                // Sender callbacks, including synchronous completion, are the transport completion boundary.
                P2PConnectorMetricsCollector completed_collector;
                completed_collector.prefill_worker_write_success = ec == transfer::TransferErrorCode::OK;
                completed_collector.prefill_worker_write_send_complete_time_us = currentTimeUs() - pool_task_start_us;
                if (reporter) {
                    reporter->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, &completed_collector);
                }
                (void)buffer_keepalive_local;
                done_cb(ec, message);
            };
            auto report_submit = [&] {
                submit_collector.prefill_worker_write_send_submit_time_us = currentTimeUs() - pool_task_start_us;
                if (reporter) {
                    reporter->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, &submit_collector);
                }
            };
            try {
                sender->send(*send_req_local, measured_done_cb);
            } catch (const std::exception& e) {
                RTP_LLM_LOG_WARNING("P2PWorkerPrefillRead async send threw, route_layer_key=%s chunk_id=%d error=%s",
                                    route_layer_key.c_str(),
                                    chunk_id,
                                    e.what());
                report_submit();
                done_cb(transfer::TransferErrorCode::UNKNOWN, e.what());
                return;
            } catch (...) {
                RTP_LLM_LOG_WARNING("P2PWorkerPrefillRead async send threw unknown exception, "
                                    "route_layer_key=%s chunk_id=%d",
                                    route_layer_key.c_str(),
                                    chunk_id);
                report_submit();
                done_cb(transfer::TransferErrorCode::UNKNOWN, "unknown async send exception");
                return;
            }
            report_submit();
            const int64_t pool_task_cost_us = currentTimeUs() - pool_task_start_us;
            if (pool_task_cost_us >= 100000) {
                RTP_LLM_LOG_WARNING("async_sender_pool task slow, "
                                    "route_layer_key=%s chunk_id=%d pool_task_cost_us=%ld",
                                    route_layer_key.c_str(),
                                    chunk_id,
                                    pool_task_cost_us);
            }

        };
        // Blocking enqueue cannot observe this request's cancellation/deadline.
        // Never fall back to inline send: its synchronous GPU copy can also block
        // the dispatcher. Reject overload and use the normal request cleanup path.
        const auto enqueue_error = async_sender_pool_ ?
                                       async_sender_pool_->pushTask(
                                           std::move(task), /*isBlocked=*/false, /*executeWhenFail=*/false) :
                                       autil::ThreadPoolBase::ERROR_POOL_HAS_STOP;
        if (enqueue_error != autil::ThreadPoolBase::ERROR_NONE) {
            const std::string reason = enqueue_error == autil::ThreadPoolBase::ERROR_POOL_QUEUE_FULL ?
                                           "sender queue full" : "sender pool unavailable";
            const std::string message = "sendKVCache: " + reason + " enqueue_error="
                                        + std::to_string(static_cast<int>(enqueue_error))
                                        + " key=" + route_layer_key + " chunk_id=" + std::to_string(chunk_id);
            RTP_LLM_LOG_WARNING("%s", message.c_str());
            task_state->releaseIfNotStarted();
            mark_dispatch_failure(ErrorCode::P2P_CONNECTOR_TRANSFER_QUEUE_REJECTED, message);
            // count already includes this rejected task. No pool task will run,
            // so complete it exactly once here; preserve the dispatch error above.
            done_cb(transfer::TransferErrorCode::QUEUE_REJECTED, message);
            return count;
        }
    }
    return count;
}

bool P2PWorkerPrefillRead::waitSendCallbacksWithTimeout(const std::shared_ptr<SendTransferResult>& transfer_result,
                                                             int     sent_transfer_count,
                                                             int64_t return_deadline_ms,
                                                             const std::shared_ptr<std::atomic<bool>>& cancel_flag) const {
    if (!transfer_result->all_success.load()) {
        return false;
    }
    const int64_t callback_deadline_ms = return_deadline_ms;
    std::unique_lock<std::mutex> lock(transfer_result->result_mutex);
    while (transfer_result->done_count.load(std::memory_order_relaxed) < sent_transfer_count) {
        // Honor cancel_flag here too — without this, a CANCEL_HANDLE_READ RPC
        // arriving while we wait for RDMA send callbacks has nowhere to land,
        // and we'd block until return_deadline_ms (≈ business deadline, up to 1h).
        // determineSendResult() will see cancel_flag.load() and return
        // P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED instead of TIMEOUT.
        if (!transfer_result->all_success.load()) {
            return false;
        }
        if (cancel_flag && cancel_flag->load()) {
            RTP_LLM_LOG_WARNING(
                "waitSendCallbacksWithTimeout cancelled, done_count: %ld, expected: %d, return_deadline_ms: %ld",
                transfer_result->done_count.load(std::memory_order_relaxed),
                sent_transfer_count,
                return_deadline_ms);
            return false;
        }
        const int64_t now = currentTimeMs();
        if (now >= callback_deadline_ms) {
            RTP_LLM_LOG_WARNING(
                "waitSendCallbacksWithTimeout timeout, done_count: %ld, expected: %d, callback_deadline_ms: %ld",
                transfer_result->done_count.load(std::memory_order_relaxed),
                sent_transfer_count,
                callback_deadline_ms);
            return false;
        }
        const int64_t wait_ms = callback_deadline_ms - now;
        if (wait_ms <= 0) {
            return false;
        }
        const bool ready = transfer_result->result_cv.wait_for(
            lock,
            std::chrono::milliseconds(wait_ms),
            [&transfer_result, sent_transfer_count, &cancel_flag]() {
                // Wake up early when cancel_flag flips so we don't have to wait
                // out the full load deadline before re-checking it. This
                // requires cancelRequest() to notify result_cv
                // after setting cancel_flag — see below.
                return transfer_result->done_count.load(std::memory_order_relaxed) >= sent_transfer_count
                       || !transfer_result->all_success.load() || (cancel_flag && cancel_flag->load());
            });
        if (!transfer_result->all_success.load()) {
            return false;
        }
        if (cancel_flag && cancel_flag->load()) {
            // Loop will re-check and return false on next iteration; exit early
            // here too in case the cv predicate fired due to cancel.
            continue;
        }
        if (ready) {
            return true;
        }
        return false;
    }
    return true;
}

ErrorInfo
P2PWorkerPrefillRead::sendKVCache(int64_t                   request_id,
                                       const std::string&        unique_key,
                                       int64_t                   deadline_ms,
                                       const P2PWorkerRoutePlan& worker_plan,
                                       int64_t                   request_deadline_ms) {
    if (request_deadline_ms <= 0 || request_deadline_ms == std::numeric_limits<int64_t>::max()
        || deadline_ms <= currentTimeMs() || deadline_ms == std::numeric_limits<int64_t>::max()) {
        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, "sendKVCache: invalid or expired deadlines");
    }
    // Register lifecycle even if StartLoad reaches this worker before the
    // first layer callback. removed_requests_ is the single terminal source
    // for all later add/register attempts.
    const auto active_deadline =
        computed_buffers_->activateRequestHorizon(request_id, deadline_ms, request_deadline_ms);
    if (!active_deadline) {
        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT,
                         "sendKVCache: request already terminal, unique_key: " + unique_key);
    }
    deadline_ms = *active_deadline;
    if (!config_.topology) {
        computed_buffers_->removeBuffer(request_id, request_deadline_ms);
        return ErrorInfo(ErrorCode::P2P_CONNECTOR_SCHEDULER_CALL_WORKER_FAILED, "sendKVCache: invalid topology");
    }
    // 「MLA 只有主 rank 发送」这条规则已由 planner 的副本类选举取代：落选的 rank 收到
    // 零条 route，在此直接返回。worker 不再自行判断自己是不是主 rank。
    if (worker_plan.empty()) {
        RTP_LLM_LOG_DEBUG("sendKVCache [P2P]: no routes for this worker, request_id=%ld, unique_key=%s, tp_rank=%ld",
                          request_id,
                          unique_key.c_str(),
                          config_.tp_rank);
        computed_buffers_->removeBuffer(request_id, request_deadline_ms);
        return ErrorInfo::OkStatus();
    }

    RTP_LLM_LOG_DEBUG(
        "sendKVCache [P2P]: start request_id=%ld, unique_key=%s, deadline_ms=%ld, routes=%zu",
        request_id,
        unique_key.c_str(),
        deadline_ms,
        worker_plan.routes.size());
    const int64_t start_time_us = currentTimeUs();
    auto          collector     = std::make_shared<P2PConnectorMetricsCollector>();

    // 不对称 TP / CP 的配对已由编排层算完，worker 只按 route 执行。expected
    // buffer metadata 在 worker 初始化时预计算，CacheTopology 生命周期内不变。
    const auto& expected_buffer_keys = expected_buffer_keys_;

    // 每个 (layer, tag) 上的传输数 = 该 tag 的 route 数，故按 tag 逐个累加，
    // 不能再用「buffer_key 数 × 单一 partition 数」。
    int total_transfers = 0;
    for (const auto& tag : expected_buffer_tags_) {
        total_transfers += worker_plan.routeCountForTag(tag);
    }
    if (total_transfers == 0) {
        computed_buffers_->removeBuffer(request_id, request_deadline_ms);
        RTP_LLM_LOG_DEBUG("sendKVCache [P2P]: no local blocks in requested range, request_id=%ld, unique_key=%s",
                          request_id,
                          unique_key.c_str());
        return ErrorInfo::OkStatus();
    }
    auto transfer_result = std::make_shared<SendTransferResult>();

    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
    {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        // transfer_result is held in this stack frame; weak_ptr is fine because
        // we only dereference it from cancelRequest() while this frame is alive.
        handle_cancel_flags_[unique_key] = {cancel_flag, std::weak_ptr<SendTransferResult>(transfer_result)};
    }

    // [PD-DIAG] Three-stage timing for sendKVCache: addBuffer / dispatch / waitCallbacks.
    // Each can independently account for the dominant cost depending on the failure mode:
    //  - addBuffer slow → ComputedLayerCacheBufferStore mutex contention
    //  - dispatch slow → sender_->send synchronous blocking or layer cache scan slow
    //  - waitCallbacks slow → RDMA callback never arrives (the actual stuck RPC scenario)
    const int64_t add_buffer_start_us                  = currentTimeUs();
    auto          computed_layer_cache_buffer = computed_buffers_->addBuffer(request_id, nullptr, request_deadline_ms);
    const int64_t add_buffer_cost_us                   = currentTimeUs() - add_buffer_start_us;
    collector->prefill_worker_write_add_buffer_time_us = add_buffer_cost_us;

    if (!computed_layer_cache_buffer) {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        handle_cancel_flags_.erase(unique_key);
        const std::string error_msg =
            "sendKVCache: computed layers already expired and removed, unique_key: " + unique_key;
        RTP_LLM_LOG_WARNING("%s", error_msg.c_str());
        if (metrics_reporter_) {
            collector->prefill_worker_write_success            = false;
            collector->prefill_worker_write_total_cost_time_us = currentTimeUs() - start_time_us;
            metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, collector.get());
        }
        // GENERATE_TIMEOUT (603) signals "request's business deadline already
        // expired by the time prefill could serve the StartLoad". Decode side
        // should surface this as a timeout to the client rather than retry.
        // Distinct from P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT (8316), which
        // means the in-flight RDMA transfer itself missed its deadline.
        return ErrorInfo(ErrorCode::GENERATE_TIMEOUT, error_msg);
    }

    // A failed layer must also wake a dispatcher blocked on the outstanding-send budget.
    computed_layer_cache_buffer->setErrorHandler(
        [weak_result = std::weak_ptr<SendTransferResult>(transfer_result)](const ErrorInfo& error) {
            if (auto result = weak_result.lock()) {
                std::lock_guard<std::mutex> lock(result->result_mutex);
                if (result->all_success.exchange(false)) {
                    result->error_code = error.code();
                    result->error_msg  = error.ToString();
                }
                result->dispatch_failed.store(true);
                result->result_cv.notify_all();
            }
        });

    const int64_t dispatch_start_us = currentTimeUs();
    transfer_result->computed_buffer = computed_layer_cache_buffer;
    const int sent_transfer_count = dispatchPendingLayerTransfers(computed_layer_cache_buffer,
                                                                  worker_plan,
                                                                  unique_key,
                                                                  deadline_ms,
                                                                  cancel_flag,
                                                                  transfer_result,
                                                                  expected_buffer_keys);
    const bool publication_drained = computed_layer_cache_buffer->sealedAndEmpty();
    {
        std::lock_guard<std::mutex> lock(transfer_result->result_mutex);
        if (!publication_drained && transfer_result->all_success.exchange(false)) {
            transfer_result->error_code =
                cancel_flag->load() ? ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED :
                (currentTimeMs() >= deadline_ms ? ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT :
                                                 ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TRANSFER_FAILED);
            transfer_result->error_msg = "sendKVCache: Prefill publication did not drain, sent="
                                         + std::to_string(sent_transfer_count) + " routes=" + describeRoutes(worker_plan)
                                         + ", unique_key: " + unique_key;
        }
    }
    const int64_t dispatch_cost_us                   = currentTimeUs() - dispatch_start_us;
    collector->prefill_worker_write_dispatch_time_us = dispatch_cost_us;
    if (transfer_result->first_buffer_ready_us >= 0) {
        collector->prefill_worker_write_first_layer_wait_time_us =
            transfer_result->first_buffer_ready_us - start_time_us;
    }
    if (publication_drained && transfer_result->last_buffer_ready_us >= 0) {
        collector->prefill_worker_write_last_layer_wait_time_us = transfer_result->last_buffer_ready_us - start_time_us;
    }

    // NOTE: do NOT erase handle_cancel_flags_[unique_key] here. The wait below
    // is the dominant phase (RDMA callbacks may take seconds to minutes), and
    // erasing the cancel_flag now would make any late-arriving CANCEL_HANDLE_READ
    // RPC silently miss its target — see Bug ① in the analysis doc.
    // We erase only after waitSendCallbacksWithTimeout returns.

    const int64_t wait_cb_start_us = currentTimeUs();
    const bool    all_callbacks_received =
        waitSendCallbacksWithTimeout(transfer_result, sent_transfer_count, deadline_ms, cancel_flag);
    const int64_t wait_cb_cost_us                         = currentTimeUs() - wait_cb_start_us;
    collector->prefill_worker_write_callback_wait_time_us = wait_cb_cost_us;
    const bool timeout_cancelled_pending_tasks =
        !all_callbacks_received && !cancel_flag->load(std::memory_order_relaxed);

    if (timeout_cancelled_pending_tasks) {
        {
            std::lock_guard<std::mutex> lock(transfer_result->result_mutex);
            if (transfer_result->all_success.exchange(false)) {
                transfer_result->error_code = ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT;
                transfer_result->error_msg  = "Prefill send callback deadline exceeded key=" + unique_key
                                             + " sent=" + std::to_string(sent_transfer_count)
                                             + " planned=" + std::to_string(total_transfers);
            }
        }
        cancel_flag->store(true, std::memory_order_relaxed);
        std::shared_ptr<SendTransferResult> wake_result = transfer_result;
        const int released_pending_task_count = releasePendingAsyncSendTasks(unique_key, &wake_result);
        {
            std::lock_guard<std::mutex> lk(wake_result->result_mutex);
            wake_result->result_cv.notify_all();
        }
        RTP_LLM_LOG_WARNING(
            "sendKVCache failure released %d queued async sender tasks, request_id: %ld, unique_key: %s",
            released_pending_task_count,
            request_id,
            unique_key.c_str());
    }

    if (!all_callbacks_received) {
        RTP_LLM_LOG_WARNING(
            "sendKVCache transfer callback wait ended before load deadline, request_id: %ld, unique_key: %s, cancelled: %d",
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

    // Mark the request key as removed, preventing late-arriving layers from
    // StoreWaitContextChecker from creating orphan description entries.
    computed_buffers_->removeBuffer(request_id, request_deadline_ms);

    auto send_result = determineSendResult(transfer_result,
                                           cancel_flag,
                                           timeout_cancelled_pending_tasks,
                                           all_callbacks_received,
                                           sent_transfer_count,
                                           worker_plan,
                                           unique_key);

    if (metrics_reporter_) {
        collector->prefill_worker_write_success            = send_result.success;
        collector->prefill_worker_write_total_cost_time_us = currentTimeUs() - start_time_us;
        metrics_reporter_->report<P2PConnectorMetrics, P2PConnectorMetricsCollector>(nullptr, collector.get());
    }

    if (!send_result.success) {
        RTP_LLM_LOG_WARNING("P2P Prefill sendKVCache failed, request_id=%ld unique_key=%s rank=%d "
                            "error_code=%d error_name=%s error=%s",
                            request_id,
                            unique_key.c_str(),
                            static_cast<int>(config_.tp_rank),
                            static_cast<int>(send_result.error_code),
                            ErrorCodeToString(send_result.error_code).c_str(),
                            send_result.error_msg.c_str());
        return ErrorInfo(send_result.error_code, send_result.error_msg);
    }

    RTP_LLM_LOG_DEBUG("sendKVCache [P2P]: done request_id=%ld, unique_key=%s, sent_batches=%d, cost_us=%ld",
                      request_id,
                      unique_key.c_str(),
                      sent_transfer_count,
                      currentTimeUs() - start_time_us);
    return ErrorInfo::OkStatus();
}

void P2PWorkerPrefillRead::finishPublication(int64_t request_id, int64_t request_deadline_ms) {
    store_wait_context_checker_->finishPublication(request_id, request_deadline_ms);
}

void P2PWorkerPrefillRead::completeNoTransfer(int64_t request_id,
                                                   int64_t /*deadline_ms*/,
                                                   int64_t request_deadline_ms) {
    computed_buffers_->removeBuffer(request_id, request_deadline_ms);
    RTP_LLM_LOG_DEBUG("sendKVCache [P2P]: no-transfer request completed, request_id=%ld", request_id);
}

bool P2PWorkerPrefillRead::cancelRequest(int64_t            request_id,
                                              const std::string& unique_key,
                                              int64_t            /*deadline_ms*/,
                                              int64_t            request_deadline_ms) {
    // removeBuffer also records a (request_id, request_deadline_ms) tombstone. Layers or StartLoad
    // arriving after cancellation are rejected instead of creating an empty
    // computed buffer and waiting until the request deadline.
    computed_buffers_->removeBuffer(request_id, request_deadline_ms);
    RTP_LLM_LOG_DEBUG("cancelRequest start, request_id=%ld, unique_key=%s", request_id, unique_key.c_str());

    std::shared_ptr<std::atomic<bool>> cancel_flag;
    std::shared_ptr<SendTransferResult> transfer_result;
    {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        auto                        it = handle_cancel_flags_.find(unique_key);
        if (it == handle_cancel_flags_.end()) {
            RTP_LLM_LOG_INFO("cancelRequest: unique_key not found: %s (best-effort)", unique_key.c_str());
            return true;
        }
        cancel_flag     = it->second.cancel_flag;
        transfer_result = it->second.transfer_result.lock();
    }
    if (transfer_result) {
        std::lock_guard<std::mutex> lock(transfer_result->result_mutex);
        if (transfer_result->all_success.exchange(false)) {
            transfer_result->error_code = ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED;
            transfer_result->error_msg  = "Prefill send cancelled key=" + unique_key;
        }
    }
    cancel_flag->store(true, std::memory_order_relaxed);
    const int released_pending_task_count = releasePendingAsyncSendTasks(unique_key, &transfer_result);
    // Wake up waitSendCallbacksWithTimeout immediately so it sees the flag,
    // instead of waiting for the load deadline before re-checking.
    if (transfer_result) {
        std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
        transfer_result->result_cv.notify_all();
    }
    RTP_LLM_LOG_INFO("cancelRequest success, request_id=%ld, unique_key=%s, released_pending_tasks=%d, notified_cv=%d",
                     request_id,
                     unique_key.c_str(),
                     released_pending_task_count,
                     transfer_result ? 1 : 0);
    return true;
}

P2PWorkerPrefillRead::SendResultInfo
P2PWorkerPrefillRead::determineSendResult(const std::shared_ptr<SendTransferResult>& transfer_result,
                                               const std::shared_ptr<std::atomic<bool>>&  cancel_flag,
                                               bool                                       timeout_cancelled_pending_tasks,
                                               bool                                       all_callbacks_received,
                                               int                                        sent_transfer_count,
                                               const P2PWorkerRoutePlan&                  worker_plan,
                                               const std::string&                         unique_key) const {

    if (!transfer_result->all_success.load()) {
        std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
        return {false, transfer_result->error_code, transfer_result->error_msg};
    }
    if (cancel_flag->load()) {
        return {false,
                ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_CANCELLED,
                "sendKVCache cancelled, unique_key: " + unique_key};
    }
    const std::string transfer_summary = "sent=" + std::to_string(sent_transfer_count)
                                         + " routes=" + describeRoutes(worker_plan);
    if (timeout_cancelled_pending_tasks) {
        return {false,
                ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT,
                "sendKVCache: transfer callback wait timeout, " + transfer_summary
                    + ", unique_key: " + unique_key};
    }
    if (!all_callbacks_received) {
        return {false,
                ErrorCode::P2P_CONNECTOR_WORKER_HANDLE_READ_TIMEOUT,
                "sendKVCache: transfer callback wait timeout, " + transfer_summary
                    + ", unique_key: " + unique_key};
    }
    if (!transfer_result->all_success.load()) {
        std::lock_guard<std::mutex> lk(transfer_result->result_mutex);
        return {false, transfer_result->error_code, transfer_result->error_msg};
    }
    return {};
}

}  // namespace rtp_llm
