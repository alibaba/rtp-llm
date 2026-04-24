#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorWorkerPrefill.h"

#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PKeyUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBufferUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"
#include <algorithm>
#include <chrono>
#include <set>
#include <thread>

namespace rtp_llm {

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
}

bool P2PConnectorWorkerPrefill::init(int64_t store_wait_timeout_ms) {
    store_wait_context_checker_ = std::make_shared<StoreWaitContextChecker>(metrics_reporter_, computed_buffers_);

    cleanup_thread_ = autil::LoopThread::createLoopThread(
        std::bind(&P2PConnectorWorkerPrefill::loopCheckProc, this), 1000, "P2PConnectorWorkerCleanupThread");
    if (!cleanup_thread_) {
        RTP_LLM_LOG_ERROR("init failed: cleanup_thread is null");
        return false;
    }

    store_wait_timeout_ms_ = store_wait_timeout_ms;
    return true;
}

bool P2PConnectorWorkerPrefill::writeByLayer(int                       layer_id,
                                             const KVCacheResourcePtr& resource,
                                             int64_t                   request_id,
                                             std::optional<c10::Event> event) {
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
    collector->total_block_count = layer_cache_buffer->blockIdMap().size();

    int64_t deadline_ms = currentTimeMs() + store_wait_timeout_ms_;
    store_wait_context_checker_->addContext(
        StoreWaitContext(request_id, std::move(event), layer_cache_buffer, deadline_ms, collector));
    if (layer_id == 0) {
        RTP_LLM_LOG_INFO("writeByLayer [P2P Prefill]: queued request_id=%ld, layer_id=%d, blocks=%zu",
                         request_id,
                         layer_id,
                         layer_cache_buffer->blockIdMap().size());
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
                layer_cache_buffer, tp_partition_ctxs, unique_key, return_deadline_ms, transfer_result);
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
                                                     const std::shared_ptr<SendTransferResult>& transfer_result) {
    int       count    = 0;
    const int layer_id = layer_cache_buffer->getLayerId();

    for (const auto& partition_ctx : tp_partition_ctxs) {
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
        sender_->send(send_req,
                      [transfer_result, partition_layer_key](transfer::TransferErrorCode transfer_ec,
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
                      });
    }
    return count;
}

bool P2PConnectorWorkerPrefill::waitSendCallbacksWithTimeout(const std::shared_ptr<SendTransferResult>& transfer_result,
                                                             int     sent_transfer_count,
                                                             int64_t return_deadline_ms) const {
    const int64_t                rdma_cap_ms = config_.transfer_backend_config.rdma_transfer_wait_timeout_ms;
    std::unique_lock<std::mutex> lock(transfer_result->result_mutex);
    while (transfer_result->done_count.load(std::memory_order_relaxed) < sent_transfer_count) {
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
            lock, std::chrono::milliseconds(wait_ms), [&transfer_result, sent_transfer_count]() {
                return transfer_result->done_count.load(std::memory_order_relaxed) >= sent_transfer_count;
            });
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

    // 计算总传输量
    const int total_transfers = static_cast<int>(config_.layer_all_num) * static_cast<int>(tp_partition_ctxs.size());
    auto      transfer_result = std::make_shared<SendTransferResult>();

    auto cancel_flag = std::make_shared<std::atomic<bool>>(false);
    {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        handle_cancel_flags_[unique_key] = cancel_flag;
    }

    auto computed_layer_cache_buffer    = computed_buffers_->addBuffer(request_id, nullptr, deadline_ms);
    collector->first_layer_wait_time_us = currentTimeUs() - start_time_us;

    std::set<int> sent_layer_ids;
    const int     sent_transfer_count  = dispatchPendingLayerTransfers(computed_layer_cache_buffer,
                                                                  tp_partition_ctxs,
                                                                  unique_key,
                                                                  return_deadline_ms,
                                                                  cancel_flag,
                                                                  transfer_result,
                                                                  sent_layer_ids,
                                                                  total_transfers);
    collector->last_layer_wait_time_us = currentTimeUs() - start_time_us;

    {
        std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
        handle_cancel_flags_.erase(unique_key);
    }

    const bool all_callbacks_received =
        waitSendCallbacksWithTimeout(transfer_result, sent_transfer_count, return_deadline_ms);
    if (!all_callbacks_received) {
        RTP_LLM_LOG_WARNING(
            "sendKVCache transfer callback wait ended before return_deadline_ms or rdma cap, request_id: %ld, unique_key: %s",
            request_id,
            unique_key.c_str());
    }

    auto send_result = determineSendResult(transfer_result,
                                           cancel_flag,
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

    RTP_LLM_LOG_INFO("sendKVCache [P2P]: done request_id=%ld, unique_key=%s, sent=%d/%d, cost_us=%ld",
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
                                               bool                                       all_callbacks_received,
                                               int                                        sent_transfer_count,
                                               int                                        total_transfers,
                                               int64_t                                    return_deadline_ms,
                                               const std::string&                         unique_key) const {

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
    std::lock_guard<std::mutex> lock(handle_cancel_mutex_);
    auto                        it = handle_cancel_flags_.find(unique_key);
    if (it != handle_cancel_flags_.end()) {
        it->second->store(true);
        RTP_LLM_LOG_INFO("cancelSend success, unique_key: %s", unique_key.c_str());
        return true;
    }
    RTP_LLM_LOG_INFO("cancelSend: unique_key not found: %s (best-effort)", unique_key.c_str());
    return true;
}

}  // namespace rtp_llm
