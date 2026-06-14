#pragma once

#include "rtp_llm/cpp/cache/BatchKVCacheResource.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorConfig.h"
#include <torch/extension.h>
#include "rtp_llm/cpp/cache/connector/p2p/AsymmetricTpUtil.h"
#include "rtp_llm/cpp/cache/connector/p2p/ComputedLayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/P2PConnectorMetrics.h"
#include "rtp_llm/cpp/cache/connector/p2p/StoreWaitContext.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerBlockConverter.h"
#include "rtp_llm/cpp/cache/connector/p2p/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheSender.h"
#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "autil/LoopThread.h"
#include "autil/ThreadPool.h"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace rtp_llm {

class P2PConnectorWorkerPrefill {
public:
    P2PConnectorWorkerPrefill(P2PConnectorWorkerConfig                    config,
                              const std::shared_ptr<LayerBlockConverter>& layer_block_converter,
                              const kmonitor::MetricsReporterPtr&         metrics_reporter,
                              const transfer::IKVCacheSenderPtr&          sender);
    ~P2PConnectorWorkerPrefill();

public:
    bool init(int64_t store_wait_timeout_ms);

    bool writeByLayer(int                           layer_id,
                      const KVCacheResourcePtr&     resource,
                      int64_t                       request_id,
                      std::shared_ptr<torch::Event> event,
                      int64_t                       deadline_ms);

    ErrorInfo sendKVCache(int64_t                                              request_id,
                          const std::string&                                   unique_key,
                          int64_t                                              deadline_ms,
                          const std::vector<std::pair<std::string, uint32_t>>& decode_transfer_servers);

    bool cancelSend(const std::string& unique_key);

    std::shared_ptr<ComputedLayerCacheBufferStore> getComputedBuffersStore() const {
        return computed_buffers_;
    }
    void setStoreWaitTimeoutMs(int64_t store_wait_timeout_ms) {
        store_wait_timeout_ms_ = store_wait_timeout_ms;
    }

private:
    void loopCheckProc();

    struct SendTransferResult {
        std::atomic<int>        done_count{0};
        // Async sender-pool tasks that are queued/running but have not yet
        // returned from sender->send(). Throttling dispatch must use this
        // counter instead of remote transfer completion done_count.
        std::atomic<int>        async_send_task_count{0};
        std::atomic<bool>       all_success{true};
        mutable std::mutex      result_mutex;
        std::condition_variable result_cv;
        ErrorCode               error_code{ErrorCode::NONE_ERROR};
        std::string             error_msg;
    };

    /// return_deadline_ms：须在此刻前结束 dispatch 与 send（与 decode recv_req.deadline_ms 对齐，均为 D -
    /// p2p_read_return_before_deadline_ms）
    int dispatchPendingLayerTransfers(const std::shared_ptr<ComputedLayerCacheBuffer>& computed_buffer,
                                      const std::vector<AsymmetricTPContext>&          tp_partition_ctxs,
                                      const std::string&                               unique_key,
                                      int64_t                                          return_deadline_ms,
                                      const std::shared_ptr<std::atomic<bool>>&        cancel_flag,
                                      const std::shared_ptr<SendTransferResult>&       transfer_result,
                                      std::set<int>&                                   sent_layer_ids,
                                      int                                              total_transfers);

    int sendLayerToPartitions(const std::shared_ptr<LayerCacheBuffer>&   layer_cache_buffer,
                              const std::vector<AsymmetricTPContext>&    tp_partition_ctxs,
                              const std::string&                         unique_key,
                              int64_t                                    transfer_deadline_ms,
                              int                                        max_outstanding_tasks,
                              const std::shared_ptr<std::atomic<bool>>&  cancel_flag,
                              const std::shared_ptr<SendTransferResult>& transfer_result);

    bool waitForAsyncSendSlot(const std::shared_ptr<SendTransferResult>& transfer_result,
                              int                                        max_outstanding_tasks,
                              int64_t                                    return_deadline_ms,
                              const std::shared_ptr<std::atomic<bool>>&  cancel_flag) const;

    bool waitSendCallbacksWithTimeout(const std::shared_ptr<SendTransferResult>& transfer_result,
                                      int                                        sent_transfer_count,
                                      int64_t                                    return_deadline_ms,
                                      const std::shared_ptr<std::atomic<bool>>&  cancel_flag) const;

    struct SendResultInfo {
        bool        success = true;
        ErrorCode   error_code{ErrorCode::NONE_ERROR};
        std::string error_msg;
    };

    SendResultInfo determineSendResult(const std::shared_ptr<SendTransferResult>& transfer_result,
                                       const std::shared_ptr<std::atomic<bool>>&  cancel_flag,
                                       bool                                       timeout_cancelled_pending_tasks,
                                       bool                                       all_callbacks_received,
                                       int                                        sent_transfer_count,
                                       int                                        total_transfers,
                                       int64_t                                    return_deadline_ms,
                                       const std::string&                         unique_key) const;

    struct AsyncSendTaskState {
        bool takeForStart(transfer::SendRequestPtr*              send_request_out,
                          std::shared_ptr<LayerCacheBuffer>*     buffer_keepalive_out);
        bool releaseIfNotStarted();

        mutable std::mutex               mutex;
        transfer::SendRequestPtr         send_request;
        std::shared_ptr<LayerCacheBuffer> buffer_keepalive;
        bool                             started{false};
        bool                             released{false};
    };

    void registerAsyncSendTask(const std::string&                     unique_key,
                               const std::shared_ptr<AsyncSendTaskState>& task_state);
    int  releasePendingAsyncSendTasks(const std::string& unique_key,
                                      std::shared_ptr<SendTransferResult>* transfer_result_out = nullptr);
    static int releaseNotStartedTaskStates(const std::vector<std::shared_ptr<AsyncSendTaskState>>& task_states);

private:
    // IMPORTANT: Declaration order determines initialization order in the constructor
    // initializer list. config_ MUST be declared before asymmetric_tp_util_ because
    // the constructor reads config_.tp_size/tp_rank to initialize asymmetric_tp_util_.
    P2PConnectorWorkerConfig                                            config_;
    std::shared_ptr<LayerBlockConverter>                                layer_block_converter_;
    kmonitor::MetricsReporterPtr                                        metrics_reporter_;
    transfer::IKVCacheSenderPtr                                         sender_;
    std::shared_ptr<AsymmetricTpUtil>                                   asymmetric_tp_util_;  // depends on config_
    std::shared_ptr<ComputedLayerCacheBufferStore>                      computed_buffers_;
    int64_t                                                             store_wait_timeout_ms_ = 10 * 1000;
    std::shared_ptr<StoreWaitContextChecker>                            store_wait_context_checker_;
    autil::LoopThreadPtr                                                cleanup_thread_;
    // Per in-flight sendKVCache, hold both the cancel signal and a weak handle
    // to its SendTransferResult. The weak handle lets cancelSend() wake up the
    // wait_for loop in waitSendCallbacksWithTimeout via cv.notify_all() instead
    // of letting cancel_flag sit unchecked for up to rdma_transfer_wait_timeout_ms
    // (180s by default). Weak ref avoids extending the lifetime of the result.
    struct HandleCancelEntry {
        std::shared_ptr<std::atomic<bool>> cancel_flag;
        std::weak_ptr<SendTransferResult>  transfer_result;
        std::vector<std::weak_ptr<AsyncSendTaskState>> async_send_tasks;
    };
    mutable std::mutex                                  handle_cancel_mutex_;
    std::unordered_map<std::string, HandleCancelEntry>  handle_cancel_flags_;

    // OPT-A2: offload sender_->send to a dedicated thread pool. The dispatcher
    // (sendKVCache main thread) used to call sender_->send synchronously, but
    // that call includes makeTransferRequest -> cudaStreamSynchronize which
    // blocks the dispatcher on every layer until the GPU->CPU copy completes.
    // Under cuda runtime contention or GPU hang, dispatch_us balloons to
    // seconds (max 1993s observed on 5/22). With this pool the dispatcher
    // only pays the push cost; the cuda sync runs on pool worker threads.
    // Pool sized at 4 threads: small enough not to oversubscribe cuda runtime,
    // large enough to overlap with 2 prefill ranks on the same device.
    // Queue 10000 keeps headroom for the per-layer × per-partition fanout.
    autil::ThreadPoolBasePtr async_sender_pool_;
};

}  // namespace rtp_llm
