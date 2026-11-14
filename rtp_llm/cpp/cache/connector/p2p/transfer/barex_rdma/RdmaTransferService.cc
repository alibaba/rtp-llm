#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaTransferService.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/utils/TimeUtil.h"

#include <limits>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

RdmaTransferService::RdmaTransferService(const std::shared_ptr<TransferTaskStore>& task_store,
                                         const std::shared_ptr<IRdmaClient>& rdma_client,
                                         const kmonitor::MetricsReporterPtr& metrics_reporter,
                                         const std::shared_ptr<RdmaStagingBufferPool>& staging_pool,
                                         int64_t rdma_disconnect_after_deadline_ms):
    task_store_(task_store),
    rdma_client_(rdma_client),
    metrics_reporter_(metrics_reporter),
    staging_pool_(staging_pool),
    cuda_copy_util_(staging_pool ? std::make_shared<tcp::CudaCopyUtil>() : nullptr),
    rdma_disconnect_after_deadline_ms_(std::max<int64_t>(0, rdma_disconnect_after_deadline_ms)) {}

RdmaTransferService::~RdmaTransferService() {
    if (wait_check_loop_thread_) {
        wait_check_loop_thread_->stop();
    }
    if (transfer_deadline_loop_thread_) {
        transfer_deadline_loop_thread_->stop();
    }
    std::list<std::shared_ptr<RdmaTaskContext>> pending;
    {
        std::lock_guard<std::mutex> lock(wait_tasks_mutex_);
        pending.swap(wait_tasks_);
    }
    for (const auto& ctx : pending) {
        ctx->run(false, TransferErrorCode::CANCELLED, "RdmaTransferService stopped");
    }
    {
        std::lock_guard<std::mutex> lock(active_transfers_mutex_);
        active_transfers_.clear();
    }
    if (worker_thread_pool_) {
        worker_thread_pool_->stop();
    }
}

bool RdmaTransferService::init(int64_t wait_check_interval_us,
                               int worker_thread_count,
                               uint64_t per_connection_block_count,
                               int worker_queue_size) {
    per_connection_block_count_ = per_connection_block_count;

    wait_check_loop_thread_ =
        autil::LoopThread::createLoopThread(std::bind(&RdmaTransferService::waitCheckProc, this),
                                            wait_check_interval_us,
                                            "RdmaTransferServiceWaitCheck");
    if (!wait_check_loop_thread_) {
        RTP_LLM_LOG_ERROR("create wait check loop thread failed");
        return false;
    }

    worker_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        worker_thread_count, worker_queue_size, nullptr, "RdmaTransferServiceWorker", false);
    if (!worker_thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("start worker thread pool failed");
        return false;
    }

    transfer_deadline_loop_thread_ =
        autil::LoopThread::createLoopThread(std::bind(&RdmaTransferService::transferDeadlineCheckProc, this),
                                            10 * 1000,  // 10ms
                                            "RdmaTransferDeadlineCheck");
    if (!transfer_deadline_loop_thread_) {
        RTP_LLM_LOG_ERROR("create transfer deadline check loop thread failed");
        return false;
    }

    return true;
}

void RdmaTransferService::transfer(::google::protobuf::RpcController*                    controller,
                                    const ::rdma_transfer::RdmaLayerBlockTransferRequest* request,
                                    ::rdma_transfer::RdmaLayerBlockTransferResponse*      response,
                                    ::google::protobuf::Closure*                          done) {
    if (!wait_check_loop_thread_ || !worker_thread_pool_) {
        response->set_error_code(::rdma_transfer::RDMA_TRANSFER_NOT_INITIALIZED);
        response->set_error_message("RdmaTransferService not initialized");
        done->Run();
        return;
    }

    auto ctx = std::make_shared<RdmaTaskContext>(controller, request, response, done, metrics_reporter_);
    std::lock_guard<std::mutex> lock(wait_tasks_mutex_);
    wait_tasks_.push_back(ctx);
}

void RdmaTransferService::waitCheckProc() {
    using DoneEntry = std::pair<std::shared_ptr<RdmaTaskContext>, std::shared_ptr<TransferTask>>;
    std::vector<std::shared_ptr<RdmaTaskContext>> ready_ctxs;
    std::vector<std::shared_ptr<RdmaTaskContext>> timeout_ctxs;
    std::vector<DoneEntry>                        done_ctxs;
    {
        std::lock_guard<std::mutex> lock(wait_tasks_mutex_);
        for (auto iter = wait_tasks_.begin(); iter != wait_tasks_.end();) {
            auto ctx = *iter;
            if (ctx->isTimeout()) {
                timeout_ctxs.push_back(ctx);
                iter = wait_tasks_.erase(iter);
            } else if (auto task = task_store_->getTask(ctx->getUniqueKey())) {
                ctx->setTask(task);
                iter = wait_tasks_.erase(iter);
                if (task->done()) {
                    done_ctxs.push_back({ctx, task});
                } else {
                    ready_ctxs.push_back(ctx);
                }
            } else {
                ++iter;
            }
        }
    }
    for (auto& ctx : timeout_ctxs) {
        ctx->run(false, TransferErrorCode::TIMEOUT,
                 "rdma transfer context timeout: no matching recv task within deadline");
    }
    for (auto& [ctx, task] : done_ctxs) {
        auto ec = task->errorCode();
        if (ec == TransferErrorCode::OK) {
            ctx->run(true);
        } else {
            ctx->run(false, ec, task->errorMessage());
        }
    }
    for (auto& ctx : ready_ctxs) {
        auto ret = worker_thread_pool_->pushTask([this, ctx]() { this->transferViaRdma(ctx); });
        if (ret != autil::ThreadPoolBase::ERROR_NONE) {
            ctx->run(false, TransferErrorCode::QUEUE_REJECTED, "push rdma task to thread pool failed");
        }
    }
}

void RdmaTransferService::transferViaRdma(const std::shared_ptr<RdmaTaskContext>& ctx) {
    if (!rdma_client_) {
        ctx->run(false, TransferErrorCode::NOT_INITIALIZED, "rdma client is null");
        return;
    }

    auto build_result = ctx->buildRdmaBlockPairs(staging_pool_);
    if (!build_result.success) {
        ctx->run(false, build_result.error_code, build_result.error_message);
        return;
    }
    auto block_pairs = std::move(build_result.pairs);
    if (block_pairs.empty()) {
        if (ctx->startTransfer()) {
            ctx->run(true);
        } else {
            ctx->run(false, TransferErrorCode::CANCELLED, "recv task cancelled before transfer started");
        }
        return;
    }
    auto staging_transfer = std::move(build_result.staging_transfer);

    auto conn = rdma_client_->getConnection(ctx->getRdmaServerIp(), ctx->getRdmaServerPort());
    if (!conn) {
        ctx->run(false, TransferErrorCode::CONNECTION_FAILED, "rdma get connection failed");
        return;
    }

    if (!ctx->startTransfer()) {
        ctx->run(false, TransferErrorCode::CANCELLED, "recv task cancelled before transfer started");
        return;
    }

    auto already_run = std::make_shared<std::atomic<bool>>(false);
    auto physical_done = std::make_shared<std::atomic<bool>>(false);
    auto disconnect_triggered = std::make_shared<std::atomic<bool>>(false);
    auto completed_count = std::make_shared<std::atomic<size_t>>(0);
    std::string unique_key = ctx->getUniqueKey();
    int64_t posted_at_us = currentTimeUs();

    auto cuda_copy_util = cuda_copy_util_;
    auto guardedRun = [ctx, already_run, physical_done, unique_key, posted_at_us, staging_transfer, cuda_copy_util](
                          bool success, TransferErrorCode ec = TransferErrorCode::OK, const std::string& msg = "") {
        std::string final_message = msg;
        if (success && staging_transfer
            && (!cuda_copy_util || !cuda_copy_util->batchCopyToDevice(staging_transfer->h2d_copy_tasks))) {
            success = false;
            ec = TransferErrorCode::COPY_FAILED;
            final_message = "rdma staging h2d copy failed";
        }
        bool expected = false;
        const bool owns_rpc_completion = already_run->compare_exchange_strong(expected, true);
        // This callback is the physical RDMA completion boundary. Always mark
        // the recv task complete here, even if the RPC watchdog replied earlier.
        ctx->notifyTransferDone(success, ec, final_message);
        physical_done->store(true, std::memory_order_release);
        if (owns_rpc_completion) {
            int64_t total_us = currentTimeUs() - posted_at_us;
            if (success) {
                RTP_LLM_LOG_DEBUG(
                    "transferViaRdma done, unique_key=%s, success=true, total_us=%ld", unique_key.c_str(), total_us);
                ctx->finishRpc(true, TransferErrorCode::OK, {});
            } else {
                RTP_LLM_LOG_DEBUG(
                    "transferViaRdma physically complete, unique_key=%s, total_us=%ld", unique_key.c_str(), total_us);
                ctx->finishRpc(false, ec, final_message);
            }
        }
    };

    size_t total_blocks = block_pairs.size();
    size_t batch_count = (total_blocks <= per_connection_block_count_) ?
                             1 :
                             (total_blocks + per_connection_block_count_ - 1) / per_connection_block_count_;

    RTP_LLM_LOG_DEBUG("transferViaRdma start, unique_key=%s, total_blocks=%zu, batch_count=%zu, "
                      "deadline_ms=%lu, posted_at_us=%ld",
                      unique_key.c_str(),
                      total_blocks,
                      batch_count,
                      ctx->getDeadlineMs(),
                      posted_at_us);

    {
        std::lock_guard<std::mutex> lock(active_transfers_mutex_);
        active_transfers_.push_back({ctx,
                                     conn,
                                     ctx->getDeadlineMs(),
                                     already_run,
                                     physical_done,
                                     disconnect_triggered,
                                     completed_count,
                                     batch_count,
                                     posted_at_us});
    }

    if (block_pairs.size() <= per_connection_block_count_) {
        conn->read(
            block_pairs,
            [ctx, guardedRun, completed_count, unique_key, posted_at_us](TransferErrorCode ec,
                                                                         const std::string& message) {
                const bool success = ec == TransferErrorCode::OK;
                const auto detail = "RDMA key=" + unique_key + " batch=0/1: " + message;
                ctx->recordTransferError(ec, detail);
                completed_count->store(1);
                int64_t cb_latency_us = currentTimeUs() - posted_at_us;
                RTP_LLM_LOG_DEBUG("transferViaRdma callback, unique_key=%s, batch_idx=0/1, "
                                  "success=%d, cb_latency_us=%ld",
                                  unique_key.c_str(),
                                  success,
                                  cb_latency_us);
                if (success) {
                    guardedRun(true);
                } else {
                    guardedRun(false, ec, detail);
                }
            },
            ctx->getDeadlineMs());
        return;
    }

    struct BatchTransferState {
        std::atomic<size_t> completed_count{0};
        std::atomic<bool> any_failed{false};
        size_t total_batches{0};
    };
    auto shared_state = std::make_shared<BatchTransferState>();
    shared_state->total_batches = batch_count;

    for (size_t i = 0; i < batch_count; ++i) {
        size_t start_idx = i * per_connection_block_count_;
        size_t end_idx = std::min(start_idx + per_connection_block_count_, total_blocks);
        std::vector<std::pair<torch::Tensor, std::shared_ptr<RemoteBuffer>>> batch(
            block_pairs.begin() + start_idx,
            block_pairs.begin() + end_idx);

        int64_t batch_posted_us = currentTimeUs();

        conn->read(
            batch,
            [ctx, guardedRun, shared_state, completed_count, unique_key, i, batch_count, batch_posted_us](
                TransferErrorCode ec, const std::string& message) {
                const bool success = ec == TransferErrorCode::OK;
                ctx->recordTransferError(ec,
                                         "RDMA key=" + unique_key + " batch=" + std::to_string(i) + "/"
                                             + std::to_string(batch_count) + ": " + message);
                int64_t cb_latency_us = currentTimeUs() - batch_posted_us;
                if (!success) {
                    shared_state->any_failed.store(true);
                }
                size_t prev_completed = shared_state->completed_count.fetch_add(1);
                size_t done_now = prev_completed + 1;
                completed_count->store(done_now);

                RTP_LLM_LOG_DEBUG("transferViaRdma callback, unique_key=%s, batch_idx=%zu/%zu, "
                                  "success=%d, cb_latency_us=%ld, completed=%zu/%zu",
                                  unique_key.c_str(),
                                  i,
                                  batch_count,
                                  success,
                                  cb_latency_us,
                                  done_now,
                                  batch_count);

                if (done_now == shared_state->total_batches) {
                    if (shared_state->any_failed.load()) {
                        guardedRun(false, TransferErrorCode::RDMA_FAILED, "");
                    } else {
                        guardedRun(true);
                    }
                }
            },
            ctx->getDeadlineMs());
    }
}

void RdmaTransferService::transferDeadlineCheckProc() {
    int64_t now_ms = currentTimeMs();
    std::vector<ActiveTransfer> rpc_expired;
    std::vector<std::shared_ptr<IRdmaConnection>> connections_to_close;
    {
        std::lock_guard<std::mutex> lock(active_transfers_mutex_);
        for (auto it = active_transfers_.begin(); it != active_transfers_.end();) {
            if (it->physical_done->load(std::memory_order_acquire)) {
                it = active_transfers_.erase(it);
            } else {
                if (static_cast<int64_t>(it->deadline_ms) <= now_ms) {
                    bool expected = false;
                    if (it->already_run->compare_exchange_strong(expected, true)) {
                        rpc_expired.push_back(*it);
                    }
                }
                const int64_t deadline_ms = static_cast<int64_t>(it->deadline_ms);
                const int64_t disconnect_at_ms =
                    rdma_disconnect_after_deadline_ms_ > std::numeric_limits<int64_t>::max() - deadline_ms ?
                        std::numeric_limits<int64_t>::max() :
                        deadline_ms + rdma_disconnect_after_deadline_ms_;
                if (now_ms >= disconnect_at_ms) {
                    bool expected = false;
                    if (it->disconnect_triggered->compare_exchange_strong(expected, true)) {
                        connections_to_close.push_back(it->connection);
                    }
                }
                ++it;
            }
        }
    }
    for (auto& entry : rpc_expired) {
        size_t completed = entry.completed_count ? entry.completed_count->load() : 0;
        int64_t inflight_us = currentTimeUs() - entry.posted_at_us;
        if (auto ctx = entry.ctx.lock()) {
            RTP_LLM_LOG_DEBUG("rdma transfer deadline watchdog: callbacks still pending, "
                              "unique_key=%s, completed_batches=%zu/%zu, "
                              "inflight_us=%ld, deadline_ms=%lu",
                              ctx->getUniqueKey().c_str(),
                              completed,
                              entry.total_batches,
                              inflight_us,
                              entry.deadline_ms);
            ctx->finishRpc(false,
                           TransferErrorCode::TIMEOUT,
                           "rdma transfer deadline watchdog: RDMA read callback not received within deadline");
        }
    }
    for (const auto& connection : connections_to_close) {
        if (connection) {
            connection->close();
        }
    }
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
