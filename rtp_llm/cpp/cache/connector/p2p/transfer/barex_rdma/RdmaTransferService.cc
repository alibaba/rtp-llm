#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaTransferService.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

RdmaTransferService::RdmaTransferService(const std::shared_ptr<TransferTaskStore>& task_store,
                                         const std::shared_ptr<IRdmaClient>&       rdma_client,
                                         const kmonitor::MetricsReporterPtr&       metrics_reporter):
    task_store_(task_store),
    rdma_client_(rdma_client),
    metrics_reporter_(metrics_reporter) {}

RdmaTransferService::~RdmaTransferService() {
    if (wait_check_loop_thread_) {
        wait_check_loop_thread_->stop();
    }
    std::list<std::shared_ptr<RdmaTaskContext>> pending;
    {
        std::lock_guard<std::mutex> lock(wait_tasks_mutex_);
        pending.swap(wait_tasks_);
    }
    for (const auto& ctx : pending) {
        ctx->run(false, TransferErrorCode::CANCELLED, "RdmaTransferService stopped");
    }
    if (worker_thread_pool_) {
        worker_thread_pool_->stop();
    }
}

bool RdmaTransferService::init(int64_t wait_check_interval_us,
                                int worker_thread_count,
                                uint64_t per_connection_block_count) {
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
        worker_thread_count, 20, nullptr, "RdmaTransferServiceWorker", false);
    if (!worker_thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("start worker thread pool failed");
        return false;
    }
    return true;
}

void RdmaTransferService::transfer(::google::protobuf::RpcController*                    controller,
                                    const ::rdma_transfer::RdmaLayerBlockTransferRequest* request,
                                    ::rdma_transfer::RdmaLayerBlockTransferResponse*      response,
                                    ::google::protobuf::Closure*                          done) {
    if (!wait_check_loop_thread_ || !worker_thread_pool_) {
        response->set_error_code(::rdma_transfer::RDMA_TRANSFER_UNKNOWN_ERROR);
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
            ctx->run(false, TransferErrorCode::UNKNOWN, "push rdma task to thread pool failed");
        }
    }
}

void RdmaTransferService::transferViaRdma(const std::shared_ptr<RdmaTaskContext>& ctx) {
    if (!rdma_client_) {
        ctx->run(false, TransferErrorCode::RDMA_FAILED, "rdma client is null");
        return;
    }

    auto block_pairs = ctx->getRdmaBlockPair();
    if (block_pairs.empty()) {
        ctx->run(false, TransferErrorCode::BUFFER_MISMATCH, "no rdma block pair to transfer");
        return;
    }

    auto conn = rdma_client_->getConnection(ctx->getRdmaServerIp(), ctx->getRdmaServerPort());
    if (!conn) {
        ctx->run(false, TransferErrorCode::RDMA_FAILED, "rdma get connection failed");
        return;
    }

    if (!ctx->startTransfer()) {
        ctx->run(false, TransferErrorCode::CANCELLED, "recv task cancelled before transfer started");
        return;
    }

    // 如果 block 数量不超过限制，走原有逻辑
    if (block_pairs.size() <= per_connection_block_count_) {
        conn->read(block_pairs,
                   [ctx](bool success) {
                       if (success) {
                           ctx->run(true);
                       } else {
                           ctx->run(false, TransferErrorCode::RDMA_FAILED, "rdma read failed");
                       }
                   },
                   ctx->getDeadlineMs());
        return;
    }

    // 分批传输逻辑：等待所有批次完成后再调用 ctx->run
    size_t total_blocks = block_pairs.size();
    size_t batch_count = (total_blocks + per_connection_block_count_ - 1) / per_connection_block_count_;

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

        conn->read(batch,
                   [ctx, shared_state](bool success) {
                       if (!success) {
                           shared_state->any_failed.store(true);
                       }

                       // 等所有批次完成后再调用 ctx->run
                       size_t prev_completed = shared_state->completed_count.fetch_add(1);
                       if (prev_completed + 1 == shared_state->total_batches) {
                           // 最后一个完成的批次负责调用 ctx->run
                           if (shared_state->any_failed.load()) {
                               ctx->run(false, TransferErrorCode::RDMA_FAILED, "rdma read failed in batch transfer");
                           } else {
                               ctx->run(true);
                           }
                       }
                   },
                   ctx->getDeadlineMs());
    }
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
