#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTransferService.h"

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace tcp {

TcpTransferService::TcpTransferService(const std::shared_ptr<TransferTaskStore>& task_store,
                                       const kmonitor::MetricsReporterPtr&       metrics_reporter):
    task_store_(task_store), metrics_reporter_(metrics_reporter), cuda_copy_util_(std::make_unique<CudaCopyUtil>()) {}

TcpTransferService::~TcpTransferService() {
    if (wait_check_loop_thread_) {
        wait_check_loop_thread_->stop();
    }
    std::list<std::shared_ptr<TcpTaskContext>> pending;
    {
        std::lock_guard<std::mutex> lock(wait_tasks_mutex_);
        pending.swap(wait_tasks_);
    }
    for (const auto& ctx : pending) {
        ctx->run(false, TransferErrorCode::CANCELLED, "TcpTransferService stopped");
    }
    if (worker_thread_pool_) {
        worker_thread_pool_->stop();
    }
}

bool TcpTransferService::init(int64_t wait_check_interval_us, int worker_thread_count) {
    wait_check_loop_thread_ = autil::LoopThread::createLoopThread(
        std::bind(&TcpTransferService::waitCheckProc, this), wait_check_interval_us, "TcpTransferServiceWaitCheck");
    if (!wait_check_loop_thread_) {
        RTP_LLM_LOG_ERROR("create wait check loop thread failed");
        return false;
    }

    worker_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        worker_thread_count, 20, nullptr, "TcpTransferServiceWorker", false);
    if (!worker_thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("start worker thread pool failed");
        return false;
    }
    return true;
}

void TcpTransferService::transfer(::google::protobuf::RpcController*                  controller,
                                  const ::tcp_transfer::TcpLayerBlockTransferRequest* request,
                                  ::tcp_transfer::TcpLayerBlockTransferResponse*      response,
                                  ::google::protobuf::Closure*                        done) {
    if (!wait_check_loop_thread_ || !worker_thread_pool_) {
        RTP_LLM_LOG_WARNING("TcpTransferService transfer failed: service not initialized");
        response->set_error_code(::tcp_transfer::TCP_TRANSFER_UNKNOWN_ERROR);
        response->set_error_message("TcpTransferService not initialized");
        done->Run();
        return;
    }
    auto ctx = std::make_shared<TcpTaskContext>(controller, request, response, done, metrics_reporter_);
    std::lock_guard<std::mutex> lock(wait_tasks_mutex_);
    wait_tasks_.push_back(ctx);
}

void TcpTransferService::waitCheckProc() {
    using DoneEntry = std::pair<std::shared_ptr<TcpTaskContext>, std::shared_ptr<TransferTask>>;
    std::vector<std::shared_ptr<TcpTaskContext>> ready_ctxs;
    std::vector<std::shared_ptr<TcpTaskContext>> timeout_ctxs;
    std::vector<DoneEntry>                       done_ctxs;
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
        ctx->run(
            false, TransferErrorCode::TIMEOUT, "tcp transfer context timeout: no matching recv task within deadline");
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
        auto ret = worker_thread_pool_->pushTask([this, ctx]() { this->transferViaTcp(ctx); });
        if (ret != autil::ThreadPoolBase::ERROR_NONE) {
            RTP_LLM_LOG_WARNING("TcpTransferService push transfer task to thread pool failed, unique_key: %s",
                                ctx->getUniqueKey().c_str());
            ctx->run(false, TransferErrorCode::UNKNOWN, "push transfer task to thread pool failed");
        }
    }
}

void TcpTransferService::transferViaTcp(const std::shared_ptr<TcpTaskContext>& ctx) {
    if (!ctx->startTransfer()) {
        ctx->run(false, TransferErrorCode::CANCELLED, "recv task cancelled before transfer started");
        return;
    }
    if (!ctx->executeCopy(*cuda_copy_util_)) {
        ctx->run(false, TransferErrorCode::BUFFER_MISMATCH, "copy blocks to device failed");
        return;
    }
    ctx->run(true);
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
