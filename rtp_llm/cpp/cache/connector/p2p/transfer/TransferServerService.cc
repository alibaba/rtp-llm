#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferServerService.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

TransferServerService::TransferServerService(
    const std::shared_ptr<LayerCacheBufferTaskStore>& layer_cache_buffer_task_store,
    const std::shared_ptr<LayerBlockConvertor>&       layer_block_convector,
    const std::shared_ptr<IRdmaClient>&               rdma_client,
    const kmonitor::MetricsReporterPtr&               metrics_reporter):
    layer_cache_buffer_task_store_(layer_cache_buffer_task_store),
    layer_block_convector_(layer_block_convector),
    rdma_client_(rdma_client),
    metrics_reporter_(metrics_reporter),
    cuda_copy_util_(std::make_unique<CudaCopyUtil>()) {}

TransferServerService::~TransferServerService() {
    // stop wait check loop thread
    if (wait_check_loop_thread_) {
        wait_check_loop_thread_->stop();
    }

    // clear wait tasks
    {
        std::lock_guard<std::mutex> lock(wait_tasks_mutex_);
        for (const auto& wait_task : wait_tasks_) {
            wait_task->run(false, "transfer server service stopped, wait task cancelled");
        }
        wait_tasks_.clear();
    }

    // stop worker thread pool
    if (worker_thread_pool_) {
        worker_thread_pool_->stop();
    }
}

bool TransferServerService::init(int64_t wait_check_interval_us, int worker_thread_count) {
    wait_check_loop_thread_ =
        autil::LoopThread::createLoopThread(std::bind(&TransferServerService::waitCheckProc, this),
                                            wait_check_interval_us,
                                            "TransferServerServiceWaitCheckLoopThread");
    if (!wait_check_loop_thread_) {
        RTP_LLM_LOG_ERROR("create wait check loop thread failed");
        return false;
    }

    worker_thread_pool_ = std::make_shared<autil::LockFreeThreadPool>(
        worker_thread_count, 20, nullptr, "TransferServerServiceWorkerThreadPool", false);
    if (!worker_thread_pool_->start()) {
        RTP_LLM_LOG_ERROR("start worker thread pool failed");
        return false;
    }

    RTP_LLM_LOG_INFO("TransferServerService init success, wait check interval us: %ld, worker thread count: %d",
                     wait_check_interval_us,
                     worker_thread_count);
    return true;
}

void TransferServerService::transfer(::google::protobuf::RpcController*           controller,
                                     const ::transfer::LayerBlockTransferRequest* request,
                                     ::transfer::LayerBlockTransferResponse*      response,
                                     ::google::protobuf::Closure*                 done) {
    RTP_LLM_LOG_DEBUG("TransferServerService transfer start, unique_key: %s", request->unique_key().c_str());
    if (!wait_check_loop_thread_ || !worker_thread_pool_) {
        response->set_success(false);
        response->set_info("TransferServerService not initialized");
        done->Run();
        RTP_LLM_LOG_ERROR("TransferServerService not initialized, unique_key: %s", request->unique_key().c_str());
        return;
    }

    auto transfer_task_context = std::make_shared<TransferTaskContext>(
        controller, request, response, done, layer_block_convector_, metrics_reporter_);
    {
        std::lock_guard<std::mutex> lock(wait_tasks_mutex_);
        wait_tasks_.push_back(transfer_task_context);
    }
}

void TransferServerService::waitCheckProc() {
    std::lock_guard<std::mutex> lock(wait_tasks_mutex_);
    auto                        iter = wait_tasks_.begin();
    while (iter != wait_tasks_.end()) {
        auto transfer_task_context = *iter;
        if (transfer_task_context->isTimeout()) {
            RTP_LLM_LOG_WARNING("transfer task context timeout, unique_key: %s",
                                transfer_task_context->getUniqueKey().c_str());
            transfer_task_context->run(false, "transfer task context timeout");
            iter = wait_tasks_.erase(iter);
            continue;
        }

        auto task = layer_cache_buffer_task_store_->getTask(transfer_task_context->getUniqueKey());
        if (!task) {
            iter++;
            continue;
        }
        iter = wait_tasks_.erase(iter);
        transfer_task_context->addTask(task);

        auto ret = worker_thread_pool_->pushTask([this, transfer_task_context]() {
            if (this->rdma_client_) {
                this->transferViaRdma(transfer_task_context);
            } else {
                this->transferViaTcp(transfer_task_context);
            }
        });
        if (ret != autil::ThreadPoolBase::ERROR_NONE) {
            RTP_LLM_LOG_ERROR("push transfer task to worker thread pool failed, unique_key: %s",
                              transfer_task_context->getUniqueKey().c_str());
            transfer_task_context->run(false, "push transfer task to worker thread pool failed");
        }
        RTP_LLM_LOG_DEBUG("TransferServerService waitCheckProc transfer task context end, unique_key: %s",
                          transfer_task_context->getUniqueKey().c_str());
    }
}

void TransferServerService::transferViaTcp(const std::shared_ptr<TransferTaskContext>& transfer_task_context) {
    RTP_LLM_LOG_DEBUG("TransferServerService transferViaTcp start, unique_key: %s",
                      transfer_task_context->getUniqueKey().c_str());
    auto block_pairs = transfer_task_context->getTcpBlockPair();
    if (block_pairs.empty()) {
        RTP_LLM_LOG_WARNING("no block pair to transfer, unique_key: %s", transfer_task_context->getUniqueKey().c_str());
        transfer_task_context->run(false, "no block pair to transfer");
        return;
    }

    // 收集所有待拷贝任务
    std::vector<CopyTask> copy_tasks;
    copy_tasks.reserve(block_pairs.size());

    for (const auto& block_pair : block_pairs) {
        auto buffer            = block_pair.first;
        auto block_buffer_info = block_pair.second;

        CopyTask task;
        task.src_ptr = const_cast<char*>(block_buffer_info->content().data());
        task.size    = block_buffer_info->len();
        task.dst_ptr = static_cast<char*>(buffer->data());
        copy_tasks.push_back(task);
    }

    // 批量执行 CPU -> GPU 拷贝
    if (!cuda_copy_util_->batchCopyToDevice(copy_tasks)) {
        RTP_LLM_LOG_WARNING("batch copy to device failed, unique_key: %s",
                            transfer_task_context->getUniqueKey().c_str());
        transfer_task_context->run(false, "batch copy to device failed");
        return;
    }

    transfer_task_context->run(true, "transfer via tcp success");
    RTP_LLM_LOG_DEBUG("TransferServerService transferViaTcp end, unique_key: %s",
                      transfer_task_context->getUniqueKey().c_str());
}

void TransferServerService::transferViaRdma(const std::shared_ptr<TransferTaskContext>& transfer_task_context) {
    RTP_LLM_LOG_DEBUG("TransferServerService transferViaRdma start, unique_key: %s",
                      transfer_task_context->getUniqueKey().c_str());
    auto block_pair = transfer_task_context->getRdmaBlockPair();
    if (block_pair.empty()) {
        RTP_LLM_LOG_WARNING("no block pair to transfer, unique_key: %s", transfer_task_context->getUniqueKey().c_str());
        transfer_task_context->run(false, "no block pair to transfer");
        return;
    }

    auto [server_ip, server_port] = transfer_task_context->getServerRdmaInfo();
    if (server_ip.empty() || server_port == 0) {
        RTP_LLM_LOG_WARNING("server rdma info is empty, unique_key: %s", transfer_task_context->getUniqueKey().c_str());
        transfer_task_context->run(false, "server rdma info is empty");
        return;
    }

    auto connection = rdma_client_->getConnection(server_ip, server_port);
    if (!connection) {
        RTP_LLM_LOG_WARNING("get rdma connection failed, ip: %s, port: %d", server_ip.c_str(), server_port);
        transfer_task_context->run(false, "get rdma connection failed");
        return;
    }

    // 捕获 connection 和 rdma_client_ 以便在回调中归还连接
    auto rdma_client = rdma_client_;
    connection->read(
        block_pair,
        [transfer_task_context, connection, rdma_client](bool success) {
            RTP_LLM_LOG_DEBUG("TransferServerService transferViaRdma read callback, unique_key: %s, success: %d",
                              transfer_task_context->getUniqueKey().c_str(),
                              success);
            // 归还连接到连接池（如果连接失败，会自动被移除）
            if (rdma_client) {
                rdma_client->recycleConnection(connection);
            }
            if (!success) {
                transfer_task_context->run(false, "rdma read failed");
                return;
            }
            transfer_task_context->run(true, "rdma read success");
        },
        transfer_task_context->getDeadlineMs());
    RTP_LLM_LOG_DEBUG("TransferServerService transferViaRdma end, unique_key: %s",
                      transfer_task_context->getUniqueKey().c_str());
}

}  // namespace rtp_llm
