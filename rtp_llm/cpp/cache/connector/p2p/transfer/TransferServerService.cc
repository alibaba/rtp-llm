#include <algorithm>
#include <atomic>

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferServerService.h"

#include "rtp_llm/cpp/utils/ErrorCode.h"
#include "rtp_llm/cpp/utils/Logger.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/proto/service.pb.h"

namespace rtp_llm {

TransferServerService::TransferServerService(const std::shared_ptr<TransferTaskStore>&   transfer_task_store,
                                             const std::shared_ptr<LayerBlockConvertor>& layer_block_convector,
                                             const std::shared_ptr<IRdmaClient>&         rdma_client,
                                             const kmonitor::MetricsReporterPtr&         metrics_reporter,
                                             int max_block_pairs_per_connection):
    transfer_task_store_(transfer_task_store),
    layer_block_convector_(layer_block_convector),
    rdma_client_(rdma_client),
    metrics_reporter_(metrics_reporter),
    cuda_copy_util_(std::make_unique<CudaCopyUtil>()),
    max_block_pairs_per_connection_(max_block_pairs_per_connection) {}

TransferServerService::~TransferServerService() {
    // stop wait check loop thread
    if (wait_check_loop_thread_) {
        wait_check_loop_thread_->stop();
    }

    // clear wait tasks
    {
        std::lock_guard<std::mutex> lock(wait_tasks_mutex_);
        for (const auto& wait_task : wait_tasks_) {
            wait_task->run(
                false, ::transfer::TRANSFER_UNKNOWN_ERROR, "transfer server service stopped, wait task cancelled");
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
        response->set_error_code(::transfer::TRANSFER_UNKNOWN_ERROR);
        response->set_error_message("TransferServerService not initialized");
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
            transfer_task_context->run(false, ::transfer::TRANSFER_UNKNOWN_ERROR, "transfer task context timeout");
            iter = wait_tasks_.erase(iter);
            continue;
        }

        auto task = transfer_task_store_->getTask(transfer_task_context->getUniqueKey());
        if (!task) {
            iter++;
            continue;
        }
        iter = wait_tasks_.erase(iter);
        transfer_task_context->setTask(task);

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
            transfer_task_context->run(
                false, ::transfer::TRANSFER_UNKNOWN_ERROR, "push transfer task to worker thread pool failed");
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
        transfer_task_context->run(false, ::transfer::TRANSFER_BUFFER_MISMATCH, "no block pair to transfer");
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
        transfer_task_context->run(false, ::transfer::TRANSFER_BUFFER_MISMATCH, "batch copy to device failed");
        return;
    }

    transfer_task_context->run(true, ::transfer::TRANSFER_NONE_ERROR, "");
    RTP_LLM_LOG_DEBUG("TransferServerService transferViaTcp end, unique_key: %s",
                      transfer_task_context->getUniqueKey().c_str());
}

void TransferServerService::transferViaRdma(const std::shared_ptr<TransferTaskContext>& transfer_task_context) {
    RTP_LLM_LOG_DEBUG("TransferServerService transferViaRdma start, unique_key: %s",
                      transfer_task_context->getUniqueKey().c_str());
    auto block_pair = transfer_task_context->getRdmaBlockPair();
    if (block_pair.empty()) {
        RTP_LLM_LOG_WARNING("no block pair to transfer, unique_key: %s", transfer_task_context->getUniqueKey().c_str());
        transfer_task_context->run(false, ::transfer::TRANSFER_BUFFER_MISMATCH, "no block pair to transfer");
        return;
    }

    auto [server_ip, server_port] = transfer_task_context->getServerRdmaInfo();
    if (server_ip.empty() || server_port == 0) {
        RTP_LLM_LOG_WARNING("server rdma info is empty, unique_key: %s", transfer_task_context->getUniqueKey().c_str());
        transfer_task_context->run(false, ::transfer::TRANSFER_RDMA_FAILED, "server rdma info is empty");
        return;
    }
    const size_t chunk_limit =
        max_block_pairs_per_connection_ > 0 ? static_cast<size_t>(max_block_pairs_per_connection_) : 0;
    const auto deadline_ms = transfer_task_context->getDeadlineMs();
    if (chunk_limit <= 0 || block_pair.size() <= chunk_limit) {
        auto callback = [transfer_task_context](bool success) {
            if (!success) {
                transfer_task_context->run(false, ::transfer::TRANSFER_RDMA_FAILED, "rdma read failed");
            }
            transfer_task_context->run(true, ::transfer::TRANSFER_NONE_ERROR, "");
        };
        sendBlockPair(server_ip, server_port, transfer_task_context, block_pair, deadline_ms, callback);
        return;
    }

    auto                        chunk_count = (block_pair.size() + chunk_limit - 1) / chunk_limit;
    std::shared_ptr<int>        done_count  = std::make_shared<int>(chunk_count);
    std::shared_ptr<bool>       success     = std::make_shared<bool>(true);
    std::shared_ptr<std::mutex> mutex       = std::make_shared<std::mutex>();
    auto                        callback    = [transfer_task_context, done_count, success, mutex](bool read_success) {
        std::lock_guard<std::mutex> lock(*mutex);
        auto                        remaining = --(*done_count);
        if (!read_success) {
            *success = false;
        }
        if (remaining > 0) {
            return;
        }
        if (*success) {
            RTP_LLM_LOG_DEBUG("TransferServerService transferViaRdma read success, unique_key: %s",
                              transfer_task_context->getUniqueKey().c_str());
            transfer_task_context->run(true, ::transfer::TRANSFER_NONE_ERROR, "");
        } else {
            RTP_LLM_LOG_WARNING("TransferServerService transferViaRdma read failed, unique_key: %s",
                                transfer_task_context->getUniqueKey().c_str());
            transfer_task_context->run(false, ::transfer::TRANSFER_RDMA_FAILED, "rdma read failed");
        }
    };

    for (size_t start = 0; start < block_pair.size(); start += chunk_limit) {
        size_t end = std::min(start + chunk_limit, block_pair.size());
        std::vector<std::pair<BufferPtr, std::shared_ptr<RemoteBuffer>>> chunk_block_pair(block_pair.begin() + start,
                                                                                          block_pair.begin() + end);
        sendBlockPair(server_ip, server_port, transfer_task_context, chunk_block_pair, deadline_ms, callback);
    }
}

void TransferServerService::sendBlockPair(
    const std::string&                                                      server_ip,
    const uint32_t                                                          server_port,
    const std::shared_ptr<TransferTaskContext>&                             transfer_task_context,
    const std::vector<std::pair<BufferPtr, std::shared_ptr<RemoteBuffer>>>& block_pair,
    int64_t                                                                 deadline_ms,
    std::function<void(bool success)>                                       callback) {
    auto connection = rdma_client_->getConnection(server_ip, server_port);
    if (!connection) {
        RTP_LLM_LOG_WARNING("get rdma connection failed, ip: %s, port: %d", server_ip.c_str(), server_port);
        callback(false);
        return;
    }
    RTP_LLM_LOG_DEBUG("TransferServerService transferViaRdma read start, unique_key: %s, connection: %p",
                      transfer_task_context->getUniqueKey().c_str(),
                      connection.get());
    connection->read(
        block_pair,
        [transfer_task_context, callback](bool success) {
            RTP_LLM_LOG_DEBUG("TransferServerService transferViaRdma read callback, unique_key: %s, success: %d",
                              transfer_task_context->getUniqueKey().c_str(),
                              success);
            callback(success);
        },
        deadline_ms);
}

}  // namespace rtp_llm
