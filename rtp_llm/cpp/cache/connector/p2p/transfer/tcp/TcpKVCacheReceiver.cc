#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpKVCacheReceiver.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {
namespace transfer {
namespace tcp {

TcpKVCacheReceiver::TcpKVCacheReceiver(const kmonitor::MetricsReporterPtr& metrics_reporter):
    metrics_reporter_(metrics_reporter) {
    task_store_ = std::make_shared<TransferTaskStore>();
}

TcpKVCacheReceiver::~TcpKVCacheReceiver() {
    tcp_server_.reset();
    transfer_service_.reset();
}

bool TcpKVCacheReceiver::init(uint32_t listen_port,
                              int      io_thread_count,
                              int      worker_thread_count,
                              uint32_t anet_rpc_thread_num,
                              uint32_t anet_rpc_queue_num,
                              int64_t  wait_check_interval_us) {
    transfer_service_         = std::make_shared<TcpTransferService>(task_store_, metrics_reporter_);
    const int64_t interval_us = wait_check_interval_us > 0 ? wait_check_interval_us : int64_t{1000};
    if (!transfer_service_->init(interval_us, worker_thread_count)) {
        RTP_LLM_LOG_WARNING("init transfer service failed");
        return false;
    }

    tcp_server_ = std::make_shared<transfer::TcpServer>();
    if (!tcp_server_->init(
            io_thread_count, worker_thread_count, listen_port, true, anet_rpc_thread_num, anet_rpc_queue_num)) {
        RTP_LLM_LOG_WARNING("create tcp server failed");
        return false;
    }

    // TODO, use service thread pool
    if (!tcp_server_->registerService(transfer_service_.get())) {
        RTP_LLM_LOG_WARNING("register transfer service failed");
        return false;
    }

    if (!tcp_server_->start()) {
        RTP_LLM_LOG_WARNING("start tcp server failed");
        return false;
    }

    RTP_LLM_LOG_INFO("init success, listen_port: %u", listen_port);
    return true;
}

bool TcpKVCacheReceiver::regMem(const BlockInfo& /*block_info*/, uint64_t /*aligned_size*/) {
    return true;
}

transfer::IKVCacheRecvTaskPtr TcpKVCacheReceiver::recv(const transfer::RecvRequest& request) {
    return task_store_->addTask(request.unique_key, request.block_info, request.deadline_ms);
}

void TcpKVCacheReceiver::stealTask(const std::string& unique_key) {
    task_store_->stealTask(unique_key);
}

transfer::IKVCacheRecvTaskPtr TcpKVCacheReceiver::getTask(const std::string& unique_key) {
    return task_store_->getTask(unique_key);
}

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
