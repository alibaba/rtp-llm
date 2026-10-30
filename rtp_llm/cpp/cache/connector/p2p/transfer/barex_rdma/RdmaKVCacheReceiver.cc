#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaKVCacheReceiver.h"

#include "rtp_llm/cpp/utils/Logger.h"

#include <cstdint>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

RdmaKVCacheReceiver::RdmaKVCacheReceiver(const std::shared_ptr<IRdmaMemoryManager>& rdma_memory_manager,
                                         const kmonitor::MetricsReporterPtr&        metrics_reporter):
    rdma_memory_manager_(rdma_memory_manager), metrics_reporter_(metrics_reporter) {
    task_store_ = std::make_shared<TransferTaskStore>();
}

RdmaKVCacheReceiver::~RdmaKVCacheReceiver() {
    tcp_server_.reset();
    transfer_service_.reset();
    rdma_client_.reset();
}

bool RdmaKVCacheReceiver::startTcpServer(uint32_t listen_port,
                                         int      tcp_io_thread_count,
                                         int      tcp_worker_thread_count,
                                         int      tcp_anet_rpc_thread_num,
                                         int      tcp_anet_rpc_queue_num) {
    tcp_server_ = std::make_shared<transfer::TcpServer>();
    if (!tcp_server_->init(tcp_io_thread_count,
                           tcp_worker_thread_count,
                           listen_port,
                           true,
                           static_cast<uint32_t>(tcp_anet_rpc_thread_num),
                           static_cast<uint32_t>(tcp_anet_rpc_queue_num))) {
        RTP_LLM_LOG_WARNING("create tcp server failed");
        return false;
    }
    if (!tcp_server_->registerService(transfer_service_.get())) {
        RTP_LLM_LOG_WARNING("register rdma transfer service failed");
        return false;
    }
    if (!tcp_server_->start()) {
        RTP_LLM_LOG_WARNING("start tcp server failed");
        return false;
    }
    return true;
}

bool RdmaKVCacheReceiver::init(uint32_t listen_port,
                               int      tcp_io_thread_count,
                               int      tcp_worker_thread_count,
                               int      rdma_io_thread_count,
                               int      rdma_worker_thread_count,
                               uint32_t rdma_connections_per_host,
                               int      rdma_max_block_pairs_per_connection,
                               int      tcp_anet_rpc_thread_num,
                               int      tcp_anet_rpc_queue_num,
                               int64_t  wait_check_interval_us) {
    rdma_client_ = createRdmaClient(
        rdma_memory_manager_, rdma_io_thread_count, rdma_worker_thread_count, rdma_connections_per_host);
    if (!rdma_client_) {
        RTP_LLM_LOG_WARNING("create rdma client failed");
        return false;
    }

    transfer_service_ = std::make_shared<RdmaTransferService>(task_store_, rdma_client_, metrics_reporter_);
    const uint64_t per_connection_block_count =
        (rdma_max_block_pairs_per_connection > 0)
            ? static_cast<uint64_t>(rdma_max_block_pairs_per_connection)
            : UINT64_MAX;
    const int64_t interval_us = wait_check_interval_us > 0 ? wait_check_interval_us : int64_t{1000};
    if (!transfer_service_->init(interval_us, /*worker_thread_count=*/4, per_connection_block_count)) {
        RTP_LLM_LOG_WARNING("init rdma transfer service failed");
        return false;
    }

    if (!startTcpServer(listen_port,
                        tcp_io_thread_count,
                        tcp_worker_thread_count,
                        tcp_anet_rpc_thread_num,
                        tcp_anet_rpc_queue_num)) {
        return false;
    }

    RTP_LLM_LOG_INFO("init success, listen_port: %u", listen_port);
    return true;
}

bool RdmaKVCacheReceiver::initForTest(uint32_t                            listen_port,
                                      int                                 tcp_io_thread_count,
                                      int                                 tcp_worker_thread_count,
                                      const std::shared_ptr<IRdmaClient>& rdma_client) {
    rdma_client_ = rdma_client;
    if (!rdma_client_) {
        RTP_LLM_LOG_WARNING("called with null rdma_client");
        return false;
    }

    transfer_service_ = std::make_shared<RdmaTransferService>(task_store_, rdma_client_, metrics_reporter_);
    if (!transfer_service_->init()) {
        RTP_LLM_LOG_WARNING("init rdma transfer service failed");
        return false;
    }

    if (!startTcpServer(listen_port, tcp_io_thread_count, tcp_worker_thread_count)) {
        return false;
    }

    RTP_LLM_LOG_INFO("success, listen_port: %u", listen_port);
    return true;
}

bool RdmaKVCacheReceiver::regMem(const BlockInfo& block_info, uint64_t aligned_size) {
    if (!rdma_memory_manager_) {
        return true;
    }
    return rdma_memory_manager_->regUserMr(block_info, aligned_size);
}

transfer::IKVCacheRecvTaskPtr RdmaKVCacheReceiver::recv(const transfer::RecvRequest& request) {
    return task_store_->addTask(request.unique_key, request.block_info, request.deadline_ms);
}

void RdmaKVCacheReceiver::stealTask(const std::string& unique_key) {
    task_store_->stealTask(unique_key);
}

transfer::IKVCacheRecvTaskPtr RdmaKVCacheReceiver::getTask(const std::string& unique_key) {
    return task_store_->getTask(unique_key);
}

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
