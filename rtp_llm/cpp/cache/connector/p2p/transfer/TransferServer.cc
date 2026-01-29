#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferServer.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

TransferServer::TransferServer(const std::shared_ptr<LayerBlockConvertor>& layer_block_convector,
                               const std::shared_ptr<IRdmaMemoryManager>&  rdma_memory_manager,
                               const kmonitor::MetricsReporterPtr&         metrics_reporter):
    layer_block_convector_(layer_block_convector),
    rdma_memory_manager_(rdma_memory_manager),
    metrics_reporter_(metrics_reporter) {
    transfer_task_store_ = std::make_shared<TransferTaskStore>();
}

TransferServer::~TransferServer() {
    if (tcp_server_) {
        tcp_server_.reset();
    }
    if (transfer_server_service_) {
        transfer_server_service_.reset();
    }
    if (rdma_client_) {
        rdma_client_.reset();
    }
    if (rdma_memory_manager_) {
        rdma_memory_manager_.reset();
    }
}

bool TransferServer::init(bool     use_rdma,
                          uint32_t listen_port,
                          int      tcp_io_thread_count,
                          int      tcp_worker_thread_count,
                          int      rdma_io_thread_count,
                          int      rdma_worker_thread_count,
                          uint32_t rdma_connections_per_host,
                          int      connect_timeout_ms,
                          int      rdma_max_block_pairs_per_connection) {
    tcp_server_ = std::make_shared<transfer::TcpServer>();
    if (!tcp_server_->init(tcp_io_thread_count, tcp_worker_thread_count, listen_port, true)) {
        RTP_LLM_LOG_WARNING("create tcp server failed");
        return false;
    }

    if (use_rdma) {
        if (!rdma_memory_manager_) {
            rdma_memory_manager_ = createRdmaMemoryManager();
            if (rdma_memory_manager_ == nullptr) {
                RTP_LLM_LOG_WARNING("create rdma memory manager failed");
                return false;
            }
        }
        rdma_client_ = createRdmaClient(rdma_memory_manager_,
                                        rdma_io_thread_count,
                                        rdma_worker_thread_count,
                                        rdma_connections_per_host,
                                        connect_timeout_ms);
        if (rdma_client_ == nullptr) {
            RTP_LLM_LOG_WARNING("create rdma client failed");
            return false;
        }
        RTP_LLM_LOG_INFO("create rdma client success");
    }

    transfer_server_service_ = std::make_shared<TransferServerService>(transfer_task_store_,
                                                                       layer_block_convector_,
                                                                       rdma_client_,
                                                                       metrics_reporter_,
                                                                       rdma_max_block_pairs_per_connection);
    if (!transfer_server_service_->init()) {
        RTP_LLM_LOG_WARNING("init transfer server service failed");
        return false;
    }

    if (!tcp_server_->registerService(transfer_server_service_.get())) {
        RTP_LLM_LOG_WARNING("register transfer server service failed");
        return false;
    }

    if (!tcp_server_->start()) {
        RTP_LLM_LOG_WARNING("start tcp server failed");
        return false;
    }
    RTP_LLM_LOG_INFO("transfer server init success, listen port: %u", listen_port);
    return true;
}

bool TransferServer::registerUserMr(const BufferPtr& buffer, uint64_t aligned_size) {
    if (rdma_memory_manager_ == nullptr) {
        return true;
    }
    return rdma_memory_manager_->regUserMr(buffer, aligned_size);
}

}  // namespace rtp_llm
