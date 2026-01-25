#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferServerService.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerBlockConvertor.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/LayerCacheBuffer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpClient.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpServer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/RdmaInterface.h"
#include <memory>
#include "rtp_llm/cpp/metrics/RtpLLMMetrics.h"

namespace rtp_llm {

/// @brief TransferServer 包含 TransferServerService、LayerBlockConvertor 和 TransferTaskStore
class TransferServer {
public:
    TransferServer(const std::shared_ptr<LayerBlockConvertor>& layer_block_convector,
                   const std::shared_ptr<IRdmaMemoryManager>&  rdma_memory_manager = nullptr,
                   const kmonitor::MetricsReporterPtr&         metrics_reporter    = nullptr);
    ~TransferServer();

public:
    /// @brief 初始化 TransferServer
    /// @return 是否成功
    bool init(bool     use_rdma,
              uint32_t listen_port,
              int      tcp_io_thread_count,
              int      tcp_worker_thread_count,
              int      rdma_io_thread_count,
              int      rdma_worker_thread_count,
              uint32_t rdma_connections_per_host,
              int      connect_timeout_ms);

    /// @brief 获取 TransferTaskStore
    /// @return TransferTaskStore 指针
    std::shared_ptr<TransferTaskStore> getTransferTaskStore() const {
        return transfer_task_store_;
    }

    bool registerUserMr(const BufferPtr& buffer, uint64_t aligned_size = 0);

    const std::shared_ptr<IRdmaMemoryManager>& getRdmaMemoryManager() const {
        return rdma_memory_manager_;
    }

private:
    std::shared_ptr<transfer::TcpServer>   tcp_server_;
    std::shared_ptr<IRdmaClient>           rdma_client_;
    std::shared_ptr<LayerBlockConvertor>   layer_block_convector_;
    std::shared_ptr<IRdmaMemoryManager>    rdma_memory_manager_;
    std::shared_ptr<TransferTaskStore>     transfer_task_store_;
    std::shared_ptr<TransferServerService> transfer_server_service_;
    kmonitor::MetricsReporterPtr           metrics_reporter_;
};

}  // namespace rtp_llm
