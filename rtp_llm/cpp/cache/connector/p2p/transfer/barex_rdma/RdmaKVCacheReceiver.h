#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpServer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaTransferService.h"
#include <memory>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

/// @brief Decode-side receiver: receives RDMA rkeys via TCP RPC, then issues RDMA Read from Prefill.
class RdmaKVCacheReceiver : public IKVCacheReceiver {
public:
    RdmaKVCacheReceiver(const std::shared_ptr<IRdmaMemoryManager>& rdma_memory_manager,
                        const kmonitor::MetricsReporterPtr&         metrics_reporter = nullptr);
    ~RdmaKVCacheReceiver();

public:
    /// @brief 启动 TCP server 注册 RDMA 传输服务，并创建 RDMA client 用于主动读取
    bool init(uint32_t listen_port,
              int      tcp_io_thread_count,
              int      tcp_worker_thread_count,
              int      rdma_io_thread_count,
              int      rdma_worker_thread_count,
              uint32_t rdma_connections_per_host,
              int      rdma_max_block_pairs_per_connection = 0,
              int      tcp_anet_rpc_thread_num               = 3,
              int      tcp_anet_rpc_queue_num                = 100,
              int64_t  wait_check_interval_us                = 1000);

    /// @brief 将 block 内存注册为 RDMA MR
    bool                regMem(const BlockInfo& block_info, uint64_t aligned_size = 0) override;
    /// @brief 注册一次接收任务，返回任务句柄
    IKVCacheRecvTaskPtr recv(const RecvRequest& request) override;
    void                          stealTask(const std::string& unique_key) override;
    transfer::IKVCacheRecvTaskPtr getTask(const std::string& unique_key) override;

    const std::shared_ptr<IRdmaMemoryManager>& getRdmaMemoryManager() const {
        return rdma_memory_manager_;
    }

    const std::shared_ptr<TransferTaskStore>& getTransferTaskStore() const {
        return task_store_;
    }

private:
    /// @brief Test-only: bypass createRdmaClient() and inject a pre-built IRdmaClient.
    bool initForTest(uint32_t                            listen_port,
                     int                                 tcp_io_thread_count,
                     int                                 tcp_worker_thread_count,
                     const std::shared_ptr<IRdmaClient>& rdma_client);


    bool startTcpServer(uint32_t listen_port,
                        int      tcp_io_thread_count,
                        int      tcp_worker_thread_count,
                        int      tcp_anet_rpc_thread_num = 3,
                        int      tcp_anet_rpc_queue_num  = 100);

private:
    std::shared_ptr<TcpServer>           tcp_server_;
    std::shared_ptr<IRdmaClient>         rdma_client_;
    std::shared_ptr<IRdmaMemoryManager>  rdma_memory_manager_;
    std::shared_ptr<TransferTaskStore>   task_store_;
    std::shared_ptr<RdmaTransferService> transfer_service_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
