#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/IKVCacheReceiver.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TransferTask.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/TcpServer.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/tcp/TcpTransferService.h"
#include <memory>

namespace rtp_llm {
namespace transfer {
namespace tcp {

/// @brief Decode-side receiver: listens on a TCP port for incoming block data from TcpKVCacheSender.
class TcpKVCacheReceiver: public transfer::IKVCacheReceiver {
public:
    explicit TcpKVCacheReceiver(const kmonitor::MetricsReporterPtr& metrics_reporter = nullptr);
    ~TcpKVCacheReceiver();

public:
    /// @brief 启动 TCP server 并注册传输服务，开始监听
    bool init(uint32_t listen_port,
              int      io_thread_count,
              int      worker_thread_count,
              uint32_t anet_rpc_thread_num    = 3,
              uint32_t anet_rpc_queue_num     = 100,
              int64_t  wait_check_interval_us = 1000);

    /// @brief No-op for TCP mode: memory registration is not required.
    bool regMem(const BlockInfo& block_info, uint64_t aligned_size = 0) override;

    /// @brief 注册一次接收任务，返回任务句柄
    transfer::IKVCacheRecvTaskPtr recv(const transfer::RecvRequest& request) override;

    void                          stealTask(const std::string& unique_key) override;
    transfer::IKVCacheRecvTaskPtr getTask(const std::string& unique_key) override;

    const std::shared_ptr<TransferTaskStore>& getTransferTaskStore() const {
        return task_store_;
    }

private:
    std::shared_ptr<transfer::TcpServer> tcp_server_;
    std::shared_ptr<TransferTaskStore>   task_store_;
    std::shared_ptr<TcpTransferService>  transfer_service_;
    kmonitor::MetricsReporterPtr         metrics_reporter_;
};

}  // namespace tcp
}  // namespace transfer
}  // namespace rtp_llm
