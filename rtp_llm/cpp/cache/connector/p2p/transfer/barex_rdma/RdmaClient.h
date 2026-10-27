#pragma once

#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaInterface.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaHostConnectionPool.h"
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaConnection.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xclient_transport.h"
#include <memory>
#include <string>
#include <shared_mutex>

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

/// @brief RDMA 客户端，管理到多个远端的连接池并提供连接复用
class RdmaClient: public IRdmaClient {
public:
    RdmaClient(const std::shared_ptr<IRdmaMemoryManager>&        memory_manager,
               uint32_t                                          rdma_connections_per_host = 2,
               const std::shared_ptr<kmonitor::MetricsReporter>& metrics_reporter          = nullptr);
    ~RdmaClient();

public:
    /// @brief 初始化 XClientTransport 及底层 RDMA mempool
    bool init(int io_thread_count, int worker_thread_count);

    /// @brief 获取到指定地址的 RDMA 连接（连接复用，轮询机制），失败返回 nullptr
    std::shared_ptr<IRdmaConnection> getConnection(const std::string& ip, uint32_t port) override;

private:
    std::shared_ptr<RdmaHostConnectionPool> getOrCreateConnectionPool(const std::string& ip, uint32_t port);
    std::shared_ptr<RdmaConnection> createNewConnection(const std::string& ip, uint32_t port);

private:
    // 连接管理参数
    uint32_t                                   rdma_connections_per_host_ = 2;
    std::shared_ptr<kmonitor::MetricsReporter> metrics_reporter_;

    // RDMA 基础设施
    std::shared_ptr<arpc::RdmaMempool>               rdma_mempool_;
    std::shared_ptr<::accl::barex::XClientTransport> client_;

    // 连接管理（每个 host:port 对应一个连接池）
    mutable std::shared_mutex                                      host_connections_mutex_;
    std::map<std::string, std::shared_ptr<RdmaHostConnectionPool>> host_connection_pools_;

    std::shared_ptr<IRdmaMemoryManager> memory_manager_;

    // 初始化标志
    bool initialized_ = false;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
