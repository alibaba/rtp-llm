#pragma once

#include <atomic>
#include <memory>
#include <string>
#include <mutex>
#include "rtp_llm/cpp/cache/connector/p2p/transfer/barex_rdma/RdmaConnection.h"

namespace rtp_llm {
namespace transfer {
namespace barex_rdma {

struct ConnectionSlot {
    std::shared_ptr<RdmaConnection> connection;
    size_t                          index;
};

struct ConnectionInstallResult {
    std::shared_ptr<RdmaConnection> connection;
    bool                            installed;
};

/// @brief 管理特定 host:port 的固定大小 RDMA 连接池（Round-Robin 复用，懒创建）
class RdmaHostConnectionPool {
public:
    RdmaHostConnectionPool(const std::string& ip, uint32_t port, uint32_t pool_size);

    /// @brief Round-Robin 获取下一个可用连接；若连接为空或已失败则返回 {nullptr, idx}
    ConnectionSlot getNextConnection();

    /// @brief 仅在 slot 为空或 FAILED 时安装 new_conn；返回实际使用的连接及本次是否安装成功
    ConnectionInstallResult installOrGetExistingAt(size_t idx,
                                                   const std::shared_ptr<RdmaConnection>& new_conn);

    uint32_t getPoolSize() const { return pool_size_; }

    /// @brief Sweep expired pending requests on all connections in this pool.
    void sweepAllConnections();

    const std::string& getIp() const { return ip_; }

    uint32_t getPort() const { return port_; }

private:
    std::string  ip_;
    uint32_t     port_;
    uint32_t     pool_size_;

    std::atomic<size_t>                          round_robin_idx_{0};
    mutable std::mutex                           mutex_;
    std::vector<std::shared_ptr<RdmaConnection>> connections_;
};

}  // namespace barex_rdma
}  // namespace transfer
}  // namespace rtp_llm
