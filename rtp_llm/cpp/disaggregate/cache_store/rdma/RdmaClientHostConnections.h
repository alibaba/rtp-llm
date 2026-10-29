#pragma once

#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xclient_transport.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"

namespace rtp_llm {

class RdmaClientHostConnections: public std::enable_shared_from_this<RdmaClientHostConnections> {
public:
    RdmaClientHostConnections(const std::shared_ptr<MemoryUtil>& memory_util_,
                              const std::string&                 host,
                              uint32_t                           port,
                              uint32_t                           qp_count_per_rdma_connection = 2,
                              uint32_t                           connect_timeout_ms           = 250);
    ~RdmaClientHostConnections();

public:
    std::shared_ptr<RdmaConnection> getConnection(const std::shared_ptr<::accl::barex::XClientTransport>& client);
    void                            recycleConnection(const std::shared_ptr<RdmaConnection>& connection);
    void                            destroyConnection(const std::shared_ptr<RdmaConnection>& connection);
    bool                            fillConnectionPool(const std::shared_ptr<::accl::barex::XClientTransport>& client, int count);

    void connectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection);
    void connectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection);

    int getAvailableConnectionCount();

private:
    std::shared_ptr<RdmaConnection>
    getRdmaConnectionFromConnectionPool(std::set<std::shared_ptr<::accl::barex::XConnection>>& connection_pool,
                                        bool                                                   all_connected);
    std::shared_ptr<RdmaConnection> newRdmaConnection(const std::shared_ptr<::accl::barex::XClientTransport>& client);
    void doRecycleConnections(const std::vector<std::shared_ptr<::accl::barex::XConnection>>& connections);

private:
    std::shared_ptr<MemoryUtil> memory_util_;
    std::string                 host_;
    uint32_t                    port_;
    uint32_t                    qp_count_per_rdma_connection_;
    uint32_t                    connect_timeout_ms_;

    std::mutex                                            mutex_;
    std::set<std::shared_ptr<::accl::barex::XConnection>> newly_connections_;      // 新建连接已被预订
    std::set<std::shared_ptr<::accl::barex::XConnection>> available_connections_;  // 可复用
    std::set<std::shared_ptr<::accl::barex::XConnection>> busy_connections_;       // 正在用

};

}  // namespace rtp_llm