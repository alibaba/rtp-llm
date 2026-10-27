#pragma once

#include "aios/network/accl-barex/include/accl/barex/highlevel/xclient_transport.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientHostConnections.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaTransferConnection.h"

namespace rtp_llm {

class RdmaClient {
public:
    RdmaClient(const std::shared_ptr<MemoryUtil>& memory_util,
               int                                connect_timeout_ms      = 250,
               int                                qp_count_per_connection = 2);

public:
    bool                            init(uint32_t io_thread_count, uint32_t worker_thread_count);
    std::shared_ptr<RdmaConnection> getConnection(const std::string& ip, uint32_t port);
    void                            recycleConnection(const std::shared_ptr<RdmaConnection>& connection);
    void                            destroyConnection(const std::shared_ptr<RdmaConnection>& connection);
    int                             getAvailableConnectionCount(const std::string& ip, uint32_t port);
    bool                            fillConnectionPool(const std::string& ip, uint32_t port, int count);

    std::shared_ptr<RdmaTransferConnection> getTransferConnection(const std::string& ip, uint32_t port);

private:
    std::shared_ptr<RdmaClientHostConnections> getOrInsertConnection(const std::string& ip, uint32_t port);

private:
    std::shared_ptr<MemoryUtil> memory_util_;
    int                         connect_timeout_ms_      = 250;
    int                         qp_count_per_connection_ = 2;

    std::shared_ptr<arpc::RdmaMempool>                                mempool_;
    std::shared_ptr<::accl::barex::XClientTransport>                  client_;
    mutable std::shared_mutex                                         host_connections_mutex_;
    std::map<std::string, std::shared_ptr<RdmaClientHostConnections>> host_connections_;
};

}  // namespace rtp_llm