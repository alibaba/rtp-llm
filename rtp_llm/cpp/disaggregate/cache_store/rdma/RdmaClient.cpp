#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClient.h"

#include "aios/network/rdma/accl/RdmaMempoolFactory.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xtransport_config.h"
#include "aios/network/accl-barex/metric/KMonitorAcclMetricReporter.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RdmaClient::RdmaClient(const std::shared_ptr<MemoryUtil>& memory_util,
                       int                                connect_timeout_ms,
                       int                                qp_count_per_connection):
    memory_util_(memory_util),
    connect_timeout_ms_(connect_timeout_ms),
    qp_count_per_connection_(qp_count_per_connection) {}


bool RdmaClient::init(uint32_t io_thread_count, uint32_t worker_thread_count) {
    mempool_ = arpc::RdmaMempoolFactory::GetMempool();
    if (!mempool_) {
        RTP_LLM_LOG_ERROR("messager server init failed, new rdma mempool singleton failed");
        return false;
    }

    auto xmempool = mempool_->GetXMempool();
    if (!xmempool) {
        RTP_LLM_LOG_ERROR("messager server init failed, get xmempool failed");
        return false;
    }

    auto metrics_reporter = std::make_shared<::accl::barex::KMonitorAcclMetricReporter>(true, kmonitor::FATAL);
    if (!metrics_reporter->init()) {
        RTP_LLM_LOG_ERROR("rdma reporter init failed");
        return false;
    }

    ::accl::barex::XClientTransportConfig config(io_thread_count, worker_thread_count, xmempool, metrics_reporter);
    client_ = ::accl::barex::XClientTransport::NewInstance(config);
    if (!client_) {
        RTP_LLM_LOG_ERROR("messager server init failed, accl client init failed");
        return false;
    }
    RTP_LLM_LOG_INFO(
        "rdma client init success, io thread count %u, worker thread count %u, connect timeout %d ms, qp count per connection %d",
        io_thread_count,
        worker_thread_count,
        connect_timeout_ms_,
        qp_count_per_connection_);
    return true;
}

std::shared_ptr<RdmaConnection> RdmaClient::getConnection(const std::string& ip, uint32_t port) {
    auto host_connections = getOrInsertConnection(ip, port);
    if (host_connections == nullptr) {
        RTP_LLM_LOG_WARNING("rdma client try get connection to %s:%u failed", ip.c_str(), port);
        return nullptr;
    }

    return host_connections->getConnection(client_);
}

void RdmaClient::recycleConnection(const std::shared_ptr<RdmaConnection>& connection) {
    auto host_connections = getOrInsertConnection(connection->getIp(), connection->getPort());
    if (host_connections == nullptr) {
        RTP_LLM_LOG_WARNING(
            "rdma client try get connection to %s:%u failed", connection->getIp().c_str(), connection->getPort());
        return;
    }
    return host_connections->recycleConnection(connection);
}

void RdmaClient::destroyConnection(const std::shared_ptr<RdmaConnection>& connection) {
    auto host_connections = getOrInsertConnection(connection->getIp(), connection->getPort());
    if (host_connections == nullptr) {
        RTP_LLM_LOG_WARNING(
            "rdma client try get connection to %s:%u failed", connection->getIp().c_str(), connection->getPort());
        return;
    }
    return host_connections->destroyConnection(connection);
}

std::shared_ptr<RdmaClientHostConnections> RdmaClient::getOrInsertConnection(const std::string& ip, uint32_t port) {
    std::string spec = ip + ":" + std::to_string(port);

    std::unique_lock<std::shared_mutex> lock(host_connections_mutex_);
    auto                                it = host_connections_.find(spec);
    if (it == host_connections_.end()) {
        auto host_connections = std::make_shared<RdmaClientHostConnections>(
            memory_util_, ip, port, qp_count_per_connection_, connect_timeout_ms_);
        host_connections_.insert(std::make_pair(spec, host_connections));
        return host_connections;
    }
    return it->second;
}

std::shared_ptr<RdmaTransferConnection> RdmaClient::getTransferConnection(const std::string& ip, uint32_t port) {
    auto host_connections = getOrInsertConnection(ip, port);
    if (host_connections == nullptr) {
        RTP_LLM_LOG_WARNING("rdma client try get connection to %s:%u failed", ip.c_str(), port);
        return nullptr;
    }

    auto connection = host_connections->getConnection(client_);
    if (connection == nullptr) {
        RTP_LLM_LOG_WARNING("rdma client try get connection to %s:%u failed", ip.c_str(), port);
        return nullptr;
    }

    std::shared_ptr<RdmaTransferConnection> transfer_connection(
        new RdmaTransferConnection(connection),
        [host_connections, connection](RdmaTransferConnection* transfer_connection) {
            host_connections->recycleConnection(connection);
        });
    return transfer_connection;
}

int RdmaClient::getAvailableConnectionCount(const std::string& ip, uint32_t port) {
    auto host_connections = getOrInsertConnection(ip, port);
    if (host_connections == nullptr) {
        RTP_LLM_LOG_WARNING("rdma client try get connection to %s:%u failed", ip.c_str(), port);
        return 0;
    }
    return host_connections->getAvailableConnectionCount();
}

bool RdmaClient::fillConnectionPool(const std::string& ip, uint32_t port, int count) {
    auto host_connections = getOrInsertConnection(ip, port);
    if (host_connections == nullptr) {
        RTP_LLM_LOG_WARNING("rdma client try get connection to %s:%u failed", ip.c_str(), port);
        return false;
    }
    return host_connections->fillConnectionPool(client_, count);
}

}  // namespace rtp_llm
