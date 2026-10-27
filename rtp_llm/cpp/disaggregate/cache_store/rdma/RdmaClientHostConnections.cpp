#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientHostConnections.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientConnectionCallback.h"

#include "autil/EnvUtil.h"

#include <unistd.h>

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RdmaClientHostConnections::RdmaClientHostConnections(const std::shared_ptr<MemoryUtil>& memory_util,
                                                     const std::string&                 host,
                                                     uint32_t                           port,
                                                     uint32_t                           qp_count_per_rdma_connection,
                                                     uint32_t                           connect_timeout_ms):
    memory_util_(memory_util),
    host_(host),
    port_(port),
    qp_count_per_rdma_connection_(qp_count_per_rdma_connection),
    connect_timeout_ms_(connect_timeout_ms) {}

RdmaClientHostConnections::~RdmaClientHostConnections() {
    for (auto& connection : newly_connections_) {
        connection->Close();
    }
    newly_connections_.clear();

    for (auto& connection : available_connections_) {
        connection->Close();
    }
    available_connections_.clear();

    if (busy_connections_.size() > 0) {
        RTP_LLM_LOG_WARNING("still has inprocessing rdma connection while rdma client dtor");
    }
    for (auto& connection : busy_connections_) {
        connection->Close();
    }
    busy_connections_.clear();
}

std::shared_ptr<RdmaConnection>
RdmaClientHostConnections::getConnection(const std::shared_ptr<::accl::barex::XClientTransport>& client) {
    auto rdma_connection = getRdmaConnectionFromConnectionPool(available_connections_, true);
    if (rdma_connection != nullptr) {
        return rdma_connection;
    }

    // A connection callback may arrive after the request that initiated it has
    // timed out. Such connections stay in newly_connections_. Reuse them
    // before issuing more Connect calls. The timeout path in newRdmaConnection
    // already permits a partial QP group, so keep the same behavior here.
    rdma_connection = getRdmaConnectionFromConnectionPool(newly_connections_, false);
    if (rdma_connection != nullptr) {
        return rdma_connection;
    }

    rdma_connection = newRdmaConnection(client);
    if (rdma_connection == nullptr) {
        RTP_LLM_LOG_WARNING("new available connections to %s:%u failed", host_.c_str(), port_);
    }
    return rdma_connection;
}

std::shared_ptr<RdmaConnection> RdmaClientHostConnections::getRdmaConnectionFromConnectionPool(
    std::set<std::shared_ptr<::accl::barex::XConnection>>& connection_pool, bool all_connected) {
    std::vector<std::shared_ptr<::accl::barex::XConnection>> connections;

    std::lock_guard<std::mutex> lock(mutex_);
    if (connection_pool.empty()) {
        return nullptr;
    }

    if (all_connected && connection_pool.size() < qp_count_per_rdma_connection_) {
        return nullptr;
    }

    for (int i = 0; i < qp_count_per_rdma_connection_ && !connection_pool.empty(); i++) {
        auto iter       = connection_pool.begin();
        auto connection = *iter;
        connection_pool.erase(connection);
        busy_connections_.insert(connection);
        connections.push_back(connection);
    }
    return std::make_shared<RdmaConnection>(memory_util_, host_, port_, connections);
}

std::shared_ptr<RdmaConnection>
RdmaClientHostConnections::newRdmaConnection(const std::shared_ptr<::accl::barex::XClientTransport>& client) {
    // new xconnection first
    auto callback = std::make_shared<RdmaClientConnectionCallback>(shared_from_this());
    for (int i = 0; i < qp_count_per_rdma_connection_; i++) {  // 多建一些连接
        client->Connect(::accl::barex::TCP_V4, host_, port_, callback);
    }

    std::shared_ptr<RdmaConnection> rdma_connection;
    for (int i = 0; i < connect_timeout_ms_; i++) {
        auto rdma_connection = getRdmaConnectionFromConnectionPool(newly_connections_, true);
        if (rdma_connection != nullptr) {
            return rdma_connection;
        }
        usleep(1000);
    }

    return getRdmaConnectionFromConnectionPool(newly_connections_, false);
}

void RdmaClientHostConnections::recycleConnection(const std::shared_ptr<RdmaConnection>& connection) {
    if (connection == nullptr) {
        return;
    }
    auto connections = connection->getAllConnections();
    doRecycleConnections(connections);
}

void RdmaClientHostConnections::doRecycleConnections(
    const std::vector<std::shared_ptr<::accl::barex::XConnection>>& connections) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& connection : connections) {
        if (connection == nullptr) {
            continue;
        }
        // connection may already broken and call failed callback
        if (busy_connections_.find(connection) != busy_connections_.end()) {
            busy_connections_.erase(connection);
            available_connections_.insert(connection);
        }
    }
}

void RdmaClientHostConnections::destroyConnection(const std::shared_ptr<RdmaConnection>& rdma_connection) {
    if (rdma_connection == nullptr) {
        return;
    }

    auto connections = rdma_connection->getAllConnections();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& connection : connections) {
            if (busy_connections_.find(connection) != busy_connections_.end()) {
                busy_connections_.erase(connection);
                RTP_LLM_LOG_INFO("rdma connection to %s:%u call disconnect", host_.c_str(), port_);
            }
        }
    }

    // trigger close
    for (auto& connection : connections) {
        connection->Close();
    }
}

void RdmaClientHostConnections::connectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    if (connection == nullptr) {
        return;
    }
    RTP_LLM_LOG_INFO("rdma connection to %s:%u connect success", host_.c_str(), port_);
    std::lock_guard<std::mutex> lock(mutex_);
    newly_connections_.insert(connection);
}

void RdmaClientHostConnections::connectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    if (connection == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto iter = newly_connections_.find(connection);
    if (iter != newly_connections_.end()) {
        newly_connections_.erase(iter);
        RTP_LLM_LOG_INFO("rdma connection to %s:%u disconnect", host_.c_str(), port_);
        return;
    }

    iter = available_connections_.find(connection);
    if (iter != available_connections_.end()) {
        available_connections_.erase(iter);
        RTP_LLM_LOG_INFO("rdma connection to %s:%u disconnect", host_.c_str(), port_);
        return;
    }

    iter = busy_connections_.find(connection);
    if (iter != busy_connections_.end()) {
        busy_connections_.erase(iter);
        RTP_LLM_LOG_WARNING("rdma connection to %s:%u disconnect while inuse", host_.c_str(), port_);
        return;
    }
    RTP_LLM_LOG_INFO("rdma connection to %s:%u disconnect", host_.c_str(), port_);
}

int RdmaClientHostConnections::getAvailableConnectionCount() {
    std::lock_guard<std::mutex> lock(mutex_);
    return available_connections_.size();
}

bool RdmaClientHostConnections::fillConnectionPool(const std::shared_ptr<::accl::barex::XClientTransport>& client, int count) {
    for (int i = 0; i < count; i++) {
        auto rdma_connection = newRdmaConnection(client);
        if (rdma_connection == nullptr) {
            RTP_LLM_LOG_WARNING("new available connections to %s:%u failed", host_.c_str(), port_);
            return false;
        }
        recycleConnection(rdma_connection);
    }
    RTP_LLM_LOG_INFO("rdma client host connections to %s:%u fill connection pool success, count %d", host_.c_str(), port_, count);
    return true;
}

}  // namespace rtp_llm
