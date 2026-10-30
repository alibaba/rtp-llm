#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServerHostConnections.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RdmaServerHostConnections::RdmaServerHostConnections(const std::string& ip): ip_(ip) {}

RdmaServerHostConnections::~RdmaServerHostConnections() {
    for (auto iter = port_connections_.begin(); iter != port_connections_.end(); iter++) {
        iter->second->Close();
    }
}

std::shared_ptr<::accl::barex::XConnection> RdmaServerHostConnections::getConnection(uint32_t port, uint64_t peer_channel) const {
    std::shared_lock<std::shared_mutex> lock(port_connections_mutex_);
    if (peer_channel != 0) {
        auto iter = port_connections_.find(PortChannelKey{port, peer_channel});
        if (iter == port_connections_.end()) {
            return nullptr;
        }
        return iter->second;
    }
    // backward compat: peer_channel==0 means match by port only
    for (auto& [key, conn] : port_connections_) {
        if (key.port == port) {
            return conn;
        }
    }
    return nullptr;
}

void RdmaServerHostConnections::addConnection(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    std::unique_lock<std::shared_mutex> lock(port_connections_mutex_);
    auto                                port       = connection->GetPeerAddrAndPort().second;
    auto                                channel_id = reinterpret_cast<uint64_t>(connection->GetChannel());
    PortChannelKey                      key{(uint32_t)port, channel_id};
    auto                                iter = port_connections_.find(key);
    if (iter != port_connections_.end()) {
        port_connections_.erase(key);
    }
    port_connections_.insert(std::make_pair(key, connection));
}

void RdmaServerHostConnections::removeConnection(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    std::unique_lock<std::shared_mutex> lock(port_connections_mutex_);
    auto port       = connection->GetPeerAddrAndPort().second;
    auto channel_id = reinterpret_cast<uint64_t>(connection->GetChannel());
    port_connections_.erase(PortChannelKey{(uint32_t)port, channel_id});
}

RdmaServerHostConnectionsMap::RdmaServerHostConnectionsMap(const std::shared_ptr<MemoryUtil>& memory_util):
    memory_util_(memory_util) {}

std::shared_ptr<RdmaConnection>
RdmaServerHostConnectionsMap::getConnection(const std::vector<AddrPortChannel>& addr_ports) const {

    std::vector<std::shared_ptr<::accl::barex::XConnection>> connections;
    for (auto& apc : addr_ports) {
        auto host_connection = getHostConnection(apc.ip);
        if (host_connection == nullptr) {
            RTP_LLM_LOG_WARNING(
                "rdma server try get connection to %s:%u failed", apc.ip.c_str(), apc.port);
            continue;
        }
        auto connection = host_connection->getConnection(apc.port, apc.peer_channel);
        if (connection == nullptr) {
            RTP_LLM_LOG_WARNING(
                "rdma server try get connection to %s:%u failed", apc.ip.c_str(), apc.port);
            continue;
        }
        connections.push_back(connection);
    }
    if (connections.empty()) {
        return nullptr;
    }
    return std::make_shared<RdmaConnection>(memory_util_, addr_ports[0].ip, addr_ports[0].port, connections);
}

void RdmaServerHostConnectionsMap::connectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    auto peer = connection->GetPeerAddrAndPort();
    if (peer.first.empty()) {
        return;
    }

    auto host_connection = getOrInsertHostConnections(peer.first);
    if (host_connection == nullptr) {
        RTP_LLM_LOG_WARNING("rdma server try get connection to %s failed", peer.first.c_str());
        return;
    }
    host_connection->addConnection(connection);
    RTP_LLM_LOG_INFO("rdma server connect to %s:%u success", peer.first.c_str(), peer.second);
}

void RdmaServerHostConnectionsMap::connectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    auto peer = connection->GetPeerAddrAndPort();
    if (peer.first.empty()) {
        return;
    }

    auto host_connection = getOrInsertHostConnections(peer.first);
    if (host_connection == nullptr) {
        RTP_LLM_LOG_WARNING("rdma server try get connection to %s failed", peer.first.c_str());
        return;
    }
    host_connection->removeConnection(connection);
    RTP_LLM_LOG_INFO("rdma server disconnect to %s:%u success", peer.first.c_str(), peer.second);
}

std::shared_ptr<RdmaServerHostConnections>
RdmaServerHostConnectionsMap::getOrInsertHostConnections(const std::string& ip) {
    std::unique_lock<std::shared_mutex> lock(host_connections_mutex_);
    auto                                it = host_connections_.find(ip);
    if (it == host_connections_.end()) {
        auto host_connections = std::make_shared<RdmaServerHostConnections>(ip);
        host_connections_.insert(std::make_pair(ip, host_connections));
        return host_connections;
    }
    return it->second;
}

std::shared_ptr<RdmaServerHostConnections>
RdmaServerHostConnectionsMap::getHostConnection(const std::string& ip) const {
    std::shared_lock<std::shared_mutex> lock(host_connections_mutex_);
    auto                                it = host_connections_.find(ip);
    if (it == host_connections_.end()) {
        return nullptr;
    }
    return it->second;
}

}  // namespace rtp_llm