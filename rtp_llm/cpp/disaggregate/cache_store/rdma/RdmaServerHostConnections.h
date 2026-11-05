#pragma once

#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"

namespace rtp_llm {

struct PortChannelKey {
    uint32_t port;
    uint64_t channel_id;
    bool operator<(const PortChannelKey& rhs) const {
        if (port != rhs.port) return port < rhs.port;
        return channel_id < rhs.channel_id;
    }
};

struct AddrPortChannel {
    std::string ip;
    uint32_t    port;
    uint64_t    peer_channel;
};

class RdmaServerHostConnections {
public:
    RdmaServerHostConnections(const std::string& ip);
    ~RdmaServerHostConnections();

public:
    std::shared_ptr<::accl::barex::XConnection> getConnection(uint32_t port, uint64_t peer_channel) const;
    void addConnection(const std::shared_ptr<::accl::barex::XConnection>& connection);
    void removeConnection(const std::shared_ptr<::accl::barex::XConnection>& connection);

private:
    std::string                                                     ip_;
    mutable std::shared_mutex                                       port_connections_mutex_;
    std::map<PortChannelKey, std::shared_ptr<::accl::barex::XConnection>> port_connections_;
};

class RdmaServerHostConnectionsMap {
public:
    RdmaServerHostConnectionsMap(const std::shared_ptr<MemoryUtil>& memory_util);

public:
    std::shared_ptr<RdmaConnection>
         getConnection(const std::vector<AddrPortChannel>& addr_ports) const;
    void connectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection);
    void connectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection);

private:
    std::shared_ptr<RdmaServerHostConnections> getOrInsertHostConnections(const std::string& ip);
    std::shared_ptr<RdmaServerHostConnections> getHostConnection(const std::string& ip) const;

private:
    std::shared_ptr<MemoryUtil> memory_util_;

    mutable std::shared_mutex                                         host_connections_mutex_;
    std::map<std::string, std::shared_ptr<RdmaServerHostConnections>> host_connections_;
};

}  // namespace rtp_llm