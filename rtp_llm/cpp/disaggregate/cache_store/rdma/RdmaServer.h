#pragma once

#include "aios/network/accl-barex/include/accl/barex/highlevel/xserver_transport.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "rtp_llm/cpp/disaggregate/cache_store/MemoryUtil.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaConnection.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServerHostConnections.h"

namespace rtp_llm {

class RdmaServer {
public:
    RdmaServer(const std::shared_ptr<MemoryUtil>& memory_util);

public:
    bool                            init(uint32_t listen_port, uint32_t io_thread_count, uint32_t worker_thread_count);
    std::shared_ptr<RdmaConnection> getConnection(const std::vector<AddrPortChannel>& addr_ports);

private:
    std::shared_ptr<MemoryUtil>                      memory_util_;
    uint32_t                                         port_{0};
    std::shared_ptr<arpc::RdmaMempool>               mempool_;
    std::shared_ptr<::accl::barex::XServerTransport> server_;
    std::shared_ptr<RdmaServerHostConnectionsMap>    host_connections_map_;
};

}  // namespace rtp_llm