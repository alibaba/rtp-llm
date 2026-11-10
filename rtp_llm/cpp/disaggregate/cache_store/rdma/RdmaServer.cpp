#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServer.h"

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServerConnectionCallback.h"

#include "aios/network/rdma/accl/RdmaMempoolFactory.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xtransport_config.h"
#include "aios/network/accl-barex/include/accl/barex/xsimple_mempool.h"
#include "aios/network/accl-barex/metric/KMonitorAcclMetricReporter.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RdmaServer::RdmaServer(const std::shared_ptr<MemoryUtil>& memory_util):
    memory_util_(memory_util), host_connections_map_(new RdmaServerHostConnectionsMap(memory_util)) {}

bool RdmaServer::init(uint32_t listen_port, uint32_t io_thread_count, uint32_t worker_thread_count) {
    auto mempool = arpc::RdmaMempoolFactory::GetMempool();
    if (!mempool) {
        RTP_LLM_LOG_ERROR("rdma server init failed, new rdmamempool singleton failed");
        return false;
    }

    auto xmempool = mempool->GetXMempool();
    if (!xmempool) {
        RTP_LLM_LOG_ERROR("rdma server init failed, get xmempool failed");
        return false;
    }

    auto metric_reporter = std::make_shared<::accl::barex::KMonitorAcclMetricReporter>(true, kmonitor::FATAL);
    if (!metric_reporter->init()) {
        RTP_LLM_LOG_ERROR("rdma reporter init failed");
        return false;
    }

    ::accl::barex::XServerTransportConfig config(io_thread_count, worker_thread_count, xmempool, metric_reporter);

    server_ = ::accl::barex::XServerTransport::NewInstance(config);
    if (!server_) {
        RTP_LLM_LOG_ERROR("rdma server init failed, new accl server failed");
        return false;
    }

    auto callback = std::make_shared<RdmaServerConnectionCallback>(host_connections_map_);
    if (server_ && server_->Listen(::accl::barex::TCP_V4, listen_port, callback) == ::accl::barex::BAREX_SUCCESS) {
        RTP_LLM_LOG_INFO("rdma server init rdma server success, accl server listen port %d success", listen_port);
        return true;
    }
    RTP_LLM_LOG_ERROR("rdma server init rdma server failed, accl server listen port %d failed", listen_port);
    return false;
}

std::shared_ptr<RdmaConnection>
RdmaServer::getConnection(const std::vector<AddrPortChannel>& addr_ports) {
    return host_connections_map_->getConnection(addr_ports);
}

}  // namespace rtp_llm