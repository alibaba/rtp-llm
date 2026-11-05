#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServerConnectionCallback.h"

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServerHostConnections.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RdmaServerConnectionCallback::RdmaServerConnectionCallback(
    const std::shared_ptr<RdmaServerHostConnectionsMap>& host_connections_map):
    host_connections_map_(host_connections_map) {}

void RdmaServerConnectionCallback::OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>& conn,
                                              const ::accl::barex::XMessage&                     msgs) {
    RTP_LLM_LOG_ERROR("rdma server recv call from %s, should not happen", conn->GetPeerAddrAndPort().first.c_str());
}

void RdmaServerConnectionCallback::OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& conn) {
    auto peer                 = conn->GetPeerAddrAndPort();
    auto channel              = conn->GetChannel();
    uint64_t channel_id       = channel ? reinterpret_cast<uint64_t>(channel) : 0;
    auto host_connections_map = host_connections_map_.lock();
    if (!host_connections_map) {
        RTP_LLM_LOG_WARNING(
            "rdma server connect success from %s:%u channel=0x%lx, but server released",
            peer.first.c_str(), peer.second, channel_id);
        return;
    }
    host_connections_map->connectSuccess(conn);
}

void RdmaServerConnectionCallback::OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& conn) {
    auto peer                 = conn->GetPeerAddrAndPort();
    auto channel              = conn->GetChannel();
    uint64_t channel_id       = channel ? reinterpret_cast<uint64_t>(channel) : 0;
    auto host_connections_map = host_connections_map_.lock();
    if (!host_connections_map) {
        RTP_LLM_LOG_INFO(
            "rdma server connect failed from %s:%u channel=0x%lx, but server released",
            peer.first.c_str(), peer.second, channel_id);
        return;
    }
    host_connections_map->connectFailed(conn);
}

}  // namespace rtp_llm