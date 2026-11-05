#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientConnectionCallback.h"

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientHostConnections.h"

#include "rtp_llm/cpp/utils/Logger.h"

namespace rtp_llm {

RdmaClientConnectionCallback::RdmaClientConnectionCallback(
    const std::shared_ptr<RdmaClientHostConnections>& connection):
    connection_(connection) {}

void RdmaClientConnectionCallback::OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>& conn,
                                              const ::accl::barex::XMessage&                     msgs) {
    RTP_LLM_LOG_ERROR("recv call, should not happen, request from %s", conn->GetPeerAddrAndPort().first.c_str());
}

void RdmaClientConnectionCallback::OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& conn) {
    auto connection = connection_.lock();
    if (!connection) {
        RTP_LLM_LOG_WARNING("connection has been released before connection to %s success",
                            conn->GetPeerAddrAndPort().first.c_str());
        return;
    }
    connection->connectSuccess(conn);
}

void RdmaClientConnectionCallback::OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& conn) {
    auto connection = connection_.lock();
    if (!connection) {
        RTP_LLM_LOG_WARNING("connection has been released before connection to %s success",
                            conn->GetPeerAddrAndPort().first.c_str());
        return;
    }
    connection->connectFailed(conn);
}

}  // namespace rtp_llm