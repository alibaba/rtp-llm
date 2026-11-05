#include "aios/network/rdma/accl/RdmaChannelClientCallback.h"

#include "aios/network/rdma/accl/RdmaChannelClient.h"
#include "aios/network/rdma/accl/RdmaMempool.h"

ARPC_BEGIN_NAMESPACE(arpc);
AUTIL_LOG_SETUP(arpc, RdmaChannelClientCallback);

RdmaChannelClientCallback::RdmaChannelClientCallback(const std::weak_ptr<RdmaChannelClient>& channel,
                                                     const std::shared_ptr<RdmaMempool>&     mempool):
    _channel(channel), _mempool(mempool) {}

RdmaChannelClientCallback::~RdmaChannelClientCallback() {}

void RdmaChannelClientCallback::OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>& connection,
                                           const ::accl::barex::XMessage&                     msgs) {
    auto channel = _channel.lock();
    if (channel) {
        channel->HandleRecvResponse(connection, msgs);
        return;
    }
    if (_mempool) {
        for (int i = 0; i < msgs.IovCount(); i++) {
            auto iov = msgs.Get(i);
            if (iov.iov_base) {
                _mempool->ReleaseBuffer(iov.iov_base, RdmaMempool::ConvertFromAcclDeviceType(iov.device));
            }
        }
        return;
    }
    AUTIL_LOG(WARN, "call expired client recv request, msg may leak");
}

void RdmaChannelClientCallback::OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    auto channel = _channel.lock();
    if (channel) {
        channel->ConnectSuccess(connection);
        return;
    }
    AUTIL_LOG(WARN, "call expired client connection success");
}

void RdmaChannelClientCallback::OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    auto channel = _channel.lock();
    if (channel) {
        channel->ConnectFailed(connection);
        return;
    }
    AUTIL_LOG(WARN, "call expired client connection failed");
}

ARPC_END_NAMESPACE(arpc);