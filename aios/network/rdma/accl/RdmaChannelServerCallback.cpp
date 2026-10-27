#include "aios/network/rdma/accl/RdmaChannelServerCallback.h"

#include "aios/network/accl-barex/include/accl/barex/barex_types.h"
#include "aios/network/rdma/accl/RdmaChannelServer.h"
#include "aios/network/rdma/accl/RdmaMempool.h"

ARPC_BEGIN_NAMESPACE(arpc);
AUTIL_LOG_SETUP(arpc, RdmaChannelServerCallback);

RdmaChannelServerCallback::RdmaChannelServerCallback(const std::weak_ptr<RdmaChannelServer>& channel,
                                                     const std::shared_ptr<RdmaMempool>&     mempool):
    _channel(channel), _mempool(mempool) {}

RdmaChannelServerCallback::~RdmaChannelServerCallback() {}

void RdmaChannelServerCallback::OnRecvCall(const std::shared_ptr<::accl::barex::XConnection>& connection,
                                           const ::accl::barex::XMessage&                     msgs) {
    // adapter to channel callback
    auto channel = _channel.lock();
    if (channel) {
        channel->HandleRecvRequest(connection, msgs);
    } else if (_mempool) {
        for (int i = 0; i < msgs.IovCount(); i++) {
            auto iov = msgs.Get(i);
            if (iov.iov_base) {
                _mempool->ReleaseBuffer(iov.iov_base, RdmaMempool::ConvertFromAcclDeviceType(iov.device));
            }
        }
    } else {
        AUTIL_LOG(WARN, "server receive on invalid channel, mem may leak");
    }
}

void RdmaChannelServerCallback::OnConnectSuccess(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    AUTIL_LOG(INFO, "server receive connect success");
}

void RdmaChannelServerCallback::OnConnectFailed(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    AUTIL_LOG(INFO, "server receive connect failed");
}

ARPC_END_NAMESPACE(arpc);