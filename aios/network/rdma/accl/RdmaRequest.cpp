#include "aios/network/rdma/accl/RdmaRequest.h"

#include "aios/network/accl-barex/include/accl/barex/barex.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xconnection.h"
#include "aios/network/rdma/accl/RdmaMessage.h"

ARPC_BEGIN_NAMESPACE(arpc);
AUTIL_LOG_SETUP(arpc, RdmaRequest);

#define DATA_MAGIC 0x55aa55aa

RdmaRequest::RdmaRequest(): _timeoutMs(0), _requestId(0), _peerAddress("unknown") {}

RdmaRequest::~RdmaRequest() {}

void RdmaRequest::SetPeerAddress(const std::shared_ptr<::accl::barex::XConnection>& connection) {
    if (connection) {
        auto peerAddress = connection->GetPeerAddrAndPort();
        if (!std::get<0>(peerAddress).empty()) {
            _peerAddress = std::get<0>(peerAddress) + ":" + std::to_string(std::get<1>(peerAddress));
        }
    }
}

ARPC_END_NAMESPACE(arpc);
