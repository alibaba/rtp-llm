#include "aios/network/rdma/accl/RdmaChannelServer.h"

#include "aios/network/accl-barex/include/accl/barex/barex.h"
#include "aios/network/anet/addrspec.h"
#include "aios/network/arpc/arpc/metric/ServerMetricReporter.h"
#include "aios/network/arpc/arpc/metric/ServerRPCStats.h"
#include "aios/network/arpc/arpc/proto/rpc_extensions.pb.h"
#include "aios/network/rdma/accl/AcclServerWrapper.h"
#include "aios/network/rdma/accl/IRdmaServerAdapter.h"
#include "aios/network/rdma/accl/RdmaChannelServerCallback.h"
#include "aios/network/rdma/accl/RdmaMessage.h"
#include "aios/network/rdma/accl/RdmaRequest.h"
#include "autil/StringUtil.h"

ARPC_BEGIN_NAMESPACE(arpc);
AUTIL_LOG_SETUP(arpc, RdmaChannelServer);

RdmaChannelServer::RdmaChannelServer(const std::string&                           address,
                                     const std::shared_ptr<IRdmaServerAdapter>&   adapter,
                                     const std::shared_ptr<AcclServerWrapper>&    server,
                                     const std::shared_ptr<ServerMetricReporter>& metricReporter):
    _serverAdapter(adapter), _address(address), _server(server), _metricReporter(metricReporter) {

    if (server) {
        _mempool = server->GetMempool();
    }
}

RdmaChannelServer::~RdmaChannelServer() {}

bool RdmaChannelServer::Listen() {
    auto server = _server.lock();
    if (!server) {
        AUTIL_LOG(WARN, "rdma channel server listen failed, accl stoped");
        return false;
    }
    anet::AddrSpec addrspec;
    if (addrspec.setAddrSpec(_address.c_str()) != 0) {
        AUTIL_LOG(WARN, "channel server address %s failed, can not parse", _address.c_str());
        return false;
    }

    if (addrspec.getProtocolFamily() == AF_INET6) {
        _protocol = ::accl::barex::TCP_V6;
    }
    _port = addrspec.getPort();

    _callback = std::make_shared<RdmaChannelServerCallback>(weak_from_this(), _mempool);
    if (!server->Listen(_protocol, _port, _callback)) {
        AUTIL_LOG(WARN, "rdma channel server listen failed, accl server listen failed");
        return false;
    }
    return true;
}

void RdmaChannelServer::HandleRecvRequest(const std::shared_ptr<::accl::barex::XConnection>& connection,
                                          const ::accl::barex::XMessage&                     msgs) {
    std::shared_ptr<ServerRPCStats> stats;
    if (_metricReporter) {
        stats = _metricReporter->makeRPCStats();
    }
    if (stats) {
        stats->markRequestBegin();
    }

    auto message = std::make_shared<RdmaMessage>();
    message->SetConnection(connection);
    message->SetMempool(_mempool);
    if (!message->SetRecvMessage(msgs)) {
        AUTIL_LOG(ERROR, "rdma message decode from request failed");
        if (stats) {
            stats->markRequestDone(ARPC_ERROR_DECODE_PACKET);
        }
        return;
    }

    if (message->IsFailedMessage()) {
        AUTIL_LOG(ERROR, "rdma message recv failed request");
        if (stats) {
            stats->markRequestDone(ARPC_ERROR_BAD_PACKET);
        }
        return;
    }

    auto ret = _serverAdapter->HandlePacket(message, shared_from_this(), stats);
    if (ret != 0) {
        AUTIL_LOG(WARN, "rdma channel server adapter handle packet failed");
        SendErrorResponse(connection, message->GetRequestId(), stats);
    }
}

void RdmaChannelServer::SendErrorResponse(const std::shared_ptr<::accl::barex::XConnection>& connection,
                                          uint32_t                                           requestId,
                                          const std::shared_ptr<ServerRPCStats>&             stats) {
    if (!connection || _server.expired()) {
        AUTIL_LOG(WARN, "rdma channel server send error response failed, accl stoped");
        return;
    }

    auto failedMessage = std::make_shared<RdmaMessage>();
    failedMessage->SetRequestId(requestId);
    failedMessage->SetConnection(connection);
    failedMessage->SetMempool(_mempool);
    failedMessage->SetFailedMessage(true);

    if (stats) {
        stats->markResponseSendBegin();
    }
    auto sendDone = [failedMessage, stats](::accl::barex::Status status) {
        if (stats) {
            stats->markResponseSendDone();
            if (status.IsOk()) {
                stats->markRequestDone(ARPC_ERROR_PUSH_WORKITEM);
            } else {
                stats->markRequestDone(ARPC_ERROR_NO_RESPONSE_SENT);
            }
        }

        if (!status.IsOk()) {
            AUTIL_LOG(WARN, "accl server send response failed, failed send");
        }
        failedMessage->ReleaseMessage();
    };

    connection->Send(failedMessage->GetMessage(), sendDone);
}

void RdmaChannelServer::SendNormalResponse(const std::shared_ptr<::accl::barex::XConnection>& connection,
                                           const std::shared_ptr<RdmaMessage>&                message,
                                           const std::shared_ptr<ServerRPCStats>&             stats) {
    if (!connection) {
        AUTIL_LOG(WARN, "rdma channel server send response failed, null connection");
        return;
    }

    if (_server.expired()) {
        AUTIL_LOG(WARN, "rdma channel server send normal response failed, accl stoped");
        return;
    }
    if (!message) {
        AUTIL_LOG(WARN, "rdma channel server send normal response failed, rdma message is null");
        return;
    }
    auto xmessage = message->GetMessage();
    if (stats) {
        stats->markResponseSendBegin();
        message->reportServerRPCStats(stats, xmessage);
    }
    auto sendDone = [message, stats](::accl::barex::Status status) {
        if (stats) {
            stats->markResponseSendDone();
            if (status.IsOk()) {
                stats->markRequestDone(ARPC_ERROR_NONE);
            } else {
                stats->markRequestDone(ARPC_ERROR_NO_RESPONSE_SENT);
            }
        }

        if (!status.IsOk()) {
            AUTIL_LOG(WARN, "accl server send response failed, failed send");
        }
        message->ReleaseMessage();
    };
    connection->Send(xmessage, sendDone);
}

ARPC_END_NAMESPACE(arpc);