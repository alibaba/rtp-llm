#pragma once

#include <memory>

#include "aios/network/accl-barex/include/accl/barex/barex.h"
#include "aios/network/arpc/arpc/CommonMacros.h"
#include "aios/network/arpc/arpc/metric/ServerRPCStats.h"
#include "autil/Log.h"

namespace accl {
namespace barex {
class XConnection;
class XMessage;
}  // namespace barex
}  // namespace accl

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaMessage;
class RdmaMempool;
class IRdmaServerAdapter;
class RdmaChannelServerCallback;
class AcclServerWrapper;

// RdmaChannelServer use accl xserver for listen
// user regist adapter to server and on receive request adapter will be called
// user should release receive request on process success
// user should not release response, response might still need send
class RdmaChannelServer: public std::enable_shared_from_this<RdmaChannelServer> {
public:
    RdmaChannelServer(const std::string&                           address,
                      const std::shared_ptr<IRdmaServerAdapter>&   adapter,
                      const std::shared_ptr<AcclServerWrapper>&    server,
                      const std::shared_ptr<ServerMetricReporter>& metricReporter = nullptr);
    virtual ~RdmaChannelServer();

public:
    // Listen on server address
    // Return true if start listen
    bool Listen();

    // callback on receive request
    virtual void HandleRecvRequest(const std::shared_ptr<::accl::barex::XConnection>& connection,
                                   const ::accl::barex::XMessage&                     msgs);

    // support adapter send success response
    virtual void SendNormalResponse(const std::shared_ptr<::accl::barex::XConnection>& connection,
                                    const std::shared_ptr<RdmaMessage>&                message,
                                    const std::shared_ptr<ServerRPCStats>&             stats);

    // return listen port
    int GetListenPort() const {
        return _port;
    }

private:
    // on process request error, should send error response
    void SendErrorResponse(const std::shared_ptr<::accl::barex::XConnection>& connection,
                           uint32_t                                           requestId,
                           const std::shared_ptr<ServerRPCStats>&             stats);

protected:
    std::shared_ptr<IRdmaServerAdapter> _serverAdapter;

private:
    std::string                                _address;
    std::weak_ptr<AcclServerWrapper>           _server;
    ::accl::barex::TcpProtoType                _protocol{::accl::barex::TcpProtoType::TCP_V4};
    std::string                                _ip;
    int32_t                                    _port;
    std::shared_ptr<RdmaMempool>               _mempool;
    std::shared_ptr<RdmaChannelServerCallback> _callback;
    std::shared_ptr<ServerMetricReporter>      _metricReporter;

private:
    AUTIL_LOG_DECLARE();
    friend class RdmaChannelServerTest;
    friend class RdmaChannelClientOnMockTest;
    friend class RdmaChannelServerOnMockTest_testParseSpec_Test;
    friend class RdmaChannelServerOnMockTest_testHandlePacket_Test;
};

ARPC_END_NAMESPACE(arpc)