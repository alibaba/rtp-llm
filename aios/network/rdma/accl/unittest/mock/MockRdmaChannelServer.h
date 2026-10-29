#pragma once

#include "aios/network/accl-barex/include/accl/barex/highlevel/xmessage.h"
#include "aios/network/rdma/accl/RdmaChannelServer.h"
#include "unittest/unittest.h"

ARPC_BEGIN_NAMESPACE(arpc)

class MockRdmaChannelServer: public RdmaChannelServer {
public:
    MockRdmaChannelServer(const std::string&                           address,
                          const std::shared_ptr<IRdmaServerAdapter>&   adapter,
                          const std::shared_ptr<AcclServerWrapper>&    server,
                          const std::shared_ptr<ServerMetricReporter>& metricReporter = nullptr):
        RdmaChannelServer(address, adapter, server, metricReporter) {}

    virtual ~MockRdmaChannelServer() {}

public:
    MOCK_METHOD2(HandleRecvRequest,
                 void(const std::shared_ptr<::accl::barex::XConnection>& connection,
                      const ::accl::barex::XMessage&                     message));
    MOCK_METHOD3(SendNormalResponse,
                 void(const std::shared_ptr<::accl::barex::XConnection>&,
                      const std::shared_ptr<RdmaMessage>&,
                      const std::shared_ptr<ServerRPCStats>&));
};

ARPC_END_NAMESPACE(arpc)