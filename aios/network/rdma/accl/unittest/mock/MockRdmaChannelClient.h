#pragma once

#include "aios/network/rdma/accl/RdmaChannelClient.h"
#include "unittest/unittest.h"

ARPC_BEGIN_NAMESPACE(arpc);

class MockRdmaChannelClient: public RdmaChannelClient {
public:
    MockRdmaChannelClient(::accl::barex::TcpProtoType               protocol,
                          const std::string&                        ip,
                          uint32_t                                  port,
                          const std::shared_ptr<TimerManager>&      timerManager,
                          const std::shared_ptr<AcclClientWrapper>& client):
        RdmaChannelClient(protocol, ip, port, timerManager, client) {}
    virtual ~MockRdmaChannelClient() {}

public:
    MOCK_METHOD2(HandleRecvResponse,
                 void(const std::shared_ptr<::accl::barex::XConnection>&, const ::accl::barex::XMessage&));
    MOCK_METHOD1(ConnectSuccess, void(const std::shared_ptr<::accl::barex::XConnection>&));
    MOCK_METHOD1(ConnectFailed, void(const std::shared_ptr<::accl::barex::XConnection>&));
    MOCK_METHOD0(Disconnect, void());
    MOCK_METHOD0(Close, void());
    MOCK_METHOD3(Post,
                 void(const std::shared_ptr<RdmaRequest>&,
                      const std::shared_ptr<RdmaChannelClosure>&,
                      const std::shared_ptr<ClientRPCStats>&));
};

ARPC_END_NAMESPACE(arpc);
