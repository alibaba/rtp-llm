#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/accl-barex/include/accl/barex/highlevel/xmessage.h"
#include "aios/network/arpc/arpc/metric/mock/MockServerMetricReporter.h"
#include "aios/network/rdma/accl/AcclServerWrapper.h"
#include "aios/network/rdma/accl/RdmaChannelServer.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMessage.h"
#include "aios/network/rdma/accl/unittest/mock/MockIServerAdapter.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"
#include "aios/network/rdma/accl/unittest/mock/MockXServerTransport.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

using namespace ::testing;
ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelServerTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    std::shared_ptr<MockIRdmaServerAdapter> _mockAdapter;
    std::shared_ptr<IRdmaServerAdapter>     _adapter;

    std::shared_ptr<::accl::barex::MockXServerTransport> _mockTransport;
    std::shared_ptr<::accl::barex::XServerTransport>     _transport;
    std::shared_ptr<AcclServerWrapper>                   _wrapper;

    std::shared_ptr<RdmaChannelServer> _server;

    std::shared_ptr<::accl::barex::MockXSimpleMempool> _mockXMempool;
    std::shared_ptr<::accl::barex::XSimpleMempool>     _xMempool;
    std::shared_ptr<RdmaMempool>                       _mempool;
};

void RdmaChannelServerTestOnMock::setUp() {
    _mockXMempool = std::make_shared<::accl::barex::MockXSimpleMempool>();
    _xMempool     = std::dynamic_pointer_cast<::accl::barex::XSimpleMempool>(_mockXMempool);
    EXPECT_CALL(*_mockXMempool, Shutdown()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, WaitStop()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool            = std::make_shared<RdmaMempool>();
    _mempool->_xmempool = _xMempool;

    _mockAdapter = std::make_shared<MockIRdmaServerAdapter>();
    _adapter     = std::dynamic_pointer_cast<IRdmaServerAdapter>(_mockAdapter);

    _mockTransport = std::make_shared<::accl::barex::MockXServerTransport>();
    _transport     = std::dynamic_pointer_cast<::accl::barex::XServerTransport>(_mockTransport);

    _wrapper           = std::make_shared<AcclServerWrapper>();
    _wrapper->_server  = _transport;
    _wrapper->_mempool = _mempool;

    _server = std::make_shared<RdmaChannelServer>("tcp:123.4.5.6:12345", _adapter, _wrapper);
}

void RdmaChannelServerTestOnMock::tearDown() {}

TEST_F(RdmaChannelServerTestOnMock, testListen_NoServer) {
    std::shared_ptr<AcclServerWrapper> wrapper;
    auto server = std::make_shared<RdmaChannelServer>("tcp:123.4.5.6:12345", _adapter, wrapper);

    EXPECT_FALSE(server->Listen());
}

TEST_F(RdmaChannelServerTestOnMock, testListen_ListenFailed) {
    EXPECT_CALL(*_mockTransport, Listen(::accl::barex::TcpProtoType::TCP_V4, 12345, _))
        .WillOnce(Return(::accl::barex::BAREX_ERR_ARG));

    EXPECT_FALSE(_server->Listen());
}

TEST_F(RdmaChannelServerTestOnMock, testListen_Success) {
    EXPECT_CALL(*_mockTransport, Listen(::accl::barex::TcpProtoType::TCP_V4, 12345, _))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_TRUE(_server->Listen());
}

TEST_F(RdmaChannelServerTestOnMock, testHandleRecvRequest_ParseXMessageFailed) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    ::accl::barex::XMessage xmessage;
    xmessage.Add(nullptr, 0, ::accl::barex::CPU, ::accl::barex::CPU);

    _server->HandleRecvRequest(connection, xmessage);
}

TEST_F(RdmaChannelServerTestOnMock, testHandleRecvRequest_FailedMessage) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    ::accl::barex::XMessage xmessage;
    xmessage.SetUserFlag0(0);
    xmessage.SetUserFlag1(1);

    _server->HandleRecvRequest(connection, xmessage);
}

TEST_F(RdmaChannelServerTestOnMock, testHandleRecvRequest_HandlePacketFailed_SendFailed) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    ::accl::barex::XMessage xmessage;
    xmessage.SetUserFlag0(0);
    xmessage.SetUserFlag1(0);

    EXPECT_CALL(*_mockAdapter, HandlePacket(_, _, _)).WillOnce(Return(-1));
    EXPECT_CALL(*mockConnection, Send(_, _, _)).WillOnce(Return());

    _server->HandleRecvRequest(connection, xmessage);

    _adapter.reset();
    _mockAdapter.reset();
}

TEST_F(RdmaChannelServerTestOnMock, testHandleRecvRequest_Success) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    ::accl::barex::XMessage xmessage;
    xmessage.SetUserFlag0(0);
    xmessage.SetUserFlag1(0);

    EXPECT_CALL(*_mockAdapter, HandlePacket(_, _, _)).WillOnce(Return(0));

    _server->HandleRecvRequest(connection, xmessage);

    _adapter.reset();
    _mockAdapter.reset();
}

TEST_F(RdmaChannelServerTestOnMock, testSendNormalResponse_NullConnection) {
    std::shared_ptr<::accl::barex::XConnection> connection;

    auto message = std::make_shared<RdmaMessage>();
    _server->SendNormalResponse(connection, message, nullptr);
}

TEST_F(RdmaChannelServerTestOnMock, testSendNormalResponse_ChannelExpires) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    _wrapper.reset();
    auto message = std::make_shared<RdmaMessage>();
    _server->SendNormalResponse(connection, message, nullptr);
}

TEST_F(RdmaChannelServerTestOnMock, testSendNormalResponse_Send) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Send(_, _, _))
        .WillOnce(Invoke([](const accl::barex::XMessage&,
                            accl::barex::DoneCallback callback,
                            ::accl::barex::XConnection::SendMode) { callback(accl::barex::Status::OK()); }));

    auto message = std::make_shared<RdmaMessage>();
    _server->SendNormalResponse(connection, message, nullptr);
}

TEST_F(RdmaChannelServerTestOnMock, testErrorNormalResponse_Send) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Send(_, _, _))
        .WillOnce(Invoke([](const accl::barex::XMessage&,
                            accl::barex::DoneCallback callback,
                            ::accl::barex::XConnection::SendMode) { callback(accl::barex::Status::OK()); }));

    _server->SendErrorResponse(connection, 1, nullptr);
}

ARPC_END_NAMESPACE(arpc);