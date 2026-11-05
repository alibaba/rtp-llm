#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/arpc/arpc/metric/mock/MockClientMetricReporter.h"
#include "aios/network/rdma/accl/AcclClientWrapper.h"
#include "aios/network/rdma/accl/RdmaChannelClient.h"
#include "aios/network/rdma/accl/RdmaChannelContext.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"
#include "aios/network/rdma/accl/TimerManager.h"
#include "aios/network/rdma/accl/unittest/mock/MockRdmaChannelClosure.h"
#include "aios/network/rdma/accl/unittest/mock/MockRdmaMessage.h"
#include "aios/network/rdma/accl/unittest/mock/MockRdmaRequest.h"
#include "aios/network/rdma/accl/unittest/mock/MockXClientTransport.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

using namespace ::testing;
ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelClientTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    ::accl::barex::TcpProtoType                          _protocol;
    std::string                                          _ip;
    uint32_t                                             _port;
    std::shared_ptr<TimerManager>                        _timerManager;
    std::shared_ptr<::accl::barex::MockXClientTransport> _mockTransport;
    std::shared_ptr<::accl::barex::XClientTransport>     _transport;
    std::shared_ptr<AcclClientWrapper>                   _wrapper;
    std::shared_ptr<RdmaChannelClient>                   _client;
    std::shared_ptr<::accl::barex::MockXSimpleMempool>   _mockXMempool;
    std::shared_ptr<RdmaMempool>                         _mempool;
};

void RdmaChannelClientTestOnMock::setUp() {
    RdmaMempoolFactory::EnableInit(false);

    _protocol = ::accl::barex::TcpProtoType::TCP_V4;
    _ip       = "127.0.0.1";
    _port     = 12345;

    _timerManager = std::make_shared<TimerManager>();

    _mockTransport = std::make_shared<::accl::barex::MockXClientTransport>();
    _transport     = std::dynamic_pointer_cast<::accl::barex::XClientTransport>(_mockTransport);

    _wrapper          = std::make_shared<AcclClientWrapper>(RdmaClientConfig{1, 1});
    _wrapper->_client = _transport;

    _mockXMempool = std::make_shared<::accl::barex::MockXSimpleMempool>();
    EXPECT_CALL(*_mockXMempool, Shutdown()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, WaitStop()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool = std::make_shared<RdmaMempool>();
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    _mempool->_xmempool = std::dynamic_pointer_cast<::accl::barex::XSimpleMempool>(_mockXMempool);

    _wrapper->_mempool = _mempool;

    _client = std::make_shared<RdmaChannelClient>(_protocol, _ip, _port, _timerManager, _wrapper);
}

void RdmaChannelClientTestOnMock::tearDown() {
    _client.reset();
    _timerManager.reset();
    _wrapper.reset();
    _transport.reset();
    _mockTransport.reset();

    RdmaMempoolFactory::EnableInit(true);
    RdmaMempoolFactory::SetMempool(std::shared_ptr<RdmaMempool>());
}

TEST_F(RdmaChannelClientTestOnMock, testConstructAndDestruct) {
    EXPECT_EQ(_protocol, _client->_protocol);
    EXPECT_EQ(_ip, _client->_ip);
    EXPECT_EQ(_port, _client->_port);
    EXPECT_EQ(0, _client->_requestId.load());
    EXPECT_EQ(CHANNEL_INIT, _client->_state);
    EXPECT_EQ(_timerManager, _client->_timerManager);
    EXPECT_EQ(_wrapper, _client->_client.lock());
    EXPECT_TRUE(_client->_inflightRequests.empty());
    EXPECT_TRUE(_client->_connection == nullptr);
    EXPECT_TRUE(_client->_callback == nullptr);
}

TEST_F(RdmaChannelClientTestOnMock, testInit_NoClient) {
    std::shared_ptr<AcclClientWrapper> wrapper;

    auto client = std::make_shared<RdmaChannelClient>(_protocol, _ip, _port, _timerManager, wrapper);
    EXPECT_TRUE(client->_callback == nullptr);
    EXPECT_EQ(CHANNEL_BROKEN, client->_state);

    client->Init();
    EXPECT_TRUE(client->_callback == nullptr);
    EXPECT_EQ(CHANNEL_BROKEN, client->_state);
}

TEST_F(RdmaChannelClientTestOnMock, testInit_ConnectFailed) {
    _wrapper->_client.reset();
    auto client = std::make_shared<RdmaChannelClient>(_protocol, _ip, _port, _timerManager, _wrapper);
    EXPECT_TRUE(client->_callback == nullptr);
    EXPECT_EQ(CHANNEL_INIT, client->_state);

    client->Init();
    EXPECT_TRUE(client->_callback != nullptr);
    EXPECT_EQ(CHANNEL_BROKEN, client->_state);
}

TEST_F(RdmaChannelClientTestOnMock, testInit_Success) {
    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());

    EXPECT_TRUE(_client->_callback == nullptr);
    EXPECT_EQ(CHANNEL_INIT, _client->_state);

    _client->Init();
    EXPECT_TRUE(_client->_callback != nullptr);
    EXPECT_EQ(CHANNEL_CONNECTING, _client->_state);
}

TEST_F(RdmaChannelClientTestOnMock, testInit_Success_CallTwice) {
    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());

    EXPECT_TRUE(_client->_callback == nullptr);
    EXPECT_EQ(CHANNEL_INIT, _client->_state);

    _client->Init();
    EXPECT_TRUE(_client->_callback != nullptr);
    EXPECT_EQ(CHANNEL_CONNECTING, _client->_state);

    _client->Init();
    EXPECT_TRUE(_client->_callback != nullptr);
    EXPECT_EQ(CHANNEL_CONNECTING, _client->_state);
}

TEST_F(RdmaChannelClientTestOnMock, testDisconnect_Success) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());

    EXPECT_CALL(*_mockTransport, Disconnect(connection)).WillOnce(Return(true));

    _client->_connection = connection;
    EXPECT_TRUE(_client->_callback == nullptr);
    EXPECT_EQ(CHANNEL_INIT, _client->_state);

    _client->Disconnect();
}

TEST_F(RdmaChannelClientTestOnMock, testConnectFailed_NoConnection) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    _client->ConnectFailed(connection);
}

TEST_F(RdmaChannelClientTestOnMock, testConnectFailed_Success) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());

    _client->_connection = connection;
    _client->ConnectFailed(connection);
}

TEST_F(RdmaChannelClientTestOnMock, testPost_NullParams_NullRrequest) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_BAD_PACKET)).WillOnce(Return());

    std::shared_ptr<RdmaRequest> request;

    _client->Post(request, closure, nullptr);
}

TEST_F(RdmaChannelClientTestOnMock, testPost_NullParams_NullClosure) {
    std::shared_ptr<RdmaChannelClosure> closure;

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);

    _client->Post(request, closure, nullptr);
}

TEST_F(RdmaChannelClientTestOnMock, testPost_ChannelStateError) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_CONNECTION_FAILED)).WillOnce(Return());

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);

    _client->Close();
    ASSERT_EQ(ChannelState::CHANNEL_BROKEN, _client->_state);

    _client->_requestId = 10;

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    EXPECT_EQ(0, _client->_pendingSendTasks.size());
    EXPECT_TRUE(_client->_inflightRequests.empty());
    EXPECT_EQ(10, stats->_requestId);
}

TEST_F(RdmaChannelClientTestOnMock, testPost_ChannelStateConnecting) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_CONNECTION_FAILED)).WillOnce(Return());

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();

    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    EXPECT_EQ(1, _client->_pendingSendTasks.size());
    EXPECT_TRUE(_client->_inflightRequests.empty());
    EXPECT_NE(0, stats->_requestPendingBeginTimeUs);

    _client->Close();
}

TEST_F(RdmaChannelClientTestOnMock, testPost_RequestEncodeFailed) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_BAD_PACKET)).WillOnce(Return());

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);
    request->SetTimeoutMs(100);
    std::shared_ptr<RdmaMessage> message;
    EXPECT_CALL(*mockRequest, Encode()).WillOnce(Return(message));

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());
    _client->ConnectSuccess(connection);
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTED, _client->_state);

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    EXPECT_EQ(0, _client->_pendingSendTasks.size());
    EXPECT_TRUE(_client->_inflightRequests.empty());
    EXPECT_NE(0, stats->_requestPendingDoneTimeUs);
    EXPECT_NE(0, stats->_requestPackBeginTimeUs);
}

TEST_F(RdmaChannelClientTestOnMock, testPost_OnSendSuccess_TimeoutBeforeSendDoneCallback) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_TIMEOUT)).WillOnce(Return());

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);
    request->SetTimeoutMs(100);

    auto                    mockMessage = std::make_shared<MockRdmaMessage>();
    auto                    message     = std::dynamic_pointer_cast<RdmaMessage>(mockMessage);
    ::accl::barex::XMessage xmessage;
    EXPECT_CALL(*mockMessage, GetMessage()).WillOnce(Return(xmessage));
    EXPECT_CALL(*mockMessage, ReleaseMessage()).WillOnce(Return(true));
    EXPECT_CALL(*mockRequest, Encode()).WillOnce(Return(message));

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());
    autil::ThreadPtr thread;
    mockConnection->MockOnCallSend(xmessage, thread, 2000);

    _client->ConnectSuccess(connection);
    EXPECT_EQ(CHANNEL_CONNECTED, _client->_state);
    EXPECT_EQ(0, _client->_pendingSendTasks.size());
    EXPECT_TRUE(_client->_inflightRequests.empty());

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    EXPECT_EQ(0, _client->_pendingSendTasks.size());
    EXPECT_EQ(1, _client->_inflightRequests.size());
    EXPECT_NE(0, stats->_requestPendingDoneTimeUs);
    EXPECT_NE(0, stats->_requestPackBeginTimeUs);
    EXPECT_NE(0, stats->_requestPackDoneTimeUs);
    EXPECT_NE(0, stats->_requestSendBeginTimeUs);

    thread->join();
    _client->ConnectFailed(connection);
}

// post 时尚未建联, request 放入 pending 队列, 之后一直未建联直到 timer 检测到 request 超时
TEST_F(RdmaChannelClientTestOnMock, testPost_PendingRequestTimeoutBeforeConnected) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);

    auto      mockRequest      = std::make_shared<MockRdmaRequest>();
    auto      request          = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);
    const int requestTimeoutMs = 100;  // 请求超时时间
    request->SetTimeoutMs(requestTimeoutMs);

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    auto nowUs = autil::TimeUtility::currentTimeInMicroSeconds();
    EXPECT_EQ(1, _client->_pendingSendTasks.size());
    EXPECT_EQ(0, _client->_inflightRequests.size());
    EXPECT_NEAR(nowUs, stats->_requestPendingBeginTimeUs, 1000);
    EXPECT_EQ(0, stats->_requestPendingDoneTimeUs);
    EXPECT_EQ(0, stats->_requestPackBeginTimeUs);
    EXPECT_EQ(0, stats->_requestPackDoneTimeUs);
    EXPECT_EQ(0, stats->_requestSendBeginTimeUs);

    // 200ms 后建联成功, 此时 pending request 已经超时
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_TIMEOUT)).WillOnce(Return());
    usleep(200 * 1000);
    _client->ConnectSuccess(connection);
    EXPECT_EQ(CHANNEL_CONNECTED, _client->_state);
    EXPECT_TRUE(_client->_pendingSendTasks.empty());
    EXPECT_TRUE(_client->_inflightRequests.empty());
    EXPECT_NEAR(nowUs + requestTimeoutMs * 1000, stats->_requestPendingDoneTimeUs, 5000);

    _client->ConnectFailed(connection);
}

// post 时尚未建联, request 放入 pending 队列, 之后 request 超时之前建联成功, 发送 request
TEST_F(RdmaChannelClientTestOnMock, testPost_SendPendingRequestAfterConnected) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);
    request->SetTimeoutMs(200);  // 请求超时时间

    auto                    mockMessage = std::make_shared<MockRdmaMessage>();
    auto                    message     = std::dynamic_pointer_cast<RdmaMessage>(mockMessage);
    ::accl::barex::XMessage xmessage;
    EXPECT_CALL(*mockMessage, GetMessage()).WillOnce(Return(xmessage));
    EXPECT_CALL(*mockMessage, ReleaseMessage()).WillOnce(Return(true));
    EXPECT_CALL(*mockRequest, Encode()).WillOnce(Return(message));

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());
    autil::ThreadPtr thread;
    mockConnection->MockOnCallSend(xmessage, thread, 0);

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    auto nowUs = autil::TimeUtility::currentTimeInMicroSeconds();
    EXPECT_EQ(1, _client->_pendingSendTasks.size());
    EXPECT_EQ(0, _client->_inflightRequests.size());
    EXPECT_NEAR(nowUs, stats->_requestPendingBeginTimeUs, 1000);
    EXPECT_EQ(0, stats->_requestPendingDoneTimeUs);
    EXPECT_EQ(0, stats->_requestPackBeginTimeUs);
    EXPECT_EQ(0, stats->_requestPackDoneTimeUs);
    EXPECT_EQ(0, stats->_requestSendBeginTimeUs);

    // 100ms 后建联成功, 此时 pending request 尚未超时, 将 request 从 pending 队列移除并发送
    usleep(100 * 1000);
    EXPECT_EQ(1, _client->_pendingSendTasks.size());
    _client->ConnectSuccess(connection);
    EXPECT_EQ(CHANNEL_CONNECTED, _client->_state);
    EXPECT_EQ(0, _client->_pendingSendTasks.size());
    EXPECT_EQ(1, _client->_inflightRequests.size());
    EXPECT_NE(0, stats->_requestPendingDoneTimeUs);
    EXPECT_NE(0, stats->_requestPackBeginTimeUs);
    EXPECT_NE(0, stats->_requestPackDoneTimeUs);
    EXPECT_NE(0, stats->_requestSendBeginTimeUs);

    thread->join();
    // 清理 inflight request
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_CONNECTION_FAILED)).WillOnce(Return());
    _client->ConnectFailed(connection);
}

TEST_F(RdmaChannelClientTestOnMock, testPost_OnSendSuccess_SendDoneCallbackFailed) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_BAD_PACKET)).WillOnce(Return());

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);
    request->SetTimeoutMs(1000);

    auto                    mockMessage = std::make_shared<MockRdmaMessage>();
    auto                    message     = std::dynamic_pointer_cast<RdmaMessage>(mockMessage);
    ::accl::barex::XMessage xmessage;
    EXPECT_CALL(*mockMessage, GetMessage()).WillOnce(Return(xmessage));
    EXPECT_CALL(*mockMessage, ReleaseMessage()).WillOnce(Return(true));
    EXPECT_CALL(*mockRequest, Encode()).WillOnce(Return(message));

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());
    autil::ThreadPtr thread;
    mockConnection->MockOnFailedCallSend(xmessage, thread, 200);

    _client->ConnectSuccess(connection);
    EXPECT_EQ(CHANNEL_CONNECTED, _client->_state);
    EXPECT_EQ(0, _client->_pendingSendTasks.size());

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    EXPECT_EQ(1, _client->_inflightRequests.size());
    EXPECT_EQ(0, _client->_pendingSendTasks.size());
    EXPECT_NE(0, stats->_requestPendingDoneTimeUs);
    EXPECT_NE(0, stats->_requestPackBeginTimeUs);
    EXPECT_NE(0, stats->_requestPackDoneTimeUs);
    EXPECT_NE(0, stats->_requestSendBeginTimeUs);

    thread->join();
    EXPECT_NE(0, stats->_requestSendDoneTimeUs);
    _client->ConnectFailed(connection);
}

TEST_F(RdmaChannelClientTestOnMock, testPost_OnSendSuccess_SendDoneCallbackSuccess) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_CONNECTION_FAILED)).WillOnce(Return());

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);
    request->SetTimeoutMs(1000);

    auto                    mockMessage = std::make_shared<MockRdmaMessage>();
    auto                    message     = std::dynamic_pointer_cast<RdmaMessage>(mockMessage);
    ::accl::barex::XMessage xmessage;
    EXPECT_CALL(*mockMessage, GetMessage()).WillOnce(Return(xmessage));
    EXPECT_CALL(*mockMessage, ReleaseMessage()).WillOnce(Return(true));
    EXPECT_CALL(*mockRequest, Encode()).WillOnce(Return(message));

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());
    autil::ThreadPtr thread;
    mockConnection->MockOnCallSend(xmessage, thread, 200);

    _client->ConnectSuccess(connection);
    EXPECT_EQ(CHANNEL_CONNECTED, _client->_state);
    EXPECT_TRUE(_client->_inflightRequests.empty());
    EXPECT_EQ(0, _client->_pendingSendTasks.size());

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    EXPECT_EQ(1, _client->_inflightRequests.size());
    EXPECT_EQ(0, _client->_pendingSendTasks.size());
    EXPECT_NE(0, stats->_requestPendingDoneTimeUs);
    EXPECT_NE(0, stats->_requestPackBeginTimeUs);
    EXPECT_NE(0, stats->_requestPackDoneTimeUs);
    EXPECT_NE(0, stats->_requestSendBeginTimeUs);

    thread->join();
    EXPECT_NE(0, stats->_requestSendDoneTimeUs);
    _client->ConnectFailed(connection);
}

TEST_F(RdmaChannelClientTestOnMock, testHandleRecvResponse_DecodeXMessageFailed) {
    ::accl::barex::XMessage xmessage;
    xmessage.Add(nullptr, 0, ::accl::barex::CPU, ::accl::barex::CPU);
    EXPECT_CALL(*_mockXMempool, ReleaseBuffer(nullptr, ::accl::barex::CPU))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    _client->HandleRecvResponse(connection, xmessage);
}

TEST_F(RdmaChannelClientTestOnMock, testHandleRecvResponse_ExpiredRequest) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_CONNECTION_FAILED)).WillOnce(Return());

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);
    request->SetTimeoutMs(1000);

    auto                    mockMessage = std::make_shared<MockRdmaMessage>();
    auto                    message     = std::dynamic_pointer_cast<RdmaMessage>(mockMessage);
    ::accl::barex::XMessage xmessage;
    EXPECT_CALL(*mockMessage, GetMessage()).WillOnce(Return(xmessage));
    EXPECT_CALL(*mockMessage, ReleaseMessage()).WillOnce(Return(true));
    EXPECT_CALL(*mockRequest, Encode()).WillOnce(Return(message));

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());
    autil::ThreadPtr thread;
    mockConnection->MockOnCallSend(xmessage, thread, 200);

    _client->ConnectSuccess(connection);
    ASSERT_EQ(CHANNEL_CONNECTED, _client->_state);
    EXPECT_TRUE(_client->_inflightRequests.empty());
    EXPECT_EQ(0, _client->_pendingSendTasks.size());

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    EXPECT_EQ(1, _client->_inflightRequests.size());
    EXPECT_EQ(0, _client->_pendingSendTasks.size());

    thread->join();
    {
        ::accl::barex::XMessage xmessage;
        xmessage.SetUserFlag0(101);
        xmessage.SetUserFlag1(0);

        _client->HandleRecvResponse(connection, xmessage);

        // 请求过期了，因此 stats 中不会存在相应的记录
        EXPECT_EQ(0, stats->_responsePopBeginTimeUs);
        EXPECT_EQ(0, stats->_responsePopDoneTimeUs);
    }

    _client->ConnectFailed(connection);
}

TEST_F(RdmaChannelClientTestOnMock, testHandleRecvResponse_FailedMessage) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_SERVER_ERROR)).WillOnce(Return());

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);
    request->SetTimeoutMs(1000);

    auto                    mockMessage = std::make_shared<MockRdmaMessage>();
    auto                    message     = std::dynamic_pointer_cast<RdmaMessage>(mockMessage);
    ::accl::barex::XMessage xmessage;
    EXPECT_CALL(*mockMessage, GetMessage()).WillOnce(Return(xmessage));
    EXPECT_CALL(*mockMessage, ReleaseMessage()).WillOnce(Return(true));
    EXPECT_CALL(*mockRequest, Encode()).WillOnce(Return(message));
    EXPECT_CALL(*mockRequest, Decode(_)).WillOnce(Return(true));

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());
    EXPECT_CALL(*mockConnection, GetPeerAddrAndPort()).WillOnce(Return(std::make_pair("", 0)));
    autil::ThreadPtr thread;
    mockConnection->MockOnCallSend(xmessage, thread, 200);

    _client->ConnectSuccess(connection);
    EXPECT_TRUE(_client->_inflightRequests.empty());
    ASSERT_EQ(CHANNEL_CONNECTED, _client->_state);

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    EXPECT_EQ(1, _client->_inflightRequests.size());
    EXPECT_EQ(0, _client->_pendingSendTasks.size());

    thread->join();

    {
        ::accl::barex::XMessage xmessage2;
        xmessage2.SetUserFlag0(0);
        xmessage2.SetUserFlag1(1);

        _client->HandleRecvResponse(connection, xmessage2);

        EXPECT_NE(0, stats->_responsePopBeginTimeUs);
        EXPECT_NE(0, stats->_responsePopDoneTimeUs);
    }
    _client->ConnectFailed(connection);
}

TEST_F(RdmaChannelClientTestOnMock, testHandleRecvResponse_DecodeRequestFailed) {
    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_BAD_PACKET)).WillOnce(Return());

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);
    request->SetTimeoutMs(1000);

    auto                    mockMessage = std::make_shared<MockRdmaMessage>();
    auto                    message     = std::dynamic_pointer_cast<RdmaMessage>(mockMessage);
    ::accl::barex::XMessage xmessage;
    EXPECT_CALL(*mockMessage, GetMessage()).WillOnce(Return(xmessage));
    EXPECT_CALL(*mockMessage, ReleaseMessage()).WillOnce(Return(true));
    EXPECT_CALL(*mockRequest, Encode()).WillOnce(Return(message));
    EXPECT_CALL(*mockRequest, Decode(_)).WillOnce(Return(false));

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());
    autil::ThreadPtr thread;
    mockConnection->MockOnCallSend(xmessage, thread, 200);

    _client->ConnectSuccess(connection);
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTED, _client->_state);
    EXPECT_TRUE(_client->_inflightRequests.empty());
    EXPECT_EQ(0, _client->_pendingSendTasks.size());

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    EXPECT_EQ(1, _client->_inflightRequests.size());
    EXPECT_EQ(0, _client->_pendingSendTasks.size());

    thread->join();

    {
        ::accl::barex::XMessage xmessage2;
        xmessage2.SetUserFlag0(0);
        xmessage2.SetUserFlag1(0);

        _client->HandleRecvResponse(connection, xmessage2);

        EXPECT_NE(0, stats->_responsePopBeginTimeUs);
        EXPECT_NE(0, stats->_responsePopDoneTimeUs);
        EXPECT_NE(0, stats->_responseUnpackBeginTimeUs);
        EXPECT_NE(0, stats->_responseUnpackDoneTimeUs);
    }
    _client->ConnectFailed(connection);
}

TEST_F(RdmaChannelClientTestOnMock, testHandleRecvResponse_Success) {

    auto mockClosure = std::make_shared<MockRdmaChannelClosure>();
    auto closure     = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    EXPECT_CALL(*mockClosure, Run(RdmaChannelClosure::ERROR_NONE)).WillOnce(Return());

    auto mockRequest = std::make_shared<MockRdmaRequest>();
    auto request     = std::dynamic_pointer_cast<RdmaRequest>(mockRequest);
    request->SetTimeoutMs(1000);

    auto                    mockMessage = std::make_shared<MockRdmaMessage>();
    auto                    message     = std::dynamic_pointer_cast<RdmaMessage>(mockMessage);
    ::accl::barex::XMessage xmessage;
    EXPECT_CALL(*mockMessage, GetMessage()).WillOnce(Return(xmessage));
    EXPECT_CALL(*mockMessage, ReleaseMessage()).WillOnce(Return(true));
    EXPECT_CALL(*mockRequest, Encode()).WillOnce(Return(message));
    EXPECT_CALL(*mockRequest, Decode(_)).WillOnce(Return(true));

    EXPECT_CALL(*_mockTransport, Connect(_protocol, _ip, _port, _)).WillOnce(Return());
    _client->Init();
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTING, _client->_state);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, Close()).WillOnce(Return());
    autil::ThreadPtr thread;
    mockConnection->MockOnCallSend(xmessage, thread, 200);

    _client->ConnectSuccess(connection);
    EXPECT_TRUE(_client->_inflightRequests.empty());
    ASSERT_EQ(ChannelState::CHANNEL_CONNECTED, _client->_state);

    auto stats = std::make_shared<ClientRPCStats>(nullptr);
    _client->Post(request, closure, stats);
    EXPECT_EQ(1, _client->_inflightRequests.size());
    EXPECT_EQ(0, _client->_pendingSendTasks.size());

    thread->join();
    {
        ::accl::barex::XMessage xmessage2;
        xmessage2.SetUserFlag0(0);
        xmessage2.SetUserFlag1(0);

        _client->HandleRecvResponse(connection, xmessage2);

        EXPECT_NE(0, stats->_requestPackBeginTimeUs);
        EXPECT_NE(0, stats->_requestPackDoneTimeUs);
        EXPECT_NE(0, stats->_requestSendBeginTimeUs);
        EXPECT_NE(0, stats->_requestSendDoneTimeUs);
        EXPECT_NE(0, stats->_responsePopBeginTimeUs);
        EXPECT_NE(0, stats->_responsePopDoneTimeUs);
        EXPECT_NE(0, stats->_responseUnpackBeginTimeUs);
        EXPECT_NE(0, stats->_responseUnpackDoneTimeUs);
    }
    _client->ConnectFailed(connection);
}

TEST_F(RdmaChannelClientTestOnMock, testChannelState) {
    _client->_state = CHANNEL_INIT;
    EXPECT_FALSE(_client->ChannelConnected());
    EXPECT_FALSE(_client->ChannelConnected());

    _client->_state = CHANNEL_CONNECTING;
    EXPECT_FALSE(_client->ChannelConnected());
    EXPECT_FALSE(_client->ChannelConnected());

    _client->_state = CHANNEL_CONNECTED;
    EXPECT_TRUE(_client->ChannelConnected());
    EXPECT_FALSE(_client->ChannelBroken());

    _client->_state = CHANNEL_BROKEN;
    EXPECT_FALSE(_client->ChannelConnected());
    EXPECT_TRUE(_client->ChannelBroken());
}

TEST_F(RdmaChannelClientTestOnMock, testCloseDeadLock) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    // xconnection close -> xchannel close -> xconnection callback connection failed -> client close
    ON_CALL(*mockConnection, Close()).WillByDefault(Invoke([client = _client, connection]() mutable {
        client->ConnectFailed(connection);
        // make sure gmock will not hold ref
        client.reset();
        connection.reset();
    }));
    EXPECT_CALL(*mockConnection, Close()).Times(1);

    _client->_connection = connection;
    _client->Close();
}

TEST_F(RdmaChannelClientTestOnMock, testPopPendingRequest) {
    auto client =
        std::make_shared<RdmaChannelClient>(::accl::barex::TcpProtoType::TCP_V4, "localhost", 12345, nullptr, nullptr);
    ASSERT_TRUE(nullptr != client);
    EXPECT_TRUE(nullptr == client->PopPendingRequest(123));

    auto realContext               = std::make_shared<RdmaChannelContext>(nullptr, nullptr, nullptr, nullptr);
    client->_pendingSendTasks[123] = realContext;
    auto expectContext             = client->PopPendingRequest(123);
    EXPECT_EQ(realContext, expectContext);
}

TEST_F(RdmaChannelClientTestOnMock, testHandleTimeoutRequest) {
    auto client =
        std::make_shared<RdmaChannelClient>(::accl::barex::TcpProtoType::TCP_V4, "localhost", 12345, nullptr, nullptr);
    ASSERT_TRUE(nullptr != client);

    auto mockClosure               = std::make_shared<MockRdmaChannelClosure>();
    auto closure                   = std::dynamic_pointer_cast<RdmaChannelClosure>(mockClosure);
    auto realContext               = std::make_shared<RdmaChannelContext>(nullptr, closure, nullptr, nullptr);
    client->_pendingSendTasks[123] = realContext;  // pending request
    client->_inflightRequests[456] = realContext;  // inflight request

    EXPECT_CALL(*mockClosure, Run).Times(1).WillOnce(Invoke([](RdmaChannelClosure::ErrorState errorState) {
        EXPECT_EQ(RdmaChannelClosure::ERROR_TIMEOUT, errorState);
    }));
    client->HandleTimeoutRequest(123, 0);

    EXPECT_CALL(*mockClosure, Run).Times(1).WillOnce(Invoke([](RdmaChannelClosure::ErrorState errorState) {
        EXPECT_EQ(RdmaChannelClosure::ERROR_TIMEOUT, errorState);
    }));
    client->HandleTimeoutRequest(456, 0);
}

ARPC_END_NAMESPACE(arpc);