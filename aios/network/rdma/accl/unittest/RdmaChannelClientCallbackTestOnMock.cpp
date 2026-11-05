#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/RdmaChannelClientCallback.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/unittest/mock/MockRdmaChannelClient.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

using namespace ::testing;
ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelClientCallbackTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    std::shared_ptr<RdmaMempool>                       _mempool;
    std::shared_ptr<::accl::barex::MockXSimpleMempool> _mockXMempool;
};

void RdmaChannelClientCallbackTestOnMock::setUp() {
    _mockXMempool = std::make_shared<::accl::barex::MockXSimpleMempool>();
    EXPECT_CALL(*_mockXMempool, Shutdown()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, WaitStop()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool = std::make_shared<RdmaMempool>();
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    _mempool->_xmempool = std::dynamic_pointer_cast<::accl::barex::XSimpleMempool>(_mockXMempool);
}

void RdmaChannelClientCallbackTestOnMock::tearDown() {}

TEST_F(RdmaChannelClientCallbackTestOnMock, testOnRecvCall_ExpiredChannel) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();
    std::shared_ptr<RdmaChannelClientCallback> callback;
    {
        std::string                        ip   = "123.4.5.6";
        int32_t                            port = 12345;
        std::shared_ptr<TimerManager>      timerManager;
        std::shared_ptr<AcclClientWrapper> wrapper;

        auto mockClient =
            std::make_shared<MockRdmaChannelClient>(::accl::barex::TCP_V4, ip, port, timerManager, wrapper);
        auto client = std::dynamic_pointer_cast<RdmaChannelClient>(mockClient);

        callback = std::make_shared<RdmaChannelClientCallback>(client, _mempool);
    }
    ASSERT_TRUE(callback->_channel.expired());
    std::shared_ptr<::accl::barex::XConnection> connection;
    ::accl::barex::XMessage                     xmessage;
    auto                                        buf = _mempool->AllocBuffer(1024);
    xmessage.Add(buf, 1024, accl::barex::CPU, accl::barex::CPU);
    callback->OnRecvCall(connection, xmessage);
}

TEST_F(RdmaChannelClientCallbackTestOnMock, testOnRecvCall_NullChannel) {
    std::shared_ptr<::accl::barex::XConnection> connection;
    ::accl::barex::XMessage                     xmessage;

    std::shared_ptr<RdmaChannelClient> client;
    auto                               callback = std::make_shared<RdmaChannelClientCallback>(client, _mempool);
    callback->OnRecvCall(connection, xmessage);
}

TEST_F(RdmaChannelClientCallbackTestOnMock, testOnRecvCall_Success) {
    std::string                        ip   = "123.4.5.6";
    int32_t                            port = 12345;
    std::shared_ptr<TimerManager>      timerManager;
    std::shared_ptr<AcclClientWrapper> wrapper;

    auto mockClient = std::make_shared<MockRdmaChannelClient>(::accl::barex::TCP_V4, ip, port, timerManager, wrapper);
    auto client     = std::dynamic_pointer_cast<RdmaChannelClient>(mockClient);

    std::shared_ptr<::accl::barex::XConnection> connection;
    ::accl::barex::XMessage                     xmessage;
    EXPECT_CALL(*mockClient, HandleRecvResponse(connection, _)).WillOnce(Return());

    auto callback = std::make_shared<RdmaChannelClientCallback>(client, _mempool);
    ASSERT_FALSE(callback->_channel.expired());
    callback->OnRecvCall(connection, xmessage);
}

TEST_F(RdmaChannelClientCallbackTestOnMock, testOnConnectSuccess_ExpiredChannel) {
    std::shared_ptr<RdmaChannelClientCallback> callback;
    {
        std::string                        ip   = "123.4.5.6";
        int32_t                            port = 12345;
        std::shared_ptr<TimerManager>      timerManager;
        std::shared_ptr<AcclClientWrapper> wrapper;

        auto mockClient =
            std::make_shared<MockRdmaChannelClient>(::accl::barex::TCP_V4, ip, port, timerManager, wrapper);
        auto client = std::dynamic_pointer_cast<RdmaChannelClient>(mockClient);

        callback = std::make_shared<RdmaChannelClientCallback>(client, _mempool);
    }
    ASSERT_TRUE(callback->_channel.expired());
    std::shared_ptr<::accl::barex::XConnection> connection;
    callback->OnConnectSuccess(connection);
}

TEST_F(RdmaChannelClientCallbackTestOnMock, testOnConnectSuccess_NullChannel) {
    std::shared_ptr<::accl::barex::XConnection> connection;
    ::accl::barex::XMessage                     xmessage;

    std::shared_ptr<RdmaChannelClient> client;
    auto                               callback = std::make_shared<RdmaChannelClientCallback>(client, _mempool);
    callback->OnConnectSuccess(connection);
}

TEST_F(RdmaChannelClientCallbackTestOnMock, testOnConnectSuccess_Success) {
    std::string                        ip   = "123.4.5.6";
    int32_t                            port = 12345;
    std::shared_ptr<TimerManager>      timerManager;
    std::shared_ptr<AcclClientWrapper> wrapper;

    auto mockClient = std::make_shared<MockRdmaChannelClient>(::accl::barex::TCP_V4, ip, port, timerManager, wrapper);
    auto client     = std::dynamic_pointer_cast<RdmaChannelClient>(mockClient);

    std::shared_ptr<::accl::barex::XConnection> connection;
    ::accl::barex::XMessage                     xmessage;
    EXPECT_CALL(*mockClient, ConnectSuccess(connection)).WillOnce(Return());

    auto callback = std::make_shared<RdmaChannelClientCallback>(client, _mempool);
    ASSERT_FALSE(callback->_channel.expired());
    callback->OnConnectSuccess(connection);
}

TEST_F(RdmaChannelClientCallbackTestOnMock, testOnConnectFailed_ExpiredChannel) {
    std::shared_ptr<RdmaChannelClientCallback> callback;
    {
        std::string                        ip   = "123.4.5.6";
        int32_t                            port = 12345;
        std::shared_ptr<TimerManager>      timerManager;
        std::shared_ptr<AcclClientWrapper> wrapper;

        auto mockClient =
            std::make_shared<MockRdmaChannelClient>(::accl::barex::TCP_V4, ip, port, timerManager, wrapper);
        auto client = std::dynamic_pointer_cast<RdmaChannelClient>(mockClient);

        callback = std::make_shared<RdmaChannelClientCallback>(client, _mempool);
    }
    ASSERT_TRUE(callback->_channel.expired());
    std::shared_ptr<::accl::barex::XConnection> connection;
    callback->OnConnectFailed(connection);
}

TEST_F(RdmaChannelClientCallbackTestOnMock, testOnConnectFailed_NullChannel) {
    std::shared_ptr<::accl::barex::XConnection> connection;
    ::accl::barex::XMessage                     xmessage;

    std::shared_ptr<RdmaChannelClient> client;
    auto                               callback = std::make_shared<RdmaChannelClientCallback>(client, _mempool);
    callback->OnConnectFailed(connection);
}

TEST_F(RdmaChannelClientCallbackTestOnMock, testOnConnectFailed_Success) {
    std::string                        ip   = "123.4.5.6";
    int32_t                            port = 12345;
    std::shared_ptr<TimerManager>      timerManager;
    std::shared_ptr<AcclClientWrapper> wrapper;

    auto mockClient = std::make_shared<MockRdmaChannelClient>(::accl::barex::TCP_V4, ip, port, timerManager, wrapper);
    auto client     = std::dynamic_pointer_cast<RdmaChannelClient>(mockClient);

    std::shared_ptr<::accl::barex::XConnection> connection;
    ::accl::barex::XMessage                     xmessage;
    EXPECT_CALL(*mockClient, ConnectFailed(connection)).WillOnce(Return());

    auto callback = std::make_shared<RdmaChannelClientCallback>(client, _mempool);
    ASSERT_FALSE(callback->_channel.expired());
    callback->OnConnectFailed(connection);
}

ARPC_END_NAMESPACE(arpc);