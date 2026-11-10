#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/arpc/arpc/metric/mock/MockServerMetricReporter.h"
#include "aios/network/rdma/accl/AcclServerWrapper.h"
#include "aios/network/rdma/accl/RdmaChannelServerCallback.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/unittest/mock/MockIServerAdapter.h"
#include "aios/network/rdma/accl/unittest/mock/MockRdmaChannelServer.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

using namespace ::testing;

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelServerCallbackTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    std::shared_ptr<RdmaMempool>                       _mempool;
    std::shared_ptr<::accl::barex::MockXSimpleMempool> _mockXMempool;
};

void RdmaChannelServerCallbackTestOnMock::setUp() {
    _mockXMempool = std::make_shared<::accl::barex::MockXSimpleMempool>();
    EXPECT_CALL(*_mockXMempool, Shutdown()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, WaitStop()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool = std::make_shared<RdmaMempool>();
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    _mempool->_xmempool = std::dynamic_pointer_cast<::accl::barex::XSimpleMempool>(_mockXMempool);
}

void RdmaChannelServerCallbackTestOnMock::tearDown() {}

TEST_F(RdmaChannelServerCallbackTestOnMock, testOnRecvCall_ExpiredChannel) {
    _mockXMempool->MockOnAnyCallAllocAndReleaseBuffer();
    std::shared_ptr<RdmaChannelServerCallback> callback;
    {
        std::string                         address = "tcp:123.4.5.6:12345";
        std::shared_ptr<IRdmaServerAdapter> adapter;
        std::shared_ptr<AcclServerWrapper>  wrapper;

        auto mockServer = std::make_shared<MockRdmaChannelServer>(address, adapter, wrapper);
        auto server     = std::dynamic_pointer_cast<RdmaChannelServer>(mockServer);
        callback        = std::make_shared<RdmaChannelServerCallback>(server, _mempool);
    }
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    ::accl::barex::XMessage xmessage;
    auto                    buf = _mempool->AllocBuffer(1024);
    xmessage.Add(buf, 1024, accl::barex::CPU, accl::barex::CPU);
    callback->OnRecvCall(connection, xmessage);
}

TEST_F(RdmaChannelServerCallbackTestOnMock, testOnRecvCall_Success) {
    std::string                         address = "ctp:123.4.5.6:12345";
    std::shared_ptr<IRdmaServerAdapter> adapter;
    std::shared_ptr<AcclServerWrapper>  wrapper;

    auto mockServer = std::make_shared<MockRdmaChannelServer>(address, adapter, wrapper);
    auto server     = std::dynamic_pointer_cast<RdmaChannelServer>(mockServer);

    auto callback = std::make_shared<RdmaChannelServerCallback>(server, _mempool);

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    ::accl::barex::XMessage xmessage;

    EXPECT_CALL(*mockServer, HandleRecvRequest(connection, _)).WillOnce(Return());
    callback->OnRecvCall(connection, xmessage);
}

ARPC_END_NAMESPACE(arpc);