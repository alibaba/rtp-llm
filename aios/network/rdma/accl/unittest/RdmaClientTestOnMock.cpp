#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/AcclClientWrapper.h"
#include "aios/network/rdma/accl/RdmaClient.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"
#include "aios/network/rdma/accl/unittest/mock/MockXClientTransport.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

using namespace ::testing;
ARPC_BEGIN_NAMESPACE(arpc);

class RdmaClientTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    std::shared_ptr<::accl::barex::MockXSimpleMempool> _mockXMempool;
    std::shared_ptr<RdmaMempool>                       _mempool;
};

void RdmaClientTestOnMock::setUp() {
    RdmaMempoolFactory::EnableInit(false);
    _mockXMempool = std::make_shared<::accl::barex::MockXSimpleMempool>();
    EXPECT_CALL(*_mockXMempool, Shutdown()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));
    EXPECT_CALL(*_mockXMempool, WaitStop()).WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    _mempool = std::make_shared<RdmaMempool>();
    _mempool->_curState.store(RdmaMempool::INIT_SUCCESS);
    _mempool->_xmempool = std::dynamic_pointer_cast<::accl::barex::XSimpleMempool>(_mockXMempool);
}

void RdmaClientTestOnMock::tearDown() {
    RdmaMempoolFactory::EnableInit(true);
    RdmaMempoolFactory::SetMempool(std::shared_ptr<RdmaMempool>());
}

TEST_F(RdmaClientTestOnMock, testInit_ClientInitFailed) {
    RdmaClientConfig config{1, 1};
    RdmaClient       client(config);
    ASSERT_FALSE(client.Init());
}

TEST_F(RdmaClientTestOnMock, testOpenChannel_ParseSepcFailed) {
    RdmaClientConfig config{1, 1};
    RdmaClient       client(config);

    ASSERT_FALSE(client.OpenChannel("rrrrrr"));
}

TEST_F(RdmaClientTestOnMock, testOpenChannel_Success) {
    auto mockTransport = std::make_shared<::accl::barex::MockXClientTransport>();
    auto transport     = std::dynamic_pointer_cast<::accl::barex::XClientTransport>(mockTransport);
    EXPECT_CALL(*mockTransport, Connect(::accl::barex::TcpProtoType::TCP_V4, "12.23.4.5", 12345, _)).WillOnce(Return());

    RdmaClientConfig config{1, 1};
    auto             wrapper = std::make_shared<AcclClientWrapper>(config);
    wrapper->_client         = transport;
    wrapper->_mempool        = _mempool;

    RdmaClient client(config);
    client._client = wrapper;
    auto channel   = client.OpenChannel("tcp:12.23.4.5:12345");
    ASSERT_TRUE(channel != nullptr);
}

ARPC_END_NAMESPACE(arpc);