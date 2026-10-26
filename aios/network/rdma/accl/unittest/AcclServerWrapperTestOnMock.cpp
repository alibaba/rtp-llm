#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/accl-barex/include/accl/barex/highlevel/xmessage.h"
#include "aios/network/rdma/accl/AcclServerWrapper.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"
#include "aios/network/rdma/accl/unittest/mock/MockXServerTransport.h"

using namespace ::testing;
ARPC_BEGIN_NAMESPACE(arpc);

class AcclServerWrapperTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();
};

void AcclServerWrapperTestOnMock::setUp() {
    RdmaMempoolFactory::EnableInit(false);
}

void AcclServerWrapperTestOnMock::tearDown() {
    RdmaMempoolFactory::EnableInit(true);
    RdmaMempoolFactory::SetMempool(std::shared_ptr<RdmaMempool>());
}

TEST_F(AcclServerWrapperTestOnMock, testInit_GetMempoolFailed) {
    AcclServerWrapper wrapper(RdmaServerConfig{1, 1});
    EXPECT_TRUE(RdmaMempoolFactory::GetMempool() == nullptr);
    EXPECT_FALSE(wrapper.Init());
}

TEST_F(AcclServerWrapperTestOnMock, testInit_GetXMempoolFailed) {
    AcclServerWrapper wrapper(RdmaServerConfig{1, 1});

    auto mempool = std::make_shared<RdmaMempool>();
    RdmaMempoolFactory::SetMempool(mempool);
    ASSERT_TRUE(RdmaMempoolFactory::GetMempool() != nullptr);
    EXPECT_TRUE(RdmaMempoolFactory::GetMempool()->GetXMempool() == nullptr);

    EXPECT_FALSE(wrapper.Init());
}

TEST_F(AcclServerWrapperTestOnMock, testListen_NoInit) {
    int                                                 port = 12345;
    std::shared_ptr<::accl::barex::XConnectionCallback> callback;

    AcclServerWrapper wrapper(RdmaServerConfig{1, 1});
    EXPECT_FALSE(wrapper.Listen(::accl::barex::TcpProtoType::TCP_V4, port, callback));
}

TEST_F(AcclServerWrapperTestOnMock, testListen_ListenFailed) {
    auto mockTransport = std::make_shared<::accl::barex::MockXServerTransport>();
    auto transport     = std::dynamic_pointer_cast<::accl::barex::XServerTransport>(mockTransport);

    int                                                 port = 12345;
    std::shared_ptr<::accl::barex::XConnectionCallback> callback;
    EXPECT_CALL(*mockTransport, Listen(::accl::barex::TcpProtoType::TCP_V4, port, callback))
        .WillOnce(Return(::accl::barex::BAREX_ERR_ARG));

    AcclServerWrapper wrapper(RdmaServerConfig{1, 1});
    wrapper._server = transport;
    EXPECT_FALSE(wrapper.Listen(::accl::barex::TcpProtoType::TCP_V4, port, callback));
}

TEST_F(AcclServerWrapperTestOnMock, testListen_Success) {
    auto mockTransport = std::make_shared<::accl::barex::MockXServerTransport>();
    auto transport     = std::dynamic_pointer_cast<::accl::barex::XServerTransport>(mockTransport);

    int                                                 port = 12345;
    std::shared_ptr<::accl::barex::XConnectionCallback> callback;
    EXPECT_CALL(*mockTransport, Listen(::accl::barex::TcpProtoType::TCP_V4, port, callback))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    AcclServerWrapper wrapper(RdmaServerConfig{1, 1});
    wrapper._server = transport;
    EXPECT_TRUE(wrapper.Listen(::accl::barex::TcpProtoType::TCP_V4, port, callback));
}

TEST_F(AcclServerWrapperTestOnMock, testListen_Success_V6) {
    auto mockTransport = std::make_shared<::accl::barex::MockXServerTransport>();
    auto transport     = std::dynamic_pointer_cast<::accl::barex::XServerTransport>(mockTransport);

    int                                                 port = 12345;
    std::shared_ptr<::accl::barex::XConnectionCallback> callback;
    EXPECT_CALL(*mockTransport, Listen(::accl::barex::TcpProtoType::TCP_V6, port, callback))
        .WillOnce(Return(::accl::barex::BAREX_SUCCESS));

    AcclServerWrapper wrapper(RdmaServerConfig{1, 1});
    wrapper._server = transport;
    EXPECT_TRUE(wrapper.Listen(::accl::barex::TcpProtoType::TCP_V6, port, callback));
}
ARPC_END_NAMESPACE(arpc);
