#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/AcclClientWrapper.h"
#include "aios/network/rdma/accl/RdmaMempool.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"
#include "aios/network/rdma/accl/unittest/mock/MockXClientTransport.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"
#include "aios/network/rdma/accl/unittest/mock/MockXSimpleMempool.h"

using namespace ::testing;
ARPC_BEGIN_NAMESPACE(arpc);

class AcclClientWrapperTestOnMock: public TESTBASE {
public:
    void setUp();
    void tearDown();
};

void AcclClientWrapperTestOnMock::setUp() {
    RdmaMempoolFactory::EnableInit(false);
}

void AcclClientWrapperTestOnMock::tearDown() {
    RdmaMempoolFactory::EnableInit(true);
    RdmaMempoolFactory::SetMempool(std::shared_ptr<RdmaMempool>());
}

TEST_F(AcclClientWrapperTestOnMock, testInit_GetMempoolFailed) {
    RdmaClientConfig  config{1, 1};
    AcclClientWrapper wrapper(config);
    EXPECT_TRUE(RdmaMempoolFactory::GetMempool() == nullptr);
    EXPECT_FALSE(wrapper.Init());
}

TEST_F(AcclClientWrapperTestOnMock, testInit_GetXMempoolFailed) {
    RdmaClientConfig  config{1, 1};
    AcclClientWrapper wrapper(config);

    auto mempool = std::make_shared<RdmaMempool>();
    RdmaMempoolFactory::SetMempool(mempool);
    ASSERT_TRUE(RdmaMempoolFactory::GetMempool() != nullptr);
    EXPECT_TRUE(RdmaMempoolFactory::GetMempool()->GetXMempool() == nullptr);

    EXPECT_FALSE(wrapper.Init());
}

TEST_F(AcclClientWrapperTestOnMock, testConnect_NoClient) {
    RdmaClientConfig                                    config{1, 1};
    AcclClientWrapper                                   wrapper(config);
    std::shared_ptr<::accl::barex::XConnectionCallback> callback;
    EXPECT_FALSE(wrapper.Connect(::accl::barex::TcpProtoType::TCP_V4, "127.0.0.1", 12345, callback));
}

TEST_F(AcclClientWrapperTestOnMock, testConnect_Success) {
    std::string                                         ip   = "121.1.1.1";
    int                                                 port = 12345;
    std::shared_ptr<::accl::barex::XConnectionCallback> callback;

    auto mockTransport = std::make_shared<::accl::barex::MockXClientTransport>();
    auto transport     = std::dynamic_pointer_cast<::accl::barex::XClientTransport>(mockTransport);
    EXPECT_CALL(*mockTransport, Connect(::accl::barex::TcpProtoType::TCP_V4, ip, port, callback)).WillOnce(Return());

    RdmaClientConfig  config{1, 1};
    AcclClientWrapper wrapper(config);
    wrapper._client = transport;

    EXPECT_TRUE(wrapper.Connect(::accl::barex::TcpProtoType::TCP_V4, ip, port, callback));
}

TEST_F(AcclClientWrapperTestOnMock, testConnect_Success_V6) {
    std::string                                         ip   = "121.1.1.1";
    int                                                 port = 12345;
    std::shared_ptr<::accl::barex::XConnectionCallback> callback;

    auto mockTransport = std::make_shared<::accl::barex::MockXClientTransport>();
    auto transport     = std::dynamic_pointer_cast<::accl::barex::XClientTransport>(mockTransport);
    EXPECT_CALL(*mockTransport, Connect(::accl::barex::TcpProtoType::TCP_V6, ip, port, callback)).WillOnce(Return());

    RdmaClientConfig  config{1, 1};
    AcclClientWrapper wrapper(config);
    wrapper._client = transport;

    EXPECT_TRUE(wrapper.Connect(::accl::barex::TcpProtoType::TCP_V6, ip, port, callback));
}

TEST_F(AcclClientWrapperTestOnMock, testDisconnect_NoClient) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    RdmaClientConfig  config{1, 1};
    AcclClientWrapper wrapper(config);

    wrapper.Disconnect(connection);
}

TEST_F(AcclClientWrapperTestOnMock, testDisconnect_NoConnection) {
    std::shared_ptr<::accl::barex::XConnection> connection;

    auto mockTransport = std::make_shared<::accl::barex::MockXClientTransport>();
    auto transport     = std::dynamic_pointer_cast<::accl::barex::XClientTransport>(mockTransport);

    RdmaClientConfig  config{1, 1};
    AcclClientWrapper wrapper(config);
    wrapper._client = transport;

    wrapper.Disconnect(connection);
}

TEST_F(AcclClientWrapperTestOnMock, testDisconnect_ReturnFailed) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    auto mockTransport = std::make_shared<::accl::barex::MockXClientTransport>();
    auto transport     = std::dynamic_pointer_cast<::accl::barex::XClientTransport>(mockTransport);
    EXPECT_CALL(*mockTransport, Disconnect(connection)).WillOnce(Return(false));

    RdmaClientConfig  config{1, 1};
    AcclClientWrapper wrapper(config);
    wrapper._client = transport;

    wrapper.Disconnect(connection);
}

TEST_F(AcclClientWrapperTestOnMock, testDisconnect_Success) {
    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);

    auto mockTransport = std::make_shared<::accl::barex::MockXClientTransport>();
    auto transport     = std::dynamic_pointer_cast<::accl::barex::XClientTransport>(mockTransport);

    EXPECT_CALL(*mockTransport, Disconnect(connection)).WillOnce(Return(true));

    RdmaClientConfig  config{1, 1};
    AcclClientWrapper wrapper(config);
    wrapper._client = transport;

    wrapper.Disconnect(connection);
}

ARPC_END_NAMESPACE(arpc);