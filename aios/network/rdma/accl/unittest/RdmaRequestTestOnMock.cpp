#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/RdmaRequest.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaRequest.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"

using namespace ::testing;

ARPC_BEGIN_NAMESPACE(arpc);

TEST(RdmaRequestTestOnMock, SetPeerAddress_nullConnection) {
    FakeRdmaRequest request;
    EXPECT_EQ("unknown", request.GetPeerAddress());

    request.SetPeerAddress(std::shared_ptr<::accl::barex::XConnection>());
    EXPECT_EQ("unknown", request.GetPeerAddress());
}

TEST(RdmaRequestTestOnMock, SetPeerAddress_emptyAddress) {
    FakeRdmaRequest request;
    EXPECT_EQ("unknown", request.GetPeerAddress());

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, GetPeerAddrAndPort).WillOnce(Return(std::pair<std::string, int>("", -1)));

    request.SetPeerAddress(connection);
    EXPECT_EQ("unknown", request.GetPeerAddress());
}

TEST(RdmaRequestTestOnMock, SetPeerAddress_success) {
    FakeRdmaRequest request;
    EXPECT_EQ("unknown", request.GetPeerAddress());

    auto mockConnection = std::make_shared<::accl::barex::MockXConnection>();
    auto connection     = std::dynamic_pointer_cast<::accl::barex::XConnection>(mockConnection);
    EXPECT_CALL(*mockConnection, GetPeerAddrAndPort).WillOnce(Return(std::pair<std::string, int>("127.0.1.2", 12345)));

    request.SetPeerAddress(connection);
    EXPECT_EQ("127.0.1.2:12345", request.GetPeerAddress());
}

ARPC_END_NAMESPACE(arpc);