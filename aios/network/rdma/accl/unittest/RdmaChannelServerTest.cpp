#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/RdmaChannelClient.h"
#include "aios/network/rdma/accl/RdmaChannelServer.h"
#include "aios/network/rdma/accl/RdmaClient.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaChannelClosure.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaRequest.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaServer.h"
#include "autil/NetUtil.h"

using namespace ::testing;

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelServerTest: public TESTBASE {
public:
    void setUp();
    void tearDown();
};

void RdmaChannelServerTest::setUp() {}

void RdmaChannelServerTest::tearDown() {}

TEST_F(RdmaChannelServerTest, testListenFailed) {
    auto server = std::make_shared<FakeRdmaServer>(RdmaServerConfig{1, 1});
    ASSERT_TRUE(server);

    EXPECT_FALSE(server->Listen(123456));
}

TEST_F(RdmaChannelServerTest, testOnConnect_ClientDisconnect) {
    auto port = autil::NetUtil::randomPort();

    auto server = std::make_shared<FakeRdmaServer>(RdmaServerConfig{1, 1});
    ASSERT_TRUE(server);
    ASSERT_TRUE(server->Listen(port));
    usleep(1000 * 1000);

    RdmaClientConfig config{1, 1};
    auto             client = std::make_shared<RdmaClient>(config);
    ASSERT_TRUE(client->Init());
    auto channel = client->OpenChannel("tcp:localhost:" + std::to_string(port));
    usleep(1000 * 1000);
    EXPECT_TRUE(channel->ChannelConnected());

    channel.reset();
    usleep(1000 * 1000);
}

TEST_F(RdmaChannelServerTest, testOnConnect_ServerDisconnect) {
    auto port = autil::NetUtil::randomPort();

    auto server = std::make_shared<FakeRdmaServer>(RdmaServerConfig{1, 1});
    ASSERT_TRUE(server);
    ASSERT_TRUE(server->Listen(port));
    usleep(1000 * 1000);

    RdmaClientConfig config{1, 1};
    auto             client = std::make_shared<RdmaClient>(config);
    ASSERT_TRUE(client->Init());
    auto channel = client->OpenChannel("tcp:localhost:" + std::to_string(port));
    usleep(1000 * 1000);
    EXPECT_TRUE(channel->ChannelConnected());

    server.reset();
    usleep(1000 * 1000);
}

TEST_F(RdmaChannelServerTest, testOnConnect_RecvRequest) {
    auto port = autil::NetUtil::randomPort();

    auto server = std::make_shared<FakeRdmaServer>(RdmaServerConfig{1, 1});
    ASSERT_TRUE(server);
    ASSERT_TRUE(server->Listen(port));
    usleep(1000 * 1000);

    RdmaClientConfig config{1, 1};
    auto             client = std::make_shared<RdmaClient>(config);
    ASSERT_TRUE(client->Init());
    auto channel = client->OpenChannel("tcp:localhost:" + std::to_string(port));
    usleep(1000 * 1000);
    EXPECT_TRUE(channel->ChannelConnected());

    auto request = std::make_shared<FakeRdmaRequest>();
    request->SetValue("request");
    request->SetTimeoutMs(100);

    auto closure = std::make_shared<FakeRdmaChannelClosure>();
    channel->Post(request, closure, nullptr);

    usleep(400 * 1000);
    EXPECT_TRUE(closure->Done());
}

ARPC_END_NAMESPACE(arpc);