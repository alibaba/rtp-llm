#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/AcclClientWrapper.h"
#include "aios/network/rdma/accl/RdmaChannelClient.h"
#include "aios/network/rdma/accl/RdmaMempoolFactory.h"
#include "aios/network/rdma/accl/TimerManager.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaChannelClosure.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaRequest.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaServer.h"
#include "autil/NetUtil.h"

using namespace ::testing;

ARPC_BEGIN_NAMESPACE(arpc);

class RdmaChannelClientTest: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    std::shared_ptr<RdmaChannelClient> startClient(int port);
    std::shared_ptr<FakeRdmaServer>    startServer(int port);

protected:
    std::shared_ptr<TimerManager>      _timerManager;
    std::shared_ptr<AcclClientWrapper> _wrapper;
};

void RdmaChannelClientTest::setUp() {
    _timerManager = std::make_shared<TimerManager>();
    RdmaClientConfig config{1, 1};
    _wrapper = std::make_shared<AcclClientWrapper>(config);
    ASSERT_TRUE(_wrapper->Init());
}

void RdmaChannelClientTest::tearDown() {}

std::shared_ptr<RdmaChannelClient> RdmaChannelClientTest::startClient(int port) {
    auto client = std::make_shared<RdmaChannelClient>(
        ::accl::barex::TcpProtoType::TCP_V4, "localhost", port, _timerManager, _wrapper);
    if (!client) {
        return client;
    }

    client->Init();

    // wait connect attemp
    usleep(1000 * 1000);  // wait 1s
    return client;
}

std::shared_ptr<FakeRdmaServer> RdmaChannelClientTest::startServer(int port) {
    auto server = std::make_shared<FakeRdmaServer>(RdmaServerConfig{1, 1});
    if (!server) {
        return server;
    }

    if (!server->Listen(port)) {
        server.reset();
    }
    usleep(1000 * 1000);  // wait 1s
    return server;
}

TEST_F(RdmaChannelClientTest, testBeforConnect_DoDisconnect) {
    auto client = std::make_shared<RdmaChannelClient>(
        ::accl::barex::TcpProtoType::TCP_V4, "localhost", 12345, _timerManager, _wrapper);
    ASSERT_TRUE(client);

    EXPECT_FALSE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());

    // do nothing
    client->Disconnect();

    EXPECT_TRUE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());
}

TEST_F(RdmaChannelClientTest, testBeforeConnect_DoPost) {
    auto client = std::make_shared<RdmaChannelClient>(
        ::accl::barex::TcpProtoType::TCP_V4, "localhost", 12345, _timerManager, _wrapper);
    ASSERT_TRUE(client);

    EXPECT_FALSE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());

    auto request = std::make_shared<FakeRdmaRequest>();
    request->SetValue("request");
    request->SetTimeoutMs(100);

    auto closure = std::make_shared<FakeRdmaChannelClosure>();

    client->Post(request, closure, nullptr);
    EXPECT_TRUE(closure->Done());
    EXPECT_EQ(RdmaChannelClosure::ERROR_CONNECTION_FAILED, closure->State());

    EXPECT_FALSE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());
}

TEST_F(RdmaChannelClientTest, testConnectFailed_DoDisconnect) {
    auto client = startClient(autil::NetUtil::randomPort());
    ASSERT_TRUE(client);
    EXPECT_TRUE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());

    client->Disconnect();

    EXPECT_TRUE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());
}

TEST_F(RdmaChannelClientTest, testConnectFailed_DoConnect) {
    auto client = startClient(autil::NetUtil::randomPort());
    ASSERT_TRUE(client);
    EXPECT_TRUE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());

    client->Init();

    EXPECT_TRUE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());
}

TEST_F(RdmaChannelClientTest, testConnectFailed_DoPost) {
    auto client = startClient(autil::NetUtil::randomPort());
    ASSERT_TRUE(client);
    EXPECT_TRUE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());

    auto request = std::make_shared<FakeRdmaRequest>();
    request->SetValue("request");
    request->SetTimeoutMs(100);

    auto closure = std::make_shared<FakeRdmaChannelClosure>();

    client->Post(request, closure, nullptr);
    EXPECT_TRUE(closure->Done());
    EXPECT_EQ(RdmaChannelClosure::ERROR_CONNECTION_FAILED, closure->State());
}

TEST_F(RdmaChannelClientTest, testConnectSuccess_DoDisconnect_Client) {
    auto port   = autil::NetUtil::randomPort();
    auto server = startServer(port);
    ASSERT_TRUE(server);

    auto client = startClient(port);
    ASSERT_TRUE(client);
    usleep(1000 * 1000);
    ASSERT_FALSE(client->ChannelBroken());
    ASSERT_TRUE(client->ChannelConnected());

    client->Disconnect();
    usleep(1000 * 1000);
    EXPECT_TRUE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());
}

TEST_F(RdmaChannelClientTest, testConnectSuccess_DoDisconnect_Server) {
    auto port   = autil::NetUtil::randomPort();
    auto server = startServer(port);
    ASSERT_TRUE(server);

    auto client = startClient(port);
    ASSERT_TRUE(client);
    ASSERT_FALSE(client->ChannelBroken());
    ASSERT_TRUE(client->ChannelConnected());

    server.reset();
    for (int i = 0; i < 20; i++) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        if (client->ChannelBroken()) {
            break;
        }
    }
    EXPECT_TRUE(client->ChannelBroken());
    EXPECT_FALSE(client->ChannelConnected());
}

TEST_F(RdmaChannelClientTest, testConnectSuccess_DoConnect) {
    auto port   = autil::NetUtil::randomPort();
    auto server = startServer(port);
    ASSERT_TRUE(server);

    auto client = startClient(port);
    ASSERT_TRUE(client);
    ASSERT_FALSE(client->ChannelBroken());
    ASSERT_TRUE(client->ChannelConnected());

    auto connection = client->_connection;
    client->Init();
    EXPECT_FALSE(client->ChannelBroken());
    EXPECT_TRUE(client->ChannelConnected());
    EXPECT_EQ(connection, client->_connection);
}

TEST_F(RdmaChannelClientTest, testConnectSuccess_DoPost) {
    auto port   = autil::NetUtil::randomPort();
    auto server = startServer(port);
    ASSERT_TRUE(server);

    auto client = startClient(port);
    ASSERT_TRUE(client);
    ASSERT_FALSE(client->ChannelBroken());
    ASSERT_TRUE(client->ChannelConnected());

    auto request = std::make_shared<FakeRdmaRequest>();
    request->SetValue("request");
    request->SetTimeoutMs(2000);

    auto closure = std::make_shared<FakeRdmaChannelClosure>();

    client->Post(request, closure, nullptr);
    for (int i = 0; i < 200; i++) {
        usleep(100 * 1000);
        if (closure->Done()) {
            break;
        }
    }
    EXPECT_TRUE(closure->Done());
    EXPECT_EQ(RdmaChannelClosure::ERROR_NONE, closure->State());

    client.reset();
    server.reset();
}

ARPC_END_NAMESPACE(arpc);
