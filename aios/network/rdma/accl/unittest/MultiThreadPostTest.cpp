#include "unittest/unittest.h"

// force unitetst header first, change private class member
#include "aios/network/rdma/accl/RdmaChannelClient.h"
#include "aios/network/rdma/accl/RdmaClient.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaChannelClosure.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaRequest.h"
#include "aios/network/rdma/accl/unittest/FakeRdmaServer.h"
#include "autil/NetUtil.h"

using namespace ::testing;

ARPC_BEGIN_NAMESPACE(arpc);

class MultiThreadPostTest: public TESTBASE {
public:
    void setUp();
    void tearDown();

protected:
    std::shared_ptr<RdmaClient>     startClient();
    std::shared_ptr<FakeRdmaServer> startServer(int port);
    std::vector<autil::ThreadPtr>   startBenchThreads(std::shared_ptr<RdmaChannelClient> client,
                                                      int                                clientCount,
                                                      int                                waitTimeUs,
                                                      std::atomic<int64_t>&              sendCount,
                                                      std::atomic<int64_t>&              successCount,
                                                      std::atomic<int64_t>&              timeoutCount,
                                                      std::atomic<int64_t>&              errorCount);
};

void MultiThreadPostTest::setUp() {}

void MultiThreadPostTest::tearDown() {}

std::shared_ptr<RdmaClient> MultiThreadPostTest::startClient() {
    RdmaClientConfig config{2, 2};
    auto             client = std::make_shared<RdmaClient>(config);
    if (!client->Init()) {
        return std::shared_ptr<RdmaClient>();
    }
    return client;
}

std::shared_ptr<FakeRdmaServer> MultiThreadPostTest::startServer(int port) {
    auto server = std::make_shared<FakeRdmaServer>(RdmaServerConfig{2, 2});
    if (!server) {
        return server;
    }

    if (!server->Listen(port)) {
        server.reset();
    }
    usleep(1000 * 1000);  // wait 1s
    return server;
}

std::vector<autil::ThreadPtr> MultiThreadPostTest::startBenchThreads(std::shared_ptr<RdmaChannelClient> client,
                                                                     int                                clientCount,
                                                                     int                                waitTimeUs,
                                                                     std::atomic<int64_t>&              sendCount,
                                                                     std::atomic<int64_t>&              successCount,
                                                                     std::atomic<int64_t>&              timeoutCount,
                                                                     std::atomic<int64_t>&              errorCount) {
    std::vector<autil::ThreadPtr> threads;
    for (int i = 0; i < clientCount; i++) {
        auto thread =
            autil::Thread::createThread([client, waitTimeUs, &sendCount, &successCount, &timeoutCount, &errorCount]() {
                while (!client->ChannelBroken()) {
                    auto request = std::make_shared<FakeRdmaRequest>();
                    request->SetValue("request");
                    request->SetTimeoutMs(100);

                    auto closure = std::make_shared<FakeRdmaChannelClosure>(
                        [&successCount, &timeoutCount, &errorCount](RdmaChannelClosure::ErrorState state) {
                            if (state == RdmaChannelClosure::ERROR_NONE) {
                                successCount++;
                            } else if (state == RdmaChannelClosure::ERROR_TIMEOUT) {
                                timeoutCount++;
                            } else {
                                errorCount++;
                            }
                        });
                    sendCount++;
                    client->Post(request, closure, nullptr);
                    usleep(waitTimeUs);
                }
            });
        threads.push_back(thread);
    }
    return threads;
}

TEST_F(MultiThreadPostTest, testMultiThreadPost_ClientDisconnect) {
    AUTIL_ROOT_LOG_SETLEVEL(WARN);

    auto port   = autil::NetUtil::randomPort();
    auto server = startServer(port);
    ASSERT_TRUE(server);

    auto client = startClient();
    ASSERT_TRUE(client);

    auto channel = client->OpenChannel("tcp:localhost:" + std::to_string(port));
    ASSERT_TRUE(channel);

    std::atomic<int64_t> sendCount(0);
    std::atomic<int64_t> successCount(0);
    std::atomic<int64_t> timeoutCount(0);
    std::atomic<int64_t> errorCount(0);

    std::vector<autil::ThreadPtr> threads =
        startBenchThreads(channel, 10, 1000 * 10, sendCount, successCount, timeoutCount, errorCount);

    usleep(5 * 1000 * 1000);

    channel->Disconnect();
    channel.reset();
    usleep(1000 * 1000);

    for (auto& thread : threads) {
        thread->join();
    }

    usleep(1000 * 1000);

    EXPECT_EQ(sendCount.load(), successCount.load() + timeoutCount.load() + errorCount.load());
    AUTIL_ROOT_LOG_SETLEVEL(INFO);
}

TEST_F(MultiThreadPostTest, testMultiThreadPost_ServerDisconnect) {
    AUTIL_ROOT_LOG_SETLEVEL(WARN);

    auto port   = autil::NetUtil::randomPort();
    auto server = startServer(port);
    ASSERT_TRUE(server);

    auto client = startClient();
    ASSERT_TRUE(client);

    auto channel = client->OpenChannel("tcp:localhost:" + std::to_string(port));
    ASSERT_TRUE(channel);

    std::atomic<int64_t> sendCount(0);
    std::atomic<int64_t> successCount(0);
    std::atomic<int64_t> timeoutCount(0);
    std::atomic<int64_t> errorCount(0);

    std::vector<autil::ThreadPtr> threads =
        startBenchThreads(channel, 10, 10 * 1000, sendCount, successCount, timeoutCount, errorCount);

    usleep(5 * 1000 * 1000);
    server.reset();
    usleep(1000 * 1000);

    for (auto& thread : threads) {
        thread->join();
    }

    EXPECT_TRUE(channel->ChannelBroken());
    usleep(1000 * 1000);

    EXPECT_EQ(sendCount.load(), successCount.load() + timeoutCount.load() + errorCount.load());
    AUTIL_ROOT_LOG_SETLEVEL(INFO);
}

ARPC_END_NAMESPACE(arpc);