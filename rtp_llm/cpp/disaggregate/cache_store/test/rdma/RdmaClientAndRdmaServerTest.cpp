#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClient.h"
#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServer.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "autil/NetUtil.h"

namespace rtp_llm {

class RdmaClientAndRdmaServerTest: public CacheStoreTestBase {};

TEST_F(RdmaClientAndRdmaServerTest, testRun) {
    if (!memory_util_->isRdmaMode()) {
        return;
    }
    ASSERT_TRUE(memory_util_ != nullptr);

    auto port = autil::NetUtil::randomPort();

    auto rdma_server = std::make_shared<RdmaServer>(memory_util_);
    ASSERT_TRUE(rdma_server->init(port, 1, 1));

    auto rdma_client = std::make_shared<RdmaClient>(memory_util_);
    ASSERT_TRUE(rdma_client->init(1, 1));

    // connect first time
    auto local_ip         = autil::NetUtil::getBindIp();
    auto rdma_connection1 = rdma_client->getConnection(local_ip, port);
    ASSERT_TRUE(rdma_connection1 != nullptr);

    auto all_connections = rdma_connection1->getAllConnections();
    ASSERT_EQ(2, all_connections.size());
    std::set<std::shared_ptr<::accl::barex::XConnection>> connections;
    for (auto& accl_connection : all_connections) {
        ASSERT_EQ(local_ip, accl_connection->GetPeerAddrAndPort().first);
        connections.insert(accl_connection);
    }

    // connect second time
    auto rdma_connection2 = rdma_client->getConnection(local_ip, port);
    ASSERT_TRUE(rdma_connection2 != nullptr);
    ASSERT_NE(rdma_connection1, rdma_connection2);
    all_connections = rdma_connection2->getAllConnections();
    for (auto& accl_connection : all_connections) {
        ASSERT_TRUE(connections.find(accl_connection) == connections.end());
    }

    // recycle conection
    rdma_client->recycleConnection(rdma_connection1);
    auto rdma_connection3 = rdma_client->getConnection(local_ip, port);
    ASSERT_NE(rdma_connection1, rdma_connection3);
    all_connections = rdma_connection3->getAllConnections();
    for (auto& accl_connection : all_connections) {
        ASSERT_TRUE(connections.find(accl_connection) != connections.end());
    }

    // client destroy connections
    rdma_client->destroyConnection(rdma_connection3);

    // server destroy connections
    rdma_server.reset();

    // make sure rdma client not destrory so fast
    usleep(10 * 1000);
    rdma_client.reset();
}

}  // namespace rtp_llm