#include "gtest/gtest.h"

#include <thread>

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaClientHostConnections.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"
#include "aios/network/rdma/accl/unittest/mock/MockXClientTransport.h"
#include "aios/network/accl-barex/include/accl/barex/highlevel/xtransport_config.h"

#include "autil/NetUtil.h"

using namespace testing;

namespace rtp_llm {

class RdmaClientHostConnectionsTest: public CacheStoreTestBase {};

TEST_F(RdmaClientHostConnectionsTest, testConnectSuccessOrFailed) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections =
        std::make_shared<RdmaClientHostConnections>(memory_util_, autil::NetUtil::getBindIp(), 12345);

    // connect failed on first time
    std::shared_ptr<::accl::barex::XConnection> connection(new ::accl::barex::MockXConnection);
    host_connections->connectFailed(connection);

    // connect success on first time
    host_connections->connectSuccess(connection);
    ASSERT_EQ(1, host_connections->newly_connections_.size());

    // connect failed later
    host_connections->connectFailed(connection);
    ASSERT_EQ(0, host_connections->newly_connections_.size());
}

TEST_F(RdmaClientHostConnectionsTest, testConnectSuccessOrFailed_Busy) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(0);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    // connect failed on first time
    std::shared_ptr<::accl::barex::XConnection> connection(new ::accl::barex::MockXConnection);
    host_connections->connectFailed(connection);

    // connect success on first time
    host_connections->connectSuccess(connection);
    ASSERT_EQ(1, host_connections->newly_connections_.size());

    // get rdma connection
    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(1, rdma_connection->getConnectionsCount());
    ASSERT_EQ(0, host_connections->newly_connections_.size());
    ASSERT_EQ(1, host_connections->busy_connections_.size());

    // connect failed after
    host_connections->connectFailed(connection);
    ASSERT_EQ(0, host_connections->newly_connections_.size());
    ASSERT_EQ(0, host_connections->busy_connections_.size());
    ASSERT_EQ(0, host_connections->available_connections_.size());
}

TEST_F(RdmaClientHostConnectionsTest, testConnectSuccessOrFailed_Available) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(0);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    // connect failed on first time
    std::shared_ptr<::accl::barex::XConnection> connection(new ::accl::barex::MockXConnection);
    host_connections->connectFailed(connection);

    // connect success on first time
    host_connections->connectSuccess(connection);
    ASSERT_EQ(1, host_connections->newly_connections_.size());

    // get rdma connection
    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(1, rdma_connection->getConnectionsCount());
    ASSERT_EQ(0, host_connections->newly_connections_.size());
    ASSERT_EQ(1, host_connections->busy_connections_.size());
    ASSERT_EQ(0, host_connections->available_connections_.size());

    host_connections->recycleConnection(rdma_connection);
    ASSERT_EQ(0, host_connections->newly_connections_.size());
    ASSERT_EQ(0, host_connections->busy_connections_.size());
    ASSERT_EQ(1, host_connections->available_connections_.size());

    // connect failed after
    host_connections->connectFailed(connection);
    ASSERT_EQ(0, host_connections->newly_connections_.size());
    ASSERT_EQ(0, host_connections->busy_connections_.size());
    ASSERT_EQ(0, host_connections->available_connections_.size());
}

TEST_F(RdmaClientHostConnectionsTest, testGetConnections_Success) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    std::thread                                 connect_callback_thread([connection1, connection2, host_connections]() {
        usleep(100 * 1000);
        host_connections->connectSuccess(connection1);
        host_connections->connectSuccess(connection2);
    });

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    auto connections = rdma_connection->getAllConnections();
    ASSERT_EQ(2, connections.size());
    for (auto& connection : connections) {
        ASSERT_TRUE(connection == connection1 || connection == connection2);
    }

    connect_callback_thread.join();

    EXPECT_CALL(*mock_connection1, Close()).Times(1).WillOnce(Return());
    EXPECT_CALL(*mock_connection2, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaClientHostConnectionsTest, testGetConnections_PartialSuccess) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    std::thread                                 connect_callback_thread([connection1, connection2, host_connections]() {
        usleep(100 * 1000);
        host_connections->connectSuccess(connection1);
        host_connections->connectFailed(connection2);
    });

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    auto connections = rdma_connection->getAllConnections();
    ASSERT_EQ(1, connections.size());
    for (auto& connection : connections) {
        ASSERT_TRUE(connection == connection1);
    }

    connect_callback_thread.join();

    EXPECT_CALL(*mock_connection1, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaClientHostConnectionsTest, testGetConnections_connectFailed) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    std::thread                                 connect_callback_thread([connection1, connection2, host_connections]() {
        usleep(100 * 1000);
        host_connections->connectFailed(connection1);
        host_connections->connectFailed(connection2);
    });

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection == nullptr);

    connect_callback_thread.join();
}

TEST_F(RdmaClientHostConnectionsTest, testGetConnections_connectTimeout) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection == nullptr);
}

TEST_F(RdmaClientHostConnectionsTest, testReuseLateConnectionsWithoutReconnect) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections =
        std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345, 2, 1);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    // The first request times out before either callback arrives.
    ASSERT_TRUE(host_connections->getConnection(xclient_transport) == nullptr);

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    host_connections->connectSuccess(connection1);
    host_connections->connectSuccess(connection2);

    // No additional Connect calls are needed for the next request.
    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(2, rdma_connection->getConnectionsCount());
    ASSERT_TRUE(host_connections->newly_connections_.empty());
    ASSERT_EQ(2, host_connections->busy_connections_.size());

    EXPECT_CALL(*mock_connection1, Close()).Times(1).WillOnce(Return());
    EXPECT_CALL(*mock_connection2, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaClientHostConnectionsTest, testReusePartialLateConnectionsWithoutReconnect) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections =
        std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345, 2, 1);
    auto mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    ASSERT_TRUE(host_connections->getConnection(xclient_transport) == nullptr);

    auto mock_connection = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection(mock_connection);
    host_connections->connectSuccess(connection);

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(1, rdma_connection->getConnectionsCount());
    ASSERT_TRUE(host_connections->newly_connections_.empty());
    ASSERT_EQ(1, host_connections->busy_connections_.size());

    EXPECT_CALL(*mock_connection, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaClientHostConnectionsTest, testRecycleConnections) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    std::thread                                 connect_callback_thread([connection1, connection2, host_connections]() {
        usleep(100 * 1000);
        host_connections->connectSuccess(connection1);
        host_connections->connectSuccess(connection2);
    });

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    auto connections = rdma_connection->getAllConnections();
    for (auto& connection : connections) {
        ASSERT_TRUE(connection == connection1 || connection == connection2);
    }

    connect_callback_thread.join();

    host_connections->recycleConnection(rdma_connection);

    auto rdma_connection2 = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection2 != nullptr);
    connections = rdma_connection2->getAllConnections();
    for (auto& connection : connections) {
        ASSERT_TRUE(connection == connection1 || connection == connection2);
    }

    EXPECT_CALL(*mock_connection1, Close()).Times(1).WillOnce(Return());
    EXPECT_CALL(*mock_connection2, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaClientHostConnectionsTest, testDestroyConnections) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections = std::make_shared<RdmaClientHostConnections>(memory_util_, "1.2.3.4", 12345);

    auto                                             mock_xclient_transport = new ::accl::barex::MockXClientTransport;
    std::shared_ptr<::accl::barex::XClientTransport> xclient_transport(mock_xclient_transport);
    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());

    auto                                        mock_connection1 = new ::accl::barex::MockXConnection();
    auto                                        mock_connection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_connection1);
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_connection2);
    std::thread                                 connect_callback_thread([connection1, connection2, host_connections]() {
        usleep(100 * 1000);
        host_connections->connectSuccess(connection1);
        host_connections->connectSuccess(connection2);
    });

    auto rdma_connection = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection != nullptr);
    auto connections = rdma_connection->getAllConnections();
    for (auto& connection : connections) {
        ASSERT_TRUE(connection == connection1 || connection == connection2);
    }

    connect_callback_thread.join();

    EXPECT_CALL(*mock_connection1, Close()).Times(1).WillOnce(Return());
    EXPECT_CALL(*mock_connection2, Close()).Times(1).WillOnce(Return());
    host_connections->destroyConnection(rdma_connection);

    EXPECT_CALL(*mock_xclient_transport, Connect(::accl::barex::TCP_V4, "1.2.3.4", 12345, _))
        .Times(2)
        .WillRepeatedly(Return());
    auto rdma_connection2 = host_connections->getConnection(xclient_transport);
    ASSERT_TRUE(rdma_connection2 == nullptr);
}

}  // namespace rtp_llm
