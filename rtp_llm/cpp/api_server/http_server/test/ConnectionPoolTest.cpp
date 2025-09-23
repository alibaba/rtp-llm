#include "gtest/gtest.h"
#include "autil/NetUtil.h"
#include "http_server/HttpServer.h"
#include "rtp_llm/cpp/api_server/http_server/http_client/ConnectionPool.h"

namespace http_server {
class ConnectionPoolTest: public ::testing::Test {
public:
    void SetUp() override {
        for (int i = 0; i < connection_size; i++) {
            std::string address = "tcp:0.0.0.0:" + std::to_string(autil::NetUtil::randomPort());
            auto        server  = initServer(address);
            ServerInfo  server_info;
            server_info.address = address;
            server_info.server  = server;
            http_servers_.push_back(server_info);
        }
        connection_pool_ = std::make_shared<ConnectionPool>();
    }
    void TearDown() override {
        for (auto& it : http_servers_) {
            if (it.server) {
                it.server->Stop();
                it.server.reset();
            }
        }
    }
    std::shared_ptr<ConnectionPool> connection_pool_;
    int                             connection_size = 5;
    struct ServerInfo {
        std::string                              address;
        std::shared_ptr<http_server::HttpServer> server;
    };
    std::vector<ServerInfo>                  http_servers_;
    std::shared_ptr<http_server::HttpServer> initServer(const std::string& address);
};

std::shared_ptr<http_server::HttpServer> ConnectionPoolTest::initServer(const std::string& address) {
    auto server_ = std::make_shared<http_server::HttpServer>();
    server_->Start(address);
    return server_;
}

TEST_F(ConnectionPoolTest, testMakeAndRecyleConnection) {
    std::vector<std::shared_ptr<anet::Connection>> conns;
    for (auto& it : http_servers_) {
        auto conn = connection_pool_->makeHttpConnection(it.address);
        ASSERT_TRUE(conn != nullptr);
        ASSERT_TRUE(!conn->isClosed());
        conns.push_back(conn);
    }
    ASSERT_TRUE(connection_pool_->busy_connection_pool_.size() == connection_size);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_.size() == connection_size);
    for (auto conn_list : connection_pool_->busy_connection_pool_) {
        ASSERT_TRUE(conn_list.second->size() == 1);
    }
    for (auto conn_list : connection_pool_->idle_connection_pool_) {
        ASSERT_TRUE(conn_list.second->size() == 0);
    }

    for (int i = 0; i < connection_size; i++) {
        connection_pool_->recycleHttpConnection(http_servers_[i].address, conns[i], false);
        ASSERT_TRUE(!conns[i]->isClosed());
    }
    ASSERT_TRUE(connection_pool_->busy_connection_pool_.size() == connection_size);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_.size() == connection_size);
    for (auto conn_list : connection_pool_->busy_connection_pool_) {
        ASSERT_TRUE(conn_list.second->size() == 0);
    }
    for (auto conn_list : connection_pool_->idle_connection_pool_) {
        ASSERT_TRUE(conn_list.second->size() == 1);
    }
}

TEST_F(ConnectionPoolTest, testMakeAndRecyleMultiConnection) {
    std::vector<std::shared_ptr<anet::Connection>> conns;
    std::string address = "tcp:0.0.0.0:" + std::to_string(autil::NetUtil::randomPort());
    auto        server  = initServer(address);

    for (int i = 0; i < 10; i++) {
        auto conn = connection_pool_->makeHttpConnection(address);
        conns.push_back(conn);
    }
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(!conns[i]->isClosed());
    }

    ASSERT_TRUE(connection_pool_->busy_connection_pool_.size() == 1);
    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 10);

    for (int i = 0; i < 5; i++) {
        connection_pool_->recycleHttpConnection(address, conns[i], false);
    }
    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 5);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_[address]->size() == 5);

    for (int i = 0; i < 10; i++) {
        auto conn = connection_pool_->makeHttpConnection(address);
        conns.push_back(conn);
    }
    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 15);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_[address]->size() == 0);
}

TEST_F(ConnectionPoolTest, testRecyleClosedConnection) {
    std::string address = "tcp:0.0.0.0:" + std::to_string(autil::NetUtil::randomPort());
    // connect invalid address
    auto conn = connection_pool_->makeHttpConnection(address);
    sleep(1);
    ASSERT_TRUE(conn->isClosed());

    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 1);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_[address]->size() == 0);

    connection_pool_->recycleHttpConnection(address, conn, true);
    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 0);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_[address]->size() == 0);
}

TEST_F(ConnectionPoolTest, testRecyleErrorAdressConnection) {
    std::string address = "tcp:0.0.0.0:" + std::to_string(autil::NetUtil::randomPort());
    // connect invalid address
    auto conn = connection_pool_->makeHttpConnection(address);
    sleep(1);
    ASSERT_TRUE(conn->isClosed());

    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 1);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_[address]->size() == 0);

    std::string error_address = "tcp:0.0.0.0:" + std::to_string(autil::NetUtil::randomPort());
    connection_pool_->recycleHttpConnection(error_address, conn, true);
    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 1);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_[address]->size() == 0);
}

TEST_F(ConnectionPoolTest, testMakeConnectionAfterClosed) {
    std::vector<std::shared_ptr<anet::Connection>> conns;
    std::string address = "tcp:0.0.0.0:" + std::to_string(autil::NetUtil::randomPort());
    auto        server  = initServer(address);

    for (int i = 0; i < 10; i++) {
        auto conn = connection_pool_->makeHttpConnection(address);
        conns.push_back(conn);
    }
    for (int i = 0; i < 10; i++) {
        ASSERT_TRUE(!conns[i]->isClosed());
    }
    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 10);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_[address]->size() == 0);

    for (int i = 0; i < 10; i++) {
        connection_pool_->recycleHttpConnection(address, conns[i], false);
    }
    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 0);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_[address]->size() == 10);

    // close 5 conns in idle list
    for (int i = 0; i < 5; i++) {
        conns[i]->close();
        ASSERT_TRUE(conns[i]->isClosed());
    }
    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 0);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_[address]->size() == 10);

    conns.clear();
    for (int i = 0; i < 20; i++) {
        auto conn = connection_pool_->makeHttpConnection(address);
        conns.push_back(conn);
    }
    for (int i = 0; i < 20; i++) {
        ASSERT_TRUE(!conns[i]->isClosed());
    }
    ASSERT_TRUE(connection_pool_->busy_connection_pool_[address]->size() == 20);
    ASSERT_TRUE(connection_pool_->idle_connection_pool_[address]->size() == 0);
}
}  // namespace http_server