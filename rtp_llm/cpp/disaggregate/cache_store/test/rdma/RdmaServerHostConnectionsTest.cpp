#include "gtest/gtest.h"

#include "rtp_llm/cpp/disaggregate/cache_store/rdma/RdmaServerHostConnections.h"
#include "rtp_llm/cpp/disaggregate/cache_store/test/CacheStoreTestBase.h"
#include "aios/network/rdma/accl/unittest/mock/MockXConnection.h"

namespace rtp_llm {

class RdmaServerHostConnectionsTest: public CacheStoreTestBase {
public:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(RdmaServerHostConnectionsTest, testGetConnection) {
    auto host_connections = std::make_shared<RdmaServerHostConnections>("1.2.3.4");

    // channel_id will be the pointer value of fake_channel
    ::accl::barex::XChannel* fake_channel = reinterpret_cast<::accl::barex::XChannel*>(0x100);
    uint64_t expected_channel_id = reinterpret_cast<uint64_t>(fake_channel);

    auto xconnection = host_connections->getConnection(12345, expected_channel_id);
    ASSERT_TRUE(xconnection == nullptr);

    auto mock_xconnection = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection(mock_xconnection);
    EXPECT_CALL(*mock_xconnection, GetPeerAddrAndPort())
        .WillRepeatedly(Return(std::make_pair<std::string, int>("1.2.3.4", 12345)));
    EXPECT_CALL(*mock_xconnection, GetChannel())
        .WillRepeatedly(Return(fake_channel));
    host_connections->addConnection(connection);

    xconnection = host_connections->getConnection(12345, expected_channel_id);
    ASSERT_TRUE(xconnection != nullptr);
    ASSERT_EQ(connection, xconnection);

    host_connections->removeConnection(connection);
    xconnection = host_connections->getConnection(12345, expected_channel_id);
    ASSERT_TRUE(xconnection == nullptr);
}

TEST_F(RdmaServerHostConnectionsTest, testMultipleConnectionsSamePort) {
    auto host_connections = std::make_shared<RdmaServerHostConnections>("1.2.3.4");

    ::accl::barex::XChannel* fake_channel_1 = reinterpret_cast<::accl::barex::XChannel*>(0x100);
    ::accl::barex::XChannel* fake_channel_2 = reinterpret_cast<::accl::barex::XChannel*>(0x200);

    auto mock1 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> conn1(mock1);
    EXPECT_CALL(*mock1, GetPeerAddrAndPort())
        .WillRepeatedly(Return(std::make_pair<std::string, int>("1.2.3.4", 12345)));
    EXPECT_CALL(*mock1, GetChannel()).WillRepeatedly(Return(fake_channel_1));

    auto mock2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> conn2(mock2);
    EXPECT_CALL(*mock2, GetPeerAddrAndPort())
        .WillRepeatedly(Return(std::make_pair<std::string, int>("1.2.3.4", 12345)));
    EXPECT_CALL(*mock2, GetChannel()).WillRepeatedly(Return(fake_channel_2));

    host_connections->addConnection(conn1);
    host_connections->addConnection(conn2);

    // Both connections coexist on same port
    auto result1 = host_connections->getConnection(12345, reinterpret_cast<uint64_t>(fake_channel_1));
    auto result2 = host_connections->getConnection(12345, reinterpret_cast<uint64_t>(fake_channel_2));
    ASSERT_TRUE(result1 != nullptr);
    ASSERT_TRUE(result2 != nullptr);
    ASSERT_EQ(conn1, result1);
    ASSERT_EQ(conn2, result2);

    // Remove one, other still exists
    host_connections->removeConnection(conn1);
    ASSERT_TRUE(host_connections->getConnection(12345, reinterpret_cast<uint64_t>(fake_channel_1)) == nullptr);
    ASSERT_TRUE(host_connections->getConnection(12345, reinterpret_cast<uint64_t>(fake_channel_2)) != nullptr);
}

TEST_F(RdmaServerHostConnectionsTest, testBackwardCompatZeroPeerChannel) {
    auto host_connections = std::make_shared<RdmaServerHostConnections>("1.2.3.4");

    ::accl::barex::XChannel* fake_channel = reinterpret_cast<::accl::barex::XChannel*>(0x100);

    auto mock = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> conn(mock);
    EXPECT_CALL(*mock, GetPeerAddrAndPort())
        .WillRepeatedly(Return(std::make_pair<std::string, int>("1.2.3.4", 12345)));
    EXPECT_CALL(*mock, GetChannel()).WillRepeatedly(Return(fake_channel));

    host_connections->addConnection(conn);

    // peer_channel=0 should fall back to port-only match
    auto result = host_connections->getConnection(12345, 0);
    ASSERT_TRUE(result != nullptr);
    ASSERT_EQ(conn, result);

    // Wrong port with peer_channel=0 should still return nullptr
    ASSERT_TRUE(host_connections->getConnection(99999, 0) == nullptr);
}

class RdmaServerHostConnectionsMapTest: public CacheStoreTestBase {};

TEST_F(RdmaServerHostConnectionsMapTest, testGetConnection) {
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections_map = std::make_shared<RdmaServerHostConnectionsMap>(memory_util_);

    auto rdma_connection = host_connections_map->getConnection({{"1.2.3.4", 12345, 0}, {"1.2.3.5", 12345, 0}});
    ASSERT_TRUE(rdma_connection == nullptr);

    ::accl::barex::XChannel* fake_channel_1 = reinterpret_cast<::accl::barex::XChannel*>(0x300);
    auto mock_xconnection1 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection1(mock_xconnection1);
    EXPECT_CALL(*mock_xconnection1, GetPeerAddrAndPort())
        .WillRepeatedly(Return(std::make_pair<std::string, int>("1.2.3.4", 12345)));
    EXPECT_CALL(*mock_xconnection1, GetChannel()).WillRepeatedly(Return(fake_channel_1));
    host_connections_map->connectSuccess(connection1);

    rdma_connection = host_connections_map->getConnection({{"1.2.3.4", 12345, 0}, {"1.2.3.5", 12345, 0}});
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(1, rdma_connection->getAllConnections().size());
    ASSERT_EQ(connection1, rdma_connection->getAllConnections()[0]);

    ::accl::barex::XChannel* fake_channel_2 = reinterpret_cast<::accl::barex::XChannel*>(0x400);
    auto mock_xconnection2 = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> connection2(mock_xconnection2);
    EXPECT_CALL(*mock_xconnection2, GetPeerAddrAndPort())
        .WillRepeatedly(Return(std::make_pair<std::string, int>("1.2.3.5", 12345)));
    EXPECT_CALL(*mock_xconnection2, GetChannel()).WillRepeatedly(Return(fake_channel_2));
    host_connections_map->connectSuccess(connection2);

    rdma_connection = host_connections_map->getConnection({{"1.2.3.4", 12345, 0}, {"1.2.3.5", 12345, 0}});
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(2, rdma_connection->getAllConnections().size());
    ASSERT_EQ(connection1, rdma_connection->getAllConnections()[0]);
    ASSERT_EQ(connection2, rdma_connection->getAllConnections()[1]);

    host_connections_map->connectFailed(connection1);
    rdma_connection = host_connections_map->getConnection({{"1.2.3.4", 12345, 0}, {"1.2.3.5", 12345, 0}});
    ASSERT_TRUE(rdma_connection != nullptr);
    ASSERT_EQ(1, rdma_connection->getAllConnections().size());
    ASSERT_EQ(connection2, rdma_connection->getAllConnections()[0]);
    EXPECT_CALL(*mock_xconnection2, Close()).Times(1).WillOnce(Return());
}

TEST_F(RdmaServerHostConnectionsTest, testBugScenario_SameHostSamePort_ConnectionOverwrite) {
    // 线上问题复现：两个 Decode Rank 在同一台机器上，使用相同端口但不同的 QP/NIC 连接到 Prefill Rank
    // 旧实现用 port 做 map key，第二个连接会覆盖第一个，导致 RDMA write 用错 QP → mlx5 protection error
    auto host_connections = std::make_shared<RdmaServerHostConnections>("10.0.0.1");

    // Decode Rank A: 10.0.0.1:12345, channel_A (模拟 NIC 0 的 QP)
    ::accl::barex::XChannel* channel_A = reinterpret_cast<::accl::barex::XChannel*>(0xA00);
    uint64_t channel_id_A = reinterpret_cast<uint64_t>(channel_A);

    auto mock_A = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> conn_A(mock_A);
    EXPECT_CALL(*mock_A, GetPeerAddrAndPort())
        .WillRepeatedly(Return(std::make_pair<std::string, int>("10.0.0.1", 12345)));
    EXPECT_CALL(*mock_A, GetChannel()).WillRepeatedly(Return(channel_A));

    // Decode Rank B: 同一台机器 10.0.0.1:12345, 但 channel_B 不同（模拟 NIC 1 的 QP）
    ::accl::barex::XChannel* channel_B = reinterpret_cast<::accl::barex::XChannel*>(0xB00);
    uint64_t channel_id_B = reinterpret_cast<uint64_t>(channel_B);

    auto mock_B = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> conn_B(mock_B);
    EXPECT_CALL(*mock_B, GetPeerAddrAndPort())
        .WillRepeatedly(Return(std::make_pair<std::string, int>("10.0.0.1", 12345)));
    EXPECT_CALL(*mock_B, GetChannel()).WillRepeatedly(Return(channel_B));

    // 两个 Decode Rank 先后连接到 Prefill Rank
    host_connections->addConnection(conn_A);
    host_connections->addConnection(conn_B);

    // ===== 修复后的行为：用 (port, channel_id) 做复合 key，两个连接共存 =====
    // Decode Rank A 发 load request 带 peer_channel=channel_A → 精确匹配到 conn_A
    auto result_A = host_connections->getConnection(12345, channel_id_A);
    ASSERT_TRUE(result_A != nullptr);
    ASSERT_EQ(conn_A, result_A);

    // Decode Rank B 发 load request 带 peer_channel=channel_B → 精确匹配到 conn_B
    auto result_B = host_connections->getConnection(12345, channel_id_B);
    ASSERT_TRUE(result_B != nullptr);
    ASSERT_EQ(conn_B, result_B);

    // ===== 旧 buggy 行为的模拟：peer_channel=0 退化为 port-only 匹配 =====
    // 当 peer_channel=0 时，只按 port 查找，同端口多连接下返回的是不确定的那个
    // 这正是线上 bug 的根因：Rank A 的 load request 没带 channel_id，
    // 查到的可能是 conn_B，然后用 conn_B 的 QP 去写 Rank A 的 rkey → mlx5 报错
    auto result_ambiguous = host_connections->getConnection(12345, 0);
    ASSERT_TRUE(result_ambiguous != nullptr);
    // port-only 匹配能拿到一个连接，但不能保证是哪一个 —— 这就是 bug
    ASSERT_TRUE(result_ambiguous == conn_A || result_ambiguous == conn_B);
}

TEST_F(RdmaServerHostConnectionsMapTest, testBugScenario_SameHostSamePort_MapLevel) {
    // 端到端复现：通过 RdmaServerHostConnectionsMap 模拟真实的 loadImpl 查连接流程
    // 场景：同一台 Decode 机器的两个 Rank 用相同端口、不同 channel 连到 Prefill Server
    ASSERT_TRUE(memory_util_ != nullptr);

    auto host_connections_map = std::make_shared<RdmaServerHostConnectionsMap>(memory_util_);

    // Decode Rank A: 10.0.0.1:12345, channel_A
    ::accl::barex::XChannel* channel_A = reinterpret_cast<::accl::barex::XChannel*>(0xA00);
    uint64_t channel_id_A = reinterpret_cast<uint64_t>(channel_A);

    auto mock_A = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> conn_A(mock_A);
    EXPECT_CALL(*mock_A, GetPeerAddrAndPort())
        .WillRepeatedly(Return(std::make_pair<std::string, int>("10.0.0.1", 12345)));
    EXPECT_CALL(*mock_A, GetChannel()).WillRepeatedly(Return(channel_A));

    // Decode Rank B: 同一台机器 10.0.0.1:12345, 不同 channel
    ::accl::barex::XChannel* channel_B = reinterpret_cast<::accl::barex::XChannel*>(0xB00);
    uint64_t channel_id_B = reinterpret_cast<uint64_t>(channel_B);

    auto mock_B = new ::accl::barex::MockXConnection();
    std::shared_ptr<::accl::barex::XConnection> conn_B(mock_B);
    EXPECT_CALL(*mock_B, GetPeerAddrAndPort())
        .WillRepeatedly(Return(std::make_pair<std::string, int>("10.0.0.1", 12345)));
    EXPECT_CALL(*mock_B, GetChannel()).WillRepeatedly(Return(channel_B));

    // 两个连接通过 connectSuccess 注册（模拟 RDMA 握手成功）
    host_connections_map->connectSuccess(conn_A);
    host_connections_map->connectSuccess(conn_B);

    // ===== 修复后：load request 带正确的 peer_channel，精确匹配 =====
    // 模拟 Decode Rank A 发起 load request，AddrPortChannel 中带了 peer_channel=channel_id_A
    auto rdma_conn_A = host_connections_map->getConnection({{"10.0.0.1", 12345, channel_id_A}});
    ASSERT_TRUE(rdma_conn_A != nullptr);
    ASSERT_EQ(1, rdma_conn_A->getAllConnections().size());
    ASSERT_EQ(conn_A, rdma_conn_A->getAllConnections()[0]);

    // 模拟 Decode Rank B 发起 load request，peer_channel=channel_id_B
    auto rdma_conn_B = host_connections_map->getConnection({{"10.0.0.1", 12345, channel_id_B}});
    ASSERT_TRUE(rdma_conn_B != nullptr);
    ASSERT_EQ(1, rdma_conn_B->getAllConnections().size());
    ASSERT_EQ(conn_B, rdma_conn_B->getAllConnections()[0]);

    // ===== 向后兼容：旧版 Decode 不带 peer_channel (=0)，退化为 port-only =====
    auto rdma_conn_compat = host_connections_map->getConnection({{"10.0.0.1", 12345, 0}});
    ASSERT_TRUE(rdma_conn_compat != nullptr);
    ASSERT_EQ(1, rdma_conn_compat->getAllConnections().size());
    // port-only 匹配能找到连接，但具体是哪个不确定 —— 旧版本就是这个歧义导致了线上事故
    auto compat_conn = rdma_conn_compat->getAllConnections()[0];
    ASSERT_TRUE(compat_conn == conn_A || compat_conn == conn_B);

    // ===== 验证断开一个不影响另一个 =====
    host_connections_map->connectFailed(conn_A);

    // Rank A 的连接已断开，用 channel_id_A 查不到了
    auto rdma_conn_A_gone = host_connections_map->getConnection({{"10.0.0.1", 12345, channel_id_A}});
    ASSERT_TRUE(rdma_conn_A_gone == nullptr);

    // Rank B 的连接不受影响
    auto rdma_conn_B_still = host_connections_map->getConnection({{"10.0.0.1", 12345, channel_id_B}});
    ASSERT_TRUE(rdma_conn_B_still != nullptr);
    ASSERT_EQ(conn_B, rdma_conn_B_still->getAllConnections()[0]);

    EXPECT_CALL(*mock_B, Close()).Times(1).WillOnce(Return());
}

}  // namespace rtp_llm
