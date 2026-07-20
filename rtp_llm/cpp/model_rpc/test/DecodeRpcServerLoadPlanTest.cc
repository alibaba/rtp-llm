#include "gtest/gtest.h"

#include "rtp_llm/cpp/model_rpc/DecodeRpcServer.h"

namespace rtp_llm {

class DecodeRpcServerLoadPlanTest: public ::testing::Test {
protected:
    void configureServer(DecodeRpcServer& server, std::vector<std::string> workers, CPRotateMethod cp_method) {
        server.resource_.workers = std::move(workers);
        server.maga_init_params_.parallelism_config.prefill_cp_config.method = cp_method;
    }
};

TEST_F(DecodeRpcServerLoadPlanTest, NormalDGreaterThanPrefillPartitionsRemoteBlock) {
    DecodeRpcServer server;
    configureServer(server, {"d0", "d1", "d2", "d3"}, CPRotateMethod::DISABLED);

    auto plan = server.buildRemoteLoadPlan(3, {"p0", "p1"}, false);

    ASSERT_EQ(plan.peer_addrs.size(), 1);
    EXPECT_EQ(plan.peer_addrs[0], "p1");
    EXPECT_EQ(plan.partition_count, 2);
    EXPECT_EQ(plan.partition_id, 1);
}

TEST_F(DecodeRpcServerLoadPlanTest, NormalPrefillGreaterThanDecodeLoadsMultiplePeers) {
    DecodeRpcServer server;
    configureServer(server, {"d0"}, CPRotateMethod::DISABLED);

    auto plan = server.buildRemoteLoadPlan(0, {"p0", "p1"}, false);

    ASSERT_EQ(plan.peer_addrs.size(), 2);
    EXPECT_EQ(plan.peer_addrs[0], "p0");
    EXPECT_EQ(plan.peer_addrs[1], "p1");
    EXPECT_EQ(plan.partition_count, 1);
    EXPECT_EQ(plan.partition_id, 0);
}

TEST_F(DecodeRpcServerLoadPlanTest, PrefillCpPartitionsOneFullKvPeer) {
    DecodeRpcServer server;
    configureServer(server, {"d0", "d1"}, CPRotateMethod::PREFILL_CP);

    auto plan = server.buildRemoteLoadPlan(1, {"p0", "p1"}, false);

    ASSERT_EQ(plan.peer_addrs.size(), 1);
    EXPECT_EQ(plan.peer_addrs[0], "p1");
    EXPECT_EQ(plan.partition_count, 2);
    EXPECT_EQ(plan.partition_id, 1);
}

TEST_F(DecodeRpcServerLoadPlanTest, AllGatherUsesOneFullKvPeer) {
    DecodeRpcServer server;
    configureServer(server, {"d0"}, CPRotateMethod::ALL_GATHER);

    auto plan = server.buildRemoteLoadPlan(0, {"p0", "p1"}, false);

    ASSERT_EQ(plan.peer_addrs.size(), 1);
    EXPECT_EQ(plan.peer_addrs[0], "p0");
    EXPECT_EQ(plan.partition_count, 1);
    EXPECT_EQ(plan.partition_id, 0);
}

TEST_F(DecodeRpcServerLoadPlanTest, MlaUsesOneFullKvPeerWithoutRemotePartition) {
    DecodeRpcServer server;
    configureServer(server, {"d0"}, CPRotateMethod::DISABLED);

    auto plan = server.buildRemoteLoadPlan(0, {"p0", "p1"}, true);

    ASSERT_EQ(plan.peer_addrs.size(), 1);
    EXPECT_EQ(plan.peer_addrs[0], "p0");
    EXPECT_EQ(plan.partition_count, 1);
    EXPECT_EQ(plan.partition_id, 0);
}

}  // namespace rtp_llm
