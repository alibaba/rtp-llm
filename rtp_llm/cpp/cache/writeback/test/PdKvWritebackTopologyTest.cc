#include "rtp_llm/cpp/cache/writeback/PdKvWritebackTopology.h"

#include <string>
#include <vector>

#include "absl/status/status.h"
#include "gtest/gtest.h"

namespace rtp_llm {

TEST(PdKvWritebackTopologyTest, TpEqualMapsSameRankForFourWayTp) {
    PdKvWritebackTopologyInput input;
    input.local_tp_size              = 4;
    input.source_partition_count     = 4;
    input.destination_partition_count = 4;
    input.decode_grpc_addrs          = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    input.prefill_grpc_addrs         = {"p0:1000", "p1:1000", "p2:1000", "p3:1000"};
    input.prefill_worker_addrs       = {"p0:2000:3000", "p1:2000:3000", "p2:2000:3000", "p3:2000:3000"};

    auto plan = buildPdKvWritebackTopology(input);
    ASSERT_TRUE(plan.ok()) << plan.status();
    ASSERT_EQ(plan->mode, PdKvWritebackTopologyMode::TpEqual);
    ASSERT_EQ(plan->mappings.size(), 4);
    for (size_t i = 0; i < plan->mappings.size(); ++i) {
        EXPECT_EQ(plan->mappings[i].decode_rank, static_cast<int32_t>(i));
        EXPECT_EQ(plan->mappings[i].prefill_rank, static_cast<int32_t>(i));
        EXPECT_EQ(plan->mappings[i].decode_grpc_addr, input.decode_grpc_addrs[i]);
        EXPECT_EQ(plan->mappings[i].prefill_grpc_addr, input.prefill_grpc_addrs[i]);
        EXPECT_EQ(plan->mappings[i].prefill_worker_addr, input.prefill_worker_addrs[i]);
        EXPECT_EQ(plan->mappings[i].local_partition_count, 1);
        EXPECT_EQ(plan->mappings[i].local_partition_id, 0);
        EXPECT_EQ(plan->mappings[i].remote_partition_count, 1);
        EXPECT_EQ(plan->mappings[i].remote_partition_id, 0);
    }
}

TEST(PdKvWritebackTopologyTest, RejectsMismatchedPrefillGrpcAndWorkerAddrs) {
    PdKvWritebackTopologyInput input;
    input.local_tp_size              = 4;
    input.source_partition_count     = 4;
    input.destination_partition_count = 4;
    input.decode_grpc_addrs          = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    input.prefill_grpc_addrs         = {"p0:1000", "p1:1000", "p2:1000", "p3:1000"};
    input.prefill_worker_addrs       = {"p0:2000:3000"};

    auto plan = buildPdKvWritebackTopology(input);
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), absl::StatusCode::kFailedPrecondition);
    EXPECT_NE(std::string(plan.status().message()).find("prefill address count mismatch"), std::string::npos);
}

TEST(PdKvWritebackTopologyTest, RejectsUnequalTpInPhaseOne) {
    PdKvWritebackTopologyInput input;
    input.local_tp_size              = 4;
    input.source_partition_count     = 2;
    input.destination_partition_count = 4;
    input.decode_grpc_addrs          = {"d0:1000", "d1:1000", "d2:1000", "d3:1000"};
    input.prefill_grpc_addrs         = {"p0:1000", "p1:1000"};
    input.prefill_worker_addrs       = {"p0:2000:3000", "p1:2000:3000"};

    auto plan = buildPdKvWritebackTopology(input);
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), absl::StatusCode::kUnimplemented);
    EXPECT_NE(std::string(plan.status().message()).find("unsupported_topology"), std::string::npos);
}

}  // namespace rtp_llm
