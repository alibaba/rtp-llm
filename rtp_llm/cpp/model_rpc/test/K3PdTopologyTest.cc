#include "rtp_llm/cpp/model_rpc/K3PdTopology.h"

#include <gtest/gtest.h>

namespace rtp_llm {

TEST(K3PdTopologyTest, MapsEightToEightWithoutKdaPartition) {
    for (int rank = 0; rank < 8; ++rank) {
        const auto plan = makeK3PdPartitionPlan(8, 1, 8, 8, rank);
        EXPECT_EQ(plan.prefill_peer_index, rank);
        EXPECT_EQ(plan.remote_kda_partition_count, 1);
        EXPECT_EQ(plan.remote_kda_partition_id, 0);
        EXPECT_EQ(plan.local_kda_heads, 12);
        const auto destination =
            makeK3PdDestinationPlan(plan.remote_kda_partition_count, plan.remote_kda_partition_id);
        EXPECT_EQ(destination.partition_count, 1);
        EXPECT_EQ(destination.partition_id, 0);
    }
    EXPECT_TRUE(isK3PdTopologySupported(8, 8, 8, 8));
}

TEST(K3PdTopologyTest, MapsEightToSixteenAndPartitionsKdaOnly) {
    for (int rank = 0; rank < 16; ++rank) {
        const auto plan = makeK3PdPartitionPlan(8, 1, 16, 8, rank);
        EXPECT_EQ(plan.prefill_peer_index, rank / 2);
        EXPECT_EQ(plan.remote_kda_partition_count, 2);
        EXPECT_EQ(plan.remote_kda_partition_id, rank % 2);
        EXPECT_EQ(plan.local_kda_heads, 6);
        const auto destination =
            makeK3PdDestinationPlan(plan.remote_kda_partition_count, plan.remote_kda_partition_id);
        EXPECT_EQ(destination.partition_count, 1);
        EXPECT_EQ(destination.partition_id, 0);
    }
}

TEST(K3PdTopologyTest, RejectsUnsupportedAndIncompleteTopologies) {
    EXPECT_FALSE(isK3PdTopologySupported(8, 1, 4, 8));
    EXPECT_FALSE(isK3PdTopologySupported(8, 1, 12, 8));
    EXPECT_FALSE(isK3PdTopologySupported(8, 1, 16, 7));
    EXPECT_FALSE(isK3PdTopologySupported(8, 2, 8, 8));
    EXPECT_FALSE(isK3PdTopologySupported(4, 1, 8, 4));
    EXPECT_THROW(makeK3PdPartitionPlan(8, 1, 16, 8, 16), std::invalid_argument);
    EXPECT_THROW(makeK3PdPartitionPlan(8, 1, 16, 8, 0, 95), std::invalid_argument);
    EXPECT_THROW(makeK3PdDestinationPlan(2, 2), std::invalid_argument);
}

}  // namespace rtp_llm
