#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include <gtest/gtest.h>
#include <stdexcept>

namespace rtp_llm {

TEST(KVCacheTransferPlannerTest, PrefillFourToDecodeEight) {
    for (int destination = 0; destination < 8; ++destination) {
        for (int source = 0; source < 4; ++source) {
            const auto plan = planHeadShardLoad(4, 8, source, destination);
            EXPECT_EQ(plan.selected, source == destination / 2);
            EXPECT_EQ(plan.source_partition_count, 2);
            EXPECT_EQ(plan.source_partition_id, destination % 2);
            EXPECT_EQ(plan.destination_partition_count, 1);
        }
    }
}

TEST(KVCacheTransferPlannerTest, PrefillEightToDecodeFourWithTwoOwners) {
    for (int destination_world_rank = 0; destination_world_rank < 8; ++destination_world_rank) {
        const int destination = destination_world_rank % 4;
        int selected = 0;
        for (int source = 0; source < 8; ++source) {
            const auto plan = planHeadShardLoad(8, 4, source, destination);
            if (plan.selected) {
                ++selected;
                EXPECT_EQ(plan.destination_partition_count, 2);
                EXPECT_EQ(plan.destination_partition_id, source % 2);
            }
            for (size_t page = 0; page < 32; ++page) {
                const bool destination_owns_page = page % 4 == static_cast<size_t>(destination);
                if (destination_owns_page && pageOwnedBySource(page, source, 8)) {
                    EXPECT_EQ(source, static_cast<int>(page % 8));
                }
            }
        }
        EXPECT_EQ(selected, 2);
    }
}

TEST(KVCacheTransferPlannerTest, SixteenCardDecodePlanningOnly) {
    // P8/EP8 -> D DP2/TP8/EP16: each owner keeps the same head shard.
    for (int dp_owner = 0; dp_owner < 2; ++dp_owner) {
        for (int destination = 0; destination < 8; ++destination) {
            int selected = 0;
            for (int source = 0; source < 8; ++source) {
                const auto plan = planHeadShardLoad(8, 8, source, destination);
                if (plan.selected) {
                    ++selected;
                    EXPECT_EQ(source, destination);
                    EXPECT_EQ(plan.source_partition_count, 1);
                    EXPECT_EQ(plan.destination_partition_count, 1);
                }
            }
            EXPECT_EQ(selected, 1) << "dp_owner=" << dp_owner;
        }
    }
    // P8/EP8 -> D DP4/TP4/EP16: each owner assembles two source head shards.
    for (int dp_owner = 0; dp_owner < 4; ++dp_owner) {
        for (int destination = 0; destination < 4; ++destination) {
            int selected = 0;
            for (int source = 0; source < 8; ++source) {
                const auto plan = planHeadShardLoad(8, 4, source, destination);
                if (plan.selected) {
                    ++selected;
                    EXPECT_EQ(source / 2, destination);
                    EXPECT_EQ(plan.destination_partition_count, 2);
                    EXPECT_EQ(plan.destination_partition_id, source % 2);
                }
            }
            EXPECT_EQ(selected, 2) << "dp_owner=" << dp_owner;
        }
    }
}

TEST(KVCacheTransferPlannerTest, EqualTpAndInvalidTopology) {
    for (int rank = 0; rank < 8; ++rank) {
        for (int source = 0; source < 8; ++source) {
            const auto plan = planHeadShardLoad(8, 8, source, rank);
            EXPECT_EQ(plan.selected, source == rank);
            EXPECT_EQ(plan.source_partition_count, 1);
            EXPECT_EQ(plan.destination_partition_count, 1);
        }
    }
    EXPECT_FALSE(supportsHeadShardTransfer(6, 4));
    EXPECT_THROW(planHeadShardLoad(6, 4, 0, 0), std::invalid_argument);
    EXPECT_THROW(planHeadShardLoad(8, 4, 8, 0), std::invalid_argument);
    EXPECT_THROW(pageOwnedBySource(0, 8, 8), std::invalid_argument);
}

}  // namespace rtp_llm
