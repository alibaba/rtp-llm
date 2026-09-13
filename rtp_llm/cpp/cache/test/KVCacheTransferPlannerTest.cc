#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

#include <gtest/gtest.h>
#include <stdexcept>
#include <vector>

namespace rtp_llm {
namespace {

using Pair = std::pair<int, int>;

std::vector<Pair> asPairs(const std::vector<CacheStoreBlockPair>& plan) {
    std::vector<Pair> pairs;
    pairs.reserve(plan.size());
    for (const auto& item : plan) {
        pairs.emplace_back(item.key_index, item.offset_index);
    }
    return pairs;
}

std::vector<Pair> expectedFullPageRRPairs(size_t total_logical_blocks, int cp_rank, int cp_size) {
    std::vector<Pair> expected;
    for (size_t global = static_cast<size_t>(cp_rank); global < total_logical_blocks;
         global += static_cast<size_t>(cp_size)) {
        expected.emplace_back(static_cast<int>(global), static_cast<int>(global / static_cast<size_t>(cp_size)));
    }
    return expected;
}

TEST(KVCacheTransferPlannerTest, FullGroupPublishesOnlyRequestedRange) {
    const auto plan = buildIncrementalCacheStoreBlockPlan(
        8, 0, true, CacheGroupType::FULL, 0, 1, CacheStorePublishRange{2, 5, false});
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0].key_index, 2);
    EXPECT_EQ(plan[1].key_index, 3);
    EXPECT_EQ(plan[2].key_index, 4);
}

TEST(KVCacheTransferPlannerTest, LinearGroupWaitsForTerminalPublication) {
    const auto intermediate = buildIncrementalCacheStoreBlockPlan(
        8, 0, true, CacheGroupType::LINEAR, 0, 1, CacheStorePublishRange{0, 4, false});
    EXPECT_TRUE(intermediate.empty());

    const auto terminal = buildIncrementalCacheStoreBlockPlan(
        8, 0, true, CacheGroupType::LINEAR, 0, 1, CacheStorePublishRange{4, 8, true});
    ASSERT_EQ(terminal.size(), 1u);
    EXPECT_EQ(terminal[0].key_index, 7);
    EXPECT_EQ(terminal[0].offset_index, 7);

    const auto terminal_without_hybrid_flag = buildIncrementalCacheStoreBlockPlan(
        8, 0, false, CacheGroupType::LINEAR, 0, 1, CacheStorePublishRange{4, 8, true});
    ASSERT_EQ(terminal_without_hybrid_flag.size(), 1u);
    EXPECT_EQ(terminal_without_hybrid_flag[0].key_index, 7);

    EXPECT_THROW(buildIncrementalCacheStoreBlockPlan(
                     8, 0, true, CacheGroupType::LINEAR, 0, 1, CacheStorePublishRange{4, 7, true}),
                 std::invalid_argument);
}

TEST(KVCacheTransferPlannerTest, FullGroupPreservesCpKeyOffsetMapping) {
    const auto plan = buildIncrementalCacheStoreBlockPlan(
        9, 0, true, CacheGroupType::FULL, 1, 2, CacheStorePublishRange{3, 8, false});
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0].key_index, 3);
    EXPECT_EQ(plan[0].offset_index, 1);
    EXPECT_EQ(plan[1].key_index, 5);
    EXPECT_EQ(plan[1].offset_index, 2);
    EXPECT_EQ(plan[2].key_index, 7);
    EXPECT_EQ(plan[2].offset_index, 3);
}

TEST(KVCacheTransferPlannerTest, FullPageRRTableCoversEveryRankAndTail) {
    constexpr int kCpSize = 8;
    for (const size_t total_blocks : {1u, 7u, 8u, 9u, 16u, 17u}) {
        for (int rank = 0; rank < kCpSize; ++rank) {
            const auto plan = buildCacheStoreBlockPlan(total_blocks,
                                                       /*first_full_block=*/0,
                                                       /*use_hybrid=*/true,
                                                       CacheGroupType::FULL,
                                                       rank,
                                                       kCpSize);
            EXPECT_EQ(asPairs(plan), expectedFullPageRRPairs(total_blocks, rank, kCpSize))
                << "total_blocks=" << total_blocks << " rank=" << rank;
        }
    }
}

TEST(KVCacheTransferPlannerTest, VirtualBlockLinearTerminalUsesCanonicalKeyAndLocalSlot) {
    constexpr int kCpSize = 8;
    for (const size_t total_blocks : {1u, 7u, 8u, 9u, 16u, 17u}) {
        const int expected_key    = static_cast<int>(total_blocks - 1);
        const int expected_offset = static_cast<int>((total_blocks - 1) / kCpSize);
        for (int rank = 0; rank < kCpSize; ++rank) {
            const auto plan =
                buildIncrementalCacheStoreBlockPlan(total_blocks,
                                                    /*reuse_block_size=*/0,
                                                    /*use_hybrid=*/true,
                                                    CacheGroupType::LINEAR,
                                                    rank,
                                                    kCpSize,
                                                    CacheStorePublishRange{/*begin_block=*/total_blocks - 1,
                                                                           /*end_block=*/total_blocks,
                                                                           /*terminal=*/true},
                                                    /*virtual_block_cache_layout=*/true);
            EXPECT_EQ(asPairs(plan), (std::vector<Pair>{{expected_key, expected_offset}}))
                << "total_blocks=" << total_blocks << " rank=" << rank;
            EXPECT_EQ(
                asPairs(buildCacheStoreBlockPlan(
                    total_blocks, 0, true, CacheGroupType::LINEAR, rank, kCpSize, /*virtual_block_cache_layout=*/true)),
                asPairs(plan));
        }
    }
}

TEST(KVCacheTransferPlannerTest, LinearWithoutVirtualBlockLayoutPreservesLegacyCoordinates) {
    EXPECT_EQ(asPairs(buildCacheStoreBlockPlan(/*total_logical_blocks=*/9,
                                               /*first_full_block=*/0,
                                               /*use_hybrid=*/true,
                                               CacheGroupType::LINEAR,
                                               /*cp_rank=*/3,
                                               /*cp_size=*/8,
                                               /*virtual_block_cache_layout=*/false)),
              (std::vector<Pair>{{8, 8}}));
}

TEST(KVCacheTransferPlannerTest, SwaWithoutVirtualBlockLayoutPreservesLegacyCoordinates) {
    EXPECT_EQ(asPairs(buildCacheStoreBlockPlan(/*total_logical_blocks=*/9,
                                               /*first_full_block=*/0,
                                               /*use_hybrid=*/true,
                                               CacheGroupType::SWA,
                                               /*cp_rank=*/3,
                                               /*cp_size=*/8,
                                               /*virtual_block_cache_layout=*/false)),
              (std::vector<Pair>{{7, 7}, {8, 8}}));
}

TEST(KVCacheTransferPlannerTest, IncrementalPageRRMergesToTheNonChunkedRegistrationSet) {
    constexpr size_t kTotalBlocks = 17;
    constexpr int    kCpSize      = 8;
    for (int rank = 0; rank < kCpSize; ++rank) {
        std::vector<Pair> chunked_full;
        for (const auto& range : {CacheStorePublishRange{0, 8, false},
                                  CacheStorePublishRange{8, 16, false},
                                  CacheStorePublishRange{16, 17, true}}) {
            const auto round       = buildIncrementalCacheStoreBlockPlan(kTotalBlocks,
                                                                   /*reuse_block_size=*/0,
                                                                   /*use_hybrid=*/true,
                                                                   CacheGroupType::FULL,
                                                                   rank,
                                                                   kCpSize,
                                                                   range);
            const auto round_pairs = asPairs(round);
            chunked_full.insert(chunked_full.end(), round_pairs.begin(), round_pairs.end());
        }
        const auto non_chunked_full = buildCacheStoreBlockPlan(kTotalBlocks,
                                                               /*first_full_block=*/0,
                                                               /*use_hybrid=*/true,
                                                               CacheGroupType::FULL,
                                                               rank,
                                                               kCpSize);
        EXPECT_EQ(chunked_full, asPairs(non_chunked_full)) << "rank=" << rank;

        const auto nonterminal_linear = buildIncrementalCacheStoreBlockPlan(
            kTotalBlocks,
            /*reuse_block_size=*/0,
            /*use_hybrid=*/true,
            CacheGroupType::LINEAR,
            rank,
            kCpSize,
            CacheStorePublishRange{/*begin_block=*/0, /*end_block=*/16, /*terminal=*/false},
            /*virtual_block_cache_layout=*/true);
        EXPECT_TRUE(nonterminal_linear.empty()) << "rank=" << rank;

    }
}

TEST(KVCacheTransferPlannerTest, FullPageRRPreservesFirstFullBlockAndHalfOpenRange) {
    constexpr int kCpSize     = 8;
    const auto    non_chunked = buildCacheStoreBlockPlan(/*total_logical_blocks=*/17,
                                                      /*first_full_block=*/9,
                                                      /*use_hybrid=*/true,
                                                      CacheGroupType::FULL,
                                                      /*cp_rank=*/0,
                                                      kCpSize);
    EXPECT_EQ(asPairs(non_chunked), (std::vector<Pair>{{16, 2}}));

    const auto incremental = buildIncrementalCacheStoreBlockPlan(
        /*total_logical_blocks=*/17,
        /*reuse_block_size=*/0,
        /*use_hybrid=*/true,
        CacheGroupType::FULL,
        /*cp_rank=*/0,
        kCpSize,
        CacheStorePublishRange{/*begin_block=*/8, /*end_block=*/16, /*terminal=*/false});
    EXPECT_EQ(asPairs(incremental), (std::vector<Pair>{{8, 1}}));
}

TEST(KVCacheTransferPlannerTest, RejectsInvalidRangeAndUnsupportedGroup) {
    EXPECT_THROW(buildIncrementalCacheStoreBlockPlan(
                     4, 0, true, CacheGroupType::FULL, 0, 1, CacheStorePublishRange{3, 2, false}),
                 std::invalid_argument);
    EXPECT_THROW(buildIncrementalCacheStoreBlockPlan(
                     4, 0, true, CacheGroupType::SWA, 0, 1, CacheStorePublishRange{0, 2, false}),
                 std::invalid_argument);
}

}  // namespace
}  // namespace rtp_llm
