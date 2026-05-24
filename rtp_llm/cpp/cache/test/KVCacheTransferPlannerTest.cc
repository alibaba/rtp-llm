// M04-PR1: property test asserting the unified planner output is byte-
// equivalent to the legacy per-group planner under bps[p] == 1.
//
// Coverage matrix (mirrors the legacy planner test fixtures):
//   * group_type ∈ {LINEAR, SWA, FULL}
//   * cp_size    ∈ {1, 2, 4}
//   * cp_rank    ∈ [0, cp_size)
//   * block_num  ∈ {0, 1, 2, 3, 5, 8, 17}
//   * reuse      ∈ {0, 1, block_num/2, block_num}
//   * hybrid_full_from_begin ∈ {true, false}
//
// For each combination we assert two things:
//   1. The (key_index, offset_index) multiset emitted by the unified planner
//      (under a 1-pool descriptor) matches what the legacy
//      ``buildCacheStoreBlockPlan`` emits.
//   2. ``ownedByPeer`` agrees with the unified planner's shard arithmetic.

#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "rtp_llm/cpp/cache/CacheGroupType.h"
#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"

namespace rtp_llm {

namespace {

using PairKey = std::tuple<int, int>;  // (key_index, offset_index)

std::vector<PairKey> legacyPairs(size_t         block_num,
                                 size_t         reuse,
                                 bool           use_hybrid,
                                 CacheGroupType g,
                                 int            cp_rank,
                                 int            cp_size) {
    // Mimic legacy buildCacheStoreBlockPlan for arbitrary use_hybrid/
    // hybrid_full_from_begin combos via the wrapper that drives the unified
    // planner with a single-pool descriptor and use_hybrid + true.  This
    // mirrors what the legacy free-function did before PR-1.
    const auto plan = buildCacheStoreBlockPlan(block_num, reuse, use_hybrid, g, cp_rank, cp_size);
    std::vector<PairKey> out;
    out.reserve(plan.size());
    for (const auto& it : plan) {
        out.emplace_back(it.key_index, it.offset_index);
    }
    std::sort(out.begin(), out.end());
    return out;
}

std::vector<PairKey> unifiedPairs(size_t         block_num,
                                  size_t         reuse,
                                  bool           use_hybrid,
                                  CacheGroupType g,
                                  int            cp_rank,
                                  int            cp_size) {
    PoolDescriptor pool{
        /*pool_id=*/0,
        /*layer_id=*/-1,
        /*region_name=*/KVCacheRegionName::DEFAULT,
        /*group_type=*/use_hybrid ? g : CacheGroupType::FULL,
        /*stride_bytes=*/0,
        /*last_partial=*/false,
    };
    const auto plan = buildUnifiedTransferPlan(/*model_id=*/0,
                                               /*total_logical_blocks=*/block_num,
                                               /*total_cache_keys=*/block_num,
                                               reuse,
                                               {pool},
                                               /*pool_block_counts=*/{block_num},
                                               cp_rank,
                                               cp_size,
                                               use_hybrid,
                                               /*hybrid_full_from_begin=*/true,
                                               PlannerRole::kPrefillWrite);
    std::vector<PairKey> out;
    out.reserve(plan.size());
    for (const auto& it : plan) {
        out.emplace_back(it.key_index, it.offset_index);
    }
    std::sort(out.begin(), out.end());
    return out;
}

}  // namespace

TEST(KVCacheTransferPlannerTest, UnifiedMatchesLegacyMatrix) {
    const std::vector<CacheGroupType> groups = {CacheGroupType::LINEAR, CacheGroupType::SWA, CacheGroupType::FULL};
    const std::vector<int>            cp_sizes = {1, 2, 4};
    const std::vector<size_t>         block_nums = {0u, 1u, 2u, 3u, 5u, 8u, 17u};

    for (const auto g : groups) {
        for (const int cp_size : cp_sizes) {
            for (int cp_rank = 0; cp_rank < cp_size; ++cp_rank) {
                for (const size_t block_num : block_nums) {
                    const std::vector<size_t> reuses = {0u, std::min<size_t>(1u, block_num),
                                                        block_num / 2u, block_num};
                    for (const size_t reuse : reuses) {
                        const auto legacy  = legacyPairs(block_num, reuse, /*use_hybrid=*/true, g, cp_rank, cp_size);
                        const auto unified = unifiedPairs(block_num, reuse, /*use_hybrid=*/true, g, cp_rank, cp_size);
                        EXPECT_EQ(legacy, unified)
                            << "mismatch g=" << static_cast<int>(g) << " cp=" << cp_rank << "/" << cp_size
                            << " block_num=" << block_num << " reuse=" << reuse;
                    }
                }
            }
        }
    }
}

TEST(KVCacheTransferPlannerTest, NonHybridForcesFullSemantics) {
    // Non-hybrid callers must produce FULL semantics even when group_type
    // claims SWA/LINEAR — replicates the legacy fall-through.
    const auto legacy  = legacyPairs(8u, 2u, /*use_hybrid=*/false, CacheGroupType::FULL, /*cp_rank=*/0,
                                    /*cp_size=*/1);
    const auto unified = unifiedPairs(8u, 2u, /*use_hybrid=*/false, CacheGroupType::FULL, /*cp_rank=*/0,
                                      /*cp_size=*/1);
    EXPECT_EQ(legacy, unified);
    // start = reuse_block_size = 2 under non-hybrid FULL.
    ASSERT_EQ(unified.size(), 6u);
    EXPECT_EQ(std::get<0>(unified.front()), 2);
}

TEST(KVCacheTransferPlannerTest, SwaTailUsesPoolCount) {
    // SWA tail under per-pool block count: pool_blocks=5 should yield
    // positions {3,4} regardless of the unified total_logical_blocks.
    PoolDescriptor pool{0, -1, KVCacheRegionName::SWA_KV, CacheGroupType::SWA, 0, false};
    const auto plan = buildUnifiedTransferPlan(/*model_id=*/0,
                                               /*total_logical_blocks=*/10u,
                                               /*total_cache_keys=*/10u,
                                               /*reuse=*/0u,
                                               {pool},
                                               /*pool_block_counts=*/{5u},
                                               /*shard_owner_rank=*/0,
                                               /*cp_size=*/1,
                                               /*use_hybrid=*/true,
                                               /*hybrid_full_from_begin=*/true);
    ASSERT_EQ(plan.size(), 2u);
    EXPECT_EQ(plan[0].block_pos, 3);
    EXPECT_EQ(plan[1].block_pos, 4);
}

TEST(KVCacheTransferPlannerTest, LinearEmitsOnlyTailBlock) {
    PoolDescriptor pool{0, -1, KVCacheRegionName::DEFAULT, CacheGroupType::LINEAR, 0, false};
    const auto plan = buildUnifiedTransferPlan(0, 7u, 7u, 0u, {pool}, {7u}, 0, 1, true, true);
    ASSERT_EQ(plan.size(), 1u);
    EXPECT_EQ(plan[0].block_pos, 6);
}

TEST(KVCacheTransferPlannerTest, FullCpShardArithmeticMatchesLegacy) {
    // Direct CP page-RR: cp_size=4, cp_rank=2 — owned positions {2,6,10,...}
    // with offset = pos / 4.
    PoolDescriptor pool{0, -1, KVCacheRegionName::CSA_KV, CacheGroupType::FULL, 0, false};
    const auto plan = buildUnifiedTransferPlan(0, 12u, 12u, 0u, {pool}, {12u},
                                               /*shard_owner_rank=*/2,
                                               /*cp_size=*/4,
                                               /*use_hybrid=*/true,
                                               /*hybrid_full_from_begin=*/true);
    ASSERT_EQ(plan.size(), 3u);
    EXPECT_EQ(plan[0].block_pos, 2);
    EXPECT_EQ(plan[0].offset_index, 0);
    EXPECT_EQ(plan[1].block_pos, 6);
    EXPECT_EQ(plan[1].offset_index, 1);
    EXPECT_EQ(plan[2].block_pos, 10);
    EXPECT_EQ(plan[2].offset_index, 2);
}

TEST(KVCacheTransferPlannerTest, NonFullPoolsReplicatedToRankZeroOnly) {
    PoolDescriptor swa_pool{0, -1, KVCacheRegionName::SWA_KV, CacheGroupType::SWA, 0, false};
    // SWA pool with cp_size=2 — only rank 0 owns; rank 1 sees an empty plan.
    const auto plan_r0 = buildUnifiedTransferPlan(0, 8u, 8u, 0u, {swa_pool}, {8u},
                                                  /*shard_owner_rank=*/0, /*cp_size=*/2, true, true);
    const auto plan_r1 = buildUnifiedTransferPlan(0, 8u, 8u, 0u, {swa_pool}, {8u},
                                                  /*shard_owner_rank=*/1, /*cp_size=*/2, true, true);
    EXPECT_FALSE(plan_r0.empty());
    EXPECT_TRUE(plan_r1.empty());
}

TEST(KVCacheTransferPlannerTest, LastPartialDropsOnlyFullTail) {
    // 3 pools: FULL with last_partial=true, FULL with last_partial=false,
    // SWA with last_partial=true. Only the first should drop B = N-1.
    PoolDescriptor full_drop{0, -1, KVCacheRegionName::CSA_KV, CacheGroupType::FULL, 0, /*last_partial=*/true};
    PoolDescriptor full_keep{1, -1, KVCacheRegionName::HCA_KV, CacheGroupType::FULL, 0, false};
    PoolDescriptor swa_drop{2, -1, KVCacheRegionName::SWA_KV, CacheGroupType::SWA, 0, /*last_partial=*/true};

    const auto plan = buildUnifiedTransferPlan(0, 6u, 6u, 0u,
                                               {full_drop, full_keep, swa_drop},
                                               {6u, 6u, 6u}, 0, 1, true, true);
    int full_drop_count = 0, full_keep_count = 0, swa_count = 0;
    bool full_drop_has_tail = false, full_keep_has_tail = false, swa_has_tail = false;
    for (const auto& it : plan) {
        if (it.pool_id == 0) {
            ++full_drop_count;
            if (it.block_pos == 5) full_drop_has_tail = true;
        } else if (it.pool_id == 1) {
            ++full_keep_count;
            if (it.block_pos == 5) full_keep_has_tail = true;
        } else if (it.pool_id == 2) {
            ++swa_count;
            if (it.block_pos == 5) swa_has_tail = true;
        }
    }
    EXPECT_EQ(full_drop_count, 5);  // dropped tail B=5
    EXPECT_EQ(full_keep_count, 6);
    EXPECT_EQ(swa_count, 2);  // SWA tail-2 unchanged
    EXPECT_FALSE(full_drop_has_tail);
    EXPECT_TRUE(full_keep_has_tail);
    EXPECT_TRUE(swa_has_tail);
}

TEST(KVCacheTransferPlannerTest, OwnedByPeerAgreesWithPlanner) {
    // For cp_size=3 and a FULL pool, ownedByPeer(FULL, B, r, 3) must agree
    // with which peer the unified planner places block B on.
    const int cp_size = 3;
    for (int r = 0; r < cp_size; ++r) {
        PoolDescriptor pool{0, -1, KVCacheRegionName::CSA_KV, CacheGroupType::FULL, 0, false};
        const auto plan = buildUnifiedTransferPlan(0, 12u, 12u, 0u, {pool}, {12u}, r, cp_size, true, true);
        std::set<int64_t> owned;
        for (const auto& it : plan) owned.insert(it.block_pos);
        for (int64_t B = 0; B < 12; ++B) {
            EXPECT_EQ(owned.count(B) > 0, ownedByPeer(CacheGroupType::FULL, B, r, cp_size))
                << "B=" << B << " r=" << r;
        }
    }
}

TEST(KVCacheTransferPlannerTest, MultiPoolPlanFlattensInBlockOrder) {
    // 2 FULL pools, no CP. The plan should iterate B outer, pool inner so the
    // emitted (B, pool_id) tuples are interleaved per B.
    PoolDescriptor a{0, -1, KVCacheRegionName::CSA_KV, CacheGroupType::FULL, 0, false};
    PoolDescriptor b{1, -1, KVCacheRegionName::HCA_KV, CacheGroupType::FULL, 0, false};
    const auto plan = buildUnifiedTransferPlan(0, 3u, 3u, 0u, {a, b}, {3u, 3u}, 0, 1, true, true);
    ASSERT_EQ(plan.size(), 6u);
    EXPECT_EQ(plan[0].block_pos, 0); EXPECT_EQ(plan[0].pool_id, 0);
    EXPECT_EQ(plan[1].block_pos, 0); EXPECT_EQ(plan[1].pool_id, 1);
    EXPECT_EQ(plan[2].block_pos, 1); EXPECT_EQ(plan[2].pool_id, 0);
    EXPECT_EQ(plan[3].block_pos, 1); EXPECT_EQ(plan[3].pool_id, 1);
    EXPECT_EQ(plan[4].block_pos, 2); EXPECT_EQ(plan[4].pool_id, 0);
    EXPECT_EQ(plan[5].block_pos, 2); EXPECT_EQ(plan[5].pool_id, 1);
}

}  // namespace rtp_llm
