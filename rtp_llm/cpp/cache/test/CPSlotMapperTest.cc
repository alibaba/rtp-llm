#include <gtest/gtest.h>
#include <stdexcept>
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheSpecDesc.h"
#include "rtp_llm/cpp/cache/MHAKVCacheSpec.h"

namespace rtp_llm {
namespace test {

class CPSlotMapperTest: public ::testing::Test {};

TEST_F(CPSlotMapperTest, DefaultConstructorIsNotSharded) {
    CPSlotMapper mapper;
    EXPECT_FALSE(mapper.isSharded());  // cp_size=1 → not sharded
    EXPECT_EQ(mapper.cpRank(), 0);
    EXPECT_EQ(mapper.cpSize(), 1);
    EXPECT_EQ(mapper.blockSize(), 1);
    EXPECT_EQ(mapper.virtualBlockSize(), 1);
}

TEST_F(CPSlotMapperTest, SingleRankIsNotSharded) {
    CPSlotMapper mapper(0, 1, 32);
    EXPECT_FALSE(mapper.isSharded());  // cp_size=1 → not sharded
}

TEST_F(CPSlotMapperTest, MultiRankIsSharded) {
    CPSlotMapper mapper(0, 2, 32);
    EXPECT_TRUE(mapper.isSharded());           // cp_size=2 → sharded
    EXPECT_EQ(mapper.virtualBlockSize(), 64);  // block_size * cp_size
}

TEST_F(CPSlotMapperTest, RejectsInvalidGeometry) {
    EXPECT_THROW(CPSlotMapper(0, 0, 32), std::invalid_argument);
    EXPECT_THROW(CPSlotMapper(0, 2, 0), std::invalid_argument);
    EXPECT_THROW(CPSlotMapper(-1, 2, 32), std::invalid_argument);
    EXPECT_THROW(CPSlotMapper(2, 2, 32), std::invalid_argument);
}

TEST_F(CPSlotMapperTest, LocalBlockCount) {
    const int block_size = 4;

    // cp_size=2: localBlockCount = ceil(total_blocks / cp_size), same for all ranks
    CPSlotMapper rank0(0, 2, block_size);
    CPSlotMapper rank1(1, 2, block_size);

    // seq_len=0: 0 total blocks -> 0
    EXPECT_EQ(rank0.localBlockCount(0), 0);
    EXPECT_EQ(rank1.localBlockCount(0), 0);

    // seq_len=4: 1 total block -> ceil(1/2)=1
    EXPECT_EQ(rank0.localBlockCount(4), 1);
    EXPECT_EQ(rank1.localBlockCount(4), 1);

    // seq_len=8: 2 total blocks -> ceil(2/2)=1
    EXPECT_EQ(rank0.localBlockCount(8), 1);
    EXPECT_EQ(rank1.localBlockCount(8), 1);

    // seq_len=12: 3 total blocks -> ceil(3/2)=2
    EXPECT_EQ(rank0.localBlockCount(12), 2);
    EXPECT_EQ(rank1.localBlockCount(12), 2);

    // seq_len=16: 4 total blocks -> ceil(4/2)=2
    EXPECT_EQ(rank0.localBlockCount(16), 2);
    EXPECT_EQ(rank1.localBlockCount(16), 2);

    // seq_len=5: 2 total blocks -> ceil(2/2)=1
    EXPECT_EQ(rank0.localBlockCount(5), 1);
    EXPECT_EQ(rank1.localBlockCount(5), 1);
}

TEST_F(CPSlotMapperTest, LocalBlockCountFourRanks) {
    // seq_len=55, block_size=8, cp_size=4
    // total_blocks = ceil(55/8) = 7, localBlockCount = ceil(7/4) = 2
    // All ranks get 2 — rank3 has 1 unused trailing block
    const int block_size = 8;
    const int cp_size    = 4;

    for (int r = 0; r < cp_size; ++r) {
        CPSlotMapper mapper(r, cp_size, block_size);
        EXPECT_EQ(mapper.localBlockCount(55), 2) << "rank=" << r;
    }
}

TEST_F(CPSlotMapperTest, EffectiveSeqLenForAllocIsRankIndependent) {
    const int    block_size = 4;
    CPSlotMapper rank0(0, 2, block_size);
    CPSlotMapper rank1(1, 2, block_size);

    // effectiveSeqLenForAlloc = ceil(total_blocks / cp_size) * block_size
    // This is rank-independent — always allocates max across all ranks.
    EXPECT_EQ(rank0.effectiveSeqLenForAlloc(0), 0);
    EXPECT_EQ(rank0.effectiveSeqLenForAlloc(4), 4);   // ceil(1/2)=1 block * 4
    EXPECT_EQ(rank0.effectiveSeqLenForAlloc(8), 4);   // ceil(2/2)=1 block * 4
    EXPECT_EQ(rank0.effectiveSeqLenForAlloc(12), 8);  // ceil(3/2)=2 blocks * 4
    EXPECT_EQ(rank0.effectiveSeqLenForAlloc(16), 8);  // ceil(4/2)=2 blocks * 4

    // Same results for rank1 — rank-independent
    EXPECT_EQ(rank1.effectiveSeqLenForAlloc(0), 0);
    EXPECT_EQ(rank1.effectiveSeqLenForAlloc(4), 4);
    EXPECT_EQ(rank1.effectiveSeqLenForAlloc(8), 4);
    EXPECT_EQ(rank1.effectiveSeqLenForAlloc(12), 8);
    EXPECT_EQ(rank1.effectiveSeqLenForAlloc(16), 8);
}

TEST_F(CPSlotMapperTest, EffectiveSeqLenFourRanks) {
    // seq_len=55, block_size=8, cp_size=4
    // total_blocks=7, ceil(7/4)=2, effective=16
    // All ranks get the same value
    const int block_size = 8;
    const int cp_size    = 4;

    for (int r = 0; r < cp_size; ++r) {
        CPSlotMapper mapper(r, cp_size, block_size);
        EXPECT_EQ(mapper.effectiveSeqLenForAlloc(55), 16) << "rank=" << r;
    }
}

TEST_F(CPSlotMapperTest, NonShardedPassthrough) {
    CPSlotMapper mapper;  // cp_size=1, block_size=1

    EXPECT_EQ(mapper.localBlockCount(10), 10);
    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(10), 10);
}

TEST_F(CPSlotMapperTest, BuildStorePlanNormalMappingKeepsLogicalIndices) {
    CPSlotMapper     mapper(0, 2, 4);
    CacheGroupPolicy policy = defaultCacheGroupPolicy(CacheGroupType::LINEAR);
    policy.cp_mapping       = CpBlockMappingMode::NONE;

    const auto plan = mapper.buildStorePlan(policy,
                                            /*total_logical_blocks=*/5,
                                            /*reuse_block_size=*/2,
                                            /*use_hybrid=*/false);

    ASSERT_EQ(plan.size(), 3);
    EXPECT_EQ(plan[0].cache_key_index, 2);
    EXPECT_EQ(plan[0].block_table_index, 2);
    EXPECT_EQ(plan[1].cache_key_index, 3);
    EXPECT_EQ(plan[1].block_table_index, 3);
    EXPECT_EQ(plan[2].cache_key_index, 4);
    EXPECT_EQ(plan[2].block_table_index, 4);
}

TEST_F(CPSlotMapperTest, BuildStorePlanRoundRobinMapsToRankLocalIndices) {
    CPSlotMapper     mapper(/*cp_rank=*/1, /*cp_size=*/2, /*block_size=*/4);
    CacheGroupPolicy policy = defaultCacheGroupPolicy(CacheGroupType::FULL);

    const auto plan = mapper.buildStorePlan(policy,
                                            /*total_logical_blocks=*/6,
                                            /*reuse_block_size=*/0,
                                            /*use_hybrid=*/false);

    ASSERT_EQ(plan.size(), 3);
    EXPECT_EQ(plan[0].cache_key_index, 1);
    EXPECT_EQ(plan[0].block_table_index, 0);
    EXPECT_EQ(plan[1].cache_key_index, 3);
    EXPECT_EQ(plan[1].block_table_index, 1);
    EXPECT_EQ(plan[2].cache_key_index, 5);
    EXPECT_EQ(plan[2].block_table_index, 2);
}

TEST_F(CPSlotMapperTest, BuildStorePlanUsesPolicyActiveTailBlocks) {
    CPSlotMapper mapper(0, 2, 4);

    auto default_swa = mapper.buildStorePlan(CacheGroupType::SWA,
                                             /*total_logical_blocks=*/5,
                                             /*reuse_block_size=*/0,
                                             /*use_hybrid=*/true);
    ASSERT_EQ(default_swa.size(), 2);
    EXPECT_EQ(default_swa[0].cache_key_index, 3);
    EXPECT_EQ(default_swa[0].block_table_index, 1);
    EXPECT_EQ(default_swa[1].cache_key_index, 4);
    EXPECT_EQ(default_swa[1].block_table_index, 2);

    CacheGroupPolicy policy   = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.active_tail_blocks = 1;
    auto custom_swa           = mapper.buildStorePlan(policy,
                                            /*total_logical_blocks=*/5,
                                            /*reuse_block_size=*/0,
                                            /*use_hybrid=*/true);
    ASSERT_EQ(custom_swa.size(), 1);
    EXPECT_EQ(custom_swa[0].cache_key_index, 4);
    EXPECT_EQ(custom_swa[0].block_table_index, 2);
}

TEST_F(CPSlotMapperTest, FullGroupDoesNotSliceWhileSwaUsesSpecMetadata) {
    CacheConfig config;
    config.seq_size_per_block = 8;
    config.layer_num          = 1;
    config.layer_all_num      = 1;

    KVCacheSpecDesc full_desc;
    full_desc.tag                  = "full";
    full_desc.cache_type           = KVCacheSpecType::CompressedKVCache;
    full_desc.entry_elems          = 1;
    full_desc.entry_dtype          = DataType::TYPE_UINT8;
    full_desc.entry_count_mode     = BlockEntryCountMode::EXPLICIT;
    full_desc.explicit_entry_count = 2;
    full_desc.cp                   = CacheCpPolicyDesc{};
    full_desc.cp->slice            = true;
    ParallelismConfig full_parallelism_config;
    SpecBuildContext full_build_ctx;
    full_build_ctx.parallelism_config = &full_parallelism_config;
    auto             full_spec = SpecBuilder::build(full_desc, full_build_ctx);
    ASSERT_TRUE(full_spec->cpSlice());
    GroupBase full_group;
    full_group.tag               = full_spec->tag;
    full_group.spec              = full_spec;
    full_group.layer_ids         = {0};
    full_group.policy            = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full_group.policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;

    KVCacheSpecDesc swa_desc;
    swa_desc.tag                  = "swa";
    swa_desc.cache_type           = KVCacheSpecType::SWAState;
    swa_desc.entry_elems          = 1;
    swa_desc.entry_dtype          = DataType::TYPE_UINT8;
    swa_desc.entry_count_mode     = BlockEntryCountMode::EXPLICIT;
    swa_desc.explicit_entry_count = 2;
    swa_desc.cp                   = CacheCpPolicyDesc{};
    swa_desc.cp->slice            = true;
    ParallelismConfig parallelism_config;
    SpecBuildContext  build_ctx;
    build_ctx.parallelism_config = &parallelism_config;
    auto      swa_spec           = SpecBuilder::build(swa_desc, build_ctx);
    GroupBase swa_group;
    swa_group.tag       = swa_spec->tag;
    swa_group.spec      = swa_spec;
    swa_group.layer_ids = {0};
    swa_group.policy    = defaultCacheGroupPolicy(CacheGroupType::SWA);
    config.setTopology({std::move(full_group), std::move(swa_group)}, {{0, {"full", "swa"}}});

    CPSlotMapper mapper(0, 2, 8);

    EXPECT_EQ(mapper.layoutForGroup(config, 0).mapping, CpBlockMappingMode::BLOCK_ROUND_ROBIN);
    EXPECT_FALSE(mapper.layoutForGroup(config, 0).slice);
    EXPECT_TRUE(mapper.layoutForGroup(config, 1).slice);

    KVCacheResource source;
    source.initGroups(config.topologyPtr());
    source.setCacheKeys({10, 11, 12, 13});
    source.mutableBlockIds(0).assign({100, 101, 102, 103});
    source.mutableBlockIds(1).assign({200, 201, 202, 203});
    source.setLastBlockAligned(true);

    const auto projected = mapper.projectConnectorResource(source, config, {11, 13});
    EXPECT_EQ(projected.blocks(0), (BlockIndicesType{101, 103}));
    EXPECT_EQ(projected.blocks(1), (BlockIndicesType{200, 201}));
}

TEST_F(CPSlotMapperTest, ProjectConnectorResourcePreservesUsableTailWithDummyKey) {
    CacheConfig config;
    auto        spec                = std::make_shared<MHAKVCacheSpec>();
    spec->tag                       = "full";
    spec->seq_size_per_block        = 8;
    GroupBase group;
    group.tag                       = spec->tag;
    group.spec                      = spec;
    group.layer_ids                 = {0};
    group.policy                    = defaultCacheGroupPolicy(CacheGroupType::FULL);
    group.policy.cp_mapping         = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    config.setTopology({std::move(group)}, {{0, {"full"}}});

    KVCacheResource source;
    source.initGroups(config.topologyPtr());
    source.setCacheKeys({10, 11, 12});
    source.mutableBlockIds(0).assign({100, 101, 102});
    source.setLastBlockAligned(false);

    CPSlotMapper mapper(/*cp_rank=*/1, /*cp_size=*/2, /*block_size=*/8);
    const auto   projected = mapper.projectConnectorResource(source, config, {11});
    EXPECT_EQ(projected.blocks(0), (BlockIndicesType{101}));
    EXPECT_EQ(projected.cacheKeys(), (CacheKeysType{11, 12}));
    EXPECT_FALSE(projected.lastBlockAligned());
}

}  // namespace test
}  // namespace rtp_llm
