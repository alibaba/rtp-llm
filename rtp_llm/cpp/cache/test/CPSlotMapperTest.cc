#include <gtest/gtest.h>
#include <stdexcept>
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
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

TEST_F(CPSlotMapperTest, BuildStorePlanUsesPolicyActiveTailBlocks) {
    CPSlotMapper mapper(0, 2, 4);

    auto default_swa = mapper.buildStorePlan(CacheGroupType::SWA,
                                             /*total_logical_blocks=*/5,
                                             /*reuse_block_size=*/0,
                                             /*use_hybrid=*/true);
    ASSERT_EQ(default_swa.size(), 2);
    EXPECT_EQ(default_swa[0].key_index, 3);
    EXPECT_EQ(default_swa[0].offset_index, 1);
    EXPECT_EQ(default_swa[1].key_index, 4);
    EXPECT_EQ(default_swa[1].offset_index, 2);

    CacheGroupPolicy policy   = defaultCacheGroupPolicy(CacheGroupType::SWA);
    policy.active_tail_blocks = 1;
    auto custom_swa           = mapper.buildStorePlan(policy,
                                            /*total_logical_blocks=*/5,
                                            /*reuse_block_size=*/0,
                                            /*use_hybrid=*/true);
    ASSERT_EQ(custom_swa.size(), 1);
    EXPECT_EQ(custom_swa[0].key_index, 4);
    EXPECT_EQ(custom_swa[0].offset_index, 2);
}

TEST_F(CPSlotMapperTest, FullGroupIgnoresByteSlicePolicy) {
    CacheConfig config;
    config.seq_size_per_block = 8;
    config.layer_num          = 1;

    auto full_spec = std::make_shared<MHAKVCacheSpec>(8, 8);
    full_spec->tag = "full";
    GroupBase full_group;
    full_group.tag               = full_spec->tag;
    full_group.spec              = full_spec;
    full_group.layer_ids         = {0};
    full_group.policy            = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full_group.policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    full_group.policy.cp_slice   = CpBlockSliceMode::EQUAL_BYTES;

    auto swa_spec = std::make_shared<MHAKVCacheSpec>(8, 8);
    swa_spec->tag = "swa";
    GroupBase swa_group;
    swa_group.tag             = swa_spec->tag;
    swa_group.spec            = swa_spec;
    swa_group.layer_ids       = {0};
    swa_group.policy          = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_group.policy.cp_slice = CpBlockSliceMode::EQUAL_BYTES;
    config.setTopology({std::move(full_group), std::move(swa_group)}, {{0, {"full", "swa"}}});

    CPSlotMapper mapper(0, 2, 8);

    EXPECT_EQ(mapper.layoutForGroup(config, "full").mapping, CpBlockMappingMode::BLOCK_ROUND_ROBIN);
    EXPECT_EQ(mapper.layoutForGroup(config, "full").slice, CpBlockSliceMode::NONE);
    EXPECT_EQ(mapper.layoutForGroup(config, "swa").slice, CpBlockSliceMode::EQUAL_BYTES);
}

TEST_F(CPSlotMapperTest, ConnectorProjectionUsesRawKeysWithoutMutatingSourceTimeline) {
    CacheConfig config;
    config.seq_size_per_block = 8;
    config.layer_num          = 1;

    auto full_spec = std::make_shared<MHAKVCacheSpec>(8, 8);
    full_spec->tag = "full";
    GroupBase full_group;
    full_group.tag               = full_spec->tag;
    full_group.spec              = full_spec;
    full_group.layer_ids         = {0};
    full_group.policy            = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full_group.policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    config.setTopology({std::move(full_group)}, {{0, {"full"}}});

    KVCacheResource source;
    source.initGroups(config.topologyPtr());
    source.setCacheKeys({10, 20, 30});
    source.mutableBlockIds("full").assign(BlockIndicesType{100, 101, 102});
    source.setLastBlockAligned(false);
    const auto source_dependencies = source.blockDependencies();

    CPSlotMapper mapper(/*cp_rank=*/1, /*cp_size=*/2, /*block_size=*/8);
    auto         projected = mapper.projectConnectorResource(source, config, CacheKeysType{20});

    // Key 20 is selected from the raw request timeline. Key 30 is only the transfer-view dummy partial tail.
    EXPECT_EQ(projected.cacheKeys(), (CacheKeysType{20, 30}));
    EXPECT_EQ(projected.blocks("full"), (BlockIndicesType{101}));
    EXPECT_FALSE(projected.lastBlockAligned());

    EXPECT_EQ(source.cacheKeys(), (CacheKeysType{10, 20, 30}));
    EXPECT_EQ(source.blocks("full"), (BlockIndicesType{100, 101, 102}));
    EXPECT_EQ(source.blockDependencies().size(), source_dependencies.size());
    for (size_t i = 0; i < source_dependencies.size(); ++i) {
        EXPECT_EQ(source.blockDependencies()[i].has_parent, source_dependencies[i].has_parent);
        EXPECT_EQ(source.blockDependencies()[i].parent_key, source_dependencies[i].parent_key);
        EXPECT_EQ(source.blockDependencies()[i].ordinal, source_dependencies[i].ordinal);
    }
    EXPECT_FALSE(source.lastBlockAligned());
}

}  // namespace test
}  // namespace rtp_llm
