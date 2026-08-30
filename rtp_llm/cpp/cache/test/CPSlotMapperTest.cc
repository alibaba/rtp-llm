#include <gtest/gtest.h>
#include <algorithm>
#include <stdexcept>
#include "rtp_llm/cpp/cache/CPSlotMapper.h"
#include "rtp_llm/cpp/cache/CacheConfig.h"
#include "rtp_llm/cpp/cache/KVCacheTransferPlanner.h"
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

    auto full_spec                       = std::make_shared<MHAKVCacheSpec>();
    full_spec->seq_size_per_block        = 8;
    full_spec->kernel_seq_size_per_block = 8;
    CacheGroup full_group;
    full_group.tag               = "full";
    full_group.spec              = full_spec;
    full_group.policy            = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full_group.policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    full_group.policy.cp_slice   = CpBlockSliceMode::EQUAL_BYTES;

    auto swa_spec                       = std::make_shared<MHAKVCacheSpec>();
    swa_spec->seq_size_per_block        = 8;
    swa_spec->kernel_seq_size_per_block = 8;
    CacheGroup swa_group;
    swa_group.tag             = "swa";
    swa_group.spec            = swa_spec;
    swa_group.policy          = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_group.policy.cp_slice = CpBlockSliceMode::EQUAL_BYTES;
    config = CacheConfig({std::move(full_group), std::move(swa_group)}, {{"full", "swa"}}, /*main_layer_num=*/1);

    CPSlotMapper mapper(0, 2, 8);

    EXPECT_EQ(mapper.layoutForGroup(config, "full").mapping, CpBlockMappingMode::BLOCK_ROUND_ROBIN);
    EXPECT_EQ(mapper.layoutForGroup(config, "full").slice, CpBlockSliceMode::NONE);
    EXPECT_EQ(mapper.layoutForGroup(config, "swa").slice, CpBlockSliceMode::EQUAL_BYTES);
}

TEST_F(CPSlotMapperTest, TaggedGroupsKeepGlobalKeySpanAndGroupPhysicalSpanDistinct) {
    CacheConfig config;
    config.seq_size_per_block = 8;

    auto full_spec                       = std::make_shared<MHAKVCacheSpec>();
    full_spec->seq_size_per_block        = 24;
    full_spec->kernel_seq_size_per_block = 8;
    CacheGroup full_group;
    full_group.tag               = "full";
    full_group.spec              = std::move(full_spec);
    full_group.policy            = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full_group.policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;

    auto swa_spec                       = std::make_shared<MHAKVCacheSpec>();
    swa_spec->seq_size_per_block        = 32;
    swa_spec->kernel_seq_size_per_block = 8;
    CacheGroup swa_group;
    swa_group.tag               = "swa";
    swa_group.spec              = std::move(swa_spec);
    swa_group.policy            = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa_group.policy.cp_mapping = CpBlockMappingMode::COMPACT_LAST_RANK;
    config = CacheConfig({std::move(full_group), std::move(swa_group)}, {{"full", "swa"}}, /*main_layer_num=*/1);

    CPSlotMapper mapper(/*cp_rank=*/1, /*cp_size=*/2, /*global key B=*/8);

    EXPECT_TRUE(mapper.blockRoundRobinGroup(config, "full"));
    EXPECT_EQ(config.group("full").seqSizePerBlock(), 24u);
    EXPECT_EQ(mapper.logicalSeqSizePerBlock(config, "full"), 16u);
    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(config, "full", 33), 24);

    EXPECT_TRUE(mapper.compactLastRankGroup(config, "swa"));
    EXPECT_EQ(mapper.logicalSeqSizePerBlock(config, "swa"), 32u);
    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(config, "swa", 33), 33);
}

TEST_F(CPSlotMapperTest, TaggedGroupMethodsRejectMissingIdentity) {
    CacheConfig config;
    config.seq_size_per_block = 8;

    auto spec                       = std::make_shared<MHAKVCacheSpec>();
    spec->seq_size_per_block        = 8;
    spec->kernel_seq_size_per_block = 8;
    CacheGroup group;
    group.tag    = "full";
    group.spec   = std::move(spec);
    group.policy = defaultCacheGroupPolicy(CacheGroupType::FULL);
    config       = CacheConfig({std::move(group)}, {{"full"}}, /*main_layer_num=*/1);

    CPSlotMapper mapper(0, 2, 8);
    EXPECT_ANY_THROW(mapper.layoutForGroup(config, ""));
    EXPECT_ANY_THROW(mapper.layoutForGroup(config, "missing"));
    EXPECT_ANY_THROW(mapper.buildStorePlan(config, "missing", 2, 0, true));
    EXPECT_ANY_THROW(mapper.sliceBlockForPeer(config, "missing", {}, 0));
}

TEST_F(CPSlotMapperTest, TransferPlannerReturnsDirectTailPositions) {
    EXPECT_EQ(blockPositionsForCacheTransfer(/*block_num=*/8,
                                             /*reuse_block_size=*/0,
                                             /*use_hybrid=*/true,
                                             /*transfer_tail_blocks=*/true,
                                             /*tail_block_count=*/2,
                                             /*hybrid_full_from_begin=*/true),
              (std::vector<size_t>{6, 7}));
}

TEST_F(CPSlotMapperTest, ConnectorProjectionPreservesSelectedTimelineIncludingDummyTail) {
    CacheConfig config;
    config.seq_size_per_block = 8;

    auto full_spec                       = std::make_shared<MHAKVCacheSpec>();
    full_spec->seq_size_per_block        = 8;
    full_spec->kernel_seq_size_per_block = 8;
    CacheGroup full_group;
    full_group.tag               = "full";
    full_group.spec              = full_spec;
    full_group.policy            = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full_group.policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;
    config                       = CacheConfig({std::move(full_group)}, {{"full"}}, /*main_layer_num=*/1);

    const CacheKeysType         full_keys{10, 11, 12, 13, 14};
    const BlockDependenciesType full_dependencies{
        BlockDependency{true, 900, 7},
        BlockDependency{true, 901, 13},
        BlockDependency{true, 11, 21},
        BlockDependency{true, 777, 34},
        BlockDependency{true, 13, 55},
    };
    KVCacheResource source;
    source.initGroups(config);
    source.setCacheKeysAndBlockDependencies(full_keys, full_dependencies);
    source.setLastBlockAligned(false);
    source.mutableBlockIds("full").assign(BlockIndicesType{100, 101, 102, 103, 104});

    CPSlotMapper mapper(/*cp_rank=*/1, /*cp_size=*/2, /*block_size=*/8);
    auto         projected = mapper.projectConnectorResource(source, config, mapper.canonicalCacheKeys(full_keys));

    EXPECT_EQ(projected.cacheKeys(), (CacheKeysType{11, 13, 14}));
    ASSERT_EQ(projected.blockDependencies().size(), 3u);
    EXPECT_EQ(projected.blockDependencies()[0].parent_key, 901);
    EXPECT_EQ(projected.blockDependencies()[0].ordinal, 13u);
    EXPECT_EQ(projected.blockDependencies()[1].parent_key, 777);
    EXPECT_EQ(projected.blockDependencies()[1].ordinal, 34u);
    EXPECT_EQ(projected.blockDependencies()[2].parent_key, 13);
    EXPECT_EQ(projected.blockDependencies()[2].ordinal, 55u);
    EXPECT_FALSE(projected.lastBlockAligned());
}

TEST_F(CPSlotMapperTest, ConnectorProjectionUsesTagMappedBlocks) {
    CacheConfig config;
    config.seq_size_per_block = 8;

    auto full_spec                       = std::make_shared<MHAKVCacheSpec>();
    full_spec->seq_size_per_block        = 8;
    full_spec->kernel_seq_size_per_block = 8;
    CacheGroup full;
    full.tag               = "full";
    full.spec              = std::move(full_spec);
    full.policy            = defaultCacheGroupPolicy(CacheGroupType::FULL);
    full.policy.cp_mapping = CpBlockMappingMode::BLOCK_ROUND_ROBIN;

    auto swa_spec                       = std::make_shared<MHAKVCacheSpec>();
    swa_spec->seq_size_per_block        = 8;
    swa_spec->kernel_seq_size_per_block = 8;
    CacheGroup swa;
    swa.tag               = "swa";
    swa.spec              = std::move(swa_spec);
    swa.policy            = defaultCacheGroupPolicy(CacheGroupType::SWA);
    swa.policy.cp_mapping = CpBlockMappingMode::COMPACT_LAST_RANK;
    config                = CacheConfig({std::move(full), std::move(swa)}, {{"full", "swa"}}, /*main_layer_num=*/1);

    KVCacheResource source;
    source.initGroups(config);
    source.setCacheKeys({10, 11, 12, 13});
    source.setLastBlockAligned(true);
    source.mutableBlockIds("full").assign({100, 101, 102, 103});
    source.mutableBlockIds("swa").assign({200, 201, 202, 203});
    CPSlotMapper mapper(/*cp_rank=*/1, /*cp_size=*/2, /*global key B=*/8);
    auto projected = mapper.projectConnectorResource(source, config, mapper.canonicalCacheKeys(source.cacheKeys()));

    EXPECT_EQ(projected.blocks("full"), (BlockIndicesType{101, 103}));
    EXPECT_EQ(projected.blocks("swa"), (BlockIndicesType{201, 203}));
}

}  // namespace test
}  // namespace rtp_llm
