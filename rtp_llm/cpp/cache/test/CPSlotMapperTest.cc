#include <gtest/gtest.h>
#include "rtp_llm/cpp/cache/CPSlotMapper.h"

#include <vector>

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
    EXPECT_TRUE(mapper.isOwned(0));
    EXPECT_TRUE(mapper.isOwned(63));
}

TEST_F(CPSlotMapperTest, MultiRankIsSharded) {
    CPSlotMapper mapper(0, 2, 32);
    EXPECT_TRUE(mapper.isSharded());  // cp_size=2 → sharded
}

TEST_F(CPSlotMapperTest, PageLevelTargetRank) {
    const int    block_size = 4;
    const int    cp_size    = 2;
    CPSlotMapper rank0(0, cp_size, block_size);
    CPSlotMapper rank1(1, cp_size, block_size);

    EXPECT_TRUE(rank0.isSharded());
    EXPECT_TRUE(rank1.isSharded());

    EXPECT_EQ(rank0.virtualBlockSize(), 8);  // block_size * cp_size

    // Block 0 (pos 0-3) -> rank 0
    // Block 1 (pos 4-7) -> rank 1
    // Block 2 (pos 8-11) -> rank 0
    // Block 3 (pos 12-15) -> rank 1
    for (int pos = 0; pos < 4; ++pos) {
        EXPECT_EQ(rank0.targetRank(pos), 0) << "pos=" << pos;
        EXPECT_TRUE(rank0.isOwned(pos)) << "pos=" << pos;
        EXPECT_FALSE(rank1.isOwned(pos)) << "pos=" << pos;
    }
    for (int pos = 4; pos < 8; ++pos) {
        EXPECT_EQ(rank0.targetRank(pos), 1) << "pos=" << pos;
        EXPECT_FALSE(rank0.isOwned(pos)) << "pos=" << pos;
        EXPECT_TRUE(rank1.isOwned(pos)) << "pos=" << pos;
    }
    for (int pos = 8; pos < 12; ++pos) {
        EXPECT_TRUE(rank0.isOwned(pos)) << "pos=" << pos;
    }
    for (int pos = 12; pos < 16; ++pos) {
        EXPECT_TRUE(rank1.isOwned(pos)) << "pos=" << pos;
    }
}

TEST_F(CPSlotMapperTest, EveryTokenOwnedByExactlyOneRank) {
    const int cp_size    = 4;
    const int block_size = 8;
    const int total      = 128;

    std::vector<int> owner(total, -1);
    for (int r = 0; r < cp_size; ++r) {
        CPSlotMapper mapper(r, cp_size, block_size);
        for (int pos = 0; pos < total; ++pos) {
            if (mapper.isOwned(pos)) {
                ASSERT_EQ(owner[pos], -1) << "Token " << pos << " owned by both rank " << owner[pos] << " and " << r;
                owner[pos] = r;
            }
        }
    }
    for (int i = 0; i < total; ++i) {
        EXPECT_GE(owner[i], 0) << "Token " << i << " not owned by any rank";
    }
}

TEST_F(CPSlotMapperTest, LocalBlockOffsetWithinRange) {
    const int    cp_size    = 2;
    const int    block_size = 32;
    CPSlotMapper mapper(0, cp_size, block_size);

    for (int pos = 0; pos < 256; ++pos) {
        if (mapper.isOwned(pos)) {
            int offset = mapper.localBlockOffset(pos);
            EXPECT_GE(offset, 0) << "pos=" << pos;
            EXPECT_LT(offset, block_size) << "pos=" << pos;
        }
    }
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

TEST_F(CPSlotMapperTest, VirtualBlockCountExamples) {
    CPSlotMapper mapper(0, 2, 4);  // virtual_block_size = 8

    EXPECT_EQ(mapper.virtualBlockCount(0), 0);
    EXPECT_EQ(mapper.virtualBlockCount(1), 1);
    EXPECT_EQ(mapper.virtualBlockCount(8), 1);
    EXPECT_EQ(mapper.virtualBlockCount(9), 2);
    EXPECT_EQ(mapper.virtualBlockCount(16), 2);
}

TEST_F(CPSlotMapperTest, NonShardedPassthrough) {
    CPSlotMapper mapper;  // cp_size=1, block_size=1

    EXPECT_EQ(mapper.targetRank(42), 0);
    EXPECT_TRUE(mapper.isOwned(42));
    EXPECT_EQ(mapper.localBlockOffset(5), 0);  // block_size=1, offset=5%1=0
    EXPECT_EQ(mapper.virtualBlockCount(10), 10);
    EXPECT_EQ(mapper.localBlockCount(10), 10);
    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(10), 10);
}

}  // namespace test
}  // namespace rtp_llm
