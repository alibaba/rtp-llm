#include <gtest/gtest.h>
#include "rtp_llm/cpp/cache/CPSlotMapper.h"

#include <set>
#include <vector>

namespace rtp_llm {
namespace test {

class CPSlotMapperTest: public ::testing::Test {};

TEST_F(CPSlotMapperTest, DefaultConstructorIsNotSharded) {
    CPSlotMapper mapper;
    EXPECT_FALSE(mapper.isSharded());
    EXPECT_EQ(mapper.cpRank(), 0);
    EXPECT_EQ(mapper.cpSize(), 1);
    EXPECT_EQ(mapper.blockSize(), 1);
    EXPECT_EQ(mapper.virtualBlockSize(), 1);
}

TEST_F(CPSlotMapperTest, SingleRankIsNotSharded) {
    CPSlotMapper mapper(0, 1, 32);
    EXPECT_FALSE(mapper.isSharded());
    EXPECT_TRUE(mapper.isOwned(0));
    EXPECT_TRUE(mapper.isOwned(63));
}

TEST_F(CPSlotMapperTest, BasicTokenInterleaving) {
    CPSlotMapper rank0(0, 2, 32);
    CPSlotMapper rank1(1, 2, 32);

    EXPECT_TRUE(rank0.isSharded());
    EXPECT_TRUE(rank1.isSharded());

    EXPECT_EQ(rank0.virtualBlockSize(), 64);
    EXPECT_EQ(rank1.virtualBlockSize(), 64);

    // Token 0 -> rank 0, Token 1 -> rank 1, Token 2 -> rank 0, ...
    EXPECT_TRUE(rank0.isOwned(0));
    EXPECT_FALSE(rank0.isOwned(1));
    EXPECT_TRUE(rank0.isOwned(2));
    EXPECT_FALSE(rank0.isOwned(3));

    EXPECT_FALSE(rank1.isOwned(0));
    EXPECT_TRUE(rank1.isOwned(1));
    EXPECT_FALSE(rank1.isOwned(2));
    EXPECT_TRUE(rank1.isOwned(3));
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

TEST_F(CPSlotMapperTest, UniqueSlots) {
    const int cp_size    = 2;
    const int block_size = 4;
    const int seq_len    = 32;

    CPSlotMapper mapper(0, cp_size, block_size);

    std::set<int64_t> slots;
    int               vblocks = mapper.virtualBlockCount(seq_len);
    for (int vb = 0; vb < vblocks; ++vb) {
        int physical_block_id = vb + 100;  // arbitrary physical block ids
        int vbs               = mapper.virtualBlockSize();
        for (int pos = vb * vbs; pos < std::min((vb + 1) * vbs, seq_len); ++pos) {
            if (mapper.isOwned(pos)) {
                int64_t slot = mapper.computeSlot(pos, physical_block_id);
                EXPECT_TRUE(slots.insert(slot).second) << "Duplicate slot " << slot << " at position " << pos;
            }
        }
    }
}

TEST_F(CPSlotMapperTest, LocalBlockCountEqualsVirtualBlockCount) {
    CPSlotMapper mapper(0, 2, 32);
    for (int seq_len : {0, 1, 32, 63, 64, 65, 128, 129}) {
        EXPECT_EQ(mapper.localBlockCount(seq_len), mapper.virtualBlockCount(seq_len)) << "seq_len=" << seq_len;
    }
}

TEST_F(CPSlotMapperTest, VirtualBlockCountExamples) {
    CPSlotMapper mapper(0, 2, 32);  // virtual_block_size = 64

    EXPECT_EQ(mapper.virtualBlockCount(0), 0);
    EXPECT_EQ(mapper.virtualBlockCount(1), 1);
    EXPECT_EQ(mapper.virtualBlockCount(64), 1);
    EXPECT_EQ(mapper.virtualBlockCount(65), 2);
    EXPECT_EQ(mapper.virtualBlockCount(128), 2);
}

TEST_F(CPSlotMapperTest, CrossRequestConsistency) {
    const int cp_size    = 4;
    const int block_size = 16;

    for (int r = 0; r < cp_size; ++r) {
        CPSlotMapper mapper(r, cp_size, block_size);
        for (int pos = 0; pos < 256; ++pos) {
            bool expected = (pos % mapper.virtualBlockSize()) % cp_size == r;
            EXPECT_EQ(mapper.isOwned(pos), expected) << "rank=" << r << " pos=" << pos;
        }
    }
}

TEST_F(CPSlotMapperTest, EffectiveSeqLenForAlloc) {
    CPSlotMapper mapper(0, 2, 32);  // virtual_block_size = 64

    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(0), 0);
    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(1), 32);    // ceil(1/64) = 1 block * 32
    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(64), 32);   // ceil(64/64) = 1 block * 32
    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(65), 64);   // ceil(65/64) = 2 blocks * 32
    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(128), 64);  // ceil(128/64) = 2 blocks * 32
}

TEST_F(CPSlotMapperTest, OwnedPositions) {
    CPSlotMapper mapper(0, 2, 4);  // virtual_block_size = 8
    auto         owned = mapper.ownedPositions(0, 8);
    EXPECT_EQ(owned, (std::vector<int>{0, 2, 4, 6}));

    auto owned1 = CPSlotMapper(1, 2, 4).ownedPositions(0, 8);
    EXPECT_EQ(owned1, (std::vector<int>{1, 3, 5, 7}));
}

TEST_F(CPSlotMapperTest, OwnedBlockIndicesReturnAll) {
    CPSlotMapper mapper(0, 2, 32);
    auto         indices = mapper.ownedBlockIndices(5);
    EXPECT_EQ(indices, (std::vector<int>{0, 1, 2, 3, 4}));
    EXPECT_EQ(mapper.ownedBlockCount(5), 5);
}

TEST_F(CPSlotMapperTest, NonShardedPassthrough) {
    CPSlotMapper mapper;

    EXPECT_EQ(mapper.targetRank(42), 0);
    EXPECT_TRUE(mapper.isOwned(42));
    EXPECT_EQ(mapper.localBlockOffset(5), 0);  // block_size=1, offset=5%1=0
    EXPECT_EQ(mapper.virtualBlockCount(10), 10);
    EXPECT_EQ(mapper.localBlockCount(10), 10);
    EXPECT_EQ(mapper.effectiveSeqLenForAlloc(10), 10);
}

}  // namespace test
}  // namespace rtp_llm
